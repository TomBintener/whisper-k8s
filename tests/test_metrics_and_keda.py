#!/usr/bin/env python3
"""
Unit and integration tests for Prometheus metrics engine and KEDA autoscaling.
Verifies thread-safety, 0.0.4 exposition formatting, bridge endpoints,
worker instrumentation, and KEDA manifest validity without external dependencies.
"""

import os
import sys
import time
import json
import shutil
import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT))

import mock_dependencies  # noqa: F401
import metrics
import bridge
import queue_manager
import webhook
import video_transcriber


class TestMetricsEngine(unittest.TestCase):
    """Unit tests for pure Python Prometheus metrics primitives."""

    def setUp(self):
        self.registry = metrics.MetricsRegistry()

    def test_counter_inc_and_get(self):
        c = self.registry.counter("test_counter", "A test counter", ["status", "env"])
        c.inc(1.0, status="ok", env="prod")
        c.inc(2.5, status="ok", env="prod")
        c.inc(4.0, status="err", env="prod")

        self.assertEqual(c.get(status="ok", env="prod"), 3.5)
        self.assertEqual(c.get(status="err", env="prod"), 4.0)
        self.assertEqual(c.get(status="unknown", env="prod"), 0.0)

        with self.assertRaises(ValueError):
            c.inc(-1.0, status="ok", env="prod")

    def test_counter_thread_safety(self):
        c = self.registry.counter("concurrent_counter", "Thread safety test", ["worker"])
        threads = []
        increments_per_thread = 500
        num_threads = 10

        def worker(wid):
            for _ in range(increments_per_thread):
                c.inc(1.0, worker=f"w{wid}")

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for i in range(num_threads):
            self.assertEqual(c.get(worker=f"w{i}"), float(increments_per_thread))

    def test_gauge_set_inc_dec(self):
        g = self.registry.gauge("test_gauge", "A test gauge", ["service"])
        g.set(10.0, service="worker")
        self.assertEqual(g.get(service="worker"), 10.0)

        g.inc(5.0, service="worker")
        self.assertEqual(g.get(service="worker"), 15.0)

        g.dec(3.0, service="worker")
        self.assertEqual(g.get(service="worker"), 12.0)

    def test_gauge_dynamic_callback(self):
        call_count = 0

        def dynamic_eval():
            nonlocal call_count
            call_count += 1
            return [
                ({"state": "idle"}, 2.0),
                ({"state": "busy"}, 8.0),
            ]

        g = self.registry.gauge("dynamic_queue", "Evaluated at scrape time", ["state"], callback=dynamic_eval)
        samples = g.collect()

        self.assertEqual(call_count, 1)
        self.assertEqual(len(samples), 2)
        sample_map = {tuple(sorted(s[1].items())): s[2] for s in samples}
        self.assertEqual(sample_map.get((("state", "busy"),)), 8.0)
        self.assertEqual(sample_map.get((("state", "idle"),)), 2.0)

    def test_histogram_observations_and_buckets(self):
        h = self.registry.histogram(
            "test_duration",
            "Duration histogram",
            ["handler"],
            buckets=[10.0, 50.0, 100.0],
        )
        h.observe(5.0, handler="api")
        h.observe(25.0, handler="api")
        h.observe(75.0, handler="api")
        h.observe(120.0, handler="api")

        samples = h.collect()
        sample_dict = {}
        for s_name, s_lbls, s_val in samples:
            key = (s_name, s_lbls.get("le", "no_le"))
            sample_dict[key] = s_val

        # Bucket cumulative observations
        self.assertEqual(sample_dict[("test_duration_bucket", "10.0")], 1.0)  # 5.0
        self.assertEqual(sample_dict[("test_duration_bucket", "50.0")], 2.0)  # 5.0, 25.0
        self.assertEqual(sample_dict[("test_duration_bucket", "100.0")], 3.0)  # 5.0, 25.0, 75.0
        self.assertEqual(sample_dict[("test_duration_bucket", "+Inf")], 4.0)  # all 4
        self.assertEqual(sample_dict[("test_duration_count", "no_le")], 4.0)
        self.assertAlmostEqual(sample_dict[("test_duration_sum", "no_le")], 225.0)

    def test_prometheus_exposition_rendering(self):
        reg = metrics.MetricsRegistry()
        cnt = reg.counter("http_requests_total", "Total requests", ["path", "code"])
        cnt.inc(5.0, path="/jobs", code="200")
        cnt.inc(1.0, path="/jobs", code="500")

        gauge = reg.gauge("active_tasks", "Active tasks count")
        gauge.set(4.0)

        output = reg.render_prometheus_text()

        self.assertIn("# HELP http_requests_total Total requests\n", output)
        self.assertIn("# TYPE http_requests_total counter\n", output)
        self.assertIn('http_requests_total{code="200",path="/jobs"} 5.0\n', output)
        self.assertIn('http_requests_total{code="500",path="/jobs"} 1.0\n', output)

        self.assertIn("# HELP active_tasks Active tasks count\n", output)
        self.assertIn("# TYPE active_tasks gauge\n", output)
        self.assertIn("active_tasks 4.0\n", output)
        self.assertTrue(output.endswith("\n"))

    def test_label_value_escaping(self):
        self.assertEqual(metrics.escape_label_value('hello"world'), 'hello\\"world')
        self.assertEqual(metrics.escape_label_value('path\\to\\dir'), 'path\\\\to\\\\dir')
        self.assertEqual(metrics.escape_label_value('line1\nline2'), 'line1\\nline2')


class TestBridgeMetricsIntegration(unittest.TestCase):
    """Integration tests for bridge HTTP /metrics endpoint and metric tracking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.videos_dir = self.data_dir / "videos"
        self.subs_dir = self.data_dir / "subs"
        self.queue_dir = self.data_dir / "queue"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.subs_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        self.orig_videos = bridge.VIDEOS_DIR
        self.orig_subs = bridge.SUBS_DIR
        self.orig_data = bridge.DATA_DIR
        bridge.VIDEOS_DIR = str(self.videos_dir)
        bridge.SUBS_DIR = str(self.subs_dir)
        bridge.DATA_DIR = str(self.data_dir)

        # Reset global metrics for clean test state
        metrics.REGISTRY.counter("whisper_jobs_total", "").reset()
        metrics.REGISTRY.counter("whisper_model_cache_events_total", "").reset()
        metrics.REGISTRY.counter("whisper_webhooks_dispatched_total", "").reset()
        metrics.REGISTRY.histogram("whisper_job_duration_seconds", "").reset()

        # Wire test file queue
        self.test_queue = queue_manager.FileTaskQueue(queue_dir=str(self.queue_dir))
        self.orig_get_queue = queue_manager.get_queue
        queue_manager.get_queue = lambda *args, **kwargs: self.test_queue

        # Wire queue collector
        def _collect():
            counts = {
                "pending": float(len([f for f in self.test_queue.pending_dir.iterdir() if f.is_file() and f.suffix == ".json"])),
                "processing": float(len([f for f in self.test_queue.processing_dir.iterdir() if f.is_file() and f.suffix == ".json"])),
                "completed": float(len([f for f in self.test_queue.completed_dir.iterdir() if f.is_file() and f.suffix == ".json"])),
            }
            return [
                ({"queue": "pending"}, counts["pending"]),
                ({"queue": "processing"}, counts["processing"]),
                ({"queue": "completed"}, counts["completed"]),
            ]

        metrics.QUEUE_DEPTH.set_callback(_collect)

    def tearDown(self):
        bridge.VIDEOS_DIR = self.orig_videos
        bridge.SUBS_DIR = self.orig_subs
        bridge.DATA_DIR = self.orig_data
        queue_manager.get_queue = self.orig_get_queue
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_metrics_endpoint_headers_and_content(self):
        resp = bridge.prometheus_metrics()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/plain", resp.media_type)
        self.assertIn("version=0.0.4", resp.media_type)

        content = resp.body.decode("utf-8")
        self.assertIn("# HELP whisper_queue_depth", content)
        self.assertIn("# TYPE whisper_queue_depth gauge", content)
        self.assertIn('whisper_queue_depth{queue="pending"} 0.0', content)

    def test_job_submission_increments_counter_and_queue_depth(self):
        # Create a sample input video
        video_file = self.videos_dir / "sample_lecture.mp4"
        video_file.write_bytes(b"dummy video data")

        # Submit job in pool mode
        req = bridge.CreateJobReq(
            filename="sample_lecture.mp4",
            mode="pool",
            format="vtt",
        )
        post_resp = asyncio.run(bridge.create_job(req))
        job_id = post_resp.jobId

        # Check metrics endpoint reflects submitted job and pending queue depth = 1
        resp = bridge.prometheus_metrics()
        text = resp.body.decode("utf-8")

        self.assertIn('whisper_jobs_total{mode="pool",status="submitted"} 1.0', text)
        self.assertIn('whisper_queue_depth{queue="pending"} 1.0', text)

        # Cancel the job
        del_resp = asyncio.run(bridge.cancel_job(job_id))
        self.assertTrue(del_resp.get("ok"))

        # Check metrics endpoint reflects cancelled job and pending queue depth = 0
        resp2 = bridge.prometheus_metrics()
        text2 = resp2.body.decode("utf-8")
        self.assertIn('whisper_jobs_total{mode="cancelled",status="cancelled"} 1.0', text2)
        self.assertIn('whisper_queue_depth{queue="pending"} 0.0', text2)


class TestWorkerAndWebhookMetrics(unittest.TestCase):
    """Test metrics tracking inside video_transcriber worker daemon and webhook engine."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.videos_dir = Path(self.temp_dir) / "videos"
        self.subs_dir = Path(self.temp_dir) / "subs"
        self.queue_dir = Path(self.temp_dir) / "queue"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.subs_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        metrics.REGISTRY.counter("whisper_jobs_total", "").reset()
        metrics.REGISTRY.counter("whisper_model_cache_events_total", "").reset()
        metrics.REGISTRY.counter("whisper_webhooks_dispatched_total", "").reset()
        metrics.REGISTRY.histogram("whisper_job_duration_seconds", "").reset()
        metrics.ACTIVE_WORKERS.set(0.0, service="pool_worker")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_cache_hit_and_miss_tracking(self):
        video_path = self.videos_dir / "test.mp4"
        video_path.write_bytes(b"dummy")

        spec = {
            "item_id": "test.mp4",
            "videos_dir": str(self.videos_dir),
            "subs_dir": str(self.subs_dir),
            "model": "tiny",
            "backend": "whisper",
            "device": "cpu",
            "mode": "pool",
        }
        model_cache = {}

        with patch("video_transcriber.load_model", return_value=MagicMock()), \
             patch("video_transcriber.transcribe_one", return_value=(self.subs_dir / "test.vtt", "en")):
            # First execution -> Cache miss
            exit_code1 = video_transcriber.process_job(spec=spec, model_cache=model_cache)
            self.assertEqual(exit_code1, 0)
            self.assertEqual(metrics.MODEL_CACHE_EVENTS.get(event="miss", model="tiny", backend="whisper"), 1.0)
            self.assertEqual(metrics.MODEL_CACHE_EVENTS.get(event="hit", model="tiny", backend="whisper"), 0.0)

            # Second execution -> Cache hit
            exit_code2 = video_transcriber.process_job(spec=spec, model_cache=model_cache)
            self.assertEqual(exit_code2, 0)
            self.assertEqual(metrics.MODEL_CACHE_EVENTS.get(event="miss", model="tiny", backend="whisper"), 1.0)
            self.assertEqual(metrics.MODEL_CACHE_EVENTS.get(event="hit", model="tiny", backend="whisper"), 1.0)

            # Check job total and duration histogram
            self.assertEqual(metrics.JOBS_TOTAL.get(status="succeeded", mode="pool"), 2.0)
            h_samples = metrics.JOB_DURATION_SECONDS.collect()
            count_samples = [s for s in h_samples if s[0] == "whisper_job_duration_seconds_count"]
            self.assertTrue(len(count_samples) > 0)
            self.assertEqual(count_samples[0][2], 2.0)

    def test_active_workers_gauge_in_daemon(self):
        q = queue_manager.FileTaskQueue(queue_dir=str(self.queue_dir))
        stop_evt = threading.Event()

        self.assertEqual(metrics.ACTIVE_WORKERS.get(service="pool_worker"), 0.0)

        # Run daemon in thread and verify gauge is 1.0 while active
        daemon_thread = threading.Thread(
            target=video_transcriber.run_worker_daemon,
            kwargs={"queue": q, "stop_event": stop_evt, "poll_interval": 0.05},
        )
        daemon_thread.start()

        # Wait briefly for daemon to initialize
        time.sleep(0.1)
        self.assertEqual(metrics.ACTIVE_WORKERS.get(service="pool_worker"), 1.0)

        # Signal shutdown and join
        stop_evt.set()
        daemon_thread.join(timeout=2.0)

        # Verify gauge returned to 0.0
        self.assertEqual(metrics.ACTIVE_WORKERS.get(service="pool_worker"), 0.0)

    def test_webhook_metrics_success_and_failure(self):
        with patch("urllib.request.urlopen") as mock_urlopen, \
             patch("webhook.validate_webhook_url"):

            # 1. Success test
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            ok = webhook.send_webhook("http://example.com/callback", {"test": 1})
            self.assertTrue(ok)
            self.assertEqual(metrics.WEBHOOKS_DISPATCHED.get(status="success"), 1.0)

            # 2. Failure test
            mock_urlopen.side_effect = Exception("Network down")
            bad = webhook.send_webhook("http://example.com/callback", {"test": 2}, max_retries=1)
            self.assertFalse(bad)
            self.assertEqual(metrics.WEBHOOKS_DISPATCHED.get(status="failure"), 1.0)


class TestKedaAutoscalerManifest(unittest.TestCase):
    """Validate KEDA ScaledObject and kustomization manifests without external dependencies."""

    def setUp(self):
        self.manifest_path = REPO_ROOT / "k8s_scripts" / "worker-keda-autoscaler.yaml"
        self.kustomize_path = REPO_ROOT / "k8s_scripts" / "kustomization.yaml"

    def test_keda_manifest_structure_and_values(self):
        self.assertTrue(self.manifest_path.exists(), f"Missing KEDA manifest: {self.manifest_path}")
        content = self.manifest_path.read_text(encoding="utf-8")

        self.assertIn("apiVersion: keda.sh/v1alpha1", content)
        self.assertIn("kind: ScaledObject", content)
        self.assertIn("name: whisper-worker-autoscaler", content)
        self.assertIn("namespace: whisper", content)

        # Check target Deployment
        self.assertIn("name: whisper-worker-pool", content)
        self.assertIn("kind: Deployment", content)

        # Check autoscaling bounds (scale-to-zero)
        self.assertIn("minReplicaCount: 0", content)
        self.assertIn("maxReplicaCount: 8", content)
        self.assertIn("cooldownPeriod: 300", content)

        # Check trigger configuration
        self.assertIn("type: prometheus", content)
        self.assertIn("metricName: whisper_queue_depth", content)
        self.assertIn('threshold: "2"', content)
        self.assertIn("whisper_queue_depth", content)

    def test_kustomization_references_autoscaler(self):
        self.assertTrue(self.kustomize_path.exists(), f"Missing kustomization: {self.kustomize_path}")
        content = self.kustomize_path.read_text(encoding="utf-8")
        self.assertIn("worker-keda-autoscaler.yaml", content)


if __name__ == "__main__":
    unittest.main()
