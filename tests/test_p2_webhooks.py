#!/usr/bin/env python3
"""
Unit and integration tests for P2: Warm Worker Pool & Push Webhook Notifications.

Tests coverage:
1. TaskQueue / FileTaskQueue: Atomic FIFO operations, ordering, timeouts, concurrency.
2. ModelCache: In-memory warm model reuse across sequential transcription jobs.
3. Push Webhooks: SSRF protection, successful POST delivery, retries on 5xx, client errors on 4xx.
4. Bridge Pool Integration: Mode="pool" enqueuing, immediate response, status polling.
5. Warm Worker Daemon: Continuous job pulling and clean termination.
"""

import os
import sys
import json
import time
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor

# Add tests and app directory to path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(APP_DIR))

# Ensure mock dependencies are loaded before importing app modules
import mock_dependencies  # noqa: F401

import queue_manager
import webhook
import video_transcriber
import bridge


class TestTaskQueue(unittest.TestCase):
    """Test TaskQueue and FileTaskQueue operations."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.queue_dir = Path(self.temp_dir.name)
        self.queue = queue_manager.FileTaskQueue(queue_dir=str(self.queue_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_queue_lifecycle(self):
        """Verify enqueue, dequeue, size, peek, complete, and empty timeouts."""
        self.assertEqual(self.queue.size(), 0)

        # Enqueue two jobs
        j1 = {"item_id": "job1.mp4", "model": "base"}
        j2 = {"item_id": "job2.mp4", "model": "small"}
        id1 = self.queue.enqueue(j1)
        id2 = self.queue.enqueue(j2)

        self.assertEqual(self.queue.size(), 2)
        self.assertIsNotNone(self.queue.peek(id1))
        self.assertEqual(self.queue.peek(id1)["model"], "base")

        # Dequeue first job
        dequeued1 = self.queue.dequeue(timeout=0.2)
        self.assertIsNotNone(dequeued1)
        self.assertEqual(dequeued1["job_id"], id1)
        self.assertEqual(self.queue.size(), 1)

        # Dequeue second job
        dequeued2 = self.queue.dequeue(timeout=0.2)
        self.assertIsNotNone(dequeued2)
        self.assertEqual(dequeued2["job_id"], id2)
        self.assertEqual(self.queue.size(), 0)

        # Empty dequeue returns None on timeout
        empty = self.queue.dequeue(timeout=0.05)
        self.assertIsNone(empty)

        # Complete jobs
        self.queue.complete(id1, success=True)
        self.queue.complete(id2, success=True)

        # Processing dir should be empty, completed dir should contain both
        self.assertEqual(len(list(self.queue.processing_dir.iterdir())), 0)
        self.assertEqual(len(list(self.queue.completed_dir.iterdir())), 2)

    def test_file_queue_ordering(self):
        """Verify strict chronological FIFO ordering."""
        job_ids = []
        for i in range(5):
            jid = self.queue.enqueue({"item_id": f"vid_{i}.mp4", "index": i})
            job_ids.append(jid)
            time.sleep(0.01)  # Ensure distinct timestamp prefix

        dequeued_indices = []
        for _ in range(5):
            job = self.queue.dequeue(timeout=0.5)
            self.assertIsNotNone(job)
            dequeued_indices.append(job["index"])

        self.assertEqual(dequeued_indices, [0, 1, 2, 3, 4])

    def test_file_queue_concurrent_dequeue(self):
        """Verify atomic task claiming with no duplicates across concurrent worker threads."""
        total_jobs = 16
        for i in range(total_jobs):
            self.queue.enqueue({"job_id": f"task_{i}", "index": i})

        claimed = []
        lock = threading.Lock()

        def worker_drain():
            w_q = queue_manager.FileTaskQueue(queue_dir=str(self.queue_dir))
            retries = 0
            while retries < 3:
                j = w_q.dequeue(timeout=0.1)
                if not j:
                    retries += 1
                    time.sleep(0.02)
                    continue
                retries = 0
                with lock:
                    claimed.append(j["job_id"])
                w_q.complete(j["job_id"], success=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_drain) for _ in range(4)]
            for f in futures:
                f.result()

        # All jobs must be claimed exactly once
        self.assertEqual(len(claimed), total_jobs)
        self.assertEqual(len(set(claimed)), total_jobs)


class TestModelCacheReuse(unittest.TestCase):
    """Test in-memory warm model reuse across jobs."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.videos_dir = self.data_dir / "videos"
        self.subs_dir = self.data_dir / "subs"
        self.videos_dir.mkdir(parents=True)
        self.subs_dir.mkdir(parents=True)

        # Create dummy video file
        self.video_path = self.videos_dir / "clip.mp4"
        self.video_path.write_bytes(b"dummy mp4 content")

    def tearDown(self):
        self.temp_dir.cleanup()

    def fake_transcribe(self, *args, **kwargs):
        subs_dir = kwargs.get("subs_dir") or args[2]
        fmt = kwargs.get("fmt", "vtt")
        out = Path(subs_dir) / f"clip.{fmt}"
        out.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello warm worker\n", encoding="utf-8")
        return out, "en"

    def test_model_cache_avoids_reloading(self):
        """Warm worker reuses model instance in memory on subsequent jobs."""
        model_cache = {}
        spec = {
            "videos_dir": str(self.videos_dir),
            "subs_dir": str(self.subs_dir),
            "item_id": "clip.mp4",
            "model": "base",
            "backend": "whisper",
            "device": "cpu",
            "format": "vtt",
        }

        mock_whisper_instance = MagicMock()

        with patch("video_transcriber.load_model", return_value=mock_whisper_instance) as mock_load, \
             patch("video_transcriber.transcribe_one", side_effect=self.fake_transcribe):

            # First job: cold load
            rc1 = video_transcriber.process_job(spec=spec, model_cache=model_cache)
            self.assertEqual(rc1, 0)
            self.assertEqual(mock_load.call_count, 1)

            # Cache must now contain the model
            cache_key = ("base", "whisper", "cpu", "float32")
            self.assertIn(cache_key, model_cache)
            self.assertEqual(model_cache[cache_key], mock_whisper_instance)

            # Second job: warm reuse
            rc2 = video_transcriber.process_job(spec=spec, model_cache=model_cache)
            self.assertEqual(rc2, 0)
            # Call count must still be 1 (zero additional model loading overhead!)
            self.assertEqual(mock_load.call_count, 1)

    def test_model_cache_differentiates_parameters(self):
        """Cache re-loads when model name or backend parameters differ."""
        model_cache = {}
        spec_base = {
            "videos_dir": str(self.videos_dir),
            "subs_dir": str(self.subs_dir),
            "item_id": "clip.mp4",
            "model": "base",
            "backend": "whisper",
            "device": "cpu",
        }
        spec_small = {
            "videos_dir": str(self.videos_dir),
            "subs_dir": str(self.subs_dir),
            "item_id": "clip.mp4",
            "model": "small",
            "backend": "whisper",
            "device": "cpu",
        }

        with patch("video_transcriber.load_model", side_effect=[MagicMock(), MagicMock()]) as mock_load, \
             patch("video_transcriber.transcribe_one", side_effect=self.fake_transcribe):

            video_transcriber.process_job(spec=spec_base, model_cache=model_cache)
            self.assertEqual(mock_load.call_count, 1)

            video_transcriber.process_job(spec=spec_small, model_cache=model_cache)
            self.assertEqual(mock_load.call_count, 2)


class TestPushWebhooks(unittest.TestCase):
    """Test SSRF validation, POST delivery, and exponential backoff retry logic."""

    def test_webhook_ssrf_protection(self):
        """Verify SSRF blocks loopback, private, and metadata IP addresses."""
        blocked_urls = [
            "http://127.0.0.1:8080/callback",
            "http://localhost:3000/webhook",
            "http://10.0.0.1/notify",
            "http://192.168.1.100/notify",
            "http://169.254.169.254/latest/meta-data",
            "ftp://example.com/callback",
        ]

        for url in blocked_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    webhook.validate_webhook_url(url)
                # send_webhook should also return False safely
                self.assertFalse(webhook.send_webhook(url, {"status": "test"}))

    def test_webhook_allow_local_override(self):
        """Local testing endpoints are permitted when ALLOW_LOCAL_URLS=true."""
        with patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):
            # Should not raise
            webhook.validate_webhook_url("http://127.0.0.1:8080/callback")

    def test_webhook_successful_delivery(self):
        """Successful 200 HTTP POST delivery returns True."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
             patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):

            success = webhook.send_webhook(
                url="http://127.0.0.1:8080/callback",
                payload={"jobId": "j-123", "status": "succeeded"},
                headers={"X-Test": "123"},
            )
            self.assertTrue(success)
            self.assertEqual(mock_urlopen.call_count, 1)

    def test_webhook_server_error_exponential_backoff(self):
        """5xx server errors trigger retries with exponential backoff up to max_retries."""
        import urllib.error
        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1/hook",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen, \
             patch("time.sleep") as mock_sleep, \
             patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):

            success = webhook.send_webhook(
                url="http://127.0.0.1/hook",
                payload={"jobId": "j-fail"},
                max_retries=3,
                retry_delay=0.1,
            )
            self.assertFalse(success)
            self.assertEqual(mock_urlopen.call_count, 3)
            # Retries doubled delay: 0.1, 0.2
            mock_sleep.assert_has_calls([call(0.1), call(0.2)])

    def test_webhook_client_error_no_retry(self):
        """4xx client errors immediately abort without retrying."""
        import urllib.error
        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1/hook",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen, \
             patch("time.sleep") as mock_sleep, \
             patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):

            success = webhook.send_webhook(
                url="http://127.0.0.1/hook",
                payload={"jobId": "j-404"},
                max_retries=3,
            )
            self.assertFalse(success)
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_sleep.assert_not_called()

    def test_process_job_dispatches_webhook_on_completion(self):
        """process_job dispatches webhook with job status payload upon completion."""
        with tempfile.TemporaryDirectory() as td:
            v_dir = Path(td) / "videos"
            s_dir = Path(td) / "subs"
            v_dir.mkdir()
            s_dir.mkdir()
            (v_dir / "lecture.mp4").write_bytes(b"video bytes")

            spec = {
                "videos_dir": str(v_dir),
                "subs_dir": str(s_dir),
                "item_id": "lecture.mp4",
                "model": "base",
                "callback_url": "http://127.0.0.1:8080/callback",
            }

            def fake_tr(*args, **kwargs):
                out = s_dir / "lecture.vtt"
                out.write_text("WEBVTT\n", encoding="utf-8")
                return out, "en"

            with patch("video_transcriber.load_model", return_value=MagicMock()), \
                 patch("video_transcriber.transcribe_one", side_effect=fake_tr), \
                 patch("webhook.send_webhook") as mock_send_hook, \
                 patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):

                rc = video_transcriber.process_job(spec=spec)
                self.assertEqual(rc, 0)
                mock_send_hook.assert_called_once()
                call_args = mock_send_hook.call_args[0]
                self.assertEqual(call_args[0], "http://127.0.0.1:8080/callback")
                self.assertEqual(call_args[1]["status"], "succeeded")
                self.assertEqual(call_args[1]["jobId"], "lecture.mp4")


class TestBridgePoolIntegration(unittest.TestCase):
    """Test bridge pool mode and callbackUrl handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.videos_dir = self.data_dir / "videos"
        self.subs_dir = self.data_dir / "subs"
        self.queue_dir = self.data_dir / "queue"
        self.videos_dir.mkdir(parents=True)
        self.subs_dir.mkdir(parents=True)
        self.queue_dir.mkdir(parents=True)

        bridge.VIDEOS_DIR = str(self.videos_dir)
        bridge.SUBS_DIR = str(self.subs_dir)
        bridge.DATA_DIR = str(self.data_dir)

        # Create dummy video
        (self.videos_dir / "sample.mp4").write_bytes(b"data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_job_req_ssrf_rejection(self):
        """CreateJobReq rejects SSRF URLs in callbackUrl."""
        with self.assertRaises(Exception):
            bridge.CreateJobReq(filename="sample.mp4", callbackUrl="http://127.0.0.1:9000/bad")

    def test_create_job_pool_mode_enqueues_task(self):
        """POST /jobs with mode='pool' enqueues task and returns immediately without k8s job creation."""
        with patch.dict(os.environ, {"QUEUE_DIR": str(self.queue_dir)}), \
             patch.object(bridge.k8s_batch, "create_namespaced_job") as mock_k8s:

            import asyncio
            req = bridge.CreateJobReq(
                filename="sample.mp4",
                mode="pool",
                model="base",
                format="srt",
            )

            resp = asyncio.run(bridge.create_job(req))
            self.assertIsNotNone(resp.jobId)

            # Kubernetes batch API must NOT have been called
            mock_k8s.assert_not_called()

            # Task must be in file queue
            q = queue_manager.FileTaskQueue(queue_dir=str(self.queue_dir))
            self.assertEqual(q.size(), 1)
            queued_spec = q.peek(resp.jobId)
            self.assertIsNotNone(queued_spec)
            self.assertEqual(queued_spec["filename"], "sample.mp4")
            self.assertEqual(queued_spec["format"], "srt")

            # Check status file immediately reports queued
            status_resp = asyncio.run(bridge.get_status(resp.jobId))
            self.assertEqual(status_resp.status, "queued")

    def test_get_status_running_and_succeeded_from_disk(self):
        """GET /status reads progress and done phases directly from status files."""
        import asyncio
        job_id = "test-job-999"

        # Simulate worker updating progress
        status_file = self.subs_dir / f"{job_id}.status.json"
        status_file.write_text(json.dumps({
            "phase": "transcribing",
            "progress": 65,
            "message": "Transcribing chunk 2/3",
        }), encoding="utf-8")

        res_running = asyncio.run(bridge.get_status(job_id))
        self.assertEqual(res_running.status, "running")
        self.assertEqual(res_running.progress, 65)

        # Simulate job completion
        status_file.write_text(json.dumps({
            "phase": "done",
            "progress": 100,
            "subtitlePath": str(self.subs_dir / "sample.srt"),
            "flavor": "srt",
        }), encoding="utf-8")

        res_done = asyncio.run(bridge.get_status(job_id))
        self.assertEqual(res_done.status, "succeeded")
        self.assertEqual(res_done.progress, 100)


class TestWarmWorkerDaemon(unittest.TestCase):
    """Test warm worker daemon execution."""

    def test_worker_daemon_processes_queued_jobs(self):
        """Worker daemon processes jobs in queue and terminates cleanly."""
        with tempfile.TemporaryDirectory() as td:
            q_dir = Path(td) / "queue"
            q = queue_manager.FileTaskQueue(queue_dir=str(q_dir))

            # Enqueue two jobs
            q.enqueue({"job_id": "job-1", "item_id": "v1.mp4"})
            q.enqueue({"job_id": "job-2", "item_id": "v2.mp4"})

            with patch("video_transcriber.process_job", return_value=0) as mock_proc:
                # Run daemon with max_jobs=2 limit
                rc = video_transcriber.run_worker_daemon(queue=q, max_jobs=2, poll_interval=0.05)
                self.assertEqual(rc, 0)
                self.assertEqual(mock_proc.call_count, 2)
                self.assertEqual(q.size(), 0)


if __name__ == "__main__":
    unittest.main()
