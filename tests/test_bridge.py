import os
import sys
import json
import socket
import tempfile
import unittest
import asyncio
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, TEST_DIR)

import mock_dependencies
import bridge
from fastapi import HTTPException


class TestBridge(unittest.TestCase):
    def test_create_job_req_validation(self):
        """Test validation rules for CreateJobReq."""
        # Valid payload with filename
        req = bridge.CreateJobReq(filename="video.mp4")
        self.assertEqual(req.filename, "video.mp4")
        self.assertEqual(req.format, "srt")

        # Missing both filename and trackUrl
        with self.assertRaises(ValueError):
            bridge.CreateJobReq()

        # Invalid device
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", device="tpu")

        # Invalid format
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", format="pdf")

        # Invalid task
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", task="summarize")

        # Invalid backend
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", backend="unsupported")

    def test_ssrf_validation(self):
        """Test SSRF validation logic for remote track URLs."""
        # Loopback URL
        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("http://127.0.0.1/video.mp4")
        self.assertEqual(ctx.exception.status_code, 400)

        # Localhost URL
        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("http://localhost/video.mp4")
        self.assertEqual(ctx.exception.status_code, 400)

        # Cloud Metadata IP (169.254.169.254)
        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("http://169.254.169.254/latest/meta-data/")
        self.assertEqual(ctx.exception.status_code, 400)

        # Private RFC1918 range (10.0.0.1)
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 80))]):
            with self.assertRaises(HTTPException) as ctx:
                bridge._validate_remote_url("http://internal-server.local/video.mp4")
            self.assertEqual(ctx.exception.status_code, 400)

        # Disallowed scheme
        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("ftp://example.com/video.mp4")
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("file:///etc/passwd")
        self.assertEqual(ctx.exception.status_code, 400)

        # Allow local URLs override for local development
        with patch.dict(os.environ, {"ALLOW_LOCAL_URLS": "true"}):
            # Should not raise exception
            bridge._validate_remote_url("http://127.0.0.1/video.mp4")

    def test_safe_under_path_traversal(self):
        """Test path traversal prevention in _safe_under."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid subpath
            safe = bridge._safe_under(temp_dir, "sub/video.mp4")
            self.assertTrue(safe.startswith(os.path.realpath(temp_dir)))

            # Traversal attempt
            with self.assertRaises(HTTPException) as ctx:
                bridge._safe_under(temp_dir, "../../etc/passwd")
            self.assertEqual(ctx.exception.status_code, 400)

    def test_make_worker_job_gpu_allocation_bug_regression(self):
        """
        REGRESSION TEST: Verify that when device='gpu' in pod mode,
        _make_worker_job sets nvidia.com/gpu in both limits and requests.
        """
        job = bridge._make_worker_job(
            job_id="gpu-job-1",
            filename="demo.mp4",
            fmt="srt",
            device="gpu",
            mode="pod",
        )
        container = job.spec.template.spec.containers[0]
        self.assertIn("nvidia.com/gpu", container.resources.limits)
        self.assertEqual(container.resources.limits["nvidia.com/gpu"], "1")
        self.assertIn("nvidia.com/gpu", container.resources.requests)
        self.assertEqual(container.resources.requests["nvidia.com/gpu"], "1")

        # Verify DEVICE env var is cuda
        device_env = next(e for e in container.env if e.name == "DEVICE")
        self.assertEqual(device_env.value, "cuda")

    def test_make_worker_job_cpu_allocation(self):
        """Verify that when device='cpu', nvidia.com/gpu is NOT set."""
        job = bridge._make_worker_job(
            job_id="cpu-job-1",
            filename="demo.mp4",
            fmt="srt",
            device="cpu",
            mode="pod",
        )
        container = job.spec.template.spec.containers[0]
        self.assertNotIn("nvidia.com/gpu", container.resources.limits)
        self.assertNotIn("nvidia.com/gpu", container.resources.requests)

        device_env = next(e for e in container.env if e.name == "DEVICE")
        self.assertEqual(device_env.value, "cpu")

    def test_get_status_ttl_recovery_from_disk(self):
        """
        Test that get_status recovers completed/failed status from disk
        when the Kubernetes Job has been cleaned up by TTL controller.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "ttl-test-job-123"
            status_file = os.path.join(temp_dir, f"{job_id}.status.json")

            # Write completed status file
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "phase": "done",
                    "progress": 100,
                    "subtitlePath": f"{temp_dir}/demo.srt",
                    "flavor": "srt",
                    "message": "Job finished successfully"
                }, f)

            with patch.object(bridge, "SUBS_DIR", temp_dir), \
                 patch.object(bridge.k8s_batch, "list_namespaced_job", return_value=MagicMock(items=[])):
                resp = asyncio.run(bridge.get_status(job_id))
                self.assertEqual(resp.status, "succeeded")
                self.assertEqual(resp.progress, 100)
                self.assertEqual(resp.subtitlePath, f"{temp_dir}/demo.srt")
                self.assertEqual(resp.flavor, "srt")

    def test_download_job_output(self):
        """Test GET /jobs/{job_id}/download endpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "download-test-123"
            sub_file = os.path.join(temp_dir, "demo.srt")
            with open(sub_file, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:01,000\nHello World\n\n")

            status_file = os.path.join(temp_dir, f"{job_id}.status.json")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "phase": "done",
                    "subtitlePath": sub_file
                }, f)

            with patch.object(bridge, "SUBS_DIR", temp_dir):
                resp = asyncio.run(bridge.download_job_output(job_id))
                self.assertEqual(resp.path, sub_file)
                self.assertEqual(resp.media_type, "application/x-subrip")

            # Missing job download -> 404
            with patch.object(bridge, "SUBS_DIR", temp_dir):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(bridge.download_job_output("nonexistent-job"))
                self.assertEqual(ctx.exception.status_code, 404)

    def test_cancel_job(self):
        """Test DELETE /jobs/{job_id} endpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "cancel-test-123"
            mock_job = MagicMock()
            mock_job.metadata.name = "whisper-cancel-test-123"

            status_file = os.path.join(temp_dir, f"{job_id}.status.json")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"phase": "running", "message": "In progress"}, f)

            with patch.object(bridge, "SUBS_DIR", temp_dir), \
                 patch.object(bridge.k8s_batch, "list_namespaced_job", return_value=MagicMock(items=[mock_job])), \
                 patch.object(bridge.k8s_batch, "delete_namespaced_job") as mock_del:

                resp = asyncio.run(bridge.cancel_job(job_id))
                self.assertTrue(resp["ok"])
                mock_del.assert_called_once()

                # Status file should now be marked as cancelled
                with open(status_file, "r", encoding="utf-8") as f:
                    updated = json.load(f)
                self.assertEqual(updated["phase"], "cancelled")

    def test_download_embedded_video(self):
        """Test GET /jobs/{job_id}/download?format=embedded serves embedded MP4."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "download-embed-123"
            sub_file = os.path.join(temp_dir, "demo.srt")
            embedded_file = os.path.join(temp_dir, "demo.embedded.mp4")
            with open(sub_file, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")
            with open(embedded_file, "wb") as f:
                f.write(b"mp4-content")

            status_file = os.path.join(temp_dir, f"{job_id}.status.json")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "phase": "done",
                    "subtitlePath": sub_file
                }, f)

            with patch.object(bridge, "SUBS_DIR", temp_dir):
                resp = asyncio.run(bridge.download_job_output(job_id, format="embedded"))
                self.assertEqual(resp.path, embedded_file)
                self.assertEqual(resp.media_type, "video/mp4")

    def test_cancel_pool_mode_job(self):
        """Test DELETE /jobs/{job_id} cancels queued pool mode job."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "cancel-pool-123"
            subs_dir = os.path.join(temp_dir, "subs")
            queue_dir = os.path.join(temp_dir, "queue")
            os.makedirs(subs_dir)
            os.makedirs(queue_dir)

            # Create queued task in FileTaskQueue
            import queue_manager
            q = queue_manager.FileTaskQueue(queue_dir=queue_dir)
            q.enqueue({"job_id": job_id, "filename": "demo.mp4"})

            status_file = os.path.join(subs_dir, f"{job_id}.status.json")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"phase": "queued", "message": "Queued in pool"}, f)

            with patch.object(bridge, "SUBS_DIR", subs_dir), \
                 patch.dict(os.environ, {"QUEUE_DIR": queue_dir}), \
                 patch.object(bridge.k8s_batch, "list_namespaced_job", return_value=MagicMock(items=[])):

                resp = asyncio.run(bridge.cancel_job(job_id))
                self.assertTrue(resp["ok"])

                # Queued file should be removed from pending
                self.assertEqual(q.size(), 0)

                # Status file should be updated to cancelled
                with open(status_file, "r", encoding="utf-8") as f:
                    updated = json.load(f)
                self.assertEqual(updated["phase"], "cancelled")

    def test_standalone_mode_create_job_k8s_unavailable(self):
        """Test creating a k8s job when Kubernetes API is unavailable raises 503."""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_file = os.path.join(temp_dir, "demo.mp4")
            with open(video_file, "wb") as f:
                f.write(b"dummy")

            with patch.object(bridge, "VIDEOS_DIR", temp_dir), \
                 patch.object(bridge, "k8s_batch", None):
                req = bridge.CreateJobReq(filename="demo.mp4", mode="job")
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(bridge.create_job(req))
                self.assertEqual(ctx.exception.status_code, 503)

    def test_standalone_mode_get_status_from_disk(self):
        """Test getting job status in standalone mode reads directly from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job_id = "standalone-job-456"
            status_file = os.path.join(temp_dir, f"{job_id}.status.json")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"phase": "transcribing", "progress": 50, "job_id": job_id}, f)

            with patch.object(bridge, "k8s_batch", None), \
                 patch.object(bridge, "SUBS_DIR", temp_dir):
                res = asyncio.run(bridge.get_status(job_id))
                self.assertEqual(res.status, "running")
                self.assertEqual(res.progress, 50)


if __name__ == "__main__":
    unittest.main()
