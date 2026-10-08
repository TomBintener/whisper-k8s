import os
import sys
import json
import io
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import demo_transcribe


class TestDemoTranscribeScript(unittest.TestCase):
    def test_render_progress_bar(self):
        """Test progress bar rendering at various percentages and clamp ranges."""
        with patch.object(demo_transcribe, "_get_bar_chars", return_value=("█", "░")):
            # 0%
            bar_0 = demo_transcribe.render_progress_bar(0, "queued", "Waiting", 0.1, bar_length=10)
            self.assertIn("0%", bar_0)
            self.assertIn("[queued]", bar_0)
            self.assertIn("░" * 10, bar_0)

            # 50%
            bar_50 = demo_transcribe.render_progress_bar(50, "transcribing", "Audio", 2.5, bar_length=10)
            self.assertIn("50%", bar_50)
            self.assertIn("█" * 5, bar_50)
            self.assertIn("2.5s", bar_50)

            # 100% (and clamp over 100)
            bar_100 = demo_transcribe.render_progress_bar(150, "done", "Finished", 5.0, bar_length=10)
            self.assertIn("100%", bar_100)
            self.assertIn("█" * 10, bar_100)

    def test_check_health_success(self):
        """Test successful health check."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True, "kube": "standalone"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            data = demo_transcribe.check_health("http://localhost:8080")
            self.assertTrue(data["ok"])
            self.assertEqual(data["kube"], "standalone")

    def test_check_health_failure(self):
        """Test connection failure raises helpful ConnectionError."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with self.assertRaises(ConnectionError) as ctx:
                demo_transcribe.check_health("http://invalid-host:8080")
            self.assertIn("Cannot connect to whisper-k8s", str(ctx.exception))
            self.assertIn("docker compose up", str(ctx.exception))

    def test_submit_job_success(self):
        """Test submitting job payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "jobId": "abc-123",
            "status": "accepted",
            "videoPath": "/data/videos/demo.mp4",
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            resp = demo_transcribe.submit_job("http://localhost:8080", {"filename": "demo.mp4"})
            self.assertEqual(resp["jobId"], "abc-123")
            self.assertEqual(resp["status"], "accepted")

    def test_submit_job_error(self):
        """Test HTTP error handling on submission."""
        err_body = io.BytesIO(b'{"detail": "File not found"}')
        mock_err = urllib.error.HTTPError("http://localhost:8080/jobs", 404, "Not Found", {}, err_body)

        with patch("urllib.request.urlopen", side_effect=mock_err):
            with self.assertRaises(RuntimeError) as ctx:
                demo_transcribe.submit_job("http://localhost:8080", {"filename": "missing.mp4"})
            self.assertIn("Job submission failed (HTTP 404)", str(ctx.exception))
            self.assertIn("File not found", str(ctx.exception))

    def test_preview_subtitles(self):
        """Test parsing and printing subtitle preview."""
        sample_vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello world\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "Second cue\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            vtt_path = os.path.join(temp_dir, "test.vtt")
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(sample_vtt)

            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                demo_transcribe.preview_subtitles(demo_transcribe.Path(vtt_path), max_cues=2)
                output = mock_stdout.getvalue()
                self.assertIn("Hello world", output)
                self.assertIn("00:00:00.000 --> 00:00:02.000", output)

    def test_cli_argument_defaults(self):
        """Test CLI argument defaults and flag parsing."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", default="demo.mp4")
        parser.add_argument("--backend", default="faster-whisper")
        parser.add_argument("--model", default="base")
        parser.add_argument("--format", default="srt")
        parser.add_argument("--mode", default="pool")

        args = parser.parse_args([])
        self.assertEqual(args.file, "demo.mp4")
        self.assertEqual(args.backend, "faster-whisper")
        self.assertEqual(args.model, "base")
        self.assertEqual(args.format, "srt")
        self.assertEqual(args.mode, "pool")


if __name__ == "__main__":
    unittest.main()
