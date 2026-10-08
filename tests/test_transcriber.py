import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, TEST_DIR)

import mock_dependencies
import video_transcriber


class TestVideoTranscriber(unittest.TestCase):
    def test_as_bool(self):
        """Test truthy and falsy parsing in as_bool."""
        self.assertTrue(video_transcriber.as_bool(True))
        self.assertTrue(video_transcriber.as_bool("true"))
        self.assertTrue(video_transcriber.as_bool("1"))
        self.assertTrue(video_transcriber.as_bool("yes"))
        self.assertTrue(video_transcriber.as_bool("on"))

        self.assertFalse(video_transcriber.as_bool(False))
        self.assertFalse(video_transcriber.as_bool("false"))
        self.assertFalse(video_transcriber.as_bool("0"))
        self.assertFalse(video_transcriber.as_bool("no"))
        self.assertFalse(video_transcriber.as_bool("off"))
        self.assertFalse(video_transcriber.as_bool(None, default=False))
        self.assertTrue(video_transcriber.as_bool(None, default=True))

    def test_pick_device(self):
        """Test device selection logic."""
        self.assertEqual(video_transcriber.pick_device("cpu"), "cpu")
        with patch.object(video_transcriber, "torch", None):
            self.assertEqual(video_transcriber.pick_device("gpu"), "cpu")

    def test_write_status_records_job_and_stem(self):
        """Test status file creation and bridge_job_id synchronization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            subs_dir = Path(temp_dir)
            with patch.dict(os.environ, {"BRIDGE_JOB_ID": "job-test-456"}):
                video_transcriber.write_status(
                    subs_dir=subs_dir,
                    stem="demo",
                    phase="transcribing",
                    progress=30,
                    message="Transcribing audio",
                    subtitle_path="/data/subs/demo.srt",
                    flavor="srt",
                )

                # Check stem status file
                stem_file = subs_dir / "demo.status.json"
                self.assertTrue(stem_file.exists())
                with stem_file.open("r", encoding="utf-8") as f:
                    stem_data = json.load(f)
                self.assertEqual(stem_data["phase"], "transcribing")
                self.assertEqual(stem_data["progress"], 30)
                self.assertEqual(stem_data["job_id"], "job-test-456")
                self.assertEqual(stem_data["subtitlePath"], "/data/subs/demo.srt")

                # Check bridge_job_id status file
                job_file = subs_dir / "job-test-456.status.json"
                self.assertTrue(job_file.exists())
                with job_file.open("r", encoding="utf-8") as f:
                    job_data = json.load(f)
                self.assertEqual(job_data["phase"], "transcribing")
                self.assertEqual(job_data["progress"], 30)

    def test_main_unhandled_exception_regression(self):
        """
        REGRESSION TEST: Verify that when transcription encounters an unhandled
        exception, main() catches it, writes phase='failed' to the status file,
        and returns non-zero. Previously, it crashed leaving status frozen.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            fake_video = temp_p / "video.mp4"
            fake_video.touch()

            mock_spec = {
                "videos_dir": str(temp_p),
                "subs_dir": str(temp_p),
                "model": "base",
                "item_id": "video.mp4",
                "generate_subs": True,
            }

            with patch("video_transcriber.load_job_spec", return_value=mock_spec), \
                 patch("video_transcriber.load_model", side_effect=RuntimeError("CUDA out of memory simulation")), \
                 patch.dict(os.environ, {"BRIDGE_JOB_ID": "job-fail-test"}):

                rc = video_transcriber.main()
                self.assertEqual(rc, 1, "main() must return code 1 on caught failure")

                # Verify failure status file was written
                status_file = temp_p / "video.status.json"
                self.assertTrue(status_file.exists())
                with status_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["phase"], "failed")
                self.assertIn("CUDA out of memory", data["message"])


if __name__ == "__main__":
    unittest.main()
