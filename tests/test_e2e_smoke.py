import os
import sys
import json
import shutil
import asyncio
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
import bridge
import video_transcriber


class TestE2ESmoke(unittest.TestCase):
    def test_e2e_demo_media_transcription_pipeline(self):
        """
        End-to-End smoke test verifying the full lifecycle using the project's demo.mp4:
        1. Media placement on shared volume
        2. Worker transcription run and status progression
        3. Subtitle file generation with valid cues
        4. Bridge status retrieval and TTL status recovery
        5. Output download endpoint retrieval
        """
        demo_media_path = os.path.join(PROJECT_ROOT, "demo.mp4")
        self.assertTrue(os.path.exists(demo_media_path), "demo.mp4 must exist in project root")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            videos_dir = temp_p / "videos"
            subs_dir = temp_p / "subs"
            videos_dir.mkdir(parents=True, exist_ok=True)
            subs_dir.mkdir(parents=True, exist_ok=True)

            # 1. Place demo video in shared videos directory
            target_video = videos_dir / "demo.mp4"
            shutil.copyfile(demo_media_path, target_video)
            self.assertTrue(target_video.exists())

            job_id = "smoke-test-job-999"

            # 2. Run transcriber using mock transcription engine to test full file workflow
            mock_spec = {
                "videos_dir": str(videos_dir),
                "subs_dir": str(subs_dir),
                "model": "base",
                "item_id": "demo.mp4",
                "format": "srt",
                "generate_subs": True,
                "embed_subs": False,
            }

            # Create mock transcribe_one to produce valid subtitle content
            def fake_transcribe_one(*args, **kwargs):
                out_path = subs_dir / "demo.srt"
                with out_path.open("w", encoding="utf-8") as f:
                    f.write("1\n00:00:00,500 --> 00:00:03,000\nThis is a test subtitle for demo.mp4.\n\n")
                return out_path, "en"

            with patch("video_transcriber.load_job_spec", return_value=mock_spec), \
                 patch("video_transcriber.load_model", return_value="mock_model"), \
                 patch("video_transcriber.transcribe_one", side_effect=fake_transcribe_one), \
                 patch.dict(os.environ, {"BRIDGE_JOB_ID": job_id}):

                rc = video_transcriber.main()
                self.assertEqual(rc, 0, "video_transcriber.main() must succeed")

            # 3. Verify output subtitle file
            generated_srt = subs_dir / "demo.srt"
            self.assertTrue(generated_srt.exists())
            content = generated_srt.read_text(encoding="utf-8")
            self.assertIn("This is a test subtitle for demo.mp4.", content)
            self.assertIn("00:00:00,500 --> 00:00:03,000", content)

            # 4. Verify bridge status endpoint using TTL recovery (Job pod has finished)
            with patch.object(bridge, "SUBS_DIR", str(subs_dir)), \
                 patch.object(bridge.k8s_batch, "list_namespaced_job", return_value=MagicMock(items=[])):

                status_resp = asyncio.run(bridge.get_status(job_id))
                self.assertEqual(status_resp.status, "succeeded")
                self.assertEqual(status_resp.progress, 100)
                self.assertEqual(status_resp.subtitlePath, str(generated_srt))

            # 5. Verify bridge download endpoint
            with patch.object(bridge, "SUBS_DIR", str(subs_dir)):
                download_resp = asyncio.run(bridge.download_job_output(job_id))
                self.assertEqual(download_resp.path, str(generated_srt))
                self.assertEqual(download_resp.media_type, "application/x-subrip")


if __name__ == "__main__":
    unittest.main()
