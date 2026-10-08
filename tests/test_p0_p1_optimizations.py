import os
import sys
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
import bridge
import dispatcher


class TestP0P1Optimizations(unittest.TestCase):
    def test_p0_convert_srt_to_vtt(self):
        """Test fast SRT to VTT in-memory/file converter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            srt_path = temp_p / "sample.srt"
            vtt_path = temp_p / "sample.vtt"

            srt_content = (
                "1\n"
                "00:01:23,456 --> 00:01:28,789\n"
                "First subtitle line.\n\n"
                "2\n"
                "00:01:29,000 --> 00:01:32,500\n"
                "Second subtitle line.\n\n"
            )
            srt_path.write_text(srt_content, encoding="utf-8")

            # Run converter
            result_path = video_transcriber.convert_srt_to_vtt(srt_path, vtt_path)
            self.assertTrue(result_path.exists())

            vtt_content = result_path.read_text(encoding="utf-8")
            # Verify WebVTT header
            self.assertTrue(vtt_content.startswith("WEBVTT"))
            # Verify comma to dot timestamp conversion
            self.assertIn("00:01:23.456 --> 00:01:28.789", vtt_content)
            self.assertIn("00:01:29.000 --> 00:01:32.500", vtt_content)
            self.assertNotIn("00:01:23,456", vtt_content)
            # Verify cue text preserved
            self.assertIn("First subtitle line.", vtt_content)
            self.assertIn("Second subtitle line.", vtt_content)

    def test_p0_single_inference_pass_during_embedding_regression(self):
        """
        REGRESSION TEST: Verify that when format='srt' and embed_subs=True,
        video_transcriber runs transcribe_one EXACTLY ONCE.
        Previously, it re-ran transcribe_one a second time over the entire media file
        to get a VTT file, doubling inference duration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            fake_video = temp_p / "demo.mp4"
            fake_video.touch()

            mock_spec = {
                "videos_dir": str(temp_p),
                "subs_dir": str(temp_p),
                "model": "base",
                "item_id": "demo.mp4",
                "format": "srt",
                "generate_subs": True,
                "embed_subs": True,
            }

            def fake_transcribe_one(*args, **kwargs):
                srt_out = temp_p / "demo.srt"
                srt_out.write_text("1\n00:00:01,000 --> 00:00:04,000\nHello World\n\n", encoding="utf-8")
                return srt_out, "en"

            with patch("video_transcriber.load_job_spec", return_value=mock_spec), \
                 patch("video_transcriber.load_model", return_value="mock_model"), \
                 patch("video_transcriber.transcribe_one", side_effect=fake_transcribe_one) as mock_transcribe, \
                 patch("video_transcriber.embed_subtitles") as mock_embed, \
                 patch.dict(os.environ, {"BRIDGE_JOB_ID": "p0-test-job"}):

                rc = video_transcriber.main()
                self.assertEqual(rc, 0)

                # CRITICAL ASSERTION: transcribe_one must only be called ONCE
                self.assertEqual(
                    mock_transcribe.call_count, 1,
                    f"transcribe_one was called {mock_transcribe.call_count} times, expected exactly 1"
                )

                # embed_subtitles must be called with the fast-converted VTT file
                mock_embed.assert_called_once()
                embed_args = mock_embed.call_args[0]
                passed_vtt = embed_args[1]
                self.assertTrue(str(passed_vtt).endswith(".vtt"))
                self.assertTrue(passed_vtt.exists())

    def test_p1_bridge_model_cache_env_propagation(self):
        """Test that bridge._make_worker_job propagates persistent cache env vars."""
        job = bridge._make_worker_job(
            job_id="p1-bridge-test",
            filename="demo.mp4",
            fmt="srt",
            device="cpu",
        )
        container = job.spec.template.spec.containers[0]
        env_dict = {e.name: e.value for e in container.env}

        self.assertIn("MODELS_DIR", env_dict)
        self.assertIn("WHISPER_DOWNLOAD_ROOT", env_dict)
        self.assertIn("HF_HOME", env_dict)
        self.assertIn("TORCH_HOME", env_dict)
        self.assertIn("WHISPER_CPP_MODEL_ROOT", env_dict)

        # Check that paths reside on persistent volume
        self.assertTrue(env_dict["WHISPER_DOWNLOAD_ROOT"].startswith(env_dict["MODELS_DIR"]))
        self.assertTrue(env_dict["HF_HOME"].startswith(env_dict["MODELS_DIR"]))
        self.assertTrue(env_dict["TORCH_HOME"].startswith(env_dict["MODELS_DIR"]))

    def test_p1_dispatcher_model_cache_env_propagation(self):
        """Test that dispatcher.build_job propagates persistent cache env vars."""
        with patch.object(dispatcher, "EXECUTION_MODE", "pod"), \
             patch("dispatcher.make_projected_volume_sources", return_value=[]):
            job = dispatcher.build_job(item_id="demo.mp4")
            container = job.spec.template.spec.containers[0]
            env_dict = {e.name: e.value for e in container.env}

            self.assertIn("MODELS_DIR", env_dict)
            self.assertIn("WHISPER_DOWNLOAD_ROOT", env_dict)
            self.assertIn("HF_HOME", env_dict)
            self.assertIn("TORCH_HOME", env_dict)
            self.assertIn("WHISPER_CPP_MODEL_ROOT", env_dict)

    def test_p1_load_faster_model_download_root(self):
        """Test that load_faster_model passes download_root to WhisperModel."""
        mock_whisper_model = MagicMock()
        with patch.object(video_transcriber, "WhisperModel", mock_whisper_model):
            video_transcriber.load_faster_model("small", device="cpu", download_root="/data/models/hf")
            mock_whisper_model.assert_called_once_with(
                "small", device="cpu", compute_type="float16", download_root="/data/models/hf"
            )


if __name__ == "__main__":
    unittest.main()
