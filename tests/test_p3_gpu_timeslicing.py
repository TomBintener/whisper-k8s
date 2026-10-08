#!/usr/bin/env python3
"""
Unit and integration tests for P3: GPU Time-Slicing, Fractional Sharing,
VRAM protection, and Model Preloader Automation.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add tests and app directory to path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(APP_DIR))

# Ensure mock dependencies are loaded before importing app modules
import mock_dependencies  # noqa: F401

import video_transcriber
import bridge
import dispatcher
import entry


class TestP3Manifests(unittest.TestCase):
    """Validate Kubernetes manifests for GPU time-slicing and model preloading."""

    def test_gpu_timeslicing_manifest_structure(self):
        """Verify gpu-timeslicing.yaml exists and defines 4x and 2x time-slicing profiles."""
        manifest_path = REPO_ROOT / "k8s_scripts" / "gpu-timeslicing.yaml"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        content = manifest_path.read_text(encoding="utf-8")
        self.assertIn("name: whisper-gpu-timeslicing", content)
        self.assertIn("time-slicing-4x:", content)
        self.assertIn("time-slicing-2x:", content)
        self.assertIn("replicas: 4", content)
        self.assertIn("replicas: 2", content)
        self.assertIn("nvidia.com/gpu", content)

    def test_model_preload_job_manifest(self):
        """Verify model-preload-job.yaml exists and configures batch preloading."""
        manifest_path = REPO_ROOT / "k8s_scripts" / "model-preload-job.yaml"
        self.assertTrue(manifest_path.exists(), f"Missing manifest: {manifest_path}")

        content = manifest_path.read_text(encoding="utf-8")
        self.assertIn("kind: Job", content)
        self.assertIn("name: whisper-model-preload", content)
        self.assertIn("name: SERVICE", content)
        self.assertIn("value: preload", content)
        self.assertIn("claimName: whisper-data", content)

    def test_kustomization_includes_new_manifests(self):
        """Verify kustomization.yaml includes gpu-timeslicing and preload job."""
        kustomize_path = REPO_ROOT / "k8s_scripts" / "kustomization.yaml"
        content = kustomize_path.read_text(encoding="utf-8")
        self.assertIn("gpu-timeslicing.yaml", content)
        self.assertIn("model-preload-job.yaml", content)


class TestP3VRAMProtection(unittest.TestCase):
    """Test worker runtime VRAM memory fractions, allocator configurations, and quantization."""

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_configure_cuda_memory_valid_fraction(self):
        """Valid vram_fraction in spec or env is properly parsed and bounds-checked."""
        # Spec override
        spec = {"vram_fraction": 0.25}
        frac = video_transcriber.configure_cuda_memory(spec)
        self.assertEqual(frac, 0.25)
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")

        # Env override
        os.environ["CUDA_MEMORY_FRACTION"] = "0.5"
        frac_env = video_transcriber.configure_cuda_memory({})
        self.assertEqual(frac_env, 0.5)

    def test_configure_cuda_memory_invalid_fraction(self):
        """Invalid fractions (<=0, >1, non-numeric) are handled safely without crashing."""
        for invalid_val in [0.0, -0.25, 1.25, "not_a_number"]:
            spec = {"vram_fraction": invalid_val}
            frac = video_transcriber.configure_cuda_memory(spec)
            self.assertIsNone(frac, f"Expected None for invalid value {invalid_val}")

    @patch("video_transcriber.torch")
    def test_configure_cuda_memory_calls_torch_fraction(self, mock_torch):
        """Verify torch.cuda.set_per_process_memory_fraction is called when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.set_per_process_memory_fraction = MagicMock()

        spec = {"cuda_memory_fraction": 0.25}
        frac = video_transcriber.configure_cuda_memory(spec)
        self.assertEqual(frac, 0.25)
        mock_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.25)

    def test_load_faster_model_cpu_quantization_fallback(self):
        """Verify faster-whisper on CPU falls back from float16 to float32 to prevent crash."""
        mock_model_cls = MagicMock()
        with patch("video_transcriber.WhisperModel", mock_model_cls):
            video_transcriber.load_faster_model("base", device="cpu", compute_type="float16")
            mock_model_cls.assert_called_once_with(
                "base", device="cpu", compute_type="float32", download_root=None
            )

    def test_load_faster_model_gpu_quantization(self):
        """Verify faster-whisper on CUDA respects int8_float16 compute_type."""
        mock_model_cls = MagicMock()
        with patch("video_transcriber.WhisperModel", mock_model_cls):
            video_transcriber.load_faster_model("base", device="cuda:0", compute_type="int8_float16")
            mock_model_cls.assert_called_once_with(
                "base", device="cuda", compute_type="int8_float16", download_root=None
            )


class TestP3BridgeIntegration(unittest.TestCase):
    """Test bridge request validation and worker job specification generation for GPU sharing."""

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_create_job_req_validation(self):
        """Verify CreateJobReq validates computeType and vramFraction properly."""
        # Valid request
        req = bridge.CreateJobReq(
            filename="sample.mp4",
            device="gpu",
            computeType="int8_float16",
            vramFraction=0.25,
        )
        self.assertEqual(req.computeType, "int8_float16")
        self.assertEqual(req.vramFraction, 0.25)

        # Invalid computeType
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(
                filename="sample.mp4",
                computeType="invalid_precision",  # type: ignore
            )

        # Invalid vramFraction > 1.0
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(
                filename="sample.mp4",
                vramFraction=1.5,
            )

        # Invalid vramFraction <= 0.0
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(
                filename="sample.mp4",
                vramFraction=0.0,
            )

    def test_make_worker_job_injects_gpu_sharing_env(self):
        """Verify _make_worker_job injects CUDA_MEMORY_FRACTION and COMPUTE_TYPE."""
        job = bridge._make_worker_job(
            job_id="test-p3-job",
            filename="video.mp4",
            fmt="srt",
            device="gpu",
            compute_type="int8_float16",
            vram_fraction=0.25,
        )
        container = job.spec.template.spec.containers[0]
        env_map = {e.name: e.value for e in container.env}

        self.assertEqual(env_map.get("DEVICE"), "cuda")
        self.assertEqual(env_map.get("COMPUTE_TYPE"), "int8_float16")
        self.assertEqual(env_map.get("CUDA_MEMORY_FRACTION"), "0.25")

        # Verify GPU resources are requested
        self.assertIn("nvidia.com/gpu", container.resources.limits)
        self.assertEqual(container.resources.limits["nvidia.com/gpu"], "1")

    def test_make_worker_job_custom_gpu_resource_name(self):
        """Verify _make_worker_job supports custom GPU_RESOURCE_NAME (e.g., nvidia.com/gpu.shared)."""
        os.environ["GPU_RESOURCE_NAME"] = "nvidia.com/gpu.shared"
        job = bridge._make_worker_job(
            job_id="test-p3-custom-res",
            filename="video.mp4",
            fmt="srt",
            device="gpu",
        )
        container = job.spec.template.spec.containers[0]
        self.assertIn("nvidia.com/gpu.shared", container.resources.limits)
        self.assertEqual(container.resources.limits["nvidia.com/gpu.shared"], "1")


class TestP3DispatcherIntegration(unittest.TestCase):
    """Test dispatcher job generation with GPU time-slicing and quantization parameters."""

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_dispatcher_build_job_gpu_sharing(self):
        """Verify dispatcher.build_job includes CUDA_MEMORY_FRACTION and COMPUTE_TYPE."""
        with patch.dict(dispatcher.ENV, {
            "CUDA_MEMORY_FRACTION": "0.25",
            "COMPUTE_TYPE": "int8_float16",
        }):
            # Patch the module-level variables
            dispatcher.CUDA_MEMORY_FRACTION = "0.25"
            dispatcher.COMPUTE_TYPE = "int8_float16"

            job = dispatcher.build_job(item_id="batch-item-1.mp4")
            container = job.spec.template.spec.containers[0]
            env_map = {e.name: e.value for e in container.env}

            self.assertEqual(env_map.get("CUDA_MEMORY_FRACTION"), "0.25")
            self.assertEqual(env_map.get("COMPUTE_TYPE"), "int8_float16")


class TestP3ModelPreloader(unittest.TestCase):
    """Test model preloader functionality and CLI/environment dispatch."""

    @patch("entry.preload_models")
    def test_entry_main_cli_preload(self, mock_preload):
        """Verify python entry.py --preload base,small triggers preload_models."""
        mock_preload.return_value = 0
        with patch.object(sys, "argv", ["entry.py", "--preload", "base,small"]):
            ret = entry.main()
            self.assertEqual(ret, 0)
            mock_preload.assert_called_once_with(["base", "small"])

    @patch("entry.preload_models")
    def test_entry_main_service_preload(self, mock_preload):
        """Verify SERVICE=preload triggers preload_models from PRELOAD_MODELS."""
        mock_preload.return_value = 0
        with patch.dict(os.environ, {"SERVICE": "preload", "PRELOAD_MODELS": "medium"}):
            with patch.object(sys, "argv", ["entry.py"]):
                ret = entry.main()
                self.assertEqual(ret, 0)
                mock_preload.assert_called_once_with(["medium"])

    def test_preload_models_execution(self):
        """Verify preload_models attempts to warm models safely with mocks."""
        mock_whisper = MagicMock()
        mock_fw_model = MagicMock()
        with patch.dict(sys.modules, {
            "whisper": mock_whisper,
            "faster_whisper": MagicMock(WhisperModel=mock_fw_model),
        }):
            ret = entry.preload_models(["tiny"])
            self.assertEqual(ret, 0)
            mock_whisper.load_model.assert_called_once()
            mock_fw_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
