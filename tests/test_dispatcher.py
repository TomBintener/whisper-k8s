import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory and app directory to path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, TEST_DIR)

import mock_dependencies  # ensure mocks if dependencies not installed globally
import dispatcher


class TestDispatcher(unittest.TestCase):
    def test_dns1123_label_sanitization(self):
        """Test DNS-1123 compliant label generation."""
        self.assertEqual(dispatcher.dns1123_label("SimpleName"), "simplename")
        self.assertEqual(dispatcher.dns1123_label("video.test.mp4"), "video-test-mp4")
        self.assertEqual(dispatcher.dns1123_label("video___name--123"), "video-name-123")
        self.assertEqual(dispatcher.dns1123_label("!@#special$$$chars"), "special-chars")
        self.assertEqual(dispatcher.dns1123_label(""), "x")
        self.assertEqual(dispatcher.dns1123_label(None), "")
        # Truncation to 63 chars max
        long_str = "a" * 100
        truncated = dispatcher.dns1123_label(long_str)
        self.assertEqual(len(truncated), 63)
        self.assertEqual(truncated, "a" * 63)

    def test_job_name_for(self):
        """Test job name generation format."""
        with patch.object(dispatcher, "JOB_PREFIX", "whisper"), \
             patch.object(dispatcher, "BRIDGE_JOB_ID", "job123"):
            name = dispatcher.job_name_for("video.mp4")
            self.assertEqual(name, "whisper-job123-video-mp4")

    def test_build_job_pod_spec_volumes_bug_regression(self):
        """
        REGRESSION TEST: Verify that build_job properly assigns volumes
        to V1PodSpec. Previously, spec=client.V1PodSpec(...) was missing
        volumes=volumes, causing Kubernetes API validation errors.
        """
        with patch.object(dispatcher, "PVC_NAME", "test-pvc"), \
             patch.object(dispatcher, "EXECUTION_MODE", "pod"), \
             patch("dispatcher.make_projected_volume_sources", return_value=[]):
            job = dispatcher.build_job(item_id="sample.mp4")

            # Check container mounts
            container = job.spec.template.spec.containers[0]
            mount_names = [m.name for m in container.volume_mounts]
            self.assertIn("data", mount_names)

            # Check that pod spec volumes are present and match mount names
            pod_volumes = job.spec.template.spec.volumes
            self.assertIsNotNone(pod_volumes, "PodSpec.volumes must not be None when PVC is configured")
            vol_names = [v.name for v in pod_volumes]
            self.assertIn("data", vol_names, "PodSpec.volumes must include the 'data' PVC volume")

            # Find the data volume and verify PVC claim name
            data_vol = next(v for v in pod_volumes if v.name == "data")
            self.assertEqual(data_vol.persistent_volume_claim.claim_name, "test-pvc")

    def test_build_job_ssh_mode(self):
        """Test build_job configuration in SSH execution mode."""
        with patch.object(dispatcher, "EXECUTION_MODE", "ssh"), \
             patch.object(dispatcher, "PVC_NAME", "test-pvc"), \
             patch.object(dispatcher, "SSH_KEY_PATH", "/etc/secret/id_rsa"), \
             patch("dispatcher.make_projected_volume_sources", return_value=[]):
            job = dispatcher.build_job(item_id="sample.mp4")

            container = job.spec.template.spec.containers[0]
            self.assertEqual(container.command, ["python", "/app/ssh_worker.py"])

            # Check SSH key volume in PodSpec
            pod_volumes = job.spec.template.spec.volumes
            vol_names = [v.name for v in pod_volumes]
            self.assertIn("ssh-key", vol_names)

            # Check SSH key volume mount in container
            mount_names = [m.name for m in container.volume_mounts]
            self.assertIn("ssh-key", mount_names)

    def test_build_job_forwards_callbacks(self):
        """Test build_job forwards CALLBACK_URL and CALLBACK_HEADERS."""
        with patch.object(dispatcher, "CALLBACK_URL", "https://api.example.com/callback"), \
             patch.object(dispatcher, "CALLBACK_HEADERS", '{"X-Auth": "secret"}'), \
             patch.object(dispatcher, "PVC_NAME", "test-pvc"), \
             patch("dispatcher.make_projected_volume_sources", return_value=[]):
            job = dispatcher.build_job(item_id="sample.mp4")
            container = job.spec.template.spec.containers[0]
            env_map = {e.name: e.value for e in container.env}
            self.assertEqual(env_map.get("CALLBACK_URL"), "https://api.example.com/callback")
            self.assertEqual(env_map.get("CALLBACK_HEADERS"), '{"X-Auth": "secret"}')


if __name__ == "__main__":
    unittest.main()
