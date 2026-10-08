import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, TEST_DIR)

import mock_dependencies
import ssh_worker


class TestSSHWorker(unittest.TestCase):
    def test_rewrite_path(self):
        """Test path rewriting from pod storage to remote host storage."""
        with patch.object(ssh_worker, "LOCAL_PATH_PREFIX", "/data"), \
             patch.object(ssh_worker, "REMOTE_PATH_PREFIX", "/Users/Shared/data"):

            # Path under LOCAL_PATH_PREFIX
            rewritten = ssh_worker.rewrite_path("/data/videos/lecture.mp4")
            self.assertEqual(rewritten, "/Users/Shared/data/videos/lecture.mp4")

            # Path not under LOCAL_PATH_PREFIX
            unchanged = ssh_worker.rewrite_path("/tmp/test.mp4")
            self.assertEqual(unchanged, "/tmp/test.mp4")

            # Empty path
            self.assertEqual(ssh_worker.rewrite_path(""), "")

    def test_forward_vars_inclusion(self):
        """Verify essential variables are in FORWARD_VARS."""
        expected_vars = [
            "ITEM_ID", "BRIDGE_JOB_ID", "MODEL", "DEVICE",
            "SUB_FORMAT", "EMBED_SUBS", "LANGUAGE", "TASK", "BACKEND"
        ]
        for var in expected_vars:
            self.assertIn(var, ssh_worker.FORWARD_VARS)


if __name__ == "__main__":
    unittest.main()
