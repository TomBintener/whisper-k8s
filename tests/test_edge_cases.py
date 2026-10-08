#!/usr/bin/env python3
"""
Comprehensive edge case test suite for whisper-k8s.
Covers audio chunking, silence alignment, subtitle stitching, timestamp arithmetic,
malformed inputs, boundary values, error recovery, and security edge cases.
"""

import os
import sys
import tempfile
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

import chunking
import video_transcriber
import bridge
import dispatcher
from fastapi import HTTPException


class TestChunkingEdgeCases(unittest.TestCase):
    """Edge cases for chunk planning, timestamp parsing, and subtitle stitching."""

    def test_parse_timestamp_malformed_and_extreme(self):
        """Test timestamp parser resilience against various formats and edge inputs."""
        # Standard formats
        self.assertAlmostEqual(chunking.parse_timestamp("00:00:00,000"), 0.0)
        self.assertAlmostEqual(chunking.parse_timestamp("00:00:00.000"), 0.0)
        # Short format MM:SS.mmm
        self.assertAlmostEqual(chunking.parse_timestamp("00:05.500"), 5.5)
        # Large hours
        self.assertAlmostEqual(chunking.parse_timestamp("100:00:00,000"), 360000.0)
        # Single float string
        self.assertAlmostEqual(chunking.parse_timestamp("42.5"), 42.5)

    def test_format_timestamp_boundary_values(self):
        """Test millisecond roll-over and negative duration boundaries."""
        # Zero seconds
        self.assertEqual(chunking.format_timestamp(0.0, fmt="srt"), "00:00:00,000")
        self.assertEqual(chunking.format_timestamp(0.0, fmt="vtt"), "00:00:00.000")
        # Negative clamp
        self.assertEqual(chunking.format_timestamp(-5.0, fmt="srt"), "00:00:00,000")
        # Millisecond rounding to next whole second (59.9999 -> 01:00)
        self.assertEqual(chunking.format_timestamp(59.9999, fmt="srt"), "00:01:00,000")
        # 1 hour exactly
        self.assertEqual(chunking.format_timestamp(3600.0, fmt="srt"), "01:00:00,000")

    def test_plan_chunks_pathological_inputs(self):
        """Test plan_chunks with zero, negative, or disproportionate inputs."""
        # Zero duration
        self.assertEqual(chunking.plan_chunks(0.0), [])
        # Negative duration
        self.assertEqual(chunking.plan_chunks(-100.0), [])
        # Target duration larger than media
        single = chunking.plan_chunks(100.0, target_duration=500.0)
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0].duration, 100.0)
        # Target duration <= 0 falls back to single chunk
        fallback = chunking.plan_chunks(100.0, target_duration=0.0)
        self.assertEqual(len(fallback), 1)

    def test_plan_chunks_silences_outside_search_window(self):
        """When silences are outside search_window, falls back cleanly to target cutpoint."""
        silences = [100.0, 200.0, 900.0]  # Far from 600s
        chunks = chunking.plan_chunks(
            total_duration=1200.0,
            target_duration=600.0,
            silence_points=silences,
            search_window=30.0,  # [570, 630] has no silences
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].end_sec, 600.0)
        self.assertEqual(chunks[1].start_sec, 600.0)

    def test_stitch_subtitles_out_of_order_chunks(self):
        """Chunks arriving out of order are properly sorted and stitched chronologically."""
        chunk0 = "1\n00:00:01,000 --> 00:00:03,000\nFirst cue\n"
        chunk1 = "1\n00:00:02,000 --> 00:00:04,000\nSecond cue\n"
        chunk2 = "1\n00:00:03,000 --> 00:00:05,000\nThird cue\n"

        # Pass chunks in reverse order: Chunk 2 (offset 1200), Chunk 0 (offset 0), Chunk 1 (offset 600)
        shuffled = [
            (chunk2, 1200.0),
            (chunk0, 0.0),
            (chunk1, 600.0),
        ]

        stitched = chunking.stitch_subtitles(shuffled, fmt="srt")
        lines = [line.strip() for line in stitched.splitlines() if line.strip()]

        # Check sequential cue numbers
        self.assertEqual(lines[0], "1")
        self.assertIn("First cue", lines[2])
        self.assertEqual(lines[3], "2")
        self.assertIn("Second cue", lines[5])
        self.assertEqual(lines[6], "3")
        self.assertIn("Third cue", lines[8])

    def test_stitch_subtitles_empty_and_silent_chunks(self):
        """Empty or silent chunks (no speech detected) do not corrupt final subtitle document."""
        chunk0 = "1\n00:00:01,000 --> 00:00:03,000\nSpoken words\n"
        chunk_silent = ""
        chunk_whitespace = "   \n\t\n  "

        results = [
            (chunk0, 0.0),
            (chunk_silent, 300.0),
            (chunk_whitespace, 600.0),
        ]

        stitched = chunking.stitch_subtitles(results, fmt="srt")
        self.assertIn("1\n00:00:01,000 --> 00:00:03,000\nSpoken words", stitched)
        self.assertEqual(stitched.count("-->"), 1)

    def test_stitch_subtitles_multiline_and_unicode_cues(self):
        """Multi-line cues and international UTF-8 characters are preserved verbatim."""
        chunk0 = (
            "1\n00:00:01,000 --> 00:00:04,000\n"
            "Line 1: Willkommen in München! 🥨\n"
            "Line 2: 日本語字幕と音声テスト 🎌\n"
        )
        stitched = chunking.stitch_subtitles([(chunk0, 0.0)], fmt="srt")
        self.assertIn("Willkommen in München! 🥨", stitched)
        self.assertIn("日本語字幕と音声テスト 🎌", stitched)

    def test_stitch_subtitles_formatting_tags(self):
        """HTML formatting tags (<i>, <b>, <c.yellow>) pass through intact."""
        chunk0 = "1\n00:00:01,000 --> 00:00:03,000\n<i>Italic speech</i> and <b>bold shout</b>\n"
        stitched = chunking.stitch_subtitles([(chunk0, 0.0)], fmt="vtt")
        self.assertIn("<i>Italic speech</i> and <b>bold shout</b>", stitched)

    def test_get_media_duration_resilience(self):
        """get_media_duration returns None when ffprobe fails or media is missing."""
        non_existent = Path("/non/existent/video.mp4")
        self.assertIsNone(chunking.get_media_duration(non_existent))

        # Mock ffprobe failure
        with patch("shutil.which", return_value="/bin/ffprobe"), \
             patch("subprocess.run", side_effect=Exception("ffprobe execution crashed")):
            self.assertIsNone(chunking.get_media_duration(Path("some_file.mp4")))


class TestWorkerEdgeCases(unittest.TestCase):
    """Worker edge case handling for invalid specs, missing media, and conversion quirks."""

    def test_missing_input_video(self):
        """Worker exits with returncode 2 and logs status failure when video is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subs_dir = Path(tmpdir)
            spec = {
                "videos_dir": tmpdir,
                "subs_dir": tmpdir,
                "model": "base",
                "item_id": "ghost_file.mp4",
            }
            with patch("video_transcriber.load_job_spec", return_value=spec):
                ret = video_transcriber.main()
                self.assertEqual(ret, 2)

                status_file = subs_dir / "ghost_file.status.json"
                self.assertTrue(status_file.exists())
                content = status_file.read_text(encoding="utf-8")
                self.assertIn('"phase": "failed"', content)
                self.assertIn("Input video not found", content)

    def test_missing_required_job_spec_settings(self):
        """Worker exits with returncode 2 if required fields are missing."""
        spec = {"videos_dir": "", "subs_dir": "", "model": "", "item_id": ""}
        with patch("video_transcriber.load_job_spec", return_value=spec):
            ret = video_transcriber.main()
            self.assertEqual(ret, 2)

    def test_convert_srt_to_vtt_edge_cases(self):
        """convert_srt_to_vtt handles pre-existing headers and Windows CRLF endings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            srt_path = tmp_path / "test.srt"
            vtt_path = tmp_path / "test.vtt"

            # CRLF endings and non-standard cues
            crlf_content = (
                "1\r\n00:00:01,234 --> 00:00:03,456\r\nHello Windows!\r\n\r\n"
                "2\r\n00:00:05,000 --> 00:00:08,000\r\nLine 2\r\n"
            )
            srt_path.write_bytes(crlf_content.encode("utf-8"))

            video_transcriber.convert_srt_to_vtt(srt_path, vtt_path)
            self.assertTrue(vtt_path.exists())
            converted = vtt_path.read_text(encoding="utf-8")

            self.assertTrue(converted.startswith("WEBVTT"))
            self.assertIn("00:00:01.234 --> 00:00:03.456", converted)
            self.assertIn("00:00:05.000 --> 00:00:08.000", converted)
            # Ensure WEBVTT is not duplicated
            self.assertEqual(converted.count("WEBVTT"), 1)


class TestBridgeSecurityAndValidationEdgeCases(unittest.TestCase):
    """Bridge API security guards and extreme request validation."""

    def test_ssrf_ipv6_loopback(self):
        """IPv6 loopback [::1] is blocked by SSRF validator."""
        with self.assertRaises(HTTPException) as ctx:
            bridge._validate_remote_url("http://[::1]/video.mp4")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_path_traversal_complex_attempts(self):
        """Test directory traversal attempts with varied path patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Attempts to escape tmpdir
            bad_paths = [
                "../../etc/passwd",
                "sub/../../..",
                "dir/../../root",
            ]
            for bad_path in bad_paths:
                with self.assertRaises(HTTPException):
                    bridge._safe_under(tmpdir, bad_path)

    def test_create_job_req_boundary_values(self):
        """Test CreateJobReq boundary constraints for vramFraction and parallelChunks."""
        # Upper bound: vramFraction = 1.0 (valid)
        req = bridge.CreateJobReq(filename="video.mp4", vramFraction=1.0)
        self.assertEqual(req.vramFraction, 1.0)

        # Upper bound violation: vramFraction = 1.0001
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", vramFraction=1.0001)

        # Lower bound violation: vramFraction = 0.0
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", vramFraction=0.0)

        # Lower bound violation: vramFraction = -0.1
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", vramFraction=-0.1)

        # parallelChunks boundary: 1 (valid minimum), 16 (valid maximum)
        req_min = bridge.CreateJobReq(filename="video.mp4", parallelChunks=1)
        self.assertEqual(req_min.parallelChunks, 1)

        req_max = bridge.CreateJobReq(filename="video.mp4", parallelChunks=16)
        self.assertEqual(req_max.parallelChunks, 16)

        # parallelChunks out of bounds: 0 and 17
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", parallelChunks=0)
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", parallelChunks=17)

        # chunkDurationSec boundary: 30 (valid), 29 (invalid)
        req_dur = bridge.CreateJobReq(filename="video.mp4", chunkDurationSec=30)
        self.assertEqual(req_dur.chunkDurationSec, 30)

        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", chunkDurationSec=29)


class TestDispatcherSanitizationEdgeCases(unittest.TestCase):
    """Dispatcher name sanitization against pathological strings."""

    def test_dns1123_label_pathological_inputs(self):
        """dns1123_label sanitizes punctuation, emojis, spaces, and excessive lengths."""
        # Only punctuation -> falls back to 'x'
        self.assertEqual(dispatcher.dns1123_label("---...___"), "x")
        self.assertEqual(dispatcher.dns1123_label(""), "x")
        self.assertEqual(dispatcher.dns1123_label(None), "")

        # Long filename exceeding 63 characters is safely truncated
        long_name = "a" * 100
        sanitized = dispatcher.dns1123_label(long_name, max_len=63)
        self.assertEqual(len(sanitized), 63)
        self.assertTrue(sanitized.startswith("aaaa"))

        # Trailing dashes are stripped
        self.assertEqual(dispatcher.dns1123_label("my-job---"), "my-job")

        # Mixed special characters and spaces
        self.assertEqual(
            dispatcher.dns1123_label("Lecture 01: Introduction & Overview (2026)!"),
            "lecture-01-introduction-overview-2026",
        )

    def test_job_name_for_formatting(self):
        """job_name_for generates valid Kubernetes Job names."""
        name = dispatcher.job_name_for("sample_video.mp4")
        self.assertTrue(name.startswith(f"{dispatcher.JOB_PREFIX}-"))
        self.assertLessEqual(len(name), 63)
        self.assertNotIn(".", name)
        self.assertNotIn("_", name)


if __name__ == "__main__":
    unittest.main()
