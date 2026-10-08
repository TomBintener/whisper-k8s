#!/usr/bin/env python3
"""
Unit and integration tests for P4: Audio Chunking, Silence Boundary Planning,
Timestamp Offset Stitching, and Parallel Sub-Job Assembly.
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


class TestTimestampHelpers(unittest.TestCase):
    """Test SRT and WebVTT timestamp parsing and formatting."""

    def test_parse_timestamp_srt(self):
        self.assertAlmostEqual(chunking.parse_timestamp("00:00:01,500"), 1.5, places=3)
        self.assertAlmostEqual(chunking.parse_timestamp("01:02:03,456"), 3723.456, places=3)

    def test_parse_timestamp_vtt(self):
        self.assertAlmostEqual(chunking.parse_timestamp("00:00:01.500"), 1.5, places=3)
        self.assertAlmostEqual(chunking.parse_timestamp("02:03.456"), 123.456, places=3)

    def test_format_timestamp_srt(self):
        self.assertEqual(chunking.format_timestamp(1.5, fmt="srt"), "00:00:01,500")
        self.assertEqual(chunking.format_timestamp(3723.456, fmt="srt"), "01:02:03,456")

    def test_format_timestamp_vtt(self):
        self.assertEqual(chunking.format_timestamp(1.5, fmt="vtt"), "00:00:01.500")
        self.assertEqual(chunking.format_timestamp(3723.456, fmt="vtt"), "01:02:03.456")


class TestChunkPlanning(unittest.TestCase):
    """Test chunk boundary calculations and silence alignment."""

    def test_plan_chunks_short_media(self):
        """Media shorter than target duration yields a single full chunk."""
        chunks = chunking.plan_chunks(total_duration=300.0, target_duration=600.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[0].start_sec, 0.0)
        self.assertEqual(chunks[0].end_sec, 300.0)

    def test_plan_chunks_regular_split(self):
        """Media longer than target duration splits into multiple chunks."""
        chunks = chunking.plan_chunks(total_duration=1800.0, target_duration=600.0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start_sec, 0.0)
        self.assertEqual(chunks[0].end_sec, 600.0)
        self.assertEqual(chunks[1].start_sec, 600.0)
        self.assertEqual(chunks[1].end_sec, 1200.0)
        self.assertEqual(chunks[2].start_sec, 1200.0)
        self.assertEqual(chunks[2].end_sec, 1800.0)

    def test_plan_chunks_silence_alignment(self):
        """Cutpoints snap to nearby silence points rather than rigid interval."""
        # Target cut at 600s, silence at 592s and 608s. 592s is closer than 608s (abs diff 8 vs 8, picks first/closest)
        silences = [200.0, 594.0, 1195.0, 1500.0]
        chunks = chunking.plan_chunks(
            total_duration=1800.0,
            target_duration=600.0,
            silence_points=silences,
            search_window=30.0,
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].end_sec, 594.0)
        self.assertEqual(chunks[1].start_sec, 594.0)
        self.assertEqual(chunks[1].end_sec, 1195.0)
        self.assertEqual(chunks[2].start_sec, 1195.0)
        self.assertEqual(chunks[2].end_sec, 1800.0)

    def test_plan_chunks_min_chunk_duration_merge(self):
        """Tiny trailing segment (< min_chunk_duration) merges with previous chunk."""
        # 630s total with 600s target and 60s min_chunk_duration -> single 630s chunk
        chunks = chunking.plan_chunks(
            total_duration=630.0,
            target_duration=600.0,
            min_chunk_duration=60.0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].duration, 630.0)


class TestSubtitleStitching(unittest.TestCase):
    """Test parsing, global timestamp offset shifting, cue renumbering, and deduplication."""

    def test_parse_cues_srt_and_vtt(self):
        srt_sample = (
            "1\n00:00:01,000 --> 00:00:04,000\nFirst line\n\n"
            "2\n00:00:05,500 --> 00:00:08,000\nSecond line\n"
        )
        cues = chunking.parse_cues(srt_sample)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["start"], 1.0)
        self.assertEqual(cues[0]["end"], 4.0)
        self.assertEqual(cues[0]["text"], "First line")
        self.assertEqual(cues[1]["start"], 5.5)
        self.assertEqual(cues[1]["end"], 8.0)
        self.assertEqual(cues[1]["text"], "Second line")

    def test_stitch_subtitles_srt(self):
        """Verify chunk subtitle cues receive exact offset additions and sequential renumbering."""
        chunk0_subs = (
            "1\n00:00:02,000 --> 00:00:05,000\nIntro text\n\n"
            "2\n00:00:08,000 --> 00:00:12,000\nWelcome text\n"
        )
        chunk1_subs = (
            "1\n00:00:01,000 --> 00:00:06,000\nPart two text\n"
        )

        chunk_results = [
            (chunk0_subs, 0.0),      # Offset 0s
            (chunk1_subs, 600.0),    # Offset 600s (10 minutes)
        ]

        stitched = chunking.stitch_subtitles(chunk_results, fmt="srt")

        self.assertIn("1\n00:00:02,000 --> 00:00:05,000\nIntro text", stitched)
        self.assertIn("2\n00:00:08,000 --> 00:00:12,000\nWelcome text", stitched)
        # Part two text was at 00:00:01 + 600s = 00:10:01,000
        self.assertIn("3\n00:10:01,000 --> 00:10:06,000\nPart two text", stitched)

    def test_stitch_subtitles_vtt(self):
        """Verify WebVTT output includes standard WEBVTT header and period separators."""
        chunk0 = "00:01.000 --> 00:03.000\nHello VTT"
        stitched = chunking.stitch_subtitles([(chunk0, 0.0)], fmt="vtt")
        self.assertTrue(stitched.startswith("WEBVTT"))
        self.assertIn("00:00:01.000 --> 00:00:03.000\nHello VTT", stitched)

    def test_stitch_subtitles_boundary_deduplication(self):
        """Verify identical speech cues across overlap boundaries are deduplicated."""
        chunk0_subs = "1\n00:09:58,000 --> 00:10:02,000\nShared boundary sentence\n"
        # Overlap in chunk 1 (started at 600s, so cue at 0s is 600s = 10:00)
        chunk1_subs = "1\n00:00:00,000 --> 00:00:04,000\nShared boundary sentence\n"

        chunk_results = [
            (chunk0_subs, 0.0),
            (chunk1_subs, 600.0),
        ]

        stitched = chunking.stitch_subtitles(chunk_results, fmt="srt")
        # Should only contain 1 cue, not 2 duplicates
        self.assertEqual(stitched.count("Shared boundary sentence"), 1)


class TestBridgeDispatcherChunking(unittest.TestCase):
    """Test bridge and dispatcher configuration, validation, and env propagation."""

    def test_create_job_req_chunking_validation(self):
        """Verify CreateJobReq validates parallelChunks and chunkDurationSec."""
        # Valid chunking request
        req = bridge.CreateJobReq(
            filename="long_video.mp4",
            parallelChunks=4,
            chunkDurationSec=300,
            enableChunking=True,
        )
        self.assertEqual(req.parallelChunks, 4)
        self.assertEqual(req.chunkDurationSec, 300)
        self.assertTrue(req.enableChunking)

        # Invalid parallelChunks > 16
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", parallelChunks=20)

        # Invalid parallelChunks < 1
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", parallelChunks=0)

        # Invalid chunkDurationSec < 30s
        with self.assertRaises(ValueError):
            bridge.CreateJobReq(filename="video.mp4", chunkDurationSec=15)

    def test_make_worker_job_injects_chunking_env(self):
        """Verify _make_worker_job passes chunking parameters to worker container env."""
        job = bridge._make_worker_job(
            job_id="test-p4-job",
            filename="video.mp4",
            fmt="vtt",
            parallel_chunks=4,
            chunk_duration_sec=300,
            enable_chunking=True,
        )
        container = job.spec.template.spec.containers[0]
        env_map = {e.name: e.value for e in container.env}

        self.assertEqual(env_map.get("PARALLEL_CHUNKS"), "4")
        self.assertEqual(env_map.get("CHUNK_DURATION_SEC"), "300")
        self.assertEqual(env_map.get("ENABLE_CHUNKING"), "true")

    def test_dispatcher_build_job_chunking_env(self):
        """Verify dispatcher.build_job includes chunking env vars when configured."""
        with patch.dict(dispatcher.ENV, {
            "PARALLEL_CHUNKS": "8",
            "CHUNK_DURATION_SEC": "600",
            "ENABLE_CHUNKING": "true",
        }):
            dispatcher.PARALLEL_CHUNKS = "8"
            dispatcher.CHUNK_DURATION_SEC = "600"
            dispatcher.ENABLE_CHUNKING = "true"

            job = dispatcher.build_job(item_id="long-batch-item.mp4")
            container = job.spec.template.spec.containers[0]
            env_map = {e.name: e.value for e in container.env}

            self.assertEqual(env_map.get("PARALLEL_CHUNKS"), "8")
            self.assertEqual(env_map.get("CHUNK_DURATION_SEC"), "600")
            self.assertEqual(env_map.get("ENABLE_CHUNKING"), "true")


class TestWorkerChunkingExecution(unittest.TestCase):
    """Test full worker execution pipeline when chunking is activated."""

    def test_worker_chunking_orchestration(self):
        """Verify worker splits media, transcribes chunks in parallel, and stitches master subtitle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_file = tmp_path / "lecture.mp4"
            video_file.write_bytes(b"dummy_media_bytes")

            spec = {
                "videos_dir": str(tmp_path),
                "subs_dir": str(tmp_path),
                "model": "base",
                "item_id": "lecture.mp4",
                "format": "srt",
                "enable_chunking": True,
                "parallel_chunks": 2,
                "chunk_duration_sec": 300,
            }

            # Mock duration probe to report 600s
            mock_duration = 600.0

            # Mock chunk splitting to return 2 fake files
            chunk0 = tmp_path / "chunk_000.wav"
            chunk1 = tmp_path / "chunk_001.wav"
            chunk0.write_bytes(b"wav0")
            chunk1.write_bytes(b"wav1")

            def mock_transcribe_one(*args, **kwargs):
                video_p = kwargs.get("video_path")
                subs_d = kwargs.get("subs_dir")
                out_p = subs_d / f"{video_p.stem}.srt"
                if "000" in video_p.name:
                    out_p.write_text("1\n00:00:01,000 --> 00:00:04,000\nChunk 0 Speech\n", encoding="utf-8")
                else:
                    out_p.write_text("1\n00:00:02,000 --> 00:00:05,000\nChunk 1 Speech\n", encoding="utf-8")
                return out_p, "en"

            with patch("video_transcriber.load_job_spec", return_value=spec), \
                 patch("video_transcriber.load_model", return_value=MagicMock()), \
                 patch("chunking.get_media_duration", return_value=mock_duration), \
                 patch("chunking.detect_silence_points", return_value=[305.0]), \
                 patch("chunking.split_audio_into_chunks", return_value=[chunk0, chunk1]), \
                 patch("video_transcriber.transcribe_one", side_effect=mock_transcribe_one):

                ret = video_transcriber.main()
                self.assertEqual(ret, 0)

                # Verify master stitched subtitle was written
                master_srt = tmp_path / "lecture.srt"
                self.assertTrue(master_srt.exists(), "Master stitched SRT was not generated")
                content = master_srt.read_text(encoding="utf-8")

                self.assertIn("Chunk 0 Speech", content)
                self.assertIn("Chunk 1 Speech", content)
                # Chunk 1 starts at 305s (05:05), so cue at 2s becomes 307s (05:07)
                self.assertIn("00:05:07,000 --> 00:05:10,000", content)


if __name__ == "__main__":
    unittest.main()
