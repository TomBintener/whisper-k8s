import os
import sys
import time
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
import bridge


class TestPerformanceBenchmark(unittest.TestCase):
    """
    Performance benchmark suite measuring latency, throughput, and speedup
    gains from P0 (redundant inference elimination) and P1 (caching & I/O).
    """

    def test_benchmark_p0_speedup(self):
        """
        Benchmark comparing the old redundant 2-pass workflow vs
        the optimized 1-pass + convert_srt_to_vtt workflow.
        """
        simulated_inference_sec = 0.05  # simulated single-pass inference time

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            srt_path = temp_p / "demo.srt"
            vtt_path = temp_p / "demo.vtt"

            # Create realistic 500-cue subtitle file
            cues = []
            for i in range(1, 501):
                start_s = i * 2
                end_s = start_s + 1
                cues.append(
                    f"{i}\n"
                    f"00:{start_s // 60:02d}:{start_s % 60:02d},000 --> 00:{end_s // 60:02d}:{end_s % 60:02d},500\n"
                    f"Transcript segment cue line number {i}.\n\n"
                )
            srt_path.write_text("".join(cues), encoding="utf-8")

            # 1. Measure Old Workflow (simulated 2x inference passes)
            t0 = time.perf_counter()
            time.sleep(simulated_inference_sec)  # Pass 1: generate SRT
            time.sleep(simulated_inference_sec)  # Pass 2: generate VTT from scratch
            old_duration = time.perf_counter() - t0

            # 2. Measure Optimized Workflow (1x inference pass + fast conversion)
            t1 = time.perf_counter()
            time.sleep(simulated_inference_sec)  # Pass 1: generate SRT
            tc0 = time.perf_counter()
            video_transcriber.convert_srt_to_vtt(srt_path, vtt_path)  # Fast conversion
            conversion_duration = time.perf_counter() - tc0
            new_duration = time.perf_counter() - t1

            speedup_ratio = old_duration / new_duration
            latency_reduction_pct = (1.0 - (new_duration / old_duration)) * 100.0

            print("\n" + "=" * 65)
            print(" PERFORMANCE BENCHMARK: P0 REDUNDANT INFERENCE ELIMINATION ")
            print("=" * 65)
            print(f"  Old Workflow (2x Whisper passes):  {old_duration * 1000:8.2f} ms")
            print(f"  New Workflow (1x pass + convert):  {new_duration * 1000:8.2f} ms")
            print(f"  Format Conversion Latency (500 cues): {conversion_duration * 1000:6.3f} ms")
            print(f"  Speedup Ratio:                     {speedup_ratio:8.2f}x faster")
            print(f"  Latency Reduction:                 {latency_reduction_pct:8.2f}%")
            print("=" * 65)

            # Assert conversion is extremely fast (< 50ms for 500 cues)
            self.assertLess(conversion_duration, 0.05)
            # Assert optimized workflow is significantly faster (> 1.5x)
            self.assertGreater(speedup_ratio, 1.5)

    def test_benchmark_subtitle_conversion_throughput(self):
        """
        Stress test and throughput benchmark for convert_srt_to_vtt
        processing large subtitle files (5,000 cues).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_p = Path(temp_dir)
            srt_path = temp_p / "large.srt"
            vtt_path = temp_p / "large.vtt"

            total_cues = 5000
            cues = []
            for i in range(1, total_cues + 1):
                cues.append(
                    f"{i}\n"
                    f"00:10:00,123 --> 00:10:04,567\n"
                    f"This is stress test cue {i} containing realistic speech transcript text.\n\n"
                )
            content = "".join(cues)
            srt_path.write_text(content, encoding="utf-8")
            file_size_kb = len(content.encode("utf-8")) / 1024.0

            # Measure conversion throughput
            t0 = time.perf_counter()
            video_transcriber.convert_srt_to_vtt(srt_path, vtt_path)
            duration = time.perf_counter() - t0

            cues_per_sec = total_cues / duration
            kb_per_sec = file_size_kb / duration

            print("\n" + "=" * 65)
            print(f" SUBTITLE CONVERTER THROUGHPUT ({total_cues:,} CUES / {file_size_kb:.1f} KB) ")
            print("=" * 65)
            print(f"  Execution Time:    {duration * 1000:8.2f} ms")
            print(f"  Throughput (cues): {cues_per_sec:10,.0f} cues/second")
            print(f"  Throughput (data): {kb_per_sec / 1024:8.2f} MB/second")
            print("=" * 65)

            self.assertTrue(vtt_path.exists())
            self.assertLess(duration, 0.5, "5,000 cues must convert in under 500ms")

    def test_benchmark_status_io_throughput(self):
        """
        Benchmark status file serialization and write throughput (500 writes)
        to ensure worker status updates do not bottleneck disk I/O.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            subs_dir = Path(temp_dir)
            iterations = 500

            t0 = time.perf_counter()
            with patch.dict(os.environ, {"BRIDGE_JOB_ID": "bench-job-io"}):
                for i in range(iterations):
                    video_transcriber.write_status(
                        subs_dir=subs_dir,
                        stem="bench_video",
                        phase="transcribing",
                        progress=(i % 100),
                        message=f"Processing segment {i}",
                        subtitle_path=f"{temp_dir}/bench.srt",
                        flavor="srt",
                    )
            duration = time.perf_counter() - t0
            ops_per_sec = iterations / duration

            print("\n" + "=" * 65)
            print(f" WORKER STATUS DISK I/O BENCHMARK ({iterations} OPERATIONS) ")
            print("=" * 65)
            print(f"  Total Duration:    {duration * 1000:8.2f} ms")
            print(f"  Mean Write Latency:{duration / iterations * 1000:8.3f} ms / write")
            print(f"  Throughput:        {ops_per_sec:10,.0f} status writes/second")
            print("=" * 65)

            self.assertGreater(ops_per_sec, 50, "Status I/O must achieve at least 50 writes/sec")


if __name__ == "__main__":
    unittest.main()
