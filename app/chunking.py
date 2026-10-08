#!/usr/bin/env python3
"""
Audio chunking and subtitle stitching module for long media files.

Enables splitting long audio/video files at natural silence boundaries,
processing chunks in parallel across worker pods, and stitching the resulting
subtitle segments with precise global timestamp offsets and cue deduplication.
"""

import os
import re
import sys
import shutil
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkSpec:
    """Specification of an audio/video chunk."""
    index: int
    start_sec: float
    end_sec: float
    duration: float


# ---------------------------------------------------------------------------
# Timestamp Helpers
# ---------------------------------------------------------------------------

def parse_timestamp(ts_str: str) -> float:
    """
    Parse a subtitle timestamp string into seconds.
    Supports SRT format (00:01:23,456) and WebVTT format (00:01:23.456 or 01:23.456).
    """
    s = ts_str.strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    else:
        return float(s)


def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """
    Format a float number of seconds into an SRT (00:00:00,000) or
    WebVTT (00:00:00.000) timestamp string.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder %= 60000
    whole_secs = remainder // 1000
    millis = remainder % 1000

    sep = "," if fmt.lower() == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{whole_secs:02d}{sep}{millis:03d}"


# ---------------------------------------------------------------------------
# Subtitle Parsing & Stitching
# ---------------------------------------------------------------------------

TIMESTAMP_LINE_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3}|\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3}|\d{2}:\d{2}[,\.]\d{3})"
)


def parse_cues(content: str) -> List[Dict[str, Any]]:
    """
    Parse subtitle content (SRT or WebVTT) into structured cue dictionaries:
    [{'start': float, 'end': float, 'text': str}, ...]
    """
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = TIMESTAMP_LINE_RE.search(line)
        if m:
            start_sec = parse_timestamp(m.group(1))
            end_sec = parse_timestamp(m.group(2))
            # Text follows the timestamp line until the next blank line
            text_lines: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = "\n".join(text_lines)
            if text:
                cues.append({"start": start_sec, "end": end_sec, "text": text})
        else:
            i += 1

    return cues


def stitch_subtitles(
    chunk_results: List[Tuple[str, float]],
    fmt: str = "srt",
    dedupe_overlap_window: float = 2.0,
) -> str:
    """
    Stitch subtitle segments from multiple chunks into a single unified subtitle document.

    Parameters
    ----------
    chunk_results:
        List of tuples: (subtitle_content_str, chunk_start_offset_seconds).
    fmt:
        Output format: "srt" or "vtt".
    dedupe_overlap_window:
        Window in seconds to deduplicate identical text cues across chunk borders.

    Returns
    -------
    str
        Full combined and re-indexed subtitle file contents.
    """
    fmt = fmt.lower().strip()
    if fmt not in {"srt", "vtt"}:
        fmt = "srt"

    # Sort chunks by start offset
    sorted_chunks = sorted(chunk_results, key=lambda x: x[1])

    all_cues: List[Dict[str, Any]] = []
    for content, offset in sorted_chunks:
        if not content or not content.strip():
            continue
        cues = parse_cues(content)
        for c in cues:
            all_cues.append({
                "start": c["start"] + offset,
                "end": c["end"] + offset,
                "text": c["text"],
            })

    if not all_cues:
        return "WEBVTT\n\n" if fmt == "vtt" else ""

    # Sort all cues by global start timestamp
    all_cues.sort(key=lambda c: (c["start"], c["end"]))

    # Deduplicate overlapping boundary cues
    deduped_cues: List[Dict[str, Any]] = []
    for c in all_cues:
        if deduped_cues:
            prev = deduped_cues[-1]
            # If same text and overlapping or very close in time, merge/skip
            if prev["text"] == c["text"] and abs(c["start"] - prev["start"]) <= dedupe_overlap_window:
                # Extend previous cue end if this cue ends later
                prev["end"] = max(prev["end"], c["end"])
                continue
            # If current cue starts earlier than previous cue ends with slight overlap
            if c["start"] < prev["start"]:
                c["start"] = prev["end"]
            if c["end"] <= c["start"]:
                continue
        deduped_cues.append(c)

    # Format output
    output_lines: List[str] = []
    if fmt == "vtt":
        output_lines.append("WEBVTT\n")

    for idx, cue in enumerate(deduped_cues, start=1):
        start_str = format_timestamp(cue["start"], fmt=fmt)
        end_str = format_timestamp(cue["end"], fmt=fmt)
        if fmt == "srt":
            output_lines.append(f"{idx}\n{start_str} --> {end_str}\n{cue['text']}\n")
        else:
            output_lines.append(f"{start_str} --> {end_str}\n{cue['text']}\n")

    return "\n".join(output_lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Silence Boundary Detection & Chunk Planning
# ---------------------------------------------------------------------------

def detect_silence_points(
    media_path: Path,
    min_silence_len: float = 0.5,
    noise_threshold: str = "-30dB",
) -> List[float]:
    """
    Detect silence points in an audio/video file using ffmpeg's silencedetect filter.
    Returns a sorted list of timestamps (in seconds) representing the middle of silences.
    """
    if not shutil.which("ffmpeg") or not media_path.exists():
        return []

    cmd = [
        "ffmpeg",
        "-nostats",
        "-i", str(media_path),
        "-af", f"silencedetect=noise={noise_threshold}:d={min_silence_len}",
        "-f", "null",
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        output = proc.stderr or ""
    except Exception as e:
        logger.warning("silencedetect execution failed: %s", e)
        return []

    # Parse silence_start and silence_end
    # Format: [silencedetect @ ...] silence_start: 123.45
    # Format: [silencedetect @ ...] silence_end: 124.12 | silence_duration: 0.67
    silence_starts: List[float] = []
    silence_ends: List[float] = []
    for line in output.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([\d\.]+)", line)
            if m:
                silence_starts.append(float(m.group(1)))
        elif "silence_end:" in line:
            m = re.search(r"silence_end:\s*([\d\.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))

    midpoints: List[float] = []
    for i in range(min(len(silence_starts), len(silence_ends))):
        start = silence_starts[i]
        end = silence_ends[i]
        if end > start:
            midpoints.append(round((start + end) / 2.0, 3))

    return sorted(midpoints)


def plan_chunks(
    total_duration: float,
    target_duration: float = 600.0,
    silence_points: Optional[List[float]] = None,
    min_chunk_duration: float = 60.0,
    search_window: float = 60.0,
) -> List[ChunkSpec]:
    """
    Plan cutpoints for media of a given duration, preferring silence boundaries
    within a search window near multiples of target_duration.

    Parameters
    ----------
    total_duration:
        Total media duration in seconds.
    target_duration:
        Desired duration per chunk (e.g. 600.0s = 10 minutes).
    silence_points:
        List of known silence timestamps in seconds.
    min_chunk_duration:
        Minimum duration for a chunk to avoid tiny trailing fragments.
    search_window:
        Window (±seconds) around the target cut point to look for silence.

    Returns
    -------
    List[ChunkSpec]
        List of planned chunks spanning [0.0, total_duration].
    """
    if total_duration <= 0:
        return []

    if total_duration <= target_duration or target_duration <= 0:
        return [ChunkSpec(index=0, start_sec=0.0, end_sec=total_duration, duration=total_duration)]

    cut_points: List[float] = [0.0]
    silence_points = sorted(silence_points or [])

    current_start = 0.0
    while True:
        ideal_cut = current_start + target_duration
        if ideal_cut >= total_duration:
            break

        # Check remaining duration if we cut at ideal_cut
        remaining = total_duration - ideal_cut
        if remaining < min_chunk_duration:
            # Not enough left for another chunk, merge with the last chunk
            break

        # Search for silence in [ideal_cut - search_window, ideal_cut + search_window]
        candidate_silences = [
            s for s in silence_points
            if (ideal_cut - search_window) <= s <= (ideal_cut + search_window)
            and (s - current_start) >= min_chunk_duration
            and (total_duration - s) >= min_chunk_duration
        ]

        if candidate_silences:
            # Pick the silence point closest to ideal_cut
            best_cut = min(candidate_silences, key=lambda s: abs(s - ideal_cut))
        else:
            best_cut = ideal_cut

        cut_points.append(round(best_cut, 3))
        current_start = best_cut

    cut_points.append(round(total_duration, 3))

    # Build ChunkSpec list
    chunks: List[ChunkSpec] = []
    for i in range(len(cut_points) - 1):
        st = cut_points[i]
        en = cut_points[i + 1]
        chunks.append(ChunkSpec(
            index=i,
            start_sec=st,
            end_sec=en,
            duration=round(en - st, 3),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Audio Splitting
# ---------------------------------------------------------------------------

def get_media_duration(media_path: Path) -> Optional[float]:
    """Retrieve media duration in seconds using ffprobe."""
    if not shutil.which("ffprobe") or not media_path.exists():
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        val = proc.stdout.strip()
        if val:
            return float(val)
    except Exception as e:
        logger.warning("ffprobe duration check failed: %s", e)

    return None


def split_audio_into_chunks(
    input_path: Path,
    output_dir: Path,
    chunks: List[ChunkSpec],
    overlap_pad: float = 0.2,
) -> List[Path]:
    """
    Split input media into 16kHz mono WAV chunks for speech-to-text processing.
    Adds a small overlap padding to the chunk bounds to avoid border clipping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: List[Path] = []

    for c in chunks:
        out_file = output_dir / f"chunk_{c.index:03d}.wav"
        chunk_paths.append(out_file)

        # Pad bounds slightly within media bounds
        start_padded = max(0.0, c.start_sec - (overlap_pad if c.index > 0 else 0.0))
        duration_padded = (c.end_sec - start_padded) + overlap_pad

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start_padded:.3f}",
            "-t", f"{duration_padded:.3f}",
            "-i", str(input_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(out_file),
        ]

        logger.info("[chunk] extracting chunk %d (%.2fs - %.2fs) -> %s", c.index, c.start_sec, c.end_sec, out_file)
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            logger.error("Failed to extract chunk %d: %s", c.index, e)
            raise RuntimeError(f"Audio chunk extraction failed for chunk {c.index}: {e}")

    return chunk_paths
