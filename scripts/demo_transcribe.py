#!/usr/bin/env python3
"""
scripts/demo_transcribe.py - Interactive CLI Client for whisper-k8s

Zero-external-dependency CLI tool to submit transcription jobs, monitor live
progress with an interactive terminal progress bar, and download output subtitles.

Usage:
    python scripts/demo_transcribe.py
    python scripts/demo_transcribe.py --file demo.mp4 --format vtt --embed
    python scripts/demo_transcribe.py --model small --backend faster-whisper
"""

import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path


# Enable UTF-8 console output if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _get_bar_chars() -> tuple[str, str]:
    """Return solid/empty bar glyphs compatible with stdout encoding."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "█░".encode(enc)
        return "█", "░"
    except Exception:
        return "#", "-"


def render_progress_bar(progress: int, phase: str, message: str, elapsed: float, bar_length: int = 28) -> str:
    """Format an interactive single-line terminal progress bar."""
    fill_ch, empty_ch = _get_bar_chars()
    clamped = max(0, min(100, progress))
    filled_len = int(bar_length * clamped // 100)
    bar = fill_ch * filled_len + empty_ch * (bar_length - filled_len)
    phase_str = f"[{phase}]" if phase else ""
    msg_str = f" - {message}" if message else ""
    return f"\r  [{bar}] {clamped:3d}% {phase_str}{msg_str} ({elapsed:.1f}s)"


def check_health(base_url: str) -> dict:
    """Validate that the bridge server is online and healthy."""
    health_url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(health_url, headers={"User-Agent": "whisper-k8s-cli/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to whisper-k8s at {base_url} ({e.reason}).\n"
            "  Ensure the server is running:\n"
            "    docker compose up -d\n"
            "  Or visit docs/deployment_and_k8s.md for Kubernetes instructions."
        ) from e


def submit_job(base_url: str, payload: dict) -> dict:
    """Submit a transcription job via POST /jobs."""
    url = f"{base_url.rstrip('/')}/jobs"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "whisper-k8s-cli/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_msg)
            detail = err_json.get("detail", err_msg)
        except Exception:
            detail = err_msg
        raise RuntimeError(f"Job submission failed (HTTP {e.code}): {detail}") from e


def get_status(base_url: str, job_id: str) -> dict:
    """Fetch status for a specific job via GET /status/{job_id}."""
    url = f"{base_url.rstrip('/')}/status/{job_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-k8s-cli/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file from the server to local disk."""
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-k8s-cli/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())
    return dest_path


def preview_subtitles(file_path: Path, max_cues: int = 4) -> None:
    """Print the first few subtitle cues cleanly to the terminal."""
    if not file_path.exists():
        return

    print("\n  --- Subtitle Preview ---")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        cues_shown = 0
        current_cue = []
        for line in lines:
            if line.upper() == "WEBVTT":
                continue
            if "-->" in line:
                if current_cue:
                    print("  " + " | ".join(current_cue))
                    cues_shown += 1
                    if cues_shown >= max_cues:
                        break
                    current_cue = []
                current_cue.append(line)
            elif current_cue:
                current_cue.append(line)

        if current_cue and cues_shown < max_cues:
            print("  " + " | ".join(current_cue))
    except Exception as e:
        print(f"  (Could not read preview: {e})")
    print("  ------------------------\n")


def run_transcription(args: argparse.Namespace) -> int:
    """Main execution workflow."""
    base_url = args.url.rstrip("/")
    media_file = Path(args.file)
    filename = media_file.name

    print(f"\n[whisper-k8s] Connecting to bridge at {base_url}...")
    try:
        health = check_health(base_url)
        kube_mode = health.get("kube", "unknown")
        print(f"[whisper-k8s] Connected! Server mode: {kube_mode}")
    except ConnectionError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1

    payload = {
        "filename": filename,
        "backend": args.backend,
        "model": args.model,
        "format": args.format,
        "embed": args.embed,
        "mode": args.mode,
    }
    if args.language:
        payload["language"] = args.language
    if args.compute_type:
        payload["computeType"] = args.compute_type
    if args.vram_fraction:
        payload["vramFraction"] = args.vram_fraction

    print(f"[whisper-k8s] Submitting '{filename}' (backend={args.backend}, model={args.model}, format={args.format})...")
    try:
        resp = submit_job(base_url, payload)
        job_id = resp["jobId"]
        print(f"[whisper-k8s] Job accepted -> ID: {job_id}")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1

    # Poll status until terminal state
    start_time = time.time()
    status = "running"
    sys.stdout.write("  Starting job...\r")
    sys.stdout.flush()

    while True:
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print(f"\n[ERROR] Job timed out after {args.timeout}s", file=sys.stderr)
            return 1

        try:
            status_data = get_status(base_url, job_id)
        except Exception:
            time.sleep(args.poll_interval)
            continue

        status = status_data.get("status", "unknown")
        progress = status_data.get("progress", 0) or 0
        message = status_data.get("message", "")

        # Infer phase from message or flavor
        phase = "running"
        if "loading" in message.lower():
            phase = "loading_model"
        elif "transcribing" in message.lower():
            phase = "transcribing"
        elif "embed" in message.lower():
            phase = "embedding"
        elif status == "succeeded":
            phase = "done"
            progress = 100

        sys.stdout.write(render_progress_bar(progress, phase, message, elapsed))
        sys.stdout.flush()

        if status in ("succeeded", "failed"):
            break

        time.sleep(args.poll_interval)

    total_time = time.time() - start_time
    print()  # newline after progress bar

    if status == "failed":
        print(f"\n[FAILED] Job failed: {status_data.get('message', 'Unknown error')}", file=sys.stderr)
        return 1

    print(f"[whisper-k8s] Transcription completed successfully in {total_time:.2f}s!")

    # Download output subtitle
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_filename = f"{media_file.stem}.{args.format}"
    sub_dest = out_dir / sub_filename
    sub_download_url = f"{base_url}/jobs/{job_id}/download?format={args.format}"

    print(f"[whisper-k8s] Downloading subtitle to {sub_dest}...")
    try:
        download_file(sub_download_url, sub_dest)
        print(f"[whisper-k8s] Saved: {sub_dest} ({sub_dest.stat().st_size} bytes)")
        preview_subtitles(sub_dest)
    except Exception as e:
        print(f"[WARNING] Could not download subtitle: {e}", file=sys.stderr)

    # If embedded MP4 was requested, download it
    if args.embed:
        embed_filename = f"{media_file.stem}.embedded.mp4"
        embed_dest = out_dir / embed_filename
        embed_download_url = f"{base_url}/jobs/{job_id}/download?format=embedded"
        print(f"[whisper-k8s] Downloading embedded video to {embed_dest}...")
        try:
            download_file(embed_download_url, embed_dest)
            print(f"[whisper-k8s] Saved: {embed_dest} ({embed_dest.stat().st_size} bytes)")
        except Exception as e:
            print(f"[WARNING] Could not download embedded video: {e}", file=sys.stderr)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="whisper-k8s interactive CLI transcription client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", default="demo.mp4", help="Video/audio file to transcribe")
    parser.add_argument("--url", default="http://localhost:8080", help="Bridge server URL")
    parser.add_argument("--backend", default="faster-whisper", choices=["faster-whisper", "whisper", "whisper.cpp"], help="Whisper inference backend")
    parser.add_argument("--model", default="base", help="Whisper model name (tiny, base, small, medium, large-v3)")
    parser.add_argument("--format", default="srt", choices=["srt", "vtt"], help="Subtitle format")
    parser.add_argument("--embed", action="store_true", help="Embed subtitles directly into an MP4 video")
    parser.add_argument("--mode", default="pool", choices=["pool", "kubernetes", "pod", "ssh"], help="Execution mode")
    parser.add_argument("--language", default=None, help="Language code (e.g. 'en', 'de', or None for auto-detect)")
    parser.add_argument("--compute-type", default=None, choices=["int8_float16", "float16", "int8", "float32"], help="Compute quantization precision")
    parser.add_argument("--vram-fraction", type=float, default=None, help="VRAM limit fraction (e.g. 0.25)")
    parser.add_argument("--out", default=".", help="Local output directory for downloaded results")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=300.0, help="Maximum execution timeout in seconds")

    args = parser.parse_args()
    sys.exit(run_transcription(args))


if __name__ == "__main__":
    main()
