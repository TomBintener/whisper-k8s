#!/usr/bin/env python3
"""
Worker script that runs a single Whisper transcription job.

It reads configuration from a JSON job spec and environment variables, loads
a Whisper model on CPU, CUDA, or MPS, generates subtitles in SRT or VTT
format, and optionally embeds them into an MP4 file using ffmpeg.
"""

import os
import sys
import json
import shlex
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Lazy imports for backends that might not be installed
try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    import whisper  # type: ignore
    from whisper.utils import get_writer  # type: ignore
except ImportError:
    whisper = None
    get_writer = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

# Default decoding settings for both Whisper and faster-whisper.
# Tuned for long recordings to avoid repetition and reduce issues in silence.
DEFAULT_WHISPER_ARGS: dict[str, object] = {
    "temperature": 0.0,
    "beam_size": 5,
    "patience": 1.0,
    "no_speech_threshold": 0.7,
    # Key for long files: do not condition each window on previous text.
    "condition_on_previous_text": False,
}

logger = logging.getLogger(__name__)


def write_status(
    subs_dir: Path,
    stem: str,
    phase: str,
    progress: Optional[int] = None,
    message: Optional[str] = None,
) -> None:
    """Write a simple JSON status file for this job into the subtitle directory."""
    try:
        ensure_dir(subs_dir)
        status_path = subs_dir / f"{stem}.status.json"
        data: Dict[str, Any] = {"phase": phase}
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("[status] %s -> %s", status_path, data)
    except Exception as e:
        logger.warning("Could not write status file: %s", e)


def as_bool(v: Any, default: bool = False) -> bool:
    """Convert various truthy or falsy representations to a boolean."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def ensure_dir(p: Path) -> None:
    """Create a directory and all missing parents."""
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> int:
    """Run a subprocess command, echoing it and streaming stdout and stderr."""
    logger.info("[cmd] %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc.returncode


def load_job_spec() -> Dict[str, Any]:
    """Load a per job JSON spec from disk or environment."""
    path = os.environ.get("JOB_JSON_PATH", "/config/job.json").strip()
    try:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                spec = json.load(f)
            logger.info("[spec] loaded job spec from %s", path)
            return spec
    except Exception as e:
        logger.warning("Could not read job spec at %s: %s", path, e)

    s = os.environ.get("JOB_JSON")
    if s:
        try:
            spec = json.loads(s)
            logger.info("[spec] loaded job spec from JOB_JSON env")
            return spec
        except Exception as e:
            logger.warning("Invalid JOB_JSON: %s", e)
    return {}


def pick_device(selection: str) -> str:
    """Choose the actual Torch device to use based on a high level selection."""
    if torch is None:
        return "cpu"

    sel = (selection or "auto").strip().lower()
    if sel == "cpu":
        return "cpu"
    if sel in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("GPU requested but CUDA not available. Using CPU.")
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(name: str, device: str, download_root: Optional[str] = None):
    """Load a Whisper model with the given name on the requested device."""
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed")
    logger.info("[model] loading '%s' on %s", name, device)
    return whisper.load_model(name, device=device, download_root=download_root)


def load_faster_model(name: str, device: str, compute_type: str = "float16"):
    """Load a faster-whisper model on GPU or CPU."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")

    if device.startswith("cuda"):
        fw_device = "cuda"
    else:
        fw_device = "cpu"

    logger.info("[model] loading faster-whisper '%s' on %s", name, fw_device)
    return WhisperModel(name, device=fw_device, compute_type=compute_type)


def transcribe_whisper_cpp(
    model_name: str,
    video_path: Path,
    subs_dir: Path,
    language: Optional[str],
    task: str,
    fmt: str = "vtt",
    overwrite: bool = False,
) -> tuple[Path, Optional[str]]:
    """
    Transcribe using whisper.cpp executable.
    
    Assumes 'whisper-cli' is in the PATH.
    """
    executable = os.environ.get("WHISPER_CPP_EXEC")
    if not executable:
        executable = shutil.which("whisper-cli") or shutil.which("main")

    if not executable:
        raise RuntimeError(
            "Could not find 'whisper-cli' or 'main' (from whisper.cpp) in PATH. "
            "Please ensure whisper.cpp is compiled and its location is in the system's PATH."
        )
        
    if not os.path.exists(executable):
        # This catches cases where the env var points to a non-existent file
        raise RuntimeError(f"The configured whisper.cpp executable does not exist at: {executable}")

    fmt = fmt.lower().strip()
    if fmt not in {"srt", "vtt"}:
        fmt = "vtt"

    ensure_dir(subs_dir)
    # whisper.cpp output filename logic: <input>.vtt or <input>.srt
    # We want to control the output path, but whisper.cpp writes to the same dir as input or specific output file.
    # The CLI usually outputs <input_filename>.<fmt>
    
    # We will let whisper.cpp write to a temp location or directly to subs_dir if possible.
    # whisper.cpp -f input.wav -osrt -of output_base
    
    out_base = subs_dir / video_path.stem
    out_path = subs_dir / f"{video_path.stem}.{fmt}"
    
    if out_path.exists() and not overwrite:
        logger.info("[whisper.cpp] exists -> %s", out_path)
        return out_path, None

    # Map model name to whisper.cpp model file path
    # You need to ensure models are downloaded to a known location
    model_root = os.environ.get("WHISPER_CPP_MODEL_ROOT", "/app/models")
    model_path = Path(model_root) / f"ggml-{model_name}.bin"
    
    if not model_path.exists():
        raise RuntimeError(f"whisper.cpp model not found at {model_path}")

    # whisper.cpp requires a 16kHz, 16-bit mono WAV file.
    # We will use ffmpeg to extract a temporary WAV from the video.
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Fallback for non-interactive SSH sessions (macOS Homebrew paths)
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if os.path.exists(p):
                ffmpeg_path = p
                break
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required but not found in PATH. Please install it via Homebrew.")

    temp_wav = subs_dir / f"{video_path.stem}_temp_16k.wav"
    try:
        logger.info("[ffmpeg] extracting 16kHz audio to %s", temp_wav)
        ffmpeg_cmd = [
            ffmpeg_path, "-y",
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(temp_wav)
        ]
        
        rc = run_cmd(ffmpeg_cmd)
        if rc != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed with code {rc}")

        # Now run whisper.cpp on the perfectly formatted WAV file
        cmd = [
            executable,
            "-m", str(model_path),
            "--output-file", str(out_base),
            str(temp_wav), # Positional argument for input file
        ]
        
        if fmt == "vtt":
            cmd.append("--output-vtt")
        else:
            cmd.append("--output-srt")
            
        if language:
            cmd.extend(["-l", language])
        else:
            cmd.extend(["-l", "auto"])
            
        if task == "translate":
            cmd.append("-tr")

        logger.info("[whisper.cpp] running -> %s", " ".join(shlex.quote(c) for c in cmd))
        rc = run_cmd(cmd)
        if rc != 0:
            raise RuntimeError(f"whisper.cpp returned {rc}")
    finally:
        # Clean up the temporary WAV file immediately to save disk space
        if temp_wav.exists():
            temp_wav.unlink()
        
    # whisper.cpp appends extension automatically
    # If we passed --output-file /path/to/stem, it writes /path/to/stem.vtt
    
    if not out_path.exists():
         # Try to find what it wrote
         logger.warning("[whisper.cpp] expected output %s not found", out_path)

    return out_path, None # whisper.cpp CLI doesn't easily give us detected language in a machine readable way


def transcribe_one(
    model,
    video_path: Path,
    subs_dir: Path,
    language: Optional[str],
    task: str,
    verbose: bool,
    overwrite: bool,
    whisper_args: Optional[Dict[str, Any]] = None,
    fmt: str = "vtt",
    backend: str = "whisper",
) -> tuple[Path, Optional[str]]:
    """
    Transcribe a single video file to SRT or VTT using a Whisper model.

    Returns (subtitle_path, detected_language).
    """
    if backend == "whisper.cpp":
        # model here is actually the model name string, not a loaded object
        return transcribe_whisper_cpp(
            model_name=str(model),
            video_path=video_path,
            subs_dir=subs_dir,
            language=language,
            task=task,
            fmt=fmt,
            overwrite=overwrite
        )

    fmt = fmt.lower().strip()
    if fmt not in {"srt", "vtt"}:
        fmt = "vtt"

    ensure_dir(subs_dir)
    out_path = subs_dir / f"{video_path.stem}.{fmt}"
    if out_path.exists() and not overwrite:
        logger.info("[subs] exists -> %s", out_path)
        # We did not run Whisper again, so we have no new detected language.
        return out_path, None

    # Base args per backend.
    if backend == "faster-whisper":
        # faster-whisper does not support "verbose".
        args: Dict[str, Any] = dict(language=language, task=task)
        # For long audio, enable internal VAD by default unless the caller
        # already set a value.
        if not whisper_args or "vad_filter" not in whisper_args:
            args["vad_filter"] = True
    else:
        args = dict(language=language, task=task, verbose=verbose)

    # Apply shared decoding arguments.
    if whisper_args:
        args.update(whisper_args)

    if backend == "faster-whisper":
        logger.info("[faster-whisper] transcribing -> %s", video_path)
        segments, info = model.transcribe(str(video_path), **args)

        detected_lang = getattr(info, "language", None)

        # Convert to a Whisper style result dict for get_writer.
        seg_list: list[Dict[str, Any]] = []
        full_text_parts: list[str] = []

        for i, seg in enumerate(segments):
            full_text_parts.append(seg.text)
            seg_dict: Dict[str, Any] = {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            if getattr(seg, "words", None):
                seg_dict["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in seg.words
                ]
            seg_list.append(seg_dict)

        result: Dict[str, Any] = {
            "text": "".join(full_text_parts),
            "segments": seg_list,
            "language": detected_lang,
        }

    else:
        logger.info("[whisper] transcribing -> %s", video_path)
        result = model.transcribe(str(video_path), **args)
        detected_lang = result.get("language")

    if get_writer:
        writer = get_writer(fmt, str(subs_dir))
        writer(result, str(video_path))
    else:
        logger.warning("get_writer not available (whisper not installed). Skipping standard writer.")

    if fmt == "srt" and not out_path.exists():
        # Some writers may not produce an SRT file in the expected location.
        # As a fallback, build a simple SRT by iterating over segments.
        logger.warning(
            "[srt] writer did not create expected file %s. Writing fallback SRT.",
            out_path,
        )
        with out_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(result.get("segments", []), start=1):

                def ts(t: float) -> str:
                    h = int(t // 3600)
                    m = int((t % 3600) // 60)
                    s = int(t % 60)
                    ms = int((t - int(t)) * 1000)
                    return f"{h:01}:{m:02}:{s:02},{ms:03}"

                f.write(
                    f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text'].strip()}\n\n"
                )

    logger.info("[subs] written -> %s", out_path)
    return out_path, detected_lang


def embed_subtitles(
    input_video: Path,
    vtt_file: Path,
    output_path: Path,
    overwrite: bool,
    language: str,
) -> Path:
    """Embed subtitles into an MP4 container as a text subtitle track."""
    ensure_dir(output_path.parent)
    if output_path.exists() and not overwrite:
        logger.info("[embed] exists -> %s", output_path)
        return output_path

    # MP4 text track via mov_text codec.
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-i",
        str(vtt_file),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        f"language={language}",
        str(output_path),
    ]
    rc = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc}")
    logger.info("[embed] written -> %s", output_path)
    return output_path


def main() -> int:
    """Entry point for a single transcription job."""
    spec = load_job_spec()

    # Required settings.
    videos_dir = str(spec.get("videos_dir") or os.environ.get("VIDEOS_DIR", "")).strip()
    subs_dir = str(spec.get("subs_dir") or os.environ.get("SUBS_DIR", "")).strip()
    model_name = str(
        spec.get("model") or os.environ.get("MODEL") or os.environ.get("WORKER_MODEL", "")
    ).strip()
    item_id = str(
        spec.get("item_id")
        or spec.get("filename")
        or os.environ.get("ITEM_ID", "")
        or os.environ.get("FILENAME", "")
    ).strip()

    missing = [
        k
        for k, v in [
            ("VIDEOS_DIR", videos_dir),
            ("SUBS_DIR", subs_dir),
            ("MODEL", model_name),
            ("ITEM_ID", item_id),
        ]
        if not v
    ]
    if missing:
        logger.error("Missing required settings: %s", ", ".join(missing))
        return 2

    # Optional settings.
    fmt = str(
        (spec.get("format") or os.environ.get("SUB_FORMAT", "vtt")).strip().lower()
    )
    if fmt not in {"srt", "vtt"}:
        fmt = "vtt"

    # Backend selection: "whisper", "faster-whisper", or "whisper.cpp"
    backend = str(spec.get("backend") or os.environ.get("BACKEND", "whisper")).strip().lower()
    if backend not in {"whisper", "faster-whisper", "whisper.cpp"}:
        logger.warning("Unknown backend '%s', falling back to 'whisper'", backend)
        backend = "whisper"

    device_sel = str(spec.get("device") or os.environ.get("DEVICE", "auto"))
    language = spec.get("language", os.environ.get("LANGUAGE"))
    task = str(spec.get("task") or os.environ.get("TASK", "transcribe"))
    verbose = as_bool(spec.get("verbose"), as_bool(os.environ.get("VERBOSE"), False))
    overwrite = as_bool(
        spec.get("overwrite"), as_bool(os.environ.get("OVERWRITE"), False)
    )
    generate_subs = as_bool(
        spec.get("generate_subs"), as_bool(os.environ.get("GENERATE_SUBS"), True)
    )
    embed_flag = as_bool(
        spec.get("embed_subs"), as_bool(os.environ.get("EMBED_SUBS"), False)
    )
    download_root = spec.get("whisper_download_root", os.environ.get("WHISPER_DOWNLOAD_ROOT"))
    output_dir = spec.get("output_dir", os.environ.get("OUTPUT_DIR"))
    cleanup = as_bool(spec.get("cleanup"), as_bool(os.environ.get("CLEANUP"), False))

    # Advanced whisper parameters (temperature, beam search, etc).
    whisper_args: Dict[str, Any] = dict(DEFAULT_WHISPER_ARGS)
    for k in ["temperature", "best_of", "beam_size", "patience"]:
        ev = os.environ.get(k.upper())
        if ev is None:
            continue
        try:
            whisper_args[k] = float(ev) if "." in ev else int(ev)
        except ValueError:
            whisper_args[k] = ev
    if isinstance(spec.get("whisper_args"), dict):
        whisper_args.update(spec["whisper_args"])

    # Paths.
    videos_dir_p = Path(videos_dir)
    subs_dir_p = Path(subs_dir)
    video_path = videos_dir_p / item_id

    if not video_path.exists():
        logger.error("Input video does not exist: %s", video_path)
        write_status(subs_dir_p, video_path.stem, "failed", message="Input video not found")
        return 2

    if overwrite:
        base = video_path.stem
        for suffix in (".srt", ".vtt", ".embedded.mp4", ".status.json"):
            p = subs_dir_p / f"{base}{suffix}"
            if p.exists():
                print(f"[overwrite] removing old artifact {p}")
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[overwrite] failed to remove {p}: {e}", file=sys.stderr)

    # Model and device.
    device = pick_device(device_sel)
    logger.info(
        "[job] item_id=%s model=%s backend=%s device=%s fmt=%s overwrite=%s embed=%s",
        item_id,
        model_name,
        backend,
        device,
        fmt,
        overwrite,
        embed_flag,
    )

    try:
        write_status(
            subs_dir_p,
            video_path.stem,
            phase="loading_model",
            progress=10,
            message=f"Loading model '{model_name}' ({backend}) on {device}",
        )

        if backend == "faster-whisper":
            model = load_faster_model(model_name, device=device)
        elif backend == "whisper.cpp":
            # For whisper.cpp, we don't load a model object in Python.
            # We just pass the model name string to the transcribe function.
            model = model_name
        else:
            model = load_model(model_name, device=device, download_root=download_root)

        # Transcribe and optionally embed.
        subs_path = None
        if generate_subs or embed_flag:
            write_status(
                subs_dir_p,
                video_path.stem,
                phase="transcribing",
                progress=30,
                message="Transcribing audio",
            )

            subs_path, detected_language = transcribe_one(
                model=model,
                video_path=video_path,
                subs_dir=subs_dir_p,
                language=language,
                task=task,
                verbose=verbose,
                overwrite=overwrite,
                whisper_args=whisper_args,
                fmt=fmt,
                backend=backend,
            )

            ffmpeg_lang = str(
                spec.get("sub_lang")
                or os.environ.get("SUB_LANG")
                or language
                or detected_language
                or "en"
            )

            if embed_flag:
                write_status(
                    subs_dir_p,
                    video_path.stem,
                    phase="embedding",
                    progress=70,
                    message="Embedding subtitles into video",
                )

                if fmt != "vtt":
                    vtt_path, _ = transcribe_one(
                        model=model,
                        video_path=video_path,
                        subs_dir=subs_dir_p,
                        language=language,
                        task=task,
                        verbose=verbose,
                        overwrite=overwrite,
                        whisper_args=whisper_args,
                        fmt="vtt",
                        backend=backend,
                    )
                else:
                    write_status(
                        subs_dir_p,
                        video_path.stem,
                        phase="subs_done",
                        progress=80,
                        message="Subtitles generated",
                    )
                    vtt_path = subs_path

                if vtt_path is not None:
                    out_path = subs_dir_p / f"{video_path.stem}.embedded.mp4"
                    embed_subtitles(
                        video_path,
                        vtt_path,
                        out_path,
                        overwrite=overwrite,
                        language=ffmpeg_lang,
                    )

        if output_dir and subs_path and subs_path.exists():
            try:
                dest_dir = Path(output_dir)
                ensure_dir(dest_dir)
                dest_file = dest_dir / subs_path.name
                logger.info(f"Copying result {subs_path} to {dest_file}")
                shutil.copy(subs_path, dest_file)
            except Exception as e:
                logger.error(f"Failed to copy result to output directory: {e}")

        if cleanup:
            logger.info("Cleaning up intermediate files...")
            try:
                if subs_path and subs_path.exists():
                    subs_path.unlink()
                # Clean up other potential formats
                for f in ["srt", "vtt"]:
                    p = subs_dir_p / f"{video_path.stem}.{f}"
                    if p.exists():
                        p.unlink()

                embedded_path = subs_dir_p / f"{video_path.stem}.embedded.mp4"
                if embedded_path.exists():
                    embedded_path.unlink()

                if video_path.exists():
                    video_path.unlink()

                status_file = subs_dir_p / f"{video_path.stem}.status.json"
                if status_file.exists():
                    status_file.unlink()

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        logger.info("[done]")
        write_status(
            subs_dir_p,
            video_path.stem,
            phase="done",
            progress=100,
            message="Job finished successfully",
        )
        return 0
    except Exception as e:
        logger.exception("Transcription job failed: %s", e)
        write_status(
            subs_dir_p,
            video_path.stem,
            phase="failed",
            progress=0,
            message=f"Job failed: {str(e)}",
        )
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(main())
