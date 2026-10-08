#!/usr/bin/env python3
"""
Worker script that runs a single Whisper transcription job.

It reads configuration from a JSON job spec and environment variables, loads
a Whisper model on CPU, CUDA, or MPS, generates subtitles in SRT or VTT
format, and optionally embeds them into an MP4 file using ffmpeg.
"""

import os
import sys
import re
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

try:
    import chunking  # type: ignore
except ImportError:
    try:
        from app import chunking  # type: ignore
    except ImportError:
        chunking = None  # type: ignore

try:
    import webhook  # type: ignore
except ImportError:
    try:
        from app import webhook  # type: ignore
    except ImportError:
        webhook = None  # type: ignore

try:
    import queue_manager  # type: ignore
except ImportError:
    try:
        from app import queue_manager  # type: ignore
    except ImportError:
        queue_manager = None  # type: ignore

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
    subtitle_path: Optional[str] = None,
    flavor: Optional[str] = None,
    job_id: Optional[str] = None,
) -> None:
    """Write a simple JSON status file for this job into the subtitle directory."""
    try:
        ensure_dir(subs_dir)
        bridge_job_id = (job_id or os.environ.get("BRIDGE_JOB_ID", "")).strip()
        data: Dict[str, Any] = {"phase": phase}
        if bridge_job_id:
            data["job_id"] = bridge_job_id
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if subtitle_path is not None:
            data["subtitlePath"] = str(subtitle_path)
        if flavor is not None:
            data["flavor"] = flavor

        status_path = subs_dir / f"{stem}.status.json"
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        if bridge_job_id:
            job_status_path = subs_dir / f"{bridge_job_id}.status.json"
            with job_status_path.open("w", encoding="utf-8") as f:
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


def configure_cuda_memory(spec: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """
    Configure CUDA per-process memory limits and allocator behavior to prevent
    OOM crashes when sharing physical GPUs across multiple worker pods via time-slicing.
    """
    fraction_raw = None
    if spec:
        fraction_raw = spec.get("vram_fraction") or spec.get("cuda_memory_fraction")
    if fraction_raw is None:
        fraction_raw = os.environ.get("CUDA_MEMORY_FRACTION")

    fraction: Optional[float] = None
    if fraction_raw is not None:
        try:
            val = float(fraction_raw)
            if 0.0 < val <= 1.0:
                fraction = val
            else:
                logger.warning("[cuda] invalid CUDA_MEMORY_FRACTION '%s' (must be between 0.0 and 1.0)", fraction_raw)
        except (ValueError, TypeError):
            logger.warning("[cuda] unparseable CUDA_MEMORY_FRACTION: %s", fraction_raw)

    # Set expandable segments in PyTorch allocator to mitigate virtual memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if fraction is not None:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
                logger.info("[cuda] configured per-process memory fraction: %.2f", fraction)
            except Exception as e:
                logger.warning("[cuda] failed to set per-process memory fraction: %s", e)
        else:
            logger.info("[cuda] memory fraction %.2f parsed (CUDA device not active or unavailable)", fraction)

    return fraction


def load_model(name: str, device: str, download_root: Optional[str] = None):
    """Load a Whisper model with the given name on the requested device."""
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed")
    logger.info("[model] loading '%s' on %s", name, device)
    return whisper.load_model(name, device=device, download_root=download_root)


def load_faster_model(name: str, device: str, compute_type: str = "float16", download_root: Optional[str] = None):
    """Load a faster-whisper model on GPU or CPU with persistent download_root and quantization support."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")

    if device.startswith("cuda"):
        fw_device = "cuda"
    else:
        fw_device = "cpu"

    # CPU backends do not support float16 or int8_float16 in CTranslate2
    if fw_device == "cpu" and ("float16" in compute_type):
        logger.info("[model] CPU does not support '%s', falling back to 'float32'", compute_type)
        compute_type = "float32"

    logger.info("[model] loading faster-whisper '%s' on %s (%s, download_root=%s)", name, fw_device, compute_type, download_root)
    return WhisperModel(name, device=fw_device, compute_type=compute_type, download_root=download_root)


def convert_srt_to_vtt(srt_path: Path, vtt_path: Path) -> Path:
    """
    Convert an SRT subtitle file to WebVTT format in milliseconds.
    Avoids re-running the full speech-to-text inference pass.
    """
    ensure_dir(vtt_path.parent)
    with srt_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Replace timestamp comma with period: 00:00:01,000 -> 00:00:01.000
    vtt_content = re.sub(
        r"(\d{2}:\d{2}:\d{2}),(\d{3})",
        r"\1.\2",
        content,
    )
    # Ensure standard WebVTT header
    if not vtt_content.strip().startswith("WEBVTT"):
        vtt_content = f"WEBVTT\n\n{vtt_content.lstrip()}"

    with vtt_path.open("w", encoding="utf-8") as f:
        f.write(vtt_content)

    logger.info("[subs] fast-converted SRT to VTT -> %s", vtt_path)
    return vtt_path


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


def process_job(
    spec: Optional[Dict[str, Any]] = None,
    model_cache: Optional[Dict[Any, Any]] = None,
) -> int:
    """Execute a single transcription job with optional warm model caching."""
    if spec is None:
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
    job_id = str(
        spec.get("job_id")
        or os.environ.get("BRIDGE_JOB_ID", "")
        or item_id
    ).strip()

    callback_url = spec.get("callback_url") or spec.get("callbackUrl") or os.environ.get("CALLBACK_URL")
    callback_headers = spec.get("callback_headers") or spec.get("callbackHeaders")
    if isinstance(callback_headers, str):
        try:
            callback_headers = json.loads(callback_headers)
        except Exception:
            callback_headers = None

    def _dispatch_webhook(success: bool, s_path: Optional[Path] = None, flv: Optional[str] = None, err: Optional[str] = None) -> None:
        if not callback_url:
            return
        payload = {
            "jobId": job_id,
            "status": "succeeded" if success else "failed",
            "subtitlePath": str(s_path) if (success and s_path) else None,
            "flavor": flv if success else None,
            "progress": 100 if success else 0,
            "error": None if success else (err or "Job failed"),
        }
        try:
            if webhook is not None and hasattr(webhook, "send_webhook"):
                webhook.send_webhook(callback_url, payload, headers=callback_headers)
            else:
                logger.warning("[webhook] webhook module unavailable, skipping callback")
        except Exception as exc:
            logger.warning("[webhook] Failed to dispatch webhook: %s", exc)

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
        _dispatch_webhook(False, err="Missing required settings: " + ", ".join(missing))
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

    compute_type = str(
        spec.get("compute_type") or os.environ.get("COMPUTE_TYPE") or ""
    ).strip().lower()

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
    models_dir = spec.get("models_dir") or os.environ.get("MODELS_DIR")
    if not models_dir and videos_dir:
        potential_models = Path(videos_dir).parent / "models"
        models_dir = str(potential_models)

    download_root = (
        spec.get("whisper_download_root")
        or os.environ.get("WHISPER_DOWNLOAD_ROOT")
        or (f"{models_dir}/whisper" if models_dir else None)
    )
    if download_root:
        ensure_dir(Path(download_root))

    output_dir = spec.get("output_dir", os.environ.get("OUTPUT_DIR"))
    cleanup = as_bool(spec.get("cleanup"), as_bool(os.environ.get("CLEANUP"), False))

    enable_chunking = as_bool(
        spec.get("enable_chunking"),
        as_bool(os.environ.get("ENABLE_CHUNKING"), False),
    )
    parallel_chunks = int(
        spec.get("parallel_chunks")
        or os.environ.get("PARALLEL_CHUNKS")
        or 1
    )
    chunk_duration_sec = float(
        spec.get("chunk_duration_sec")
        or os.environ.get("CHUNK_DURATION_SEC")
        or 600.0
    )
    chunk_threshold_sec = float(
        spec.get("chunk_threshold_sec")
        or os.environ.get("CHUNK_THRESHOLD_SEC")
        or 600.0
    )

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
        write_status(subs_dir_p, video_path.stem, "failed", message="Input video not found", job_id=job_id)
        _dispatch_webhook(False, err=f"Input video not found: {video_path}")
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
    vram_fraction = None
    if device == "cuda":
        vram_fraction = configure_cuda_memory(spec)

    # Resolve compute_type default
    if not compute_type:
        if device == "cuda":
            if vram_fraction is not None and vram_fraction <= 0.5:
                compute_type = "int8_float16"
            else:
                compute_type = "float16"
        else:
            compute_type = "float32"

    logger.info(
        "[job] item_id=%s model=%s backend=%s device=%s compute_type=%s vram_fraction=%s fmt=%s overwrite=%s embed=%s chunking=%s parallel_chunks=%d",
        item_id,
        model_name,
        backend,
        device,
        compute_type,
        vram_fraction,
        fmt,
        overwrite,
        embed_flag,
        enable_chunking,
        parallel_chunks,
    )

    try:
        write_status(
            subs_dir_p,
            video_path.stem,
            phase="loading_model",
            progress=10,
            message=f"Loading model '{model_name}' ({backend}) on {device}",
            job_id=job_id,
        )

        cache_key = (model_name, backend, device, compute_type)
        if model_cache is not None and cache_key in model_cache:
            model = model_cache[cache_key]
            logger.info("[warm_worker] Reusing pre-loaded model in memory for %s (%s)", model_name, backend)
        else:
            if backend == "faster-whisper":
                fw_download_root = (
                    spec.get("hf_home")
                    or os.environ.get("HF_HOME")
                    or (f"{models_dir}/huggingface" if models_dir else download_root)
                )
                if fw_download_root:
                    ensure_dir(Path(fw_download_root))
                model = load_faster_model(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=fw_download_root,
                )
            elif backend == "whisper.cpp":
                # For whisper.cpp, we don't load a model object in Python.
                # We just pass the model name string to the transcribe function.
                model = model_name
            else:
                model = load_model(model_name, device=device, download_root=download_root)

            if model_cache is not None:
                model_cache[cache_key] = model
                logger.info("[warm_worker] Cached model instance in memory for %s (%s)", model_name, backend)

        # Transcribe and optionally embed.
        subs_path = None
        detected_language = None
        if generate_subs or embed_flag:
            write_status(
                subs_dir_p,
                video_path.stem,
                phase="transcribing",
                progress=30,
                message="Transcribing audio",
                job_id=job_id,
            )

            # Check if chunking should be applied
            media_duration = None
            if chunking is not None and (enable_chunking or parallel_chunks > 1):
                try:
                    media_duration = chunking.get_media_duration(video_path)
                except Exception as e:
                    logger.debug("Could not probe media duration: %s", e)

            use_chunking = False
            planned_chunks = []
            if chunking is not None and media_duration is not None and (
                enable_chunking or media_duration >= chunk_threshold_sec or parallel_chunks > 1
            ):
                try:
                    silences = chunking.detect_silence_points(video_path)
                    planned_chunks = chunking.plan_chunks(
                        media_duration,
                        target_duration=chunk_duration_sec,
                        silence_points=silences,
                    )
                    if len(planned_chunks) > 1:
                        use_chunking = True
                except Exception as e:
                    logger.warning("Chunk planning failed, falling back to monolithic: %s", e)

            if use_chunking:
                logger.info(
                    "[chunking] Splitting %s (%.1fs) into %d chunks (parallel_chunks=%d)",
                    video_path.name,
                    media_duration,
                    len(planned_chunks),
                    parallel_chunks,
                )
                tmp_chunks_dir = subs_dir_p / f"{video_path.stem}_chunks"
                ensure_dir(tmp_chunks_dir)
                try:
                    chunk_files = chunking.split_audio_into_chunks(
                        video_path, tmp_chunks_dir, planned_chunks
                    )

                    chunk_sub_results = []

                    def _transcribe_chunk(c_spec, c_file):
                        c_subs_path, c_lang = transcribe_one(
                            model=model,
                            video_path=c_file,
                            subs_dir=tmp_chunks_dir,
                            language=language,
                            task=task,
                            verbose=verbose,
                            overwrite=True,
                            whisper_args=whisper_args,
                            fmt=fmt,
                            backend=backend,
                        )
                        c_text = ""
                        if c_subs_path and c_subs_path.exists():
                            with c_subs_path.open("r", encoding="utf-8") as f:
                                c_text = f.read()
                        return (c_spec.start_sec, c_text, c_lang)

                    from concurrent.futures import ThreadPoolExecutor
                    max_w = max(1, min(parallel_chunks, len(planned_chunks)))
                    with ThreadPoolExecutor(max_workers=max_w) as executor:
                        futures = [
                            executor.submit(_transcribe_chunk, c_spec, c_file)
                            for c_spec, c_file in zip(planned_chunks, chunk_files)
                        ]
                        for fut in futures:
                            c_offset, c_text, c_lang = fut.result()
                            chunk_sub_results.append((c_text, c_offset))
                            if not detected_language and c_lang:
                                detected_language = c_lang
                            progress_pct = int(30 + 35 * (len(chunk_sub_results) / len(planned_chunks)))
                            write_status(
                                subs_dir_p,
                                video_path.stem,
                                phase="transcribing",
                                progress=progress_pct,
                                message=f"Transcribed chunk {len(chunk_sub_results)}/{len(planned_chunks)}",
                                job_id=job_id,
                            )

                    stitched_content = chunking.stitch_subtitles(chunk_sub_results, fmt=fmt)
                    subs_path = subs_dir_p / f"{video_path.stem}.{fmt}"
                    ensure_dir(subs_path.parent)
                    with subs_path.open("w", encoding="utf-8") as f:
                        f.write(stitched_content)

                    logger.info("[chunking] Successfully stitched %d chunks into %s", len(planned_chunks), subs_path)
                finally:
                    shutil.rmtree(tmp_chunks_dir, ignore_errors=True)
            else:
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
                    job_id=job_id,
                )

                if fmt != "vtt":
                    vtt_path = subs_dir_p / f"{video_path.stem}.vtt"
                    if subs_path and subs_path.exists():
                        convert_srt_to_vtt(subs_path, vtt_path)
                    else:
                        convert_srt_to_vtt(subs_dir_p / f"{video_path.stem}.srt", vtt_path)
                else:
                    vtt_path = subs_path

                write_status(
                    subs_dir_p,
                    video_path.stem,
                    phase="subs_done",
                    progress=80,
                    message="Subtitles generated and prepared for embedding",
                    job_id=job_id,
                )

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
        flavor = f"{fmt}+embedded" if embed_flag else fmt
        write_status(
            subs_dir_p,
            video_path.stem,
            phase="done",
            progress=100,
            message="Job finished successfully",
            subtitle_path=str(subs_path) if subs_path else None,
            flavor=flavor,
            job_id=job_id,
        )
        _dispatch_webhook(True, s_path=subs_path, flv=flavor)
        return 0
    except Exception as e:
        logger.exception("Transcription job failed: %s", e)
        write_status(
            subs_dir_p,
            video_path.stem if 'video_path' in locals() else item_id,
            phase="failed",
            progress=0,
            message=f"Job failed: {str(e)}",
            job_id=job_id,
        )
        _dispatch_webhook(False, err=str(e))
        return 1


def run_worker_daemon(
    queue: Optional[Any] = None,
    stop_event: Optional[Any] = None,
    poll_interval: float = 1.0,
    max_jobs: Optional[int] = None,
) -> int:
    """
    Run persistent warm worker daemon pulling jobs from the task queue.
    Reuses models in memory across jobs to eliminate pod cold starts.
    """
    if queue is None:
        if queue_manager is not None and hasattr(queue_manager, "get_queue"):
            queue = queue_manager.get_queue()
        else:
            logger.error("[daemon] queue_manager not available")
            return 1

    logger.info("[daemon] Starting warm worker daemon...")
    model_cache: Dict[Any, Any] = {}
    jobs_processed = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("[daemon] Stop event received, shutting down daemon.")
            break
        if max_jobs is not None and jobs_processed >= max_jobs:
            logger.info("[daemon] Reached max_jobs limit (%d), shutting down daemon.", max_jobs)
            break

        job_spec = queue.dequeue(timeout=poll_interval)
        if not job_spec:
            continue

        job_id = job_spec.get("job_id") or job_spec.get("item_id")
        logger.info("[daemon] Claimed job %s from queue, starting execution", job_id)
        exit_code = process_job(spec=job_spec, model_cache=model_cache)
        queue.complete(job_id, success=(exit_code == 0))
        jobs_processed += 1
        logger.info("[daemon] Finished job %s (exit_code=%d, total_processed=%d)", job_id, exit_code, jobs_processed)

    return 0


def main() -> int:
    """Entry point for worker execution."""
    service = os.environ.get("SERVICE", "").strip().lower()
    if service in ("pool_worker", "daemon", "worker_pool") or "--daemon" in sys.argv:
        return run_worker_daemon()
    return process_job()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(main())
