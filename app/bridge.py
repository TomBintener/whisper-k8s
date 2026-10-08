"""
Whisper bridge API.

This service exposes a small HTTP API that:
- accepts jobs to transcribe or translate a single video or audio track
- writes the media file into a shared PVC
- creates a Kubernetes Job that runs the worker on that file
- reports back status and output paths for subtitles and embedded video

Key endpoints:
- GET  /health
- POST /jobs
- GET  /status/{job_id}
"""

import os
import re
import uuid
import mimetypes
import logging
from typing import Optional, Dict, Literal

import json
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, model_validator

from kubernetes import client, config
from kubernetes.client import ApiException

logger = logging.getLogger(__name__)

# =========================
# Settings
# =========================

# Namespace where Jobs and worker pods run.
NAMESPACE = os.getenv("NAMESPACE", "default")

# Image and model used by the worker Job for Whisper inference.
WORKER_IMAGE = os.getenv("WORKER_IMAGE", "whisper-suite:latest")
WORKER_MODEL = os.getenv("WORKER_MODEL", "base")

# Service account used by worker pods (for example to read ConfigMaps).
SERVICE_ACCOUNT = os.getenv("DISPATCHER_SERVICE_ACCOUNT", "whisper-bridge")

# Shared volume mount inside worker pods.
DATA_DIR = os.getenv("DATA_DIR", "/data")  # PVC mount root
VIDEOS_DIR = f"{DATA_DIR}/videos"
SUBS_DIR = f"{DATA_DIR}/subs"
DATA_PVC_NAME = os.getenv("DATA_PVC_NAME", "whisper-data")

# Prefix for Jobs created by this bridge.
JOB_PREFIX = os.getenv("JOB_PREFIX", "whisper")

# Download limits for remote trackUrl inputs.
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "2048"))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "120"))

# Device the worker should prefer. The worker will map "gpu" to cuda internally.
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "gpu")

# Default model name for the API if the client does not provide one.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", WORKER_MODEL or "base")

# Default backend used by the worker ("whisper" or "faster-whisper").
DEFAULT_BACKEND = os.getenv("DEFAULT_BACKEND", "whisper")

# Whisper CPP Configuration
WHISPER_CPP_EXEC = os.getenv("WHISPER_CPP_EXEC")
WHISPER_CPP_MODEL_ROOT = os.getenv("WHISPER_CPP_MODEL_ROOT")

# Execution Mode: "pod" (default) or "ssh" (native host execution)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "pod").lower()

# SSH Configuration (only used if EXECUTION_MODE=ssh)
SSH_HOST_USER = os.getenv("SSH_HOST_USER", "admin")
SSH_REMOTE_PYTHON = os.getenv("SSH_REMOTE_PYTHON", "/usr/bin/python3")
SSH_REMOTE_SCRIPT = os.getenv("SSH_REMOTE_SCRIPT", "/Users/Shared/whisper-k8s/app/video_transcriber.py")
SSH_REMOTE_PATH_PREFIX = os.getenv("SSH_REMOTE_PATH_PREFIX", "/Users/Shared/data")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "/etc/secret/id_rsa")
SSH_HOST_IP = os.getenv("SSH_HOST_IP") # Optional override for local desktop testing

# =========================
# Kubernetes init
# =========================

def _load_kube() -> str:
    """
    Load Kubernetes configuration for the bridge.

    First tries in cluster configuration (when running inside a pod).
    If that fails, falls back to reading local kubeconfig.

    Returns
    -------
    str
        "incluster" or "kubeconfig" depending on what was loaded.
    """
    try:
        config.load_incluster_config()
        logger.info("Loaded in cluster Kubernetes config")
        return "incluster"
    except Exception:
        config.load_kube_config()
        logger.info("Loaded Kubernetes config from kubeconfig")
        return "kubeconfig"


KUBE_MODE = _load_kube()
k8s_batch = client.BatchV1Api()
k8s_core = client.CoreV1Api()


# =========================
# Request / response models
# =========================

class CreateJobReq(BaseModel):
    """
    Incoming payload for POST /jobs.

    Either `filename` or `trackUrl` must be provided:

    - filename: path relative to VIDEOS_DIR inside the shared PVC.
    - trackUrl: remote URL that the bridge will download into VIDEOS_DIR.

    Optional controls:
    - format: "srt" or "vtt" for the external subtitle file.
    - device: "cpu" or "gpu" (worker maps gpu to cuda).
    - embed: whether to embed subtitles into an MP4 output.
    - model: Whisper model name hint (for example "base", "small", "large").
    - overwrite:
        * None -> worker default (no overwrite)
        * True -> force recompute and overwrite outputs
        * False -> reuse existing outputs if present
    - language: language hint for Whisper, or None to let it auto detect.
    - task: "transcribe" (keep source language) or "translate" (to English).
    - outputDir: Optional path to copy the results to after completion.
    - cleanup: If True, delete input video and intermediate files after completion.
    """
    filename: Optional[str] = None
    trackUrl: Optional[str] = None
    format: Literal["srt", "vtt"] = "srt"
    device: Literal["cpu", "gpu"] = DEFAULT_DEVICE
    embed: bool = False
    model: str | None = None
    overwrite: bool | None = None
    language: str | None = None  # e.g. "en", "de", etc.
    task: Literal["transcribe", "translate"] | None = None
    backend: Literal["whisper", "faster-whisper", "whisper.cpp"] | None = None
    mode: Literal["kubernetes", "pod", "ssh"] | None = None
    outputDir: Optional[str] = None
    cleanup: bool = False

    @model_validator(mode="after")
    def _validate(self):
        """
        Enforce basic consistency on the request.

        - Either filename or trackUrl must be set.
        - format and device must be in the allowed sets.
        - task, if provided, must be one of the supported values.
        """
        if not self.trackUrl and not self.filename:
            raise ValueError("Provide trackUrl or filename")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.format not in {"srt", "vtt"}:
            raise ValueError("format must be 'srt' or 'vtt'")
        if self.task is not None and self.task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        if self.backend is not None and self.backend not in {"whisper", "faster-whisper", "whisper.cpp"}:
            raise ValueError("backend must be 'whisper' or 'faster-whisper' or 'whisper.cpp'")
        return self


class CreateJobResp(BaseModel):
    """
    Response body for a successful POST /jobs.

    Attributes
    ----------
    jobId:
        Logical bridge job identifier used for subsequent status checks.
    status:
        Current high level status for the job, initially "accepted".
    videoPath:
        Absolute path inside the worker PVC where the input file was written.
    namespace:
        Kubernetes namespace where worker Jobs are created.
    """
    jobId: str
    status: str = "accepted"
    videoPath: str
    namespace: str = NAMESPACE


class JobStatusResp(BaseModel):
    """
    Response body for GET /status/{job_id}.

    Attributes
    ----------
    status:
        "queued", "running", "succeeded", or "failed".
    progress:
        Rough progress estimate (0 to 100) for clients that want a simple bar.
    subtitlePath:
        Path to the generated subtitle file on the shared PVC, if available.
    flavor:
        A string describing the output type, for example:
        - "srt"
        - "vtt"
        - "srt+embedded"
        - "vtt+embedded"
    message:
        Optional error or status message for failed jobs or phase messages.
    """
    status: str
    progress: Optional[int] = None
    subtitlePath: Optional[str] = None
    flavor: Optional[str] = None
    message: Optional[str] = None


# =========================
# Helper functions
# =========================

def _ensure_dir(p: str) -> None:
    """
    Ensure that a directory exists, creating parents as needed.

    Parameters
    ----------
    p:
        Path to a directory that should exist after this call.
    """
    os.makedirs(p, exist_ok=True)


def _safe_under(base_dir: str, filename: str) -> str:
    """
    Safely resolve a filename under a given base directory.

    This prevents directory traversal by rejecting any path that would escape
    the base directory after resolution.

    Parameters
    ----------
    base_dir:
        Root directory path that must contain the resolved file.
    filename:
        User provided relative filename.

    Returns
    -------
    str
        Absolute path to the file under base_dir.

    Raises
    ------
    HTTPException
        If the resolved path is not under base_dir.
    """
    base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base, filename))
    if not target.startswith(base + os.sep):
        raise HTTPException(400, "invalid filename path")
    return target


def _dns1123_label(value: str, max_len: int = 63) -> str:
    """
    Convert an arbitrary string into a DNS 1123 compliant label.

    Kubernetes requires names and many labels to satisfy:
    - only lowercase alphanumeric characters and '-'
    - must start and end with an alphanumeric character
    - max length 63

    Parameters
    ----------
    value:
        Input string to sanitize.
    max_len:
        Maximum allowed length of the result.

    Returns
    -------
    str
        Sanitized label string.
    """
    v = (value or "").lower().replace(".", "-")
    v = re.sub(r"[^a-z0-9-]", "-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return (v or "x")[:max_len].rstrip("-")


async def _download(url: str, dest_path: str) -> None:
    """
    Download a remote file to dest_path with a size limit and timeout.

    The file is first written as dest_path + ".part" and atomically moved
    into place when finished. If anything goes wrong, the partial file
    is removed.

    Parameters
    ----------
    url:
        Remote URL to download from.
    dest_path:
        Target path on local disk or PVC.

    Raises
    ------
    HTTPException
        If the download fails, returns an HTTP error, or exceeds the size limit.
    """
    tmp = dest_path + ".part"
    _ensure_dir(os.path.dirname(dest_path))
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    size = 0
    logger.info("Downloading track from %s to %s (limit %d MB)", url, dest_path, MAX_DOWNLOAD_MB)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as http:
            async with http.stream("GET", url) as r:
                if r.status_code >= 400:
                    logger.error("Download failed with status %s for %s", r.status_code, url)
                    raise HTTPException(400, f"download failed: {r.status_code}")
                with open(tmp, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > limit:
                            logger.error(
                                "Download aborted, file too large (> %d MB) for %s",
                                MAX_DOWNLOAD_MB,
                                url,
                            )
                            raise HTTPException(
                                413,
                                f"file too large. limit {MAX_DOWNLOAD_MB} MB",
                            )
                        f.write(chunk)
        os.replace(tmp, dest_path)
        logger.info("Downloaded %d bytes to %s", size, dest_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _expected_paths(filename: str, fmt: str) -> Dict[str, str]:
    """
    Compute expected subtitle and embedded output paths for a given filename.

    The worker writes subtitles under SUBS_DIR, using:
    - "<name>.vtt" for VTT
    - "<name>.srt" for SRT
    and may also produce:
    - "<name>.embedded.mp4" when embedding is enabled.

    Parameters
    ----------
    filename:
        Base filename of the input media, including extension.
    fmt:
        Subtitle format chosen for the external file, "srt" or "vtt".

    Returns
    -------
    dict
        Dictionary with keys:
        - "subtitle": expected path to the external subtitle file
        - "embedded": expected path to the embedded MP4 file
    """
    base, _ = os.path.splitext(filename)
    subtitle = f"{SUBS_DIR}/{base}.vtt" if fmt == "vtt" else f"{SUBS_DIR}/{base}.srt"
    embedded = f"{SUBS_DIR}/{base}.embedded.mp4"
    return {"subtitle": subtitle, "embedded": embedded}


# =========================
# Job builder
# =========================

def _make_worker_job(
    job_id: str,
    filename: str,
    fmt: str,
    device: str = "cpu",
    embed: bool = False,
    model: str | None = None,
    overwrite: bool | None = None,
    language: str | None = None,
    task: str | None = None,
    backend: str | None = None,
    mode: str | None = None,
    output_dir: str | None = None,
    cleanup: bool = False,
) -> client.V1Job:
    """
    Build a Kubernetes Job that runs the worker on a single file.

    The Job:
    - mounts the shared data PVC at DATA_DIR
    - passes ITEM_ID, VIDEOS_DIR, SUBS_DIR and configuration via environment
    - sets labels and annotations so that /status can find outputs later

    Parameters
    ----------
    job_id:
        Logical job id for this bridge request. Used for labels and status.
    filename:
        Basename of the media file inside VIDEOS_DIR (no directory).
    fmt:
        Requested subtitle format, "srt" or "vtt".
    device:
        "cpu" or "gpu" from the API. Mapped to "cpu" or "cuda" for the worker.
    embed:
        Whether the worker should embed subtitles into an MP4 file.
    model:
        Optional Whisper model name hint. Defaults to DEFAULT_MODEL if None.
    overwrite:
        Controls overwrite semantics for the worker (see CreateJobReq docs).
    language:
        Optional language hint for Whisper.
    task:
        Optional task for Whisper, usually "transcribe" or "translate".
    output_dir:
        Optional directory to copy results to.
    cleanup:
        Whether to clean up input/output files after processing.

    Returns
    -------
    kubernetes.client.V1Job
        Fully populated Job resource ready to submit to the Kubernetes API.
    """
    job_name = _dns1123_label(
        f"{JOB_PREFIX}-{job_id}-{filename.replace('.', '-') } "
    )

    labels = {
        "app.kubernetes.io/name": "whisper-worker",
        "app.kubernetes.io/part-of": "whisper",
        "bridge-job-id": job_id,
    }
    annotations = {
        "bridge/filename": filename,
        "bridge/format": fmt,
        "bridge/embed": "true" if embed else "false",
    }

    env = [
        client.V1EnvVar(name="SERVICE", value="worker"),
        client.V1EnvVar(name="ITEM_ID", value=filename),
        client.V1EnvVar(name="BRIDGE_JOB_ID", value=job_id),
        client.V1EnvVar(name="VIDEOS_DIR", value=VIDEOS_DIR),
        client.V1EnvVar(name="SUBS_DIR", value=SUBS_DIR),
    ]

    # Model hint for the worker.
    env.append(client.V1EnvVar(name="MODEL", value=model or DEFAULT_MODEL))

    # Device mapping: API says "gpu", worker expects "cuda".
    if device == "gpu":
        env.append(client.V1EnvVar(name="DEVICE", value="cuda"))
    else:
        env.append(client.V1EnvVar(name="DEVICE", value="cpu"))

    # Subtitle format for the worker.
    env.append(client.V1EnvVar(name="SUB_FORMAT", value=fmt))

    # Embed flag controls whether the worker should mux subs into an MP4.
    env.append(
        client.V1EnvVar(
            name="EMBED_SUBS",
            value="true" if embed else "false",
        )
    )

    # Language hint forwarded to the worker.
    if language:
        env.append(client.V1EnvVar(name="LANGUAGE", value=language))

    # Overwrite behavior, if explicitly set in the API request.
    if overwrite is not None:
        env.append(client.V1EnvVar(name="OVERWRITE", value=str(overwrite).lower()))

    # Task for Whisper (transcribe vs translate).
    if task:
        env.append(client.V1EnvVar(name="TASK", value=task))

    # Backend for the worker.
    if backend:
        env.append(client.V1EnvVar(name="BACKEND", value=backend))
    else:
        env.append(client.V1EnvVar(name="BACKEND", value=DEFAULT_BACKEND))

    # Output Dir and Cleanup
    if output_dir:
        env.append(client.V1EnvVar(name="OUTPUT_DIR", value=output_dir))

    if cleanup:
        env.append(client.V1EnvVar(name="CLEANUP", value="true"))

    # Whisper CPP config
    if WHISPER_CPP_EXEC:
        env.append(client.V1EnvVar(name="WHISPER_CPP_EXEC", value=WHISPER_CPP_EXEC))
    if WHISPER_CPP_MODEL_ROOT:
        env.append(client.V1EnvVar(name="WHISPER_CPP_MODEL_ROOT", value=WHISPER_CPP_MODEL_ROOT))

    resources = client.V1ResourceRequirements(
        requests={"cpu": "500m", "memory": "4Gi"},
        limits={"cpu": "2", "memory": "8Gi"},
    )

    volumes = [
        client.V1Volume(
            name="data",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=DATA_PVC_NAME
            ),
        )
    ]

    mounts = [
        client.V1VolumeMount(
            name="data",
            mount_path=DATA_DIR,
        )
    ]

    job_exec_mode = (mode or EXECUTION_MODE).lower()
    # SSH Mode Configuration
    if job_exec_mode == "ssh":
        # Override the command to run the SSH proxy script
        command = ["python", "/app/ssh_worker.py"]
        
        # Add SSH specific env vars
        env.extend([
            client.V1EnvVar(name="HOST_USER", value=SSH_HOST_USER),
            client.V1EnvVar(name="REMOTE_PYTHON", value=SSH_REMOTE_PYTHON),
            client.V1EnvVar(name="REMOTE_SCRIPT", value=SSH_REMOTE_SCRIPT),
            client.V1EnvVar(name="REMOTE_PATH_PREFIX", value=SSH_REMOTE_PATH_PREFIX),
            client.V1EnvVar(name="LOCAL_PATH_PREFIX", value=DATA_DIR),
            client.V1EnvVar(name="SSH_KEY_PATH", value=SSH_KEY_PATH),
        ])

        if SSH_HOST_IP:
            env.append(client.V1EnvVar(name="HOST_IP", value=SSH_HOST_IP))
        else:
            # Host IP via Downward API (used for bare-metal multi-node clusters)
            env.append(client.V1EnvVar(
                name="HOST_IP",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path="status.hostIP")
                )
            ))
        
        # Mount the SSH key secret
        volumes.append(
            client.V1Volume(
                name="ssh-key",
                secret=client.V1SecretVolumeSource(
                    secret_name="ssh-key",
                    default_mode=0o400 # Read only, secure
                )
            )
        )
        mounts.append(
            client.V1VolumeMount(
                name="ssh-key",
                mount_path="/etc/secret",
                read_only=True
            )
        )
    else:
        command = None # Use default entrypoint

    container = client.V1Container(
        name="worker",
        image=WORKER_IMAGE,
        image_pull_policy="IfNotPresent",
        env=env,
        command=command,
        volume_mounts=mounts,
        resources=resources,
    )

    pod_spec = client.V1PodSpec(
        service_account_name=SERVICE_ACCOUNT,
        restart_policy="Never",
        containers=[container],
        volumes=volumes,
    )

    spec = client.V1JobSpec(
        backoff_limit=0,
        ttl_seconds_after_finished=3600,
        template=client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels, annotations=annotations),
            spec=pod_spec,
        ),
    )

    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name.strip(),
            namespace=NAMESPACE,
            labels=labels,
            annotations=annotations,
        ),
        spec=spec,
    )


# =========================
# FastAPI application
# =========================

app = FastAPI()


@app.get("/health")
def health():
    """
    Liveness and readiness endpoint.

    Returns some basic metadata so you can confirm that the bridge is running
    and can talk to Kubernetes.

    Returns
    -------
    dict
        Simple JSON object with "ok", "namespace" and "kube" mode.
    """
    logger.debug("Health check")
    return {"ok": True, "namespace": NAMESPACE, "kube": KUBE_MODE}


@app.post("/jobs", response_model=CreateJobResp)
async def create_job(req: CreateJobReq = Body(...)):
    """
    Create a new transcription or translation job.

    Behavior:

    - If `filename` is provided:
        * verify that VIDEOS_DIR/filename exists on the shared volume.
    - If only `trackUrl` is provided:
        * download the file into VIDEOS_DIR under a generated or URL based name.

    In both cases the bridge then:
    - builds a worker Job for the given file and options
    - creates the Job in Kubernetes
    - responds with a jobId that clients can use with /status

    Parameters
    ----------
    req:
        Parsed CreateJobReq payload.

    Returns
    -------
    CreateJobResp
        Basic job metadata including jobId and videoPath.

    Raises
    ------
    HTTPException
        If the input file cannot be found or the worker Job cannot be created.
    """
    job_id = uuid.uuid4().hex
    logger.info(
        "Create job request job_id=%s device=%s format=%s embed=%s model=%s overwrite=%s language=%s task=%s backend=%s mode=%s filename=%s trackUrl=%s outputDir=%s cleanup=%s",
        job_id,
        req.device,
        req.format,
        req.embed,
        req.model,
        req.overwrite,
        req.language,
        req.task,
        req.backend,
        req.mode,
        req.filename,
        str(req.trackUrl) if req.trackUrl else None,
        req.outputDir,
        req.cleanup,
    )

    # Resolve the input file.
    if req.filename:
        filename = req.filename
        video_path = _safe_under(VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            logger.error("File not found under VIDEOS_DIR for job_id=%s: %s", job_id, video_path)
            raise HTTPException(404, "file not found under VIDEOS_DIR")
    else:
        # Download from trackUrl into VIDEOS_DIR with a safe name.
        import urllib.parse

        name = os.path.basename(urllib.parse.urlparse(str(req.trackUrl)).path) or f"{job_id}.mp4"
        if not os.path.splitext(name)[1]:
            name = name + (mimetypes.guess_extension("video/mp4") or ".mp4")
        filename = name
        video_path = _safe_under(VIDEOS_DIR, filename)
        _ensure_dir(os.path.dirname(video_path))
        await _download(str(req.trackUrl), video_path)
        logger.info("Downloaded remote track for job_id=%s to %s", job_id, video_path)

    worker = _make_worker_job(
        job_id=job_id,
        filename=os.path.basename(filename),
        fmt=req.format,
        device=req.device,
        embed=req.embed,
        model=req.model,
        overwrite=req.overwrite,
        language=req.language,
        task=req.task,
        backend=req.backend,
        mode=req.mode,
        output_dir=req.outputDir,
        cleanup=req.cleanup,
    )
    logger.info("Submitting worker job for job_id=%s filename=%s", job_id, os.path.basename(filename))

    try:
        await run_in_threadpool(k8s_batch.create_namespaced_job, NAMESPACE, worker)
        logger.info("Created worker job in Kubernetes for job_id=%s", job_id)
    except ApiException as e:
        logger.error("Failed to create worker job for job_id=%s: %s", job_id, e)
        raise HTTPException(422, f"failed to create worker job: {e}")

    return CreateJobResp(jobId=job_id, videoPath=video_path)


@app.get("/status/{job_id}", response_model=JobStatusResp)
async def get_status(job_id: str):
    """
    Get the current status of a job previously created via /jobs.

    The bridge:
    - finds the Job with label bridge-job-id=job_id
    - inspects its annotations to know the filename and format
    - reads a status JSON (if present) written by the worker
    - checks on disk whether subtitles and embedded MP4 have been written
    - maps that to a high level status, progress and flavor

    Status mapping (job-centric):
    - no Job found         -> "queued", progress 5
    - Job Failed           -> "failed"
    - Job Complete         -> "succeeded"
    - Job active           -> "running"
    - Job exists but idle  -> "queued"
    """
    logger.debug("Status request for job_id=%s", job_id)
    try:
        selector = f"bridge-job-id={job_id}"
        jobs = await run_in_threadpool(
            k8s_batch.list_namespaced_job,
            NAMESPACE,
            label_selector=selector,
        )

        # No Job yet (or TTL cleaned it up) – we cannot know filename safely.
        if not jobs.items:
            logger.debug("No jobs found yet for job_id=%s", job_id)
            return JobStatusResp(
                status="queued",
                progress=5,
                message="Job queued",
            )

        job = jobs.items[0]
        ann = job.metadata.annotations or {}
        filename = ann.get("bridge/filename")
        fmt = (ann.get("bridge/format") or "srt").lower()
        embed_requested = ann.get("bridge/embed") == "true"

        if not filename:
            logger.debug(
                "Job %s missing bridge/filename annotation",
                job.metadata.name if job.metadata else "?",
            )
            return JobStatusResp(
                status="queued",
                progress=5,
                message="Job queued",
            )

        # Expected file locations and default flavor/path for the client.
        paths = _expected_paths(filename, fmt)
        subtitle = paths["subtitle"]
        embedded = paths["embedded"]
        expected_flavor = f"{fmt}+embedded" if embed_requested else fmt
        subtitle_path_for_resp = subtitle
        flavor_for_resp = expected_flavor

        # Try to read worker status file for phase / progress / message.
        status_file = Path(SUBS_DIR) / f"{Path(filename).stem}.status.json"
        phase = None
        phase_progress = None
        phase_message = None
        if status_file.exists():
            try:
                with status_file.open("r", encoding="utf-8") as f:
                    status_data = json.load(f)
                phase = status_data.get("phase")
                phase_progress = status_data.get("progress")
                phase_message = status_data.get("message")
            except Exception as e:
                logger.warning("Could not read status file %s: %s", status_file, e)

        subs_exists = os.path.exists(subtitle)
        embedded_exists = os.path.exists(embedded)

        # Examine Job conditions.
        conds = job.status.conditions or []
        failed = any(
            getattr(c, "type", "") == "Failed" and getattr(c, "status", "") == "True"
            for c in conds
        )
        complete = any(
            getattr(c, "type", "") == "Complete" and getattr(c, "status", "") == "True"
            for c in conds
        )
        active = int(getattr(job.status, "active", 0) or 0)

        # Failed overrides everything else.
        if failed:
            logger.warning("Worker reported failure for job_id=%s", job_id)
            return JobStatusResp(
                status="failed",
                progress=phase_progress,
                subtitlePath=subtitle_path_for_resp,
                flavor=flavor_for_resp,
                message=phase_message or "worker failed",
            )

        # Job completed successfully.
        if complete:
            if embedded_exists:
                logger.info(
                    "Job %s completed with embedded subtitles at %s",
                    job_id,
                    embedded,
                )
            elif subs_exists:
                logger.info(
                    "Job %s completed with external subtitles at %s",
                    job_id,
                    subtitle,
                )
            else:
                logger.info(
                    "Job %s completed but no outputs found yet for %s",
                    job_id,
                    filename,
                )

            return JobStatusResp(
                status="succeeded",
                progress=phase_progress or 100,
                subtitlePath=subtitle_path_for_resp if subs_exists else None,
                flavor=flavor_for_resp if subs_exists or embedded_exists else None,
                message=phase_message or "Job finished successfully",
            )

        # Job not complete and not failed – either running or queued.
        if active > 0:
            # Refine progress a bit based on outputs.
            if embedded_exists:
                prog = phase_progress or 90
                msg = phase_message or "Embedding almost done"
            elif subs_exists:
                if embed_requested:
                    prog = phase_progress or 80
                    msg = phase_message or "Subtitles generated, embedding into video"
                else:
                    prog = phase_progress or 80
                    msg = phase_message or "Subtitles generated"
            else:
                prog = phase_progress or 10
                msg = phase_message or "Worker running"

            return JobStatusResp(
                status="running",
                progress=prog,
                subtitlePath=subtitle_path_for_resp,
                flavor=flavor_for_resp,
                message=msg,
            )

        # Job exists but has no active pods yet and is not complete or failed.
        return JobStatusResp(
            status="queued",
            progress=phase_progress or 5,
            subtitlePath=subtitle_path_for_resp,
            flavor=flavor_for_resp,
            message=phase_message or "Job queued",
        )

    except ApiException as e:
        logger.error("Error while checking status for job_id=%s: %s", job_id, e)
        return JobStatusResp(
            status="queued",
            progress=5,
            message="Job queued",
        )
