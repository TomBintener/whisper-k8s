# Configuration & Environment Variables Reference

This catalog documents all environment variables used across the `whisper-k8s` codebase, categorized by functional subsystem.

---

## 1. Container Entrypoint (`app/entry.py`)

Controls which component executes when the container image starts up:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `SERVICE` | `bridge`, `worker`, `pool_worker`, `dispatcher`, `preload` | `""` | Primary component entrypoint. If `--preload` is passed as a CLI flag, triggers model preloading regardless of `SERVICE`. |
| `PRELOAD_MODELS` | Comma-separated list (e.g. `base,small`) | `base` | Models to download onto `/data/models` when `SERVICE=preload`. |

---

## 2. API Bridge (`app/bridge.py`)

Controls HTTP API server behavior, default parameters, and Kubernetes Job creation:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | Integer port number | `8080` | Port for the FastAPI server to listen on. |
| `NAMESPACE` | Kubernetes namespace | `default` | Namespace where worker Jobs and pods are scheduled. |
| `EXECUTION_MODE` | `pool`, `pod`, `ssh` | `pod` | Default execution mode when requests do not specify `mode`. |
| `WORKER_IMAGE` | Container image reference | `whisper-suite:latest` | Container image used for ephemeral worker batch Jobs. |
| `WORKER_MODEL` | Whisper model name | `base` | Default Whisper model size for the API bridge. |
| `DEFAULT_BACKEND` | `faster-whisper`, `whisper`, `whisper.cpp` | `whisper` | Default inference engine if unstated in requests. |
| `DEFAULT_DEVICE` | `gpu`, `cpu` | `gpu` | Default inference device requested by the bridge. |
| `DISPATCHER_SERVICE_ACCOUNT` | Kubernetes ServiceAccount | `whisper-bridge` | ServiceAccount bound to created worker Jobs. |
| `DATA_PVC_NAME` | Kubernetes PVC name | `whisper-data` | PersistentVolumeClaim mounted by worker pods. |
| `DATA_DIR` | Filesystem path | `/data` | Root directory for the shared PVC. |
| `JOB_PREFIX` | String prefix | `whisper` | Prefix used when naming Kubernetes batch Jobs. |
| `MAX_DOWNLOAD_MB` | Integer (megabytes) | `2048` | Size ceiling for remote audio downloads via `trackUrl`. |
| `DOWNLOAD_TIMEOUT`| Float (seconds) | `120.0` | Socket timeout for remote audio downloads. |
| `ALLOW_LOCAL_URLS`| `true`, `false` | `false` | Bypasses SSRF check in local test suites to permit loopback URLs. |

---

## 3. Worker Inference Engine (`app/video_transcriber.py`)

Configures inference execution, GPU memory guardrails, and subtitle muxing:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `DEVICE` | `cuda`, `cpu`, `mps`, `auto` | `auto` | PyTorch execution device. (`cuda` is used for NVIDIA GPUs). |
| `MODEL` | Whisper model name | `base` | Whisper model size loaded by the worker. |
| `BACKEND` | `faster-whisper`, `whisper`, `whisper.cpp` | `whisper` | Inference backend implementation. |
| `COMPUTE_TYPE` | `int8_float16`, `float16`, `int8`, `float32` | `""` | Quantization precision for `faster-whisper`. |
| `CUDA_MEMORY_FRACTION` | Float between `0.0` and `1.0` | `None` | Caps per-process VRAM allocation (`torch.cuda.set_per_process_memory_fraction`). |
| `PYTORCH_CUDA_ALLOC_CONF` | String config | `expandable_segments:True` | PyTorch CUDA allocator configuration to mitigate memory fragmentation. |
| `SUB_FORMAT` | `srt`, `vtt` | `vtt` | Subtitle format written by worker. |
| `EMBED_SUBS` | `true`, `false` | `false` | If true, muxes subtitles into an MP4 video (`<name>.embedded.mp4`). |
| `LANGUAGE` | 2-letter ISO code (e.g. `en`) | `None` | Language hint for Whisper. If omitted, Whisper auto-detects. |
| `TASK` | `transcribe`, `translate` | `transcribe` | Transcription mode (keep language vs. translate to English). |
| `OVERWRITE` | `true`, `false` | `false` | Whether to re-run transcription if output subtitles already exist. |
| `OUTPUT_DIR` | Filesystem directory | `None` | Optional destination path to copy final results upon completion. |
| `CLEANUP` | `true`, `false` | `false` | Deletes source media and intermediate files upon successful completion. |
| `VERBOSE` | `true`, `false` | `false` | Enables detailed Whisper segment logging in worker stdout. |

---

## 4. P4 Audio Chunking Engine (`app/chunking.py`)

Controls automated VAD silence detection and parallel processing:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENABLE_CHUNKING` | `true`, `false` | `false` | Forces P4 VAD audio chunking even for shorter recordings. |
| `PARALLEL_CHUNKS` | Integer between `1` and `16` | `1` | Number of concurrent worker threads used to process slices. |
| `CHUNK_DURATION_SEC`| Float (seconds $\ge 30$) | `600.0` | Target duration per audio slice (default 10 minutes). |
| `CHUNK_THRESHOLD_SEC`| Float (seconds) | `600.0` | Duration threshold above which chunking activates automatically. |

---

## 5. Model Weights & Cache Storage Roots

Enforces P1 persistence on the shared PVC to eliminate repeated model downloads:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `MODELS_DIR` | `/data/models` | Base directory for all downloaded model weights on the shared PVC. |
| `WHISPER_DOWNLOAD_ROOT` | `/data/models/whisper` | Storage root for OpenAI Whisper PyTorch model checkpoints. |
| `HF_HOME` | `/data/models/huggingface` | Hugging Face cache root used by `faster-whisper`. |
| `TORCH_HOME` | `/data/models/torch` | PyTorch cache directory. |
| `WHISPER_CPP_MODEL_ROOT` | `/data/models/whisper.cpp` | Directory for GGML quantized model files (`.bin`). |
| `WHISPER_CPP_EXEC` | `None` | Path to the compiled `whisper-cli` executable. |

---

## 6. Task Queue Manager (`app/queue_manager.py`)

Controls task queue operations in Warm Worker Pool mode:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `QUEUE_TYPE` | `file`, `redis` | `file` | Queue backend implementation (`FileTaskQueue` vs. `RedisTaskQueue`). |
| `QUEUE_DIR` | Filesystem directory | `/data/queue` | Base directory for `FileTaskQueue` on the shared PVC. |
| `REDIS_URL` | Redis connection URI | `redis://localhost:6379/0` | Connection string when `QUEUE_TYPE=redis`. |

---

## 7. Push Webhooks (`app/webhook.py`)

Controls asynchronous callback notification delivery:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `CALLBACK_URL` | Valid HTTP/HTTPS URL | `None` | Default webhook URL if not provided in individual job requests. |
| `CALLBACK_HEADERS` | JSON-encoded string | `None` | Default HTTP headers included with webhooks (e.g. `'{"Authorization":"Bearer ..."}'`). |
| `ALLOW_LOCAL_URLS` | `true`, `false` | `false` | Bypasses SSRF validation against loopback addresses in local test environments. |

---

## 8. Bare-Metal SSH Proxy (`app/ssh_worker.py`)

Configures remote host execution when `EXECUTION_MODE=ssh`:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `HOST_USER` | `admin` | SSH username on the target bare-metal machine. |
| `HOST_IP` | Downward API (`status.hostIP`) | IP address of the target bare-metal machine. |
| `SSH_KEY_PATH` | `/etc/secret/id_rsa` | Path to mounted private key secret inside the container. |
| `REMOTE_PYTHON` | `/usr/bin/python3` | Absolute path to Python executable on the remote host. |
| `REMOTE_SCRIPT` | `/Users/Shared/whisper-k8s/app/video_transcriber.py` | Path to `video_transcriber.py` on the remote host. |
| `REMOTE_PATH_PREFIX`| `/Users/Shared/data` | Storage root on the remote host corresponding to pod `/data`. |
| `LOCAL_PATH_PREFIX` | `/data` | Storage root inside the Kubernetes pod. |

---

## 9. Batch Dispatcher (`app/dispatcher.py`)

Controls batch scheduling of Kubernetes Jobs:

| Variable Name | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `MAX_PAR` | Integer $\ge 1$ | `3` | Maximum number of simultaneously running batch Jobs. |
| `BACKOFF_LIMIT` | Integer $\ge 0$ | `0` | Number of Kubernetes retries before marking a Job failed. |
| `ITEMS_CONFIGMAP`| ConfigMap name | `""` | ConfigMap containing a list of video files to process. |
| `BRIDGE_JOB_ID` | String | `""` | Bridge Job ID grouping the batch items. |
