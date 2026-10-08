# whisper-k8s

[![Kubernetes](https://img.shields.io/badge/Kubernetes-v1.24%2B-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA_GPU-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![KEDA](https://img.shields.io/badge/KEDA-Autoscaling-FF69B4?logo=kubernetes&logoColor=white)](https://keda.sh/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**whisper-k8s** is an enterprise-grade, high-throughput distributed speech-to-text and subtitle platform built for Kubernetes. Originally architected for university lecture recording platforms (such as Opencast) and large media archives, it coordinates transcription and translation across heterogeneous clusters (Kubernetes GPU worker pools, ephemeral batch pods, and bare-metal hardware).

---

## Key Architectural Highlights

- **P0 Redundant Inference Elimination**: Fast subtitle conversion (`convert_srt_to_vtt`) avoids running a redundant second Whisper pass during video subtitle embedding, achieving an immediate **~1.8x to 2x speedup**.
- **P1 Shared Model Cache Persistence**: Shared PVC cache roots (`/data/models`) across all worker pods eliminate repeated 1–3 GB model downloads and cut pod cold starts from 30–90 seconds to zero.
- **P2 Warm Worker Pool & VRAM Reuse**: Persistent worker daemon (`SERVICE=pool_worker`) reuses pre-loaded Whisper models in GPU memory across consecutive tasks. Job submission returns in **$<10\text{ms}$** with zero pod-creation overhead.
- **P3 GPU Time-Slicing & Fractional Sharing**: Divides physical NVIDIA GPUs into virtual slices ($4\times$ or $2\times$), prevents Out-Of-Memory errors with per-process VRAM fraction capping (`torch.cuda.set_per_process_memory_fraction`), and halves memory footprints using `faster-whisper` `int8_float16` quantization.
- **P4 VAD Silence Chunking & Parallel Dispatch**: Automatically splits long audio files at natural non-speech pauses (using ffmpeg VAD silence detection), distributes slices concurrently across worker threads, and stitches subtitles with global timestamp arithmetic and cue deduplication (**$4\times$ to $8\times$ turnaround acceleration on long recordings**).
- **Observability & Autoscaling**: Zero-dependency Prometheus `/metrics` exposition engine and a KEDA `ScaledObject` that scales GPU workers from **0 up to 8 pods** based on queue backlog, slashing cloud GPU costs to zero during idle periods.
- **Push Webhook Delivery**: Asynchronous completion notifications with exponential backoff retries and strict SSRF defenses against private network probing and cloud metadata leakage (`169.254.169.254`).

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Clients["1. API Clients & Ingestion"]
        Opencast["Opencast / Media API"] -->|POST /jobs| Bridge["FastAPI Bridge<br/>(:8080)"]
    end

    subgraph Storage_Queue["2. Storage & Queue Layer (PVC)"]
        Bridge -->|Enqueue Task| TaskQueue[("FIFO Task Queue<br/>(FileTaskQueue / Redis)")]
        Bridge -->|Write Media| SharedPVC[("Shared PVC (/data)<br/>videos / subs / models")]
    end

    subgraph Workers["3. Execution Engines"]
        TaskQueue -.->|Dequeue (<10ms)| WarmPool["Warm Worker Pool<br/>(Models Warm in VRAM)"]
        Bridge -.->|Submit Batch Job| EphemeralPod["Ephemeral K8s Job<br/>(Isolated Batch Pod)"]
        Bridge -.->|SSH Proxy| BareMetal["Bare-Metal Host<br/>(Native GPU / Mac Studio)"]
    end

    subgraph Processing["4. Core Optimization Pipeline"]
        WarmPool --> VAD["VAD Silence Chunking<br/>(app/chunking.py)"]
        VAD --> Inference["Multi-Backend Inference<br/>(faster-whisper / whisper.cpp)"]
        Inference --> Convert["Fast SRT/VTT Conversion<br/>& Single-Pass Embedding"]
        Convert --> Output["Output Subtitles & MP4<br/>(/data/subs)"]
    end

    subgraph Monitoring["5. Metrics & Autoscaling"]
        Bridge -->|GET /metrics| Prometheus["Prometheus Server"]
        Prometheus --> KEDA["KEDA Autoscaler"]
        KEDA -->|Scale 0 -> N Replicas| WarmPool
        Output -->|HTTP Callback| Webhook["Push Webhooks (SSRF-Safe)"]
    end
```

---

## Execution Modes

`whisper-k8s` provides three flexible execution modes configured per-job via `POST /jobs` or globally via `EXECUTION_MODE`:

| Execution Mode | Parameter | Description | Best For |
| :--- | :--- | :--- | :--- |
| **Warm Worker Pool** *(Recommended)* | `mode="pool"` | Immediate enqueue into `FileTaskQueue` or `RedisTaskQueue`. Persistent daemon workers hold models in GPU VRAM across consecutive tasks. | High-throughput, low-latency queues, lecture batch processing. Scales to 0 via KEDA. |
| **Ephemeral Batch Pod** | `mode="pod"` | Spawns an isolated Kubernetes batch `Job` per transcription task with dedicated CPU/memory/GPU resource limits. | Multi-tenant clusters, sporadic one-off jobs, strict job isolation. |
| **Bare-Metal SSH Proxy** | `mode="ssh"` | Proxies inference to an external bare-metal workstation (e.g., Apple Silicon Mac Studio via MPS or local RTX GPU) with automatic path translation. | Leveraging local hardware without running full Kubernetes on the host machine. |

---

## Quickstart Guide

### 1. Try It in 30 Seconds with Docker Compose
Start the platform locally with a single command (CPU worker by default; works out-of-the-box on any OS):
```bash
docker compose up -d
```

> [!TIP]
> If you have an NVIDIA GPU with the Container Toolkit installed, launch the GPU-accelerated worker pool instead:
> ```bash
> docker compose --profile gpu up -d
> ```

Transcribe the sample video immediately using the interactive CLI client:
```bash
python scripts/demo_transcribe.py --file demo.mp4 --format vtt --embed
```

### 2. Standalone Docker Run
```bash
# Build the unified container
docker build -t whisper-suite:latest .

# Run the API bridge
docker run -d -p 8080:8080 \
  -v $(pwd)/data:/data \
  -e SERVICE=bridge \
  whisper-suite:latest

# Run a warm worker daemon with GPU access
docker run -d --gpus all \
  -v $(pwd)/data:/data \
  -e SERVICE=pool_worker \
  -e DEVICE=cuda \
  -e BACKEND=faster-whisper \
  whisper-suite:latest
```

### 3. Deploy on Kubernetes with Kustomize
Deploy the entire production stack (Storage, RBAC, Bridge, Warm Workers, GPU Time-Slicing, Model Preloading, and KEDA Autoscaling) with a single command:

```bash
kubectl apply -k k8s_scripts/
```

Verify deployment status:
```bash
kubectl get pods,services,scaledobjects -n whisper
```

### 3. Submit a Transcription Job
Submit a job using `curl`:

```bash
# Submit an existing video with VAD chunking and push callback
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "lecture_01.mp4",
    "mode": "pool",
    "backend": "faster-whisper",
    "model": "base",
    "format": "vtt",
    "embed": true,
    "enableChunking": true,
    "parallelChunks": 4,
    "callbackUrl": "https://api.opencast.org/whisper/callback"
  }'
```

Response:
```json
{
  "jobId": "4c98a3b890d24e1b8b80e8f7ec409d20",
  "status": "accepted",
  "videoPath": "/data/videos/lecture_01.mp4",
  "namespace": "whisper"
}
```

Check job status:
```bash
curl http://localhost:8080/status/4c98a3b890d24e1b8b80e8f7ec409d20
```

Download completed subtitles:
```bash
curl http://localhost:8080/jobs/4c98a3b890d24e1b8b80e8f7ec409d20/download -o lecture_01.vtt
```

---

## Documentation Index

Explore the detailed manuals in the [`docs/`](docs/) directory:

- 📖 **[Architecture & Internals](docs/architecture.md)**: Deep dive into the 3 execution modes, multi-backend inference, GPU time-slicing mechanics, and the P4 VAD chunking/stitching algorithm.
- 📡 **[API Reference](docs/api_reference.md)**: Exhaustive documentation of all REST endpoints (`/jobs`, `/status`, `/download`, `/metrics`), request payloads, webhook event format, and SSRF security.
- ☸️ **[Kubernetes Deployment & Operations](docs/deployment_and_k8s.md)**: Production guide covering PVC provisioning, RBAC, NVIDIA time-slicing profiles, preloader Jobs, and KEDA scale-to-zero autoscaling.
- ⚙️ **[Configuration & Environment Variables](docs/configuration.md)**: Comprehensive reference table of all environment variables across bridge, worker, queue manager, and dispatcher.
- 🧪 **[Development, Testing & Benchmarking](docs/development_and_testing.md)**: How to run the 107 zero-dependency test suite, mock external services, run throughput benchmarks, and contribute.

---

## Verification & Testing

`whisper-k8s` includes a 116-test automated verification suite that runs in ~1.1 seconds with **zero external dependencies**:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 116 tests in 1.130s
OK
```

---

## Local Development & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TomBi/whisper-k8s.git
   cd whisper-k8s
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

3. **Install dependencies**:
   ```bash
   # Install PyTorch with your platform accelerator (e.g. CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # Install core dependencies
   pip install -r requirements.txt

   # Or install development & linting tools
   pip install -r requirements-dev.txt
   ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
