# Development, Testing & Benchmarking Guide

This guide covers local development workflows, running automated unit tests, executing performance benchmarks, and contributing to `whisper-k8s`.

---

## 1. Development Principles

1. **Zero External Test Dependencies**: The core test suite executes out-of-the-box using the Python standard library `unittest` runner. You do not need a live Kubernetes cluster, GPU drivers, Redis instance, or third-party PyPI packages installed globally to run the unit and integration tests.
2. **Deterministic Mocking**: [`tests/mock_dependencies.py`](../tests/mock_dependencies.py) intercepts and mocks missing heavy libraries (`kubernetes`, `fastapi`, `httpx`, `torch`, `whisper`, `pydantic`) if they are absent in the local development environment.
3. **Comprehensive Edge Case Defense**: Every critical algorithm (timestamp arithmetic, queue claiming, SSRF validation, subtitle stitching) is guarded by dedicated edge-case tests.

---

## 2. Test Suite Architecture

The test suite consists of 10 test modules organized by subsystem in the [`tests/`](../tests/) directory:

| Test File | Test Count | Subsystem Tested | Key Invariants Verified |
| :--- | :---: | :--- | :--- |
| `tests/test_metrics_and_keda.py` | 14 | Prometheus & KEDA | Thread-safe Counter/Gauge/Histogram, dynamic queue evaluation, 0.0.4 text format, KEDA YAML schema. |
| `tests/test_p2_webhooks.py` | 15 | Warm Worker & Webhooks | Atomic file queue locking, in-memory model cache reuse, SSRF validation, exponential backoff retries. |
| `tests/test_edge_cases.py` | 17 | Algorithm Edge Cases | Millisecond roll-overs (`59.999` $\rightarrow$ `60.000`), out-of-order cues, unicode/emojis, formatting tags, IPv6 SSRF. |
| `tests/test_p4_chunking.py` | 16 | VAD Chunking & Stitching | Silence boundary alignment, WAV extraction, timestamp offset shifting, boundary cue deduplication. |
| `tests/test_p3_gpu_timeslicing.py` | 15 | GPU Sharing & VRAM | Time-slicing profiles, per-process VRAM fraction bounds, PyTorch allocator fragmentation guard, model preloader. |
| `tests/test_p0_p1_optimizations.py`| 5 | P0 & P1 Optimizations | Elimination of 2nd inference pass, fast regex SRT-to-VTT conversion, persistent model cache env propagation. |
| `tests/test_performance_benchmark.py`| 3 | Empirical Benchmarks | P0 inference latency speedup, status disk I/O throughput, subtitle conversion cues/sec. |
| `tests/test_dispatcher.py` | 5 | K8s Batch Dispatcher | PodSpec volume mounts, DNS label sanitization, concurrency throttling (`MAX_PAR`), callback env forwarding. |
| `tests/test_bridge.py` | 10 | Bridge REST API | SSRF validation, GPU resource limits, TTL disk recovery, `/download` streaming, task cancellation. |
| `tests/test_transcriber.py` | 4 | Worker Core | Top-level exception handling, status JSON disk format, boolean environment variable parsing. |
| `tests/test_ssh_worker.py` | 2 | SSH Proxy | Pod-to-host path translation, environment variable forwarding parity. |
| `tests/test_e2e_smoke.py` | 1 | End-to-End Pipeline | Full audio transcription lifecycle using `demo.mp4`. |

---

## 3. Running Automated Tests

### Run the Full Test Suite
Execute all 107 tests across all modules:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
*Expected Execution Time: ~1.1 seconds with 100% pass rate.*

### Run an Individual Test Module
Target a specific feature area:

```bash
# Test Prometheus metrics and KEDA autoscaling
python -m unittest tests/test_metrics_and_keda.py -v

# Test audio chunking and stitching
python -m unittest tests/test_p4_chunking.py -v

# Test edge cases and formatting
python -m unittest tests/test_edge_cases.py -v
```

---

## 4. Running Performance Benchmarks

The benchmark suite in [`tests/test_performance_benchmark.py`](../tests/test_performance_benchmark.py) quantitatively verifies optimization gains:

```bash
python -m unittest tests/test_performance_benchmark.py -v
```

### Interpreting Benchmark Results

#### 1. P0 Redundant Inference Elimination
Measures the latency reduction of converting subtitles via regex (`convert_srt_to_vtt`) versus executing a second Whisper pass:
```text
=================================================================
 PERFORMANCE BENCHMARK: P0 REDUNDANT INFERENCE ELIMINATION 
=================================================================
  Old Workflow (2x Whisper passes):    100.62 ms
  New Workflow (1x pass + convert):     56.54 ms
  Format Conversion Latency (500 cues):  6.501 ms
  Speedup Ratio:                         1.78x faster
  Latency Reduction:                    43.80%
=================================================================
```

#### 2. Worker Status Disk I/O Throughput
Validates that writing real-time job progress to `/data/subs/{id}.status.json` incurs minimal filesystem overhead:
```text
=================================================================
 WORKER STATUS DISK I/O BENCHMARK (500 OPERATIONS) 
=================================================================
  Total Duration:      391.47 ms
  Mean Write Latency:   0.783 ms / write
  Throughput:             1,277 status writes/second
=================================================================
```

#### 3. Subtitle Converter Throughput
Stress-tests the SRT-to-VTT conversion engine across 5,000 cues:
```text
=================================================================
 SUBTITLE CONVERTER THROUGHPUT (5,000 CUES / 534.9 KB) 
=================================================================
  Execution Time:       12.05 ms
  Throughput (cues):    415,093 cues/second
  Throughput (data):    43.37 MB/second
=================================================================
```

---

## 5. Mocking Architecture (`mock_dependencies.py`)

When developing locally on machines without PyTorch, CUDA, or Kubernetes libraries installed, [`tests/mock_dependencies.py`](../tests/mock_dependencies.py) automatically injects mock classes into `sys.modules`:

```python
# Automatically intercepted if kubernetes is not installed
k8s_mod = ModuleType("kubernetes")
client_mod = ModuleType("kubernetes.client")
class BatchV1Api:
    def list_namespaced_job(self, *args, **kwargs): return SimpleRecord(items=[])
    def create_namespaced_job(self, *args, **kwargs): return None
```

This guarantees that:
- CI/CD pipelines can run lightweight pull request checks in $<2\text{s}$ on standard CPU runners without downloading 4 GB PyTorch CUDA wheels.
- Developers on macOS, Windows, and Linux can run the full test suite in any standard Python environment.

---

## 6. Contributing Guidelines

1. **Maintain Zero-Dependency Test Integrity**: Any new test added to `tests/` must execute without requiring external packages. Use standard library `unittest` and `mock_dependencies.py`.
2. **Atomic Commits**: Group feature modifications and tests into focused, descriptive commits following the Conventional Commits specification (e.g. `feat(chunking): ...`, `fix(bridge): ...`, `test(metrics): ...`).
3. **No Regressions**: Always run `python -m unittest discover -s tests -p "test_*.py" -v` before staging changes. All 107+ tests must pass cleanly.
