# Security Policy & Architectural Threat Model

The `whisper-k8s` project takes the security of distributed speech processing, cloud infrastructure, and enterprise media pipelines seriously. This document outlines our security architecture, built-in threat mitigations, and responsible vulnerability disclosure policy.

---

## 1. Supported Versions

Security updates and critical patches are actively provided for:

| Version | Supported |
| :--- | :---: |
| `1.0.x` | :white_check_mark: |
| `< 1.0.0` | :x: |

---

## 2. Built-in Security Architecture & Threat Mitigations

### A. Server-Side Request Forgery (SSRF) Protection
The bridge API (`POST /jobs`) accepts remote media URLs (`trackUrl`) and push notification targets (`callbackUrl`). To prevent malicious actors from using `whisper-k8s` as an internal network scanning proxy or leaking cloud instance credentials:
- **DNS Resolution & IP Inspection**: Every target hostname is resolved to its underlying IP address and validated against strict address rules prior to connection.
- **Cloud Metadata Defense**: Explicitly blocks requests to link-local cloud metadata endpoints (such as `169.254.169.254` on AWS, GCP, and Azure).
- **Private Subnet & Loopback Isolation**: Automatically rejects loopback addresses (`127.0.0.1`, `::1`, `localhost`), RFC 1918 private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local subnets (`169.254.0.0/16`), and multicast addresses.
- **Allowed Local Override**: For isolated development environments only, `ALLOW_LOCAL_URLS=true` may be set to permit localhost testing; this is strictly disabled by default.

### B. Path Traversal & Filesystem Sandboxing
All file access operations (audio ingestion, subtitle output, and custom destination directories) are strictly bounded:
- **Canonical Path Containment (`_safe_under`)**: Resolves target paths via `os.path.realpath` and verifies that the canonical path starts with the designated shared volume root (`/data`).
- Rejects any attempt to use relative traversal (`../`), null-byte injections, or arbitrary host filesystem paths outside the persistent volume.

### C. GPU Memory & Resource Isolation
- **Fractional VRAM Guardrails**: Workers enforce strict per-process GPU memory limits via `torch.cuda.set_per_process_memory_fraction` (default `0.25`), preventing individual jobs from triggering GPU-wide CUDA Out-Of-Memory (OOM) crashes across co-located tenant workers.
- **PyTorch Fragmentation Guards**: Configured with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid allocator thrashing under heavy concurrent workloads.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability in `whisper-k8s`, please do **not** report it via a public GitHub issue.

Instead, please report it via responsible disclosure:

1. **GitHub Security Advisory** (Preferred): Submit a private report via the [Security Advisories](https://github.com/TomBi/whisper-k8s/security/advisories) tab on GitHub.
2. **Direct Email**: Send encrypted details to `tom.bintener@gmail.com`.

### What to Include in Your Report
- A detailed description of the vulnerability, including affected endpoints or components.
- Step-by-step reproduction instructions or a minimal Proof of Concept (PoC).
- Any potential impact on cloud infrastructure, PVC data, or GPU hardware.

### Response Timeline
- **Initial Acknowledgment**: Within 48 hours.
- **Vulnerability Assessment & Triage**: Within 5 business days.
- **Patch Release & Security Notice**: Coordinated with the reporter before public disclosure.
