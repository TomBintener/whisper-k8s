#!/usr/bin/env python3
"""
SSH Proxy Worker.

This script runs inside a Kubernetes Pod but executes the actual transcription
logic on the remote (host) machine via SSH. This allows utilizing the host's
native hardware (Apple Silicon / Metal) which is inaccessible from standard
Docker containers.

It forwards all necessary environment variables and handles path rewriting
if the storage paths differ between the Pod and the Host.
"""

import os
import sys
import subprocess
import logging
import shlex
import stat
import shutil

# Configuration
HOST_USER = os.getenv("HOST_USER", "admin")
# In Docker Desktop/Minikube, this often resolves to the host. 
# In bare metal K8s, use the Downward API to pass status.hostIP.
HOST_IP = os.getenv("HOST_IP", "host.docker.internal")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "/etc/secret/ssh-privatekey")

# Path to the python executable and script ON THE HOST
REMOTE_PYTHON = os.getenv("REMOTE_PYTHON", "/usr/bin/python3")
REMOTE_SCRIPT = os.getenv("REMOTE_SCRIPT", "/Users/Shared/whisper-k8s/app/video_transcriber.py")

# Path mapping: Pod Path -> Host Path
# Example: Pod sees /data, Host sees /Users/Shared/data
LOCAL_PATH_PREFIX = os.getenv("LOCAL_PATH_PREFIX", "/data")
REMOTE_PATH_PREFIX = os.getenv("REMOTE_PATH_PREFIX", "/Users/Shared/data")

# Environment variables to forward to the host process
FORWARD_VARS = [
    "ITEM_ID", "BRIDGE_JOB_ID", 
    "MODEL", "DEVICE", "SUB_FORMAT", "EMBED_SUBS", "LANGUAGE",
    "OVERWRITE", "TASK", "BACKEND",
    "WHISPER_CPP_MODEL_ROOT", # If using whisper.cpp
    "WHISPER_CPP_EXEC",       # Path to whisper.cpp executable
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "OUTPUT_DIR", "CLEANUP",
    "CUDA_MEMORY_FRACTION", "COMPUTE_TYPE",
    "PARALLEL_CHUNKS", "CHUNK_DURATION_SEC", "ENABLE_CHUNKING",
    "CALLBACK_URL", "CALLBACK_HEADERS",
    "MODELS_DIR", "WHISPER_DOWNLOAD_ROOT", "HF_HOME", "TORCH_HOME",
]

logger = logging.getLogger("ssh-worker")
logging.basicConfig(level=logging.INFO)

def rewrite_path(path: str) -> str:
    """Rewrite a path from the Pod's perspective to the Host's perspective."""
    if not path:
        return ""
    if path.startswith(LOCAL_PATH_PREFIX):
        return path.replace(LOCAL_PATH_PREFIX, REMOTE_PATH_PREFIX, 1)
    return path

def main():
    logger.info(f"Preparing to SSH into {HOST_USER}@{HOST_IP}...")

    # Pre-flight check: SSH Key existence and permissions
    if not os.path.exists(SSH_KEY_PATH):
        logger.error(f"SSH Key not found at: {SSH_KEY_PATH}")
        sys.exit(1)
    
    # FIX: Copy key to internal storage to force correct permissions.
    # Docker bind-mounts often force permissions (like 0755) that SSH rejects.
    secure_key_path = "/tmp/id_rsa_secure"
    try:
        shutil.copyfile(SSH_KEY_PATH, secure_key_path)
        os.chmod(secure_key_path, 0o600)
        logger.info(f"Copied SSH key to {secure_key_path} with strict permissions.")
    except Exception as e:
        logger.warning(f"Could not copy/chmod key: {e}. Using original path.")
        secure_key_path = SSH_KEY_PATH

    # 1. Build environment exports
    env_exports = []
    
    # Handle paths specifically
    videos_dir = rewrite_path(os.getenv("VIDEOS_DIR", ""))
    subs_dir = rewrite_path(os.getenv("SUBS_DIR", ""))
    
    if videos_dir:
        env_exports.append(f"export VIDEOS_DIR='{videos_dir}'")
    if subs_dir:
        env_exports.append(f"export SUBS_DIR='{subs_dir}'")

    # Handle other vars
    for key in FORWARD_VARS:
        val = os.getenv(key)
        if val:
            # Use shlex.quote for robust shell safety
            env_exports.append(f"export {key}={shlex.quote(val)}")

    # Log the resolved environment to help user debug path issues
    logger.info("--- Remote Environment Configuration ---")
    for exp in env_exports:
        logger.info(f"  {exp}")
    logger.info("----------------------------------------")
    
    # Debug: Check if the whisper executable is visible to the SSH session
    whisper_exec = os.getenv("WHISPER_CPP_EXEC")
    if whisper_exec:
        logger.info(f"Debug: Verifying {whisper_exec} on remote host...")
        check_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-i", secure_key_path,
            f"{HOST_USER}@{HOST_IP}",
            f"ls -l {shlex.quote(whisper_exec)}"
        ]
        # We run this and ignore errors, just to print output to logs
        subprocess.run(check_cmd)
        logger.info("----------------------------------------")

    # 2. Construct the remote command
    # We chain the exports and then run the python script
    env_cmd = "; ".join(env_exports)
    full_remote_cmd = f"{env_cmd}; {REMOTE_PYTHON} {REMOTE_SCRIPT}"
    
    logger.info(f"Remote command: {REMOTE_PYTHON} {REMOTE_SCRIPT}")

    # 3. Execute SSH
    # -o StrictHostKeyChecking=no is used to avoid interactive prompts. 
    # In production, manage known_hosts properly.
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", secure_key_path,
        f"{HOST_USER}@{HOST_IP}",
        full_remote_cmd
    ]

    # Replace current process with SSH so signals (like SIGTERM from K8s) 
    # propagate to the SSH client (and hopefully the remote process).
    sys.stdout.flush()
    os.execvp("ssh", ssh_cmd)

if __name__ == "__main__":
    main()
