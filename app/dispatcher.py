#!/usr/bin/env python3
"""
Dispatcher for Whisper batch jobs on Kubernetes.

This script reads a list of item IDs (for example video filenames) from a
ConfigMap and schedules one Kubernetes Job per item. It throttles the number
of concurrently active Jobs per bridge, so you do not overwhelm the cluster.

Environment variables control:
- what bridge this dispatcher belongs to
- which image to use for worker Jobs
- where to read items from
- how many Jobs may run in parallel
- basic resource requests and limits
"""

import os
import sys
import time
import json
import re
import logging
from datetime import datetime
from typing import Optional, Dict, List

from kubernetes import client, config
from kubernetes.client import ApiException

ENV = os.environ
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required core configuration
# ---------------------------------------------------------------------------

# All Jobs created by this dispatcher are associated to a specific "bridge job".
ENV = os.environ
BRIDGE_JOB_ID = ENV.get("BRIDGE_JOB_ID", "").strip()

# Namespace where Jobs and ConfigMaps live.
NAMESPACE = ENV.get("NAMESPACE", "default")

# Container image for worker Jobs that actually run Whisper.
WORKER_IMAGE = ENV.get("WORKER_IMAGE", "whisper-suite:latest")

# PVC that should be mounted into worker Jobs (optional).
PVC_NAME = ENV.get("PVC_NAME", "")

# Logical name of the worker application used in labels and container name.
WORKER_NAME = ENV.get("WORKER_NAME", "whisper-worker")

# Prefix for Job names.
JOB_PREFIX = ENV.get("JOB_PREFIX", "whisper")

# Maximum number of active Jobs at the same time for this bridge.
MAX_PAR = int(ENV.get("MAX_PAR", "3"))

# How many times Kubernetes should retry a Job before giving up.
BACKOFF_LIMIT = int(ENV.get("BACKOFF_LIMIT", "0"))

# How long finished Jobs are kept around before being garbage collected.
TTL_SECONDS_AFTER_FINISHED = int(ENV.get("TTL_SECONDS_AFTER_FINISHED", "600"))

# ---------------------------------------------------------------------------
# Resource configuration
# ---------------------------------------------------------------------------

# CPU and memory requests for worker pods.
REQUESTS_CPU = ENV.get("REQUESTS_CPU", "250m")
REQUESTS_MEM = ENV.get("REQUESTS_MEM", "256Mi")

# CPU and memory limits for worker pods.
LIMITS_CPU = ENV.get("LIMITS_CPU", "1")
LIMITS_MEM = ENV.get("LIMITS_MEM", "1Gi")

# ---------------------------------------------------------------------------
# ConfigMap configuration
# ---------------------------------------------------------------------------

# Name of ConfigMap projected into workers for general config.
WORKER_CONFIG_CM = ENV.get("WORKER_CONFIG_CM", f"{JOB_PREFIX}-worker-config")

# ConfigMap that holds the list of item IDs for this bridge.
VIDEO_LIST_CM = ENV.get("VIDEO_LIST_CM", f"whisper-video-list-{BRIDGE_JOB_ID}")

# Optional base directories passed to workers so they know where to find videos
# and where to write subtitles on the shared volume.
VIDEOS_DIR = ENV.get("VIDEOS_DIR", "")
SUBS_DIR = ENV.get("SUBS_DIR", "")

# How often the dispatcher rechecks capacity when throttled.
SLEEP_SECONDS = int(ENV.get("SLEEP_SECONDS", "5"))

# Execution Mode: "pod" (default) or "ssh" (native host execution)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "pod").lower()

# SSH Configuration (only used if EXECUTION_MODE=ssh)
SSH_HOST_USER = os.getenv("SSH_HOST_USER", "admin")
SSH_REMOTE_PYTHON = os.getenv("SSH_REMOTE_PYTHON", "/usr/bin/python3")
SSH_REMOTE_SCRIPT = os.getenv("SSH_REMOTE_SCRIPT", "/Users/Shared/whisper-k8s/app/video_transcriber.py")
SSH_REMOTE_PATH_PREFIX = os.getenv("SSH_REMOTE_PATH_PREFIX", "/Users/Shared/data")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "/etc/secret/id_rsa")
MODELS_DIR = ENV.get("MODELS_DIR", "/data/models")
CUDA_MEMORY_FRACTION = ENV.get("CUDA_MEMORY_FRACTION")
COMPUTE_TYPE = ENV.get("COMPUTE_TYPE")
GPU_RESOURCE_NAME = ENV.get("GPU_RESOURCE_NAME", "nvidia.com/gpu")
REQUESTS_GPU = ENV.get("REQUESTS_GPU")
LIMITS_GPU = ENV.get("LIMITS_GPU")

# Whisper CPP Configuration
WHISPER_CPP_EXEC = os.getenv("WHISPER_CPP_EXEC")
WHISPER_CPP_MODEL_ROOT = os.getenv("WHISPER_CPP_MODEL_ROOT")

# ---------------------------------------------------------------------------
# Label and annotation templates
# ---------------------------------------------------------------------------

# Base labels applied to Jobs and pods.
# These are used both for bookkeeping and for selecting active Jobs.
LABELS_BASE: Dict[str, str] = {
    "app.kubernetes.io/name": WORKER_NAME,
    "app.kubernetes.io/part-of": "whisper-batch",
    "app.kubernetes.io/managed-by": "dispatcher",
    "bridge-job-id": BRIDGE_JOB_ID,
}

# Base annotations applied to Jobs and pods.
ANNOTATIONS_BASE: Dict[str, str] = {
    "dispatcher.openai.com/created-at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DNS1123_LABEL_RE = re.compile(r"[^a-z0-9-]+")


def dns1123_label(value: str, max_len: int = 63) -> str:
    """
    Convert an arbitrary string to a valid DNS 1123 label.

    The Kubernetes label and name constraints are:
    - lowercase alphanumeric characters and '-'
    - must start and end with an alphanumeric character
    - maximum length per segment is 63 characters

    This helper:
    - lowercases the input
    - replaces any invalid characters with '-'
    - collapses multiple '-' into one
    - strips leading and trailing '-'
    - truncates to max_len and strips trailing '-'

    Parameters
    ----------
    value:
        Input string to convert. If None, returns an empty string.
    max_len:
        Maximum length of the resulting label.

    Returns
    -------
    str
        A safe DNS 1123 label string. If the result would otherwise be empty,
        returns 'x'.
    """
    if value is None:
        return ""
    v = value.lower()
    v = v.replace(".", "-")  # dots are not allowed for DNS 1123 label
    v = DNS1123_LABEL_RE.sub("-", v)
    v = re.sub(r"-{2,}", "-", v)
    v = v.strip("-")
    if not v:
        v = "x"
    if len(v) > max_len:
        v = v[:max_len].rstrip("-")
    return v


def job_name_for(item_id: str) -> str:
    """
    Build a Kubernetes Job name for a given item id.

    The name is composed as:
        JOB_PREFIX-BRIDGE_JOB_ID-item_id
    and then sanitized into a DNS 1123 compliant label.

    Parameters
    ----------
    item_id:
        Logical identifier for the unit of work. Often a video filename.

    Returns
    -------
    str
        Sanitized Job name.
    """
    return dns1123_label(f"{JOB_PREFIX}-{BRIDGE_JOB_ID}-{item_id}")


def cm_name_for(name: str) -> str:
    """
    Sanitize a ConfigMap name to be DNS 1123 compatible.

    Parameters
    ----------
    name:
        Desired ConfigMap name.

    Returns
    -------
    str
        Sanitized name that satisfies DNS 1123 constraints.
    """
    return dns1123_label(name)


def load_kube() -> str:
    """
    Load Kubernetes configuration for the dispatcher process.

    Tries in this order:
    1. In cluster config (when running inside a pod).
    2. Local kubeconfig (when running outside the cluster).

    If both attempts fail, the process exits with code 2.

    Returns
    -------
    str
        Either "incluster" or "kubeconfig" depending on what was loaded.
    """
    try:
        config.load_incluster_config()
        return "incluster"
    except Exception:
        try:
            config.load_kube_config()
            return "kubeconfig"
        except Exception as e:
            logger.error("Failed to load Kubernetes config: %s", e)
            sys.exit(2)


def batch_api() -> client.BatchV1Api:
    """
    Get a BatchV1Api client.

    Returns
    -------
    kubernetes.client.BatchV1Api
        Client for batch resources such as Jobs.
    """
    return client.BatchV1Api()


def core_api() -> client.CoreV1Api:
    """
    Get a CoreV1Api client.

    Returns
    -------
    kubernetes.client.CoreV1Api
        Client for core resources such as ConfigMaps.
    """
    return client.CoreV1Api()


def list_active_jobs_for_bridge() -> List[client.V1Job]:
    """
    List active Jobs for this bridge.

    A Job is considered active if:
    - it has the base labels for this dispatcher
    - its status.active field is greater than 0

    Returns
    -------
    list[kubernetes.client.V1Job]
        List of currently active Job objects.
    """
    api = batch_api()
    selector_parts = []
    for k, v in LABELS_BASE.items():
        if "/" in k:
            # For label selectors only simple keys are supported.
            # Qualified labels can be present on the object but not used in selectors.
            continue
        selector_parts.append(f"{k}={dns1123_label(v)}")
    selector = ",".join(selector_parts)

    try:
        jobs = api.list_namespaced_job(namespace=NAMESPACE, label_selector=selector).items
    except ApiException as e:
        logger.error("Error listing jobs: %s", e)
        return []
    active: List[client.V1Job] = []
    for j in jobs:
        st = j.status
        if st and getattr(st, "active", 0) and st.active > 0:
            active.append(j)
    return active


def job_exists(name: str) -> bool:
    """
    Check whether a Job with the given name already exists in the namespace.

    Parameters
    ----------
    name:
        Name of the Job to check.

    Returns
    -------
    bool
        True if the Job exists, False if it does not exist or a non 404 error occurred.
    """
    api = batch_api()
    try:
        api.read_namespaced_job(name=name, namespace=NAMESPACE)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        logger.error("Error checking job existence for %s: %s", name, e)
        return False


def make_projected_volume_sources() -> List[client.V1VolumeProjection]:
    """
    Build a list of projected volume sources for worker pods.

    The dispatcher checks for the presence of:
    - WORKER_CONFIG_CM
    - VIDEO_LIST_CM

    Only existing ConfigMaps are added. Missing ones are silently ignored
    except for non 404 errors which are logged.

    Returns
    -------
    list[kubernetes.client.V1VolumeProjection]
        Projections that can be used in a projected volume.
    """
    api = core_api()
    cms = [cm_name_for(WORKER_CONFIG_CM), cm_name_for(VIDEO_LIST_CM)]
    projections: List[client.V1VolumeProjection] = []
    for cm in cms:
        try:
            api.read_namespaced_config_map(name=cm, namespace=NAMESPACE)
            projections.append(
                client.V1VolumeProjection(
                    config_map=client.V1ConfigMapProjection(name=cm)
                )
            )
        except ApiException as e:
            if e.status != 404:
                logger.warning("Error reading ConfigMap %s: %s", cm, e)
            # skip if missing
    return projections


def build_job(item_id: str, extra_env: Optional[Dict[str, str]] = None) -> client.V1Job:
    """
    Build a Kubernetes Job specification for a single item.

    The Job:
    - runs the worker container image
    - mounts the shared data PVC (if configured)
    - projects selected ConfigMaps into /configs
    - receives ITEM_ID, BRIDGE_JOB_ID and optional VIDEOS_DIR, SUBS_DIR
      via environment variables
    - uses the base labels and annotations so the dispatcher can track it

    Parameters
    ----------
    item_id:
        Logical work unit identifier. Passed to the worker as ITEM_ID.
    extra_env:
        Optional additional environment variables for the container.

    Returns
    -------
    kubernetes.client.V1Job
        Fully constructed Job object ready to create via the API.
    """
    name = job_name_for(item_id)
    labels = {k: dns1123_label(v) if "/" not in k else v for k, v in LABELS_BASE.items()}
    annotations = dict(ANNOTATIONS_BASE)
    annotations["dispatcher.openai.com/item-id"] = item_id

    env_list = [
        client.V1EnvVar(name="SERVICE", value="worker"),
        client.V1EnvVar(name="BRIDGE_JOB_ID", value=BRIDGE_JOB_ID),
        client.V1EnvVar(name="ITEM_ID", value=item_id),
    ]

    if VIDEOS_DIR:
        env_list.append(client.V1EnvVar(name="VIDEOS_DIR", value=VIDEOS_DIR))
    if SUBS_DIR:
        env_list.append(client.V1EnvVar(name="SUBS_DIR", value=SUBS_DIR))

    if extra_env:
        for k, v in extra_env.items():
            env_list.append(client.V1EnvVar(name=str(k), value=str(v)))

    # Persistent model cache roots on shared storage
    env_list.append(client.V1EnvVar(name="MODELS_DIR", value=MODELS_DIR))
    env_list.append(client.V1EnvVar(name="WHISPER_DOWNLOAD_ROOT", value=ENV.get("WHISPER_DOWNLOAD_ROOT", f"{MODELS_DIR}/whisper")))
    env_list.append(client.V1EnvVar(name="HF_HOME", value=ENV.get("HF_HOME", f"{MODELS_DIR}/huggingface")))
    env_list.append(client.V1EnvVar(name="TORCH_HOME", value=ENV.get("TORCH_HOME", f"{MODELS_DIR}/torch")))

    # Quantization and GPU time-slicing
    if CUDA_MEMORY_FRACTION:
        env_list.append(client.V1EnvVar(name="CUDA_MEMORY_FRACTION", value=CUDA_MEMORY_FRACTION))
    if COMPUTE_TYPE:
        env_list.append(client.V1EnvVar(name="COMPUTE_TYPE", value=COMPUTE_TYPE))

    # Whisper CPP config
    if WHISPER_CPP_EXEC:
        env_list.append(client.V1EnvVar(name="WHISPER_CPP_EXEC", value=WHISPER_CPP_EXEC))
    if WHISPER_CPP_MODEL_ROOT:
        env_list.append(client.V1EnvVar(name="WHISPER_CPP_MODEL_ROOT", value=WHISPER_CPP_MODEL_ROOT))
    else:
        env_list.append(client.V1EnvVar(name="WHISPER_CPP_MODEL_ROOT", value=f"{MODELS_DIR}/whisper.cpp"))

    requests_res = {"cpu": REQUESTS_CPU, "memory": REQUESTS_MEM}
    limits_res = {"cpu": LIMITS_CPU, "memory": LIMITS_MEM}
    if REQUESTS_GPU:
        requests_res[GPU_RESOURCE_NAME] = REQUESTS_GPU
    if LIMITS_GPU:
        limits_res[GPU_RESOURCE_NAME] = LIMITS_GPU

    resources = client.V1ResourceRequirements(
        requests=requests_res,
        limits=limits_res,
    )

    volume_mounts: List[client.V1VolumeMount] = []
    volumes: List[client.V1Volume] = []

    # Optional PVC containing input videos and output subtitles.
    if PVC_NAME:
        volumes.append(
            client.V1Volume(
                name="data",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=PVC_NAME
                )
            )
        )
        volume_mounts.append(client.V1VolumeMount(name="data", mount_path="/data"))

    # Project ConfigMaps (for example worker config and video list) into /configs.
    projections = make_projected_volume_sources()
    if projections:
        volumes.append(
            client.V1Volume(
                name="configs",
                projected=client.V1ProjectedVolumeSource(sources=projections)
            )
        )
        volume_mounts.append(
            client.V1VolumeMount(name="configs", mount_path="/configs", read_only=True)
        )

    # SSH Mode Configuration
    if EXECUTION_MODE == "ssh":
        # Override the command to run the SSH proxy script
        command = ["python", "/app/ssh_worker.py"]

        # Add SSH specific env vars
        env_list.extend([
            client.V1EnvVar(name="HOST_USER", value=SSH_HOST_USER),
            client.V1EnvVar(name="REMOTE_PYTHON", value=SSH_REMOTE_PYTHON),
            client.V1EnvVar(name="REMOTE_SCRIPT", value=SSH_REMOTE_SCRIPT),
            client.V1EnvVar(name="REMOTE_PATH_PREFIX", value=SSH_REMOTE_PATH_PREFIX),
            client.V1EnvVar(name="LOCAL_PATH_PREFIX", value="/data"), # Assuming /data is the mount point
            client.V1EnvVar(name="SSH_KEY_PATH", value=SSH_KEY_PATH),
            # Host IP via Downward API
            client.V1EnvVar(
                name="HOST_IP",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path="status.hostIP")
                )
            )
        ])

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
        volume_mounts.append(
            client.V1VolumeMount(
                name="ssh-key",
                mount_path="/etc/secret",
                read_only=True
            )
        )
    else:
        command = None # Use default entrypoint

    container = client.V1Container(
        name=WORKER_NAME,
        image=WORKER_IMAGE,
        image_pull_policy="IfNotPresent",
        env=env_list,
        command=command,
        resources=resources,
        volume_mounts=volume_mounts or None,
    )

    pod_labels = dict(labels)
    pod_annotations = dict(annotations)
    pod = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=pod_labels, annotations=pod_annotations),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            volumes=volumes or None,
        ),
    )

    spec = client.V1JobSpec(
        template=pod,
        backoff_limit=BACKOFF_LIMIT,
        ttl_seconds_after_finished=TTL_SECONDS_AFTER_FINISHED,
    )

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=NAMESPACE,
            labels=labels,
            annotations=annotations,
        ),
        spec=spec,
    )
    return job


def create_job_for_item(item_id: str) -> Optional[str]:
    """
    Create a Job for a given item id if it does not already exist.

    A Job is skipped if another Job with the same name already exists
    in the namespace. This prevents duplicate scheduling for the same item.

    Parameters
    ----------
    item_id:
        Logical work unit identifier.

    Returns
    -------
    str | None
        The Job name if a Job was created, otherwise None.
    """
    name = job_name_for(item_id)
    if job_exists(name):
        logger.info("Job %s already exists, skipping", name)
        return None
    api = batch_api()
    job = build_job(item_id=item_id)
    try:
        api.create_namespaced_job(namespace=NAMESPACE, body=job)
        logger.info("Created Job %s for item %s", name, item_id)
        return name
    except ApiException as e:
        logger.error("Failed to create job %s: %s", name, e)
        return None


def wait_for_capacity(max_par: int) -> None:
    """
    Block until there is capacity to start another Job.

    The dispatcher checks how many Jobs for this bridge are currently active
    (status.active > 0). If the number is greater than or equal to max_par,
    it sleeps SLEEP_SECONDS and checks again.

    Parameters
    ----------
    max_par:
        Maximum number of active Jobs allowed at once.
    """
    while True:
        active_jobs = list_active_jobs_for_bridge()
        count = len(active_jobs)
        if count < max_par:
            return
        logger.info(
            "Throttling bridge %s. Active %d >= MAX_PAR %d. Sleeping %ds...",
            BRIDGE_JOB_ID,
            count,
            max_par,
            SLEEP_SECONDS,
        )
        time.sleep(SLEEP_SECONDS)


def read_items_from_configmap(cm_name: str) -> List[str]:
    """
    Read work items from a ConfigMap.

    Two modes are supported:
    1. If the ConfigMap contains a key 'items.json', its value is parsed as
       JSON and must be a list of items.
    2. Otherwise, all keys of the ConfigMap are treated as items and their
       values are ignored.

    Parameters
    ----------
    cm_name:
        Name of the ConfigMap to read.

    Returns
    -------
    list[str]
        List of item identifiers. Returns an empty list if the ConfigMap
        cannot be read or parsing fails.
    """
    api = core_api()
    try:
        cm = api.read_namespaced_config_map(
            name=cm_name_for(cm_name),
            namespace=NAMESPACE,
        )
    except ApiException as e:
        logger.error("Could not read ConfigMap %s: %s", cm_name, e)
        return []
    data = cm.data or {}
    if "items.json" in data:
        try:
            arr = json.loads(data["items.json"])
            return [str(x) for x in arr]
        except Exception as e:
            logger.error("Invalid JSON in items.json of %s: %s", cm_name, e)
            return []
    # fallback: keys as items
    return list(data.keys())


def main() -> None:
    """
    Main entry point for the dispatcher.

    Steps:
    1. Load Kubernetes configuration.
    2. Read the list of items to process from VIDEO_LIST_CM.
    3. For each item:
       - wait until the number of active Jobs is below MAX_PAR
       - create a new Job if it does not already exist
    4. Log the list of Jobs that were created.

    Returns
    -------
    None
        Exits the process with code 0 on success.
    """

    if not BRIDGE_JOB_ID:
        logger.error("BRIDGE_JOB_ID is required")
        sys.exit(2)
    mode = load_kube()
    logger.info("Kubernetes config loaded via %s", mode)

    # Read work items from the per bridge video list ConfigMap.
    items = read_items_from_configmap(VIDEO_LIST_CM)
    if not items:
        logger.info("No items to process for bridge %s. Exiting.", BRIDGE_JOB_ID)
        return

    total = len(items)
    logger.info(
        "Scheduling %d items for bridge %s with MAX_PAR=%d",
        total,
        BRIDGE_JOB_ID,
        MAX_PAR,
    )

    created: List[str] = []
    for idx, item in enumerate(items, start=1):
        # Per item scheduling info: log progress and which item is being scheduled.
        logger.info(
            "Scheduling item %s (%d/%d)",
            item,
            idx,
            total,
        )
        wait_for_capacity(MAX_PAR)
        name = create_job_for_item(item)
        if name:
            created.append(name)
        else:
            logger.info(
                "No job created for item %s (might already exist or failed)",
                item,
            )

    logger.info("Finished scheduling. Created %d new jobs.", len(created))
    for n in created:
        logger.info("  job: %s", n)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
