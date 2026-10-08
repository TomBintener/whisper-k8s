#!/usr/bin/env python3
"""
Task Queue Manager for P2: Warm Worker Pool.

Provides a thread-safe and process-safe queue abstraction supporting:
1. FileTaskQueue: Zero-dependency atomic file-based FIFO queue on shared volume.
2. RedisTaskQueue: Distributed high-throughput queue using Redis (if available).
"""

import os
import time
import json
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class TaskQueue(ABC):
    """Abstract interface for Whisper task queues."""

    @abstractmethod
    def enqueue(self, job_spec: Dict[str, Any]) -> str:
        """Enqueue a job spec. Returns the assigned job_id."""
        pass

    @abstractmethod
    def dequeue(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Pop the next available job spec. Blocks up to timeout seconds."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of pending jobs in queue."""
        pass

    @abstractmethod
    def peek(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Inspect a specific job by job_id without removing it."""
        pass

    @abstractmethod
    def complete(self, job_id: str, success: bool = True) -> None:
        """Mark a job as finished."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all pending jobs from queue."""
        pass


class FileTaskQueue(TaskQueue):
    """
    Zero-dependency file-based FIFO task queue.
    Uses atomic filesystem operations (os.replace) to ensure thread-safe
    and process-safe task assignment across multiple concurrent worker pods.
    """

    def __init__(self, queue_dir: Optional[str] = None):
        base = queue_dir or os.getenv("QUEUE_DIR") or "/data/queue"
        self.base_dir = Path(base)
        self.pending_dir = self.base_dir / "pending"
        self.processing_dir = self.base_dir / "processing"
        self.completed_dir = self.base_dir / "completed"

        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def enqueue(self, job_spec: Dict[str, Any]) -> str:
        with self._lock:
            job_id = str(job_spec.get("job_id") or job_spec.get("item_id") or uuid.uuid4().hex)
            job_spec["job_id"] = job_id
            timestamp = time.time()

            filename = f"{timestamp:.6f}_{job_id}.json"
            tmp_path = self.pending_dir / f"{filename}.tmp"
            target_path = self.pending_dir / filename

            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(job_spec, f)

            # Atomic move into pending queue
            os.replace(tmp_path, target_path)
            logger.info("[queue] Enqueued job %s -> %s", job_id, target_path.name)
            return job_id

    def dequeue(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        start_time = time.time()

        while True:
            with self._lock:
                # List all pending files sorted chronologically by timestamp
                try:
                    pending_files = sorted(
                        [f for f in self.pending_dir.iterdir() if f.is_file() and f.suffix == ".json"],
                        key=lambda p: p.name,
                    )
                except Exception as e:
                    logger.warning("[queue] Error reading pending queue: %s", e)
                    pending_files = []

                for pending_file in pending_files:
                    claim_id = uuid.uuid4().hex[:8]
                    target_proc = self.processing_dir / f"{pending_file.stem}_{claim_id}.json"
                    try:
                        # Atomic move from pending to processing claims the job exclusively
                        os.replace(pending_file, target_proc)
                        with target_proc.open("r", encoding="utf-8") as f:
                            spec = json.load(f)
                        spec["_claim_file"] = target_proc.name
                        logger.info("[queue] Claimed job %s -> %s", spec.get("job_id", target_proc.stem), target_proc.name)
                        return spec
                    except (FileNotFoundError, PermissionError):
                        # Another concurrent worker claimed it first, check next file
                        continue
                    except Exception as e:
                        logger.error("[queue] Error claiming job %s: %s", pending_file.name, e)
                        continue

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None

            time.sleep(min(0.05, timeout - elapsed))

    def size(self) -> int:
        with self._lock:
            try:
                return len([f for f in self.pending_dir.iterdir() if f.is_file() and f.suffix == ".json"])
            except Exception:
                return 0

    def peek(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for d in [self.pending_dir, self.processing_dir, self.completed_dir]:
                try:
                    for f in d.iterdir():
                        if f.is_file() and (
                            f"_{job_id}_" in f.name
                            or f"_{job_id}." in f.name
                            or f.name.startswith(f"{job_id}_")
                            or f.stem == job_id
                        ):
                            with f.open("r", encoding="utf-8") as fp:
                                return json.load(fp)
                except Exception:
                    continue
            return None

    def complete(self, job_id: str, success: bool = True) -> None:
        with self._lock:
            try:
                for f in self.processing_dir.iterdir():
                    if f.is_file() and (
                        f"_{job_id}_" in f.name
                        or f"_{job_id}." in f.name
                        or f.name.startswith(f"{job_id}_")
                        or f.stem == job_id
                    ):
                        target = self.completed_dir / f.name
                        os.replace(f, target)
                        logger.info("[queue] Marked job %s completed (success=%s)", job_id, success)
                        return
            except Exception as e:
                logger.warning("[queue] Error completing job %s: %s", job_id, e)

    def clear(self) -> None:
        with self._lock:
            for d in [self.pending_dir, self.processing_dir, self.completed_dir]:
                try:
                    for f in d.iterdir():
                        if f.is_file():
                            f.unlink(missing_ok=True)
                except Exception:
                    pass


class RedisTaskQueue(TaskQueue):
    """Distributed Redis task queue using LPUSH / BRPOP for high-throughput scaling."""

    def __init__(self, redis_url: Optional[str] = None):
        import redis  # type: ignore

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.queue_key = "whisper:jobs:pending"
        self.processing_key = "whisper:jobs:processing"
        self.jobs_hash = "whisper:jobs:specs"

    def enqueue(self, job_spec: Dict[str, Any]) -> str:
        job_id = str(job_spec.get("job_id") or job_spec.get("item_id") or uuid.uuid4().hex)
        job_spec["job_id"] = job_id
        spec_json = json.dumps(job_spec)

        self.client.hset(self.jobs_hash, job_id, spec_json)
        self.client.lpush(self.queue_key, job_id)
        logger.info("[redis-queue] Enqueued job %s", job_id)
        return job_id

    def dequeue(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        timeout_int = max(1, int(round(timeout)))
        res = self.client.brpop(self.queue_key, timeout=timeout_int)
        if not res:
            return None
        _, job_id = res
        spec_json = self.client.hget(self.jobs_hash, job_id)
        if spec_json:
            self.client.sadd(self.processing_key, job_id)
            return json.loads(spec_json)
        return None

    def size(self) -> int:
        return self.client.llen(self.queue_key)

    def peek(self, job_id: str) -> Optional[Dict[str, Any]]:
        spec_json = self.client.hget(self.jobs_hash, job_id)
        if spec_json:
            return json.loads(spec_json)
        return None

    def complete(self, job_id: str, success: bool = True) -> None:
        self.client.srem(self.processing_key, job_id)
        self.client.hdel(self.jobs_hash, job_id)

    def clear(self) -> None:
        self.client.delete(self.queue_key, self.processing_key, self.jobs_hash)


def get_queue(queue_type: Optional[str] = None, queue_dir: Optional[str] = None) -> TaskQueue:
    """Factory function returning the active TaskQueue implementation."""
    q_type = (queue_type or os.getenv("QUEUE_TYPE", "file")).strip().lower()
    if q_type == "redis":
        try:
            return RedisTaskQueue()
        except Exception as e:
            logger.warning("Redis initialization failed, falling back to FileTaskQueue: %s", e)
            return FileTaskQueue(queue_dir=queue_dir)
    return FileTaskQueue(queue_dir=queue_dir)
