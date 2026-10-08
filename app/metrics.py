#!/usr/bin/env python3
"""
Zero-Dependency Prometheus Metrics Exposition Engine for whisper-k8s.

Provides thread-safe Prometheus metric primitives conforming to the
official Prometheus text exposition standard 0.0.4 (text/plain; version=0.0.4; charset=utf-8).
Runs out-of-the-box with standard library Python (zero external dependencies).
"""

import collections
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable, Union

logger = logging.getLogger(__name__)


def escape_label_value(val: Any) -> str:
    """Escape label values according to Prometheus 0.0.4 exposition standard."""
    s = str(val)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_labels(labels: Dict[str, Any]) -> str:
    """Format dictionary of labels into Prometheus format '{k1="v1",k2="v2"}' sorted by key."""
    if not labels:
        return ""
    sorted_items = sorted(labels.items(), key=lambda x: str(x[0]))
    pairs = [f'{k}="{escape_label_value(v)}"' for k, v in sorted_items]
    return "{" + ",".join(pairs) + "}"


def _labels_to_key(labels: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    """Convert label dict to an immutable sorted tuple key for indexing."""
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _key_to_labels(key: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    """Convert immutable tuple key back to a label dictionary."""
    return dict(key)


class Metric:
    """Base class for all Prometheus metric primitives."""

    def __init__(self, name: str, documentation: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.documentation = documentation
        self.label_names = tuple(label_names or [])
        self._lock = threading.Lock()

    def collect(self) -> List[Tuple[str, Dict[str, str], float]]:
        """Return list of samples as (sample_name, labels_dict, float_value)."""
        raise NotImplementedError


class Counter(Metric):
    """Monotonically increasing cumulative metric."""

    def __init__(self, name: str, documentation: str, label_names: Optional[List[str]] = None):
        super().__init__(name, documentation, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        """Increment counter by non-negative value."""
        if value < 0:
            raise ValueError("Counters can only be incremented by non-negative values")
        key = _labels_to_key(labels)
        with self._lock:
            self._values[key] += float(value)

    def get(self, **labels: Any) -> float:
        """Retrieve current value for given label combination."""
        key = _labels_to_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        """Reset counter values (useful for tests)."""
        with self._lock:
            self._values.clear()

    def collect(self) -> List[Tuple[str, Dict[str, str], float]]:
        with self._lock:
            if not self._values and not self.label_names:
                return [(self.name, {}, 0.0)]
            return [
                (self.name, _key_to_labels(k), v)
                for k, v in sorted(self._values.items(), key=lambda x: x[0])
            ]


class Gauge(Metric):
    """Instantaneous metric value that can go up and down, or dynamic callback."""

    def __init__(
        self,
        name: str,
        documentation: str,
        label_names: Optional[List[str]] = None,
        callback: Optional[Callable[[], Any]] = None,
    ):
        super().__init__(name, documentation, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)
        self._callback = callback

    def set(self, value: float, **labels: Any) -> None:
        """Set gauge to an arbitrary float value."""
        key = _labels_to_key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        """Increment gauge by value."""
        key = _labels_to_key(labels)
        with self._lock:
            self._values[key] += float(value)

    def dec(self, value: float = 1.0, **labels: Any) -> None:
        """Decrement gauge by value."""
        key = _labels_to_key(labels)
        with self._lock:
            self._values[key] -= float(value)

    def get(self, **labels: Any) -> float:
        """Retrieve current value for given label combination."""
        key = _labels_to_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def set_callback(self, callback: Optional[Callable[[], Any]]) -> None:
        """Register a dynamic evaluator function invoked at scrape time."""
        with self._lock:
            self._callback = callback

    def reset(self) -> None:
        """Reset gauge values (useful for tests)."""
        with self._lock:
            self._values.clear()

    def collect(self) -> List[Tuple[str, Dict[str, str], float]]:
        with self._lock:
            if self._callback is not None:
                try:
                    res = self._callback()
                    # Callback can return a single float/int or list of (labels_dict, value)
                    if isinstance(res, (int, float)):
                        return [(self.name, {}, float(res))]
                    elif isinstance(res, list):
                        samples = []
                        for item in res:
                            if isinstance(item, (tuple, list)) and len(item) == 2:
                                lbls, val = item
                                samples.append((self.name, dict(lbls), float(val)))
                        return samples
                    elif isinstance(res, dict):
                        return [(self.name, dict(lbls), float(val)) for lbls, val in res.items()]
                except Exception as e:
                    logger.warning("[metrics] Error executing callback for %s: %s", self.name, e)

            if not self._values and not self.label_names:
                return [(self.name, {}, 0.0)]
            return [
                (self.name, _key_to_labels(k), v)
                for k, v in sorted(self._values.items(), key=lambda x: x[0])
            ]


class Histogram(Metric):
    """Histogram tracking observations into predefined buckets, sum, and count."""

    DEFAULT_BUCKETS = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)

    def __init__(
        self,
        name: str,
        documentation: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Union[List[float], Tuple[float, ...]]] = None,
    ):
        super().__init__(name, documentation, label_names)
        raw_b = tuple(sorted(buckets)) if buckets else self.DEFAULT_BUCKETS
        self.buckets = raw_b
        self._counts: Dict[Tuple[Tuple[str, str], ...], int] = collections.defaultdict(int)
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)
        # _bucket_counts[label_key][bucket_le] = count
        self._bucket_counts: Dict[
            Tuple[Tuple[str, str], ...], Dict[float, int]
        ] = collections.defaultdict(lambda: collections.defaultdict(int))

    def observe(self, value: float, **labels: Any) -> None:
        """Record an observation in the histogram."""
        v = float(value)
        key = _labels_to_key(labels)
        with self._lock:
            self._counts[key] += 1
            self._sums[key] += v
            b_counts = self._bucket_counts[key]
            for b in self.buckets:
                if v <= b:
                    b_counts[b] += 1

    def reset(self) -> None:
        """Reset histogram observations (useful for tests)."""
        with self._lock:
            self._counts.clear()
            self._sums.clear()
            self._bucket_counts.clear()

    def collect(self) -> List[Tuple[str, Dict[str, str], float]]:
        samples: List[Tuple[str, Dict[str, str], float]] = []
        with self._lock:
            # If empty and no labels, provide default zero summary
            all_keys = list(self._counts.keys())
            if not all_keys and not self.label_names:
                all_keys = [()]

            for key in sorted(all_keys):
                base_labels = _key_to_labels(key)
                b_counts = self._bucket_counts[key]
                total_count = self._counts.get(key, 0)
                total_sum = self._sums.get(key, 0.0)

                # Output cumulative bucket samples
                for b in self.buckets:
                    b_lbls = dict(base_labels)
                    b_lbls["le"] = f"{b:.1f}" if b.is_integer() else str(b)
                    samples.append((f"{self.name}_bucket", b_lbls, float(b_counts[b])))

                # +Inf bucket equals total count
                inf_lbls = dict(base_labels)
                inf_lbls["le"] = "+Inf"
                samples.append((f"{self.name}_bucket", inf_lbls, float(total_count)))

                # Sum and count
                samples.append((f"{self.name}_sum", dict(base_labels), float(total_sum)))
                samples.append((f"{self.name}_count", dict(base_labels), float(total_count)))

        return samples


class MetricsRegistry:
    """Thread-safe registry managing metrics and formatting Prometheus exposition output."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def register(self, metric: Metric) -> Metric:
        """Register a metric instance."""
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]
            self._metrics[metric.name] = metric
            return metric

    def counter(self, name: str, documentation: str, label_names: Optional[List[str]] = None) -> Counter:
        with self._lock:
            if name in self._metrics:
                m = self._metrics[name]
                if isinstance(m, Counter):
                    return m
                raise ValueError(f"Metric '{name}' already registered with different type")
            c = Counter(name, documentation, label_names)
            self._metrics[name] = c
            return c

    def gauge(
        self,
        name: str,
        documentation: str,
        label_names: Optional[List[str]] = None,
        callback: Optional[Callable[[], Any]] = None,
    ) -> Gauge:
        with self._lock:
            if name in self._metrics:
                m = self._metrics[name]
                if isinstance(m, Gauge):
                    if callback:
                        m.set_callback(callback)
                    return m
                raise ValueError(f"Metric '{name}' already registered with different type")
            g = Gauge(name, documentation, label_names, callback=callback)
            self._metrics[name] = g
            return g

    def histogram(
        self,
        name: str,
        documentation: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Union[List[float], Tuple[float, ...]]] = None,
    ) -> Histogram:
        with self._lock:
            if name in self._metrics:
                m = self._metrics[name]
                if isinstance(m, Histogram):
                    return m
                raise ValueError(f"Metric '{name}' already registered with different type")
            h = Histogram(name, documentation, label_names, buckets=buckets)
            self._metrics[name] = h
            return h

    def get(self, name: str) -> Optional[Metric]:
        with self._lock:
            return self._metrics.get(name)

    def render_prometheus_text(self) -> str:
        """
        Format all registered metrics into official Prometheus 0.0.4 text representation:
        # HELP metric_name Documentation
        # TYPE metric_name type
        metric_name{labels} value
        """
        lines: List[str] = []
        with self._lock:
            metric_items = sorted(self._metrics.items(), key=lambda x: x[0])

        for name, metric in metric_items:
            m_type = "untyped"
            if isinstance(metric, Counter):
                m_type = "counter"
            elif isinstance(metric, Gauge):
                m_type = "gauge"
            elif isinstance(metric, Histogram):
                m_type = "histogram"

            lines.append(f"# HELP {metric.name} {metric.documentation}")
            lines.append(f"# TYPE {metric.name} {m_type}")

            samples = metric.collect()
            for sample_name, labels, val in samples:
                lbl_str = format_labels(labels)
                # Ensure float representation or integer format
                val_str = f"{val:.1f}" if val.is_integer() else f"{val}"
                lines.append(f"{sample_name}{lbl_str} {val_str}")

        return "\n".join(lines) + "\n"


# ==============================================================================
# Global Default Registry & Standard Framework Metrics
# ==============================================================================

REGISTRY = MetricsRegistry()

# 1. Total jobs processed across modes and terminal statuses
JOBS_TOTAL = REGISTRY.counter(
    "whisper_jobs_total",
    "Total submitted transcription jobs across execution modes and terminal statuses",
    ["status", "mode"],
)

# 2. End-to-end processing duration histogram
JOB_DURATION_SECONDS = REGISTRY.histogram(
    "whisper_job_duration_seconds",
    "End-to-end transcription processing duration in seconds",
    ["mode"],
    buckets=[5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0],
)

# 3. Dynamic queue backlog gauge
QUEUE_DEPTH = REGISTRY.gauge(
    "whisper_queue_depth",
    "Instantaneous count of transcription tasks across queue states",
    ["queue"],
)

# 4. Active worker pod concurrency gauge
ACTIVE_WORKERS = REGISTRY.gauge(
    "whisper_active_workers",
    "Number of active worker processes/daemons actively running jobs",
    ["service"],
)

# 5. In-memory model cache hits and misses
MODEL_CACHE_EVENTS = REGISTRY.counter(
    "whisper_model_cache_events_total",
    "In-memory Whisper model cache lookup events (hits and misses)",
    ["event", "model", "backend"],
)

# 6. Push webhook delivery dispatch outcomes
WEBHOOKS_DISPATCHED = REGISTRY.counter(
    "whisper_webhooks_dispatched_total",
    "Total push webhook delivery notifications dispatched by outcome",
    ["status"],
)


def get_metrics_registry() -> MetricsRegistry:
    """Return the global default metrics registry instance."""
    return REGISTRY
