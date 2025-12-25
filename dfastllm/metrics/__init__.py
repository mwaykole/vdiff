"""Prometheus metrics for vdiff serving."""

from dfastllm.metrics.prometheus import (
    setup_metrics,
    record_request,
    metrics_endpoint,
    update_kv_cache_hit_rate,
    update_gpu_memory,
    update_queue_metrics,
)

__all__ = [
    "setup_metrics",
    "record_request",
    "metrics_endpoint",
    "update_kv_cache_hit_rate",
    "update_gpu_memory",
    "update_queue_metrics",
]
