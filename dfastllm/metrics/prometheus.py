"""Prometheus metrics for vdiff serving.

Metrics match vLLM's naming conventions where applicable,
with additional vdiff-specific metrics.
"""

import time
from typing import Optional
import logging

from fastapi.responses import Response

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics disabled")


# Global metrics registry
_metrics_initialized = False
_model_name = "unknown"

# Counters (matching vLLM naming)
request_success_total: Optional["Counter"] = None
request_failure_total: Optional["Counter"] = None
prompt_tokens_total: Optional["Counter"] = None
generation_tokens_total: Optional["Counter"] = None

# vdiff specific counters
parallel_tokens_decoded_total: Optional["Counter"] = None

# Histograms (matching vLLM naming)
time_to_first_token_seconds: Optional["Histogram"] = None
request_latency_seconds: Optional["Histogram"] = None
time_per_output_token_seconds: Optional["Histogram"] = None

# vdiff specific histograms
parallel_batch_size: Optional["Histogram"] = None

# Gauges
kv_cache_hit_rate: Optional["Gauge"] = None
num_requests_running: Optional["Gauge"] = None
num_requests_waiting: Optional["Gauge"] = None
gpu_memory_usage: Optional["Gauge"] = None

# Info
model_info: Optional["Info"] = None


def setup_metrics(model_name: str) -> None:
    """Initialize Prometheus metrics.
    
    Metrics are named to match vLLM conventions where applicable.
    
    Args:
        model_name: Name of the model being served.
    """
    global _metrics_initialized, _model_name
    global request_success_total, request_failure_total
    global prompt_tokens_total, generation_tokens_total
    global parallel_tokens_decoded_total
    global time_to_first_token_seconds, request_latency_seconds
    global time_per_output_token_seconds, parallel_batch_size
    global kv_cache_hit_rate, num_requests_running, num_requests_waiting
    global gpu_memory_usage, model_info
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available, skipping metrics setup")
        return
    
    if _metrics_initialized:
        logger.debug("Metrics already initialized")
        return
    
    _model_name = model_name
    
    # Request counters (vLLM compatible)
    request_success_total = Counter(
        "vdiff_request_success_total",
        "Total number of successful requests",
        ["model"],
    )
    
    request_failure_total = Counter(
        "vdiff_request_failure_total",
        "Total number of failed requests",
        ["model"],
    )
    
    # Token counters (vLLM compatible)
    prompt_tokens_total = Counter(
        "vdiff_prompt_tokens_total",
        "Total number of prompt tokens processed",
        ["model"],
    )
    
    generation_tokens_total = Counter(
        "vdiff_generation_tokens_total",
        "Total number of tokens generated",
        ["model"],
    )
    
    # vdiff specific: parallel decoding tokens
    parallel_tokens_decoded_total = Counter(
        "vdiff_parallel_tokens_decoded_total",
        "Total number of tokens decoded in parallel (APD)",
        ["model"],
    )
    
    # Latency histograms (vLLM compatible)
    time_to_first_token_seconds = Histogram(
        "vdiff_time_to_first_token_seconds",
        "Time to first token in seconds",
        ["model"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    
    request_latency_seconds = Histogram(
        "vdiff_request_latency_seconds",
        "Total request latency in seconds",
        ["model"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    
    time_per_output_token_seconds = Histogram(
        "vdiff_time_per_output_token_seconds",
        "Average time per output token in seconds",
        ["model"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    
    # vdiff specific: parallel batch size distribution
    parallel_batch_size = Histogram(
        "vdiff_parallel_batch_size",
        "Distribution of parallel decoding batch sizes (APD)",
        ["model"],
        buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    
    # Gauges (vLLM compatible)
    kv_cache_hit_rate = Gauge(
        "vdiff_kv_cache_hit_rate",
        "KV cache hit rate",
        ["model"],
    )
    
    num_requests_running = Gauge(
        "vdiff_num_requests_running",
        "Number of requests currently running",
        ["model"],
    )
    
    num_requests_waiting = Gauge(
        "vdiff_num_requests_waiting",
        "Number of requests waiting in queue",
        ["model"],
    )
    
    gpu_memory_usage = Gauge(
        "vdiff_gpu_memory_usage_bytes",
        "GPU memory usage in bytes",
        ["model", "device"],
    )
    
    # Model info
    model_info = Info(
        "vdiff_model",
        "Information about the served model",
    )
    model_info.info({
        "model_name": model_name,
        "model_type": "diffusion-llm",
    })
    
    _metrics_initialized = True
    logger.info(f"Prometheus metrics initialized for model: {model_name}")


def record_request(
    success: bool,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    ttft: Optional[float] = None,
    total_latency: Optional[float] = None,
    parallel_tokens: int = 0,
) -> None:
    """Record metrics for a request.
    
    Args:
        success: Whether the request was successful.
        prompt_tokens: Number of prompt tokens.
        generated_tokens: Number of generated tokens.
        ttft: Time to first token in seconds.
        total_latency: Total request latency in seconds.
        parallel_tokens: Number of tokens decoded in parallel.
    """
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    
    try:
        if success:
            request_success_total.labels(model=_model_name).inc()
        else:
            request_failure_total.labels(model=_model_name).inc()
            return  # Don't record other metrics for failed requests
        
        # Token counts
        if prompt_tokens > 0:
            prompt_tokens_total.labels(model=_model_name).inc(prompt_tokens)
        
        if generated_tokens > 0:
            generation_tokens_total.labels(model=_model_name).inc(generated_tokens)
        
        # vdiff specific: parallel tokens
        if parallel_tokens > 0:
            parallel_tokens_decoded_total.labels(model=_model_name).inc(parallel_tokens)
            parallel_batch_size.labels(model=_model_name).observe(parallel_tokens)
        
        # Latencies
        if ttft is not None and ttft > 0:
            time_to_first_token_seconds.labels(model=_model_name).observe(ttft)
        
        if total_latency is not None and total_latency > 0:
            request_latency_seconds.labels(model=_model_name).observe(total_latency)
            
            if generated_tokens > 0:
                time_per_token = total_latency / generated_tokens
                time_per_output_token_seconds.labels(model=_model_name).observe(time_per_token)
    
    except Exception as e:
        logger.error(f"Error recording metrics: {e}")


def update_kv_cache_hit_rate(hit_rate: float) -> None:
    """Update the KV cache hit rate metric.
    
    Args:
        hit_rate: Current hit rate (0-1).
    """
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    
    try:
        kv_cache_hit_rate.labels(model=_model_name).set(hit_rate)
    except Exception as e:
        logger.error(f"Error updating KV cache hit rate: {e}")


def update_queue_metrics(running: int, waiting: int) -> None:
    """Update queue metrics.
    
    Args:
        running: Number of running requests.
        waiting: Number of waiting requests.
    """
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    
    try:
        num_requests_running.labels(model=_model_name).set(running)
        num_requests_waiting.labels(model=_model_name).set(waiting)
    except Exception as e:
        logger.error(f"Error updating queue metrics: {e}")


def update_gpu_memory(device: str, memory_bytes: int) -> None:
    """Update GPU memory usage metric.
    
    Args:
        device: GPU device identifier.
        memory_bytes: Memory usage in bytes.
    """
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    
    try:
        gpu_memory_usage.labels(model=_model_name, device=device).set(memory_bytes)
    except Exception as e:
        logger.error(f"Error updating GPU memory metric: {e}")


def metrics_endpoint() -> Response:
    """Generate Prometheus metrics response.
    
    Returns:
        FastAPI Response with metrics in Prometheus format.
    """
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="# Prometheus client not available\n",
            media_type="text/plain",
        )
    
    try:
        metrics_output = generate_latest(REGISTRY)
        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {e}\n",
            media_type="text/plain",
            status_code=500,
        )


class MetricsContext:
    """Context manager for timing requests."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prompt_tokens: int = 0
        self.generated_tokens: int = 0
        self.parallel_tokens: int = 0
        self.success: bool = True
    
    def __enter__(self) -> "MetricsContext":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        self.success = exc_type is None
        
        ttft = None
        if self.first_token_time and self.start_time:
            ttft = self.first_token_time - self.start_time
        
        total_latency = None
        if self.end_time and self.start_time:
            total_latency = self.end_time - self.start_time
        
        record_request(
            success=self.success,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            ttft=ttft,
            total_latency=total_latency,
            parallel_tokens=self.parallel_tokens,
        )
    
    def mark_first_token(self) -> None:
        """Mark when the first token was generated."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
