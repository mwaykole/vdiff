"""Integration tests for Prometheus metrics."""

import pytest
from unittest.mock import patch


class TestMetrics:
    """Integration tests for metrics collection."""
    
    def test_setup_metrics(self):
        """Test metrics initialization."""
        from dfastllm.metrics import setup_metrics
        
        # Should not raise
        setup_metrics("test-model")
    
    def test_record_request_success(self):
        """Test recording successful request."""
        from dfastllm.metrics import setup_metrics, record_request
        
        setup_metrics("test-model")
        
        # Should not raise
        record_request(
            success=True,
            prompt_tokens=10,
            generated_tokens=20,
            ttft=0.1,
            total_latency=1.0,
            parallel_tokens=15,
        )
    
    def test_record_request_failure(self):
        """Test recording failed request."""
        from dfastllm.metrics import setup_metrics, record_request
        
        setup_metrics("test-model")
        
        # Should not raise
        record_request(success=False)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns valid response."""
        from dfastllm.metrics import setup_metrics, metrics_endpoint
        
        setup_metrics("test-model")
        
        response = metrics_endpoint()
        
        assert response.status_code == 200
        content = response.body.decode() if hasattr(response.body, 'decode') else str(response.body)
        # Should contain some metrics or indication of prometheus
        assert len(content) > 0
    
    def test_update_kv_cache_hit_rate(self):
        """Test updating KV cache hit rate metric."""
        from dfastllm.metrics import setup_metrics, update_kv_cache_hit_rate
        
        setup_metrics("test-model")
        
        # Should not raise
        update_kv_cache_hit_rate(0.85)
    
    def test_metrics_context_manager(self):
        """Test MetricsContext context manager."""
        from dfastllm.metrics.prometheus import MetricsContext, setup_metrics
        
        setup_metrics("test-model")
        
        with MetricsContext() as ctx:
            ctx.prompt_tokens = 10
            ctx.generated_tokens = 20
            ctx.parallel_tokens = 15
            ctx.mark_first_token()
        
        assert ctx.first_token_time is not None
        assert ctx.end_time is not None
        assert ctx.success is True
    
    def test_metrics_context_on_error(self):
        """Test MetricsContext records failure on exception."""
        from dfastllm.metrics.prometheus import MetricsContext, setup_metrics
        
        setup_metrics("test-model")
        
        try:
            with MetricsContext() as ctx:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert ctx.success is False


class TestMetricsNaming:
    """Test that metrics follow vLLM naming conventions."""
    
    def test_metric_names(self):
        """Test that metric names match vLLM conventions."""
        from dfastllm.metrics.prometheus import (
            setup_metrics,
            request_success_total,
            request_failure_total,
            prompt_tokens_total,
            generation_tokens_total,
            time_to_first_token_seconds,
            request_latency_seconds,
            kv_cache_hit_rate,
        )
        
        setup_metrics("test-model")
        
        # Check metric names contain vdiff prefix
        # These would be the actual Prometheus metric objects if available
        if request_success_total is not None:
            assert "vdiff" in str(request_success_total._name).lower() or "request_success" in str(request_success_total._name)
