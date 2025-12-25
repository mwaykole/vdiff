"""Tests for optimization modules.

Covers:
- Attention caching
- Model quantization
- Adaptive step scheduling
- Mixed precision
"""

import pytest
import logging

logger = logging.getLogger(__name__)

# Check for torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestAttentionCache:
    """Tests for AttentionCache optimization."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_cache_config_defaults(self):
        """Test default configuration values."""
        from dfastllm.engine.attention_cache import AttentionCacheConfig
        
        config = AttentionCacheConfig()
        assert config.enabled is True
        assert config.cache_interval == 4
        assert config.max_cache_size_mb == 512
        assert config.warmup_steps == 2
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_should_recompute_warmup(self):
        """Test that attention is always recomputed during warmup."""
        from dfastllm.engine.attention_cache import AttentionCache, AttentionCacheConfig
        
        config = AttentionCacheConfig(warmup_steps=3)
        cache = AttentionCache(config)
        
        # During warmup, always recompute
        assert cache.should_recompute(0) is True
        assert cache.should_recompute(1) is True
        assert cache.should_recompute(2) is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_cache_interval_behavior(self):
        """Test that cache respects interval settings."""
        from dfastllm.engine.attention_cache import AttentionCache, AttentionCacheConfig
        
        config = AttentionCacheConfig(warmup_steps=0, cache_interval=4)
        cache = AttentionCache(config)
        
        # Step 0 - recompute (boundary)
        assert cache.should_recompute(0) is True
        # Steps 1-3 - use cache
        assert cache.should_recompute(1) is False
        assert cache.should_recompute(2) is False
        assert cache.should_recompute(3) is False
        # Step 4 - recompute (boundary)
        assert cache.should_recompute(4) is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_cache_update_and_get(self):
        """Test caching and retrieval of attention tensors."""
        from dfastllm.engine.attention_cache import AttentionCache, AttentionCacheConfig
        
        config = AttentionCacheConfig(cache_layers=[0, 1, 2])
        cache = AttentionCache(config)
        
        # Create and cache a tensor
        attention = torch.randn(2, 8, 64, 64)
        cache.update(0, attention)
        
        # Retrieve cached tensor
        cached = cache.get(0)
        assert cached is not None
        assert cached.shape == attention.shape
        
        # Check hit rate
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss from update
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_cache_memory_limit(self):
        """Test that cache respects memory limits."""
        from dfastllm.engine.attention_cache import AttentionCache, AttentionCacheConfig
        
        config = AttentionCacheConfig(max_cache_size_mb=1, cache_layers=[0, 1, 2, 3])
        cache = AttentionCache(config)
        
        # Try to cache a very large tensor (should be rejected)
        large_attention = torch.randn(256, 64, 1024, 1024)  # ~67GB
        cache.update(0, large_attention)
        
        # Should not be cached due to size limit
        cached = cache.get(0)
        # May or may not be cached depending on available memory
        assert cache._get_cache_size_mb() <= config.max_cache_size_mb or cached is None


class TestQuantization:
    """Tests for model quantization."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_quantization_config_defaults(self):
        """Test default quantization configuration."""
        from dfastllm.engine.quantization import QuantizationConfig
        
        config = QuantizationConfig()
        assert config.enabled is True
        assert config.dtype == "int8"
        assert config.quantize_linear is True
        assert config.quantize_embedding is False
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_quantization_disabled(self):
        """Test that disabled quantization returns model unchanged."""
        from dfastllm.engine.quantization import ModelQuantizer, QuantizationConfig
        
        model = nn.Linear(64, 64)
        config = QuantizationConfig(enabled=False)
        quantizer = ModelQuantizer(config)
        
        result = quantizer.quantize(model)
        assert result is model
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_dynamic_int8_quantization(self):
        """Test dynamic INT8 quantization."""
        from dfastllm.engine.quantization import ModelQuantizer, QuantizationConfig
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)
            
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        model = SimpleModel()
        config = QuantizationConfig(enabled=True, dtype="int8")
        quantizer = ModelQuantizer(config)
        
        quantized = quantizer.quantize(model)
        
        # Model should still work
        x = torch.randn(4, 64)
        output = quantized(x)
        assert output.shape == (4, 64)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_memory_savings_estimation(self):
        """Test memory savings estimation utility."""
        from dfastllm.engine.quantization import estimate_memory_savings
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1024, 1024)
        
        model = SimpleModel()
        estimates = estimate_memory_savings(model, "int8")
        
        assert "current_size_gb" in estimates
        assert "estimated_quantized_gb" in estimates
        assert "estimated_savings_gb" in estimates
        assert estimates["linear_params_millions"] > 0


class TestAdaptiveSteps:
    """Tests for adaptive step scheduling."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_adaptive_step_config_defaults(self):
        """Test default adaptive step configuration."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepConfig
        
        config = AdaptiveStepConfig()
        assert config.enabled is True
        assert config.min_steps == 8
        assert config.max_steps == 128
        assert config.confidence_threshold == 0.95
        assert config.convergence_patience == 2
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scheduler_no_early_stop_before_min_steps(self):
        """Test that scheduler doesn't stop before minimum steps."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepScheduler, AdaptiveStepConfig
        
        config = AdaptiveStepConfig(min_steps=8)
        scheduler = AdaptiveStepScheduler(config)
        scheduler.start_generation(max_steps=64)
        
        # High confidence but before min_steps
        high_conf = torch.ones(10) * 0.99
        
        for step in range(7):
            should_stop = scheduler.should_stop_early(high_conf, step, masks_remaining=5, total_masks=10)
            assert should_stop is False, f"Unexpected early stop at step {step}"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scheduler_stops_when_all_unmasked(self):
        """Test that scheduler stops when all masks are removed."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepScheduler, AdaptiveStepConfig
        
        config = AdaptiveStepConfig()
        scheduler = AdaptiveStepScheduler(config)
        scheduler.start_generation(max_steps=64)
        
        confidence = torch.rand(10)
        should_stop = scheduler.should_stop_early(confidence, step=10, masks_remaining=0, total_masks=10)
        assert should_stop is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scheduler_stops_on_high_unmasked_ratio(self):
        """Test that scheduler stops when unmasked ratio is high."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepScheduler, AdaptiveStepConfig
        
        config = AdaptiveStepConfig(min_unmasked_ratio=0.9)
        scheduler = AdaptiveStepScheduler(config)
        scheduler.start_generation(max_steps=64)
        
        # 95% unmasked (5 masks remaining out of 100)
        confidence = torch.rand(100)
        should_stop = scheduler.should_stop_early(confidence, step=32, masks_remaining=5, total_masks=100)
        assert should_stop is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scheduler_statistics_tracking(self):
        """Test that scheduler tracks statistics correctly."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepScheduler, AdaptiveStepConfig
        
        config = AdaptiveStepConfig()
        scheduler = AdaptiveStepScheduler(config)
        
        # Simulate a generation
        scheduler.start_generation(max_steps=64)
        confidence = torch.rand(10)
        for step in range(20):
            scheduler.should_stop_early(confidence, step, masks_remaining=10-step//2, total_masks=10)
        scheduler.end_generation()
        
        stats = scheduler.get_stats()
        assert stats["total_generations"] == 1
        assert stats["total_steps_executed"] == 20
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_optimal_block_length_computation(self):
        """Test optimal block length computation."""
        from dfastllm.engine.adaptive_steps import compute_optimal_block_length
        
        # Perfect divisor
        block_len, num_blocks = compute_optimal_block_length(128, max_block=32)
        assert block_len == 32
        assert num_blocks == 4
        
        # Needs adjustment
        block_len, num_blocks = compute_optimal_block_length(100, max_block=32)
        assert 100 % block_len == 0
        assert block_len * num_blocks == 100
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_recommended_steps_temperature_adjustment(self):
        """Test that recommended steps adjust based on temperature."""
        from dfastllm.engine.adaptive_steps import AdaptiveStepScheduler, AdaptiveStepConfig
        
        config = AdaptiveStepConfig(max_steps=128)
        scheduler = AdaptiveStepScheduler(config)
        
        # Lower temperature should recommend fewer steps
        greedy_steps = scheduler.get_recommended_steps(gen_length=64, temperature=0.0)
        high_temp_steps = scheduler.get_recommended_steps(gen_length=64, temperature=1.5)
        
        assert greedy_steps <= high_temp_steps


class TestMixedPrecision:
    """Tests for mixed precision support in diffusion sampler."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_sampler_config_mixed_precision_flag(self):
        """Test that sampler config includes mixed precision flag."""
        from dfastllm.engine.diffusion_sampler import DiffusionSamplerConfig
        
        config = DiffusionSamplerConfig()
        assert hasattr(config, 'use_mixed_precision')
        assert config.use_mixed_precision is True  # Default should be True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_sampler_config_adaptive_steps_flag(self):
        """Test that sampler config includes adaptive steps flag."""
        from dfastllm.engine.diffusion_sampler import DiffusionSamplerConfig
        
        config = DiffusionSamplerConfig()
        assert hasattr(config, 'use_adaptive_steps')
        assert hasattr(config, 'confidence_threshold')


class TestIntegration:
    """Integration tests for optimization modules."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_all_optimizations_in_config(self):
        """Test that all optimization flags exist in DFastLLMConfig."""
        from dfastllm.config import DFastLLMConfig
        
        config = DFastLLMConfig(model="test-model")
        
        # Check all optimization flags exist
        assert hasattr(config, 'compile_model')
        assert hasattr(config, 'use_flash_attention')
        assert hasattr(config, 'use_8bit')
        assert hasattr(config, 'use_4bit')
        assert hasattr(config, 'use_mixed_precision')
        assert hasattr(config, 'use_adaptive_steps')
        assert hasattr(config, 'confidence_threshold')
        assert hasattr(config, 'enable_early_stopping')
        assert hasattr(config, 'use_attention_cache')
        assert hasattr(config, 'use_dynamic_quantization')
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_engine_init_exports(self):
        """Test that all optimization classes are exported from engine."""
        from dfastllm.engine import (
            AttentionCache,
            AttentionCacheConfig,
            ModelQuantizer,
            QuantizationConfig,
            AdaptiveStepScheduler,
            AdaptiveStepConfig,
            compute_optimal_block_length,
        )
        
        # All classes should be importable
        assert AttentionCache is not None
        assert AttentionCacheConfig is not None
        assert ModelQuantizer is not None
        assert QuantizationConfig is not None
        assert AdaptiveStepScheduler is not None
        assert AdaptiveStepConfig is not None
        assert compute_optimal_block_length is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



