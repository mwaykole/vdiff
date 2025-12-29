"""Unit tests for dfastllm Engine."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from dfastllm.config import DFastLLMConfig
from dfastllm.engine.sampling_params import SamplingParams
from dfastllm.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics


class TestDFastLLMConfig:
    """Test cases for DFastLLMConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DFastLLMConfig(model="test-model")
        
        assert config.model == "test-model"
        assert config.tokenizer == "test-model"  # Defaults to model
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.enable_apd is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DFastLLMConfig(
            model="custom-model",
            tokenizer="custom-tokenizer",
            host="127.0.0.1",
            port=9000,
            enable_apd=False,
            apd_threshold=0.5,
        )
        
        assert config.model == "custom-model"
        assert config.tokenizer == "custom-tokenizer"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.enable_apd is False
        assert config.apd_threshold == 0.5
    
    def test_validation_gpu_memory(self):
        """Test GPU memory utilization validation."""
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            DFastLLMConfig(model="test", gpu_memory_utilization=1.5)
        
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            DFastLLMConfig(model="test", gpu_memory_utilization=0.0)
    
    def test_validation_apd_threshold(self):
        """Test APD threshold validation."""
        with pytest.raises(ValueError, match="apd_threshold"):
            DFastLLMConfig(model="test", apd_threshold=1.5)
    
    def test_validation_apd_max_parallel(self):
        """Test APD max parallel validation."""
        with pytest.raises(ValueError, match="apd_max_parallel"):
            DFastLLMConfig(model="test", apd_max_parallel=0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DFastLLMConfig(
            model="test-model",
            port=8080,
        )
        
        d = config.to_dict()
        
        assert d["model"] == "test-model"
        assert d["port"] == 8080
        assert "enable_apd" in d


class TestCompletionOutput:
    """Test cases for CompletionOutput."""
    
    def test_basic_creation(self):
        """Test basic output creation."""
        output = CompletionOutput(
            index=0,
            text="Hello, world!",
            token_ids=[1, 2, 3],
            finish_reason="stop",
        )
        
        assert output.index == 0
        assert output.text == "Hello, world!"
        assert output.token_ids == [1, 2, 3]
        assert output.finish_reason == "stop"
    
    def test_parallel_tokens(self):
        """Test parallel tokens tracking."""
        output = CompletionOutput(
            index=0,
            text="Test",
            parallel_tokens_decoded=5,
        )
        
        assert output.parallel_tokens_decoded == 5


class TestRequestOutput:
    """Test cases for RequestOutput."""
    
    def test_basic_creation(self):
        """Test basic request output creation."""
        completion = CompletionOutput(index=0, text="Response")
        output = RequestOutput(
            request_id="req-123",
            prompt="Hello",
            outputs=[completion],
            finished=True,
        )
        
        assert output.request_id == "req-123"
        assert output.prompt == "Hello"
        assert len(output.outputs) == 1
        assert output.finished is True
    
    def test_metrics_auto_created(self):
        """Test that metrics are auto-created."""
        output = RequestOutput(
            request_id="req-123",
            prompt="Hello",
        )
        
        assert output.metrics is not None


class TestRequestMetrics:
    """Test cases for RequestMetrics."""
    
    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = RequestMetrics(
            prompt_tokens=10,
            generated_tokens=20,
        )
        
        assert metrics.prompt_tokens == 10
        assert metrics.generated_tokens == 20
    
    def test_time_to_first_token(self):
        """Test TTFT calculation."""
        metrics = RequestMetrics()
        metrics.arrival_time = 100.0
        metrics.first_token_time = 100.5
        
        assert metrics.time_to_first_token == 0.5
    
    def test_total_latency(self):
        """Test total latency calculation."""
        metrics = RequestMetrics()
        metrics.arrival_time = 100.0
        metrics.finished_time = 102.0
        
        assert metrics.total_latency == 2.0
    
    def test_time_per_token(self):
        """Test time per token calculation."""
        metrics = RequestMetrics()
        metrics.arrival_time = 100.0
        metrics.finished_time = 102.0
        metrics.generated_tokens = 10
        
        assert metrics.time_per_token == 0.2
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RequestMetrics(
            prompt_tokens=10,
            generated_tokens=20,
            parallel_tokens_decoded=15,
        )
        
        d = metrics.to_dict()
        
        assert d["prompt_tokens"] == 10
        assert d["generated_tokens"] == 20
        assert d["parallel_tokens_decoded"] == 15


class TestDFastLLMEngineInit:
    """Test cases for DFastLLMEngine initialization."""
    
    @patch("dfastllm.engine.dfastllm_engine.TokenizerWrapper")
    @patch("dfastllm.engine.dfastllm_engine.ModelConfig")
    @patch("dfastllm.engine.dfastllm_engine.TORCH_AVAILABLE", False)
    def test_mock_mode_initialization(self, mock_model_config, mock_tokenizer):
        """Test engine initialization in mock mode (no PyTorch)."""
        from dfastllm.engine.dfastllm_engine import DFastLLMEngine
        
        mock_tokenizer.return_value = MagicMock()
        mock_model_config.from_pretrained.return_value = MagicMock()
        
        config = DFastLLMConfig(model="mock-model")
        engine = DFastLLMEngine(config)
        
        assert engine.is_ready is True
        assert engine.config == config
