"""Pytest configuration and fixtures for vdiff tests."""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, patch

from dfastllm.config import DFastLLMConfig as VDiffConfig
from dfastllm.engine.sampling_params import SamplingParams
from dfastllm.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> VDiffConfig:
    """Create a mock configuration for testing."""
    return VDiffConfig(
        model="mock-model",
        tokenizer="mock-model",
        host="127.0.0.1",
        port=8000,
        max_model_len=2048,
        dtype="float32",
        enable_apd=True,
        apd_max_parallel=8,
        apd_threshold=0.3,
        block_size=4,
    )


@pytest.fixture
def default_sampling_params() -> SamplingParams:
    """Create default sampling parameters."""
    return SamplingParams(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        n=1,
    )


@pytest.fixture
def mock_completion_output() -> CompletionOutput:
    """Create a mock completion output."""
    return CompletionOutput(
        index=0,
        text="This is a test response.",
        token_ids=[1, 2, 3, 4, 5],
        finish_reason="stop",
        parallel_tokens_decoded=3,
    )


@pytest.fixture
def mock_request_output(mock_completion_output) -> RequestOutput:
    """Create a mock request output."""
    return RequestOutput(
        request_id="test-request-123",
        prompt="Test prompt",
        prompt_token_ids=[10, 20, 30],
        outputs=[mock_completion_output],
        finished=True,
        metrics=RequestMetrics(
            prompt_tokens=3,
            generated_tokens=5,
            parallel_tokens_decoded=3,
        ),
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.vocab_size = 32000
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    
    # Mock generate output
    import torch
    mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model.generate.return_value = mock_output
    
    # Mock forward output
    forward_output = MagicMock()
    forward_output.logits = torch.randn(1, 10, 32000)
    model.return_value = forward_output
    
    return model


@pytest.fixture
def mock_engine(mock_config, mock_tokenizer, mock_request_output):
    """Create a mock vdiff engine."""
    engine = MagicMock()
    engine.config = mock_config
    engine.is_ready = True
    engine._tokenizer = mock_tokenizer
    engine.tokenizer = mock_tokenizer
    engine.generate.return_value = mock_request_output
    engine.generate_async.return_value = mock_request_output
    
    async def mock_stream(*args, **kwargs):
        yield mock_request_output
    
    engine.generate_stream = mock_stream
    engine.get_stats.return_value = {
        "requests_processed": 100,
        "tokens_generated": 5000,
        "parallel_tokens_decoded": 2000,
    }
    
    return engine


@pytest.fixture
async def test_client(mock_config, mock_engine):
    """Create a test client for the API server."""
    from fastapi.testclient import TestClient
    from vdiff.entrypoints.openai.api_server import create_app
    
    with patch("vdiff.entrypoints.openai.api_server.VDiffEngine") as MockEngine:
        MockEngine.return_value = mock_engine
        
        # Override lifespan for testing
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def test_lifespan(app):
            import vdiff.entrypoints.openai.api_server as server
            server.engine = mock_engine
            server.completion_serving = MagicMock()
            server.chat_serving = MagicMock()
            yield
        
        app = create_app(mock_config)
        app.router.lifespan_context = test_lifespan
        
        client = TestClient(app)
        yield client
