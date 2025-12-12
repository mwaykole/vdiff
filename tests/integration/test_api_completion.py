"""Integration tests for completion API."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from vdiff.config import VDiffConfig
from vdiff.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics


@pytest.fixture
def mock_engine():
    """Create a mock engine for testing."""
    engine = MagicMock()
    engine.is_ready = True
    
    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
    engine.tokenizer = tokenizer
    
    # Mock generate_async
    async def mock_generate(prompt, sampling_params, request_id=None):
        return RequestOutput(
            request_id=request_id or "test-id",
            prompt=prompt,
            prompt_token_ids=[1, 2, 3],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="This is a test response.",
                    token_ids=[4, 5, 6, 7, 8],
                    finish_reason="stop",
                    parallel_tokens_decoded=3,
                )
            ],
            finished=True,
            metrics=RequestMetrics(
                prompt_tokens=3,
                generated_tokens=5,
                parallel_tokens_decoded=3,
            ),
        )
    
    engine.generate_async = mock_generate
    
    # Mock generate_stream
    async def mock_stream(prompt, sampling_params, request_id=None):
        yield RequestOutput(
            request_id=request_id or "test-id",
            prompt=prompt,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="This is a test response.",
                    finish_reason="stop",
                )
            ],
            finished=True,
        )
    
    engine.generate_stream = mock_stream
    engine.get_stats.return_value = {"requests_processed": 100}
    
    return engine


@pytest.fixture
def app_client(mock_engine):
    """Create a test client with mocked engine."""
    import vdiff.entrypoints.openai.api_server as server
    from vdiff.entrypoints.openai.api_server import create_app
    from vdiff.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vdiff.entrypoints.openai.serving_chat import OpenAIServingChat
    
    # Create config
    config = VDiffConfig(model="test-model")
    
    # Create app without lifespan
    from fastapi import FastAPI
    app = FastAPI()
    
    # Set up global state
    server.engine = mock_engine
    server.config = config
    server.completion_serving = OpenAIServingCompletion(
        engine=mock_engine,
        model_name="test-model",
    )
    server.chat_serving = OpenAIServingChat(
        engine=mock_engine,
        model_name="test-model",
    )
    
    # Register routes manually
    from vdiff.entrypoints.openai.protocol import (
        CompletionRequest,
        ChatCompletionRequest,
        HealthResponse,
        ModelList,
    )
    from fastapi.responses import StreamingResponse
    from vdiff.metrics import setup_metrics, metrics_endpoint
    
    setup_metrics("test-model")
    
    @app.get("/health")
    async def health():
        return HealthResponse(status="healthy")
    
    @app.get("/v1/models")
    async def list_models():
        return server.completion_serving.show_available_models()
    
    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        result = await server.completion_serving.create_completion(request)
        if request.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        result = await server.chat_serving.create_chat_completion(request)
        if request.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result
    
    @app.get("/metrics")
    async def get_metrics():
        return metrics_endpoint()
    
    return TestClient(app)


class TestCompletionAPI:
    """Integration tests for /v1/completions endpoint."""
    
    def test_basic_completion(self, app_client):
        """Test basic completion request."""
        response = app_client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello, world!",
                "max_tokens": 50,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "This is a test response."
        assert data["choices"][0]["finish_reason"] == "stop"
    
    def test_completion_with_params(self, app_client):
        """Test completion with various parameters."""
        response = app_client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Test prompt",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "n": 1,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 3
        assert data["usage"]["completion_tokens"] == 5
    
    def test_completion_usage_info(self, app_client):
        """Test that usage info is returned."""
        response = app_client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Test",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


class TestModelsAPI:
    """Integration tests for /v1/models endpoint."""
    
    def test_list_models(self, app_client):
        """Test listing models."""
        response = app_client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["owned_by"] == "vdiff"


class TestHealthAPI:
    """Integration tests for /health endpoint."""
    
    def test_health_check(self, app_client):
        """Test health check endpoint."""
        response = app_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"


class TestMetricsAPI:
    """Integration tests for /metrics endpoint."""
    
    def test_metrics_endpoint(self, app_client):
        """Test Prometheus metrics endpoint."""
        response = app_client.get("/metrics")
        
        assert response.status_code == 200
        # Metrics should be in Prometheus format
        content = response.text
        assert "vdiff" in content or "Prometheus" in content or "#" in content
