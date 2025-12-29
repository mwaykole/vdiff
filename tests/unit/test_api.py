"""Unit tests for API endpoints."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

from dfastllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ChatMessage,
)
from dfastllm.entrypoints.openai.api_server import create_error_response

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed"
)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    @pytest.fixture
    def mock_engine(self):
        engine = Mock()
        engine.is_ready = True
        engine.get_health_status.return_value = {
            "status": "healthy",
            "state": "ready",
            "model_loaded": True,
            "device": "cuda",
        }
        return engine
    
    def test_health_check_healthy(self, mock_engine):
        """Test health check returns healthy status."""
        assert mock_engine.is_ready is True
        health_status = mock_engine.get_health_status()
        assert health_status["status"] == "healthy"
        assert health_status["model_loaded"] is True


class TestCompletionEndpoint:
    """Tests for completion endpoint."""
    
    def test_completion_request_structure(self):
        """Test the expected request structure."""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            max_tokens=50,
            temperature=0.7,
        )
        
        assert request.model == "test-model"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 50
        assert request.temperature == 0.7
    
    def test_completion_request_defaults(self):
        """Test default values in request."""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
        )
        
        assert request.stream is False
        assert request.echo is False


class TestChatCompletionEndpoint:
    """Tests for chat completion endpoint."""
    
    def test_chat_request_structure(self):
        """Test the expected chat request structure."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello!")
            ],
            max_tokens=100,
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
    
    def test_chat_request_with_system_message(self):
        """Test chat request with system message."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!")
            ],
        )
        
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"


class TestErrorResponses:
    """Tests for error response handling."""
    
    def test_error_response_structure(self):
        """Test error response structure."""
        response = create_error_response(
            status_code=400,
            message="Invalid request",
            error_type="validation_error",
            request_id="test-123",
        )
        
        assert response.status_code == 400


class TestModelNotFoundError:
    """Tests for model not found error handling."""
    
    def test_model_not_found_exception(self):
        """Test model not found error response creation."""
        response = create_error_response(
            status_code=404,
            message="Model 'xyz' not found",
            error_type="model_not_found",
        )
        assert response.status_code == 404
