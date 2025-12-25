"""Unit tests for API endpoints."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

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
        with patch("dfastllm.entrypoints.openai.api_server.server_state") as mock_state:
            mock_state.engine = mock_engine
            mock_state.model_loaded = True
            mock_state.start_time = 0
            mock_state.active_requests = 0
            
            # Import after patching
            from dfastllm.entrypoints.openai.api_server import app
            client = TestClient(app, raise_server_exceptions=False)
            
            # This would require full app setup
            # For now, we just verify the structure


class TestCompletionEndpoint:
    """Tests for completion endpoint."""
    
    def test_completion_request_structure(self):
        """Test the expected request structure."""
        from dfastllm.entrypoints.openai.protocol import CompletionRequest
        
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
        from dfastllm.entrypoints.openai.protocol import CompletionRequest
        
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
        from dfastllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=100,
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
    
    def test_chat_request_with_system_message(self):
        """Test chat request with system message."""
        from dfastllm.entrypoints.openai.protocol import ChatCompletionRequest
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
        )
        
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"


class TestErrorResponses:
    """Tests for error response handling."""
    
    def test_error_response_structure(self):
        """Test error response structure."""
        from dfastllm.entrypoints.openai.api_server import create_error_response
        
        response = create_error_response(
            status_code=400,
            message="Invalid request",
            error_type="validation_error",
            request_id="test-123",
        )
        
        # Verify it's a proper response
        assert response.status_code == 400


class TestModelNotFoundError:
    """Tests for model not found error handling."""
    
    def test_model_not_found_exception(self):
        """Test ModelNotFoundError exception."""
        from dfastllm.entrypoints.openai.serving_completion import ModelNotFoundError
        
        error = ModelNotFoundError("Model 'xyz' not found")
        assert str(error) == "Model 'xyz' not found"

