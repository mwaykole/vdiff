"""Unit tests for OpenAI protocol models."""

import pytest
from dfastllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatMessage,
    ModelList,
    ModelCard,
    UsageInfo,
    ErrorResponse,
    HealthResponse,
    VersionResponse,
)


class TestCompletionRequest:
    """Test cases for CompletionRequest."""
    
    def test_basic_request(self):
        """Test basic completion request."""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
        )
        
        assert request.model == "test-model"
        assert request.prompt == "Hello, world!"
        assert request.max_tokens == 16
        assert request.temperature == 1.0
    
    def test_custom_parameters(self):
        """Test request with custom parameters."""
        request = CompletionRequest(
            model="test-model",
            prompt="Test",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=2,
            stop=["END"],
        )
        
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.n == 2
        assert request.stop == ["END"]
    
    def test_batch_prompts(self):
        """Test request with multiple prompts."""
        request = CompletionRequest(
            model="test-model",
            prompt=["Prompt 1", "Prompt 2"],
        )
        
        assert request.prompt == ["Prompt 1", "Prompt 2"]
    
    def test_dfastllm_extensions(self):
        """Test dfastllm-specific extensions."""
        request = CompletionRequest(
            model="test-model",
            prompt="Test",
            parallel_decoding=True,
            confidence_threshold=0.85,
        )
        
        assert request.parallel_decoding is True
        assert request.confidence_threshold == 0.85


class TestCompletionResponse:
    """Test cases for CompletionResponse."""
    
    def test_basic_response(self):
        """Test basic completion response."""
        choice = CompletionResponseChoice(
            index=0,
            text="Generated text",
            finish_reason="stop",
        )
        
        response = CompletionResponse(
            model="test-model",
            choices=[choice],
        )
        
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].text == "Generated text"
        assert response.object == "text_completion"
    
    def test_with_usage(self):
        """Test response with usage info."""
        response = CompletionResponse(
            model="test-model",
            choices=[CompletionResponseChoice(index=0, text="Test")],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15


class TestChatCompletionRequest:
    """Test cases for ChatCompletionRequest."""
    
    def test_basic_request(self):
        """Test basic chat completion request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="user", content="Hello!"),
            ],
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello!"
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!"),
                ChatMessage(role="assistant", content="Hi there!"),
                ChatMessage(role="user", content="How are you?"),
            ],
        )
        
        assert len(request.messages) == 4
    
    def test_custom_parameters(self):
        """Test request with custom parameters."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=200,
            temperature=0.8,
            stream=True,
        )
        
        assert request.max_tokens == 200
        assert request.temperature == 0.8
        assert request.stream is True


class TestChatCompletionResponse:
    """Test cases for ChatCompletionResponse."""
    
    def test_basic_response(self):
        """Test basic chat completion response."""
        message = ChatCompletionResponseMessage(
            role="assistant",
            content="Hello! How can I help?",
        )
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        
        response = ChatCompletionResponse(
            model="test-model",
            choices=[choice],
        )
        
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help?"
        assert response.object == "chat.completion"


class TestModelList:
    """Test cases for ModelList."""
    
    def test_empty_list(self):
        """Test empty model list."""
        model_list = ModelList()
        
        assert model_list.object == "list"
        assert model_list.data == []
    
    def test_with_models(self):
        """Test model list with models."""
        models = [
            ModelCard(id="model-1", owned_by="dfastllm"),
            ModelCard(id="model-2", owned_by="dfastllm"),
        ]
        model_list = ModelList(data=models)
        
        assert len(model_list.data) == 2
        assert model_list.data[0].id == "model-1"


class TestErrorResponse:
    """Test cases for ErrorResponse."""
    
    def test_basic_error(self):
        """Test basic error response."""
        error = ErrorResponse(
            message="Something went wrong",
            type="server_error",
            code=500,
        )
        
        assert error.message == "Something went wrong"
        assert error.type == "server_error"
        assert error.code == 500
        assert error.object == "error"


class TestHealthResponse:
    """Test cases for HealthResponse."""
    
    def test_default_healthy(self):
        """Test default healthy response."""
        health = HealthResponse()
        assert health.status == "healthy"


class TestVersionResponse:
    """Test cases for VersionResponse."""
    
    def test_version_info(self):
        """Test version response."""
        version = VersionResponse(
            version="0.1.0",
            vllm_compat_version="0.3.0",
            model_type="diffusion-llm",
        )
        
        assert version.version == "0.1.0"
        assert version.vllm_compat_version == "0.3.0"
        assert version.model_type == "diffusion-llm"
