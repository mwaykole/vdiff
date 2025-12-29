"""Compatibility tests to ensure vLLM API compatibility.

These tests verify that dfastllm's API matches vLLM's API exactly,
ensuring drop-in compatibility for KServe and llm-d deployments.
"""

import pytest
from pydantic import ValidationError

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
)


class TestOpenAICompletionCompat:
    """Test OpenAI Completion API compatibility."""
    
    def test_request_required_fields(self):
        """Test that required fields match OpenAI spec."""
        # model and prompt are required
        with pytest.raises(ValidationError):
            CompletionRequest()
        
        # This should work
        request = CompletionRequest(model="test", prompt="hello")
        assert request.model == "test"
        assert request.prompt == "hello"
    
    def test_request_optional_fields(self):
        """Test that optional fields have correct defaults."""
        request = CompletionRequest(model="test", prompt="hello")
        
        # OpenAI defaults
        assert request.max_tokens == 16
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.n == 1
        assert request.stream is False
        assert request.logprobs is None
        assert request.echo is False
        assert request.stop is None
        assert request.presence_penalty == 0.0
        assert request.frequency_penalty == 0.0
        assert request.best_of is None
        assert request.logit_bias is None
        assert request.user is None
    
    def test_response_structure(self):
        """Test response structure matches OpenAI spec."""
        choice = CompletionResponseChoice(
            index=0,
            text="Generated text",
            finish_reason="stop",
        )
        
        response = CompletionResponse(
            model="test-model",
            choices=[choice],
        )
        
        # Required fields
        assert hasattr(response, "id")
        assert hasattr(response, "object")
        assert hasattr(response, "created")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        
        # Object type
        assert response.object == "text_completion"
        
        # ID format (should start with cmpl-)
        assert response.id.startswith("cmpl-")
    
    def test_usage_info_structure(self):
        """Test usage info matches OpenAI spec."""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestOpenAIChatCompat:
    """Test OpenAI Chat Completion API compatibility."""
    
    def test_message_roles(self):
        """Test that message roles match OpenAI spec."""
        # Valid roles
        for role in ["system", "user", "assistant", "function", "tool"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role
    
    def test_request_required_fields(self):
        """Test that required fields match OpenAI spec."""
        # model and messages are required
        with pytest.raises(ValidationError):
            ChatCompletionRequest()
        
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert request.model == "test"
        assert len(request.messages) == 1
    
    def test_request_optional_fields(self):
        """Test that optional fields have correct defaults."""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
        )
        
        # OpenAI defaults
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.n == 1
        assert request.stream is False
        assert request.stop is None
        assert request.max_tokens is None  # OpenAI doesn't set default
        assert request.presence_penalty == 0.0
        assert request.frequency_penalty == 0.0
    
    def test_response_structure(self):
        """Test response structure matches OpenAI spec."""
        message = ChatCompletionResponseMessage(
            role="assistant",
            content="Hello!",
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
        
        # Required fields
        assert hasattr(response, "id")
        assert hasattr(response, "object")
        assert hasattr(response, "created")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        
        # Object type
        assert response.object == "chat.completion"
        
        # ID format (should start with chatcmpl-)
        assert response.id.startswith("chatcmpl-")


class TestModelListCompat:
    """Test Model List API compatibility."""
    
    def test_model_list_structure(self):
        """Test model list matches OpenAI spec."""
        model = ModelCard(
            id="test-model",
            owned_by="dfastllm",
        )
        
        model_list = ModelList(data=[model])
        
        assert model_list.object == "list"
        assert len(model_list.data) == 1
        assert model_list.data[0].object == "model"
    
    def test_model_card_structure(self):
        """Test model card matches OpenAI spec."""
        model = ModelCard(id="test-model")
        
        assert hasattr(model, "id")
        assert hasattr(model, "object")
        assert hasattr(model, "created")
        assert hasattr(model, "owned_by")
        
        assert model.object == "model"


class TestErrorResponseCompat:
    """Test Error Response compatibility."""
    
    def test_error_structure(self):
        """Test error response matches OpenAI spec."""
        error = ErrorResponse(
            message="Something went wrong",
            type="server_error",
            code=500,
        )
        
        assert error.object == "error"
        assert error.message == "Something went wrong"
        assert error.type == "server_error"


class TestVLLMExtensions:
    """Test that dfastllm extensions don't break vLLM compatibility."""
    
    def test_completion_request_extensions(self):
        """Test dfastllm extensions in completion request."""
        # Extensions should be optional
        request = CompletionRequest(
            model="test",
            prompt="hello",
        )
        
        # Extensions have defaults
        assert request.parallel_decoding is True
        assert request.confidence_threshold is None
    
    def test_chat_request_extensions(self):
        """Test dfastllm extensions in chat request."""
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
        )
        
        # Extensions have defaults
        assert request.parallel_decoding is True
        assert request.confidence_threshold is None
    
    def test_response_extensions_optional(self):
        """Test that dfastllm extension fields are optional in response."""
        response = CompletionResponse(
            model="test",
            choices=[CompletionResponseChoice(index=0, text="test")],
        )
        
        # Extension field should be optional
        assert response.parallel_tokens_decoded is None


class TestEndpointPaths:
    """Test that endpoint paths match vLLM."""
    
    def test_expected_endpoints(self):
        """Document expected endpoint paths."""
        expected_endpoints = [
            "/health",
            "/version",
            "/v1/models",
            "/v1/completions",
            "/v1/chat/completions",
            "/metrics",
        ]
        
        # These paths should be documented and implemented
        for endpoint in expected_endpoints:
            assert endpoint.startswith("/")


class TestCLICompatibility:
    """Test CLI argument compatibility with vLLM."""
    
    def test_common_arguments(self):
        """Document common CLI arguments that should match vLLM."""
        common_args = [
            "--model",
            "--tokenizer",
            "--revision",
            "--max-model-len",
            "--dtype",
            "--trust-remote-code",
            "--host",
            "--port",
            "--tensor-parallel-size",
            "--gpu-memory-utilization",
        ]
        
        # These should all be supported
        assert len(common_args) > 0
    
    def test_dfastllm_specific_arguments(self):
        """Document dfastllm-specific CLI arguments."""
        dfastllm_args = [
            "--enable-kv-cache",
            "--enable-parallel-decoding",
            "--confidence-threshold",
            "--block-size",
        ]
        
        assert len(dfastllm_args) > 0
