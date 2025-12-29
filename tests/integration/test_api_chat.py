"""Integration tests for chat completion API."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from dfastllm.config import DFastLLMConfig
from dfastllm.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics


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
            prompt_token_ids=[1, 2, 3, 4, 5],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="Hello! How can I assist you today?",
                    token_ids=[6, 7, 8, 9, 10, 11, 12],
                    finish_reason="stop",
                    parallel_tokens_decoded=4,
                )
            ],
            finished=True,
            metrics=RequestMetrics(
                prompt_tokens=5,
                generated_tokens=7,
                parallel_tokens_decoded=4,
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
                    text="Hello! How can I assist you today?",
                    finish_reason="stop",
                )
            ],
            finished=True,
        )
    
    engine.generate_stream = mock_stream
    
    return engine


@pytest.fixture
def chat_client(mock_engine):
    """Create a test client for chat completions."""
    import dfastllm.entrypoints.openai.api_server as server
    from dfastllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from dfastllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    
    config = DFastLLMConfig(model="chat-model")
    
    app = FastAPI()
    
    server.engine = mock_engine
    server.config = config
    server.completion_serving = OpenAIServingCompletion(
        engine=mock_engine,
        model_name="chat-model",
    )
    server.chat_serving = OpenAIServingChat(
        engine=mock_engine,
        model_name="chat-model",
    )
    
    from dfastllm.entrypoints.openai.protocol import ChatCompletionRequest
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        result = await server.chat_serving.create_chat_completion(request)
        if request.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result
    
    return TestClient(app)


class TestChatCompletionAPI:
    """Integration tests for /v1/chat/completions endpoint."""
    
    def test_basic_chat(self, chat_client):
        """Test basic chat completion request."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "chat.completion"
        assert data["model"] == "chat-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Hello" in data["choices"][0]["message"]["content"]
        assert data["choices"][0]["finish_reason"] == "stop"
    
    def test_chat_with_system_message(self, chat_client):
        """Test chat with system message."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["choices"][0]["message"]["role"] == "assistant"
    
    def test_multi_turn_conversation(self, chat_client):
        """Test multi-turn conversation."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["choices"]) == 1
    
    def test_chat_with_params(self, chat_client):
        """Test chat with various parameters."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "user", "content": "Test"}
                ],
                "max_tokens": 200,
                "temperature": 0.8,
                "top_p": 0.95,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
    
    def test_chat_usage_info(self, chat_client):
        """Test that usage info is returned."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 7
        assert data["usage"]["total_tokens"] == 12
    
    def test_chat_parallel_tokens(self, chat_client):
        """Test that parallel tokens are tracked."""
        response = chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # vdiff extension: parallel tokens decoded
        if "parallel_tokens_decoded" in data:
            assert data["parallel_tokens_decoded"] == 4
