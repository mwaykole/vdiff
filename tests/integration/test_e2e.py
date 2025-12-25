"""End-to-end integration tests for dfastllm.

These tests require a running dfastllm server.
Set DFASTLLM_TEST_URL environment variable to the server URL.
"""

import pytest
import os
import time
import requests
from typing import Generator

# Skip all tests if no server URL is provided
DFASTLLM_URL = os.environ.get("DFASTLLM_TEST_URL", "")
pytestmark = pytest.mark.skipif(
    not DFASTLLM_URL,
    reason="DFASTLLM_TEST_URL environment variable not set"
)


@pytest.fixture
def dfastllm_url() -> str:
    """Get the dfastllm server URL."""
    return DFASTLLM_URL


class TestHealthEndpoint:
    """Test health check endpoints."""
    
    def test_health_check(self, dfastllm_url):
        """Test basic health check."""
        response = requests.get(f"{dfastllm_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ready"]
    
    def test_ready_check(self, dfastllm_url):
        """Test ready check endpoint."""
        response = requests.get(f"{dfastllm_url}/ready")
        assert response.status_code == 200


class TestCompletionEndpoint:
    """Test completion endpoint."""
    
    def test_basic_completion(self, dfastllm_url):
        """Test basic text completion."""
        response = requests.post(
            f"{dfastllm_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Hello, world!",
                "max_tokens": 10,
                "temperature": 0.7,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]
    
    def test_completion_with_stop_sequence(self, dfastllm_url):
        """Test completion with stop sequence."""
        response = requests.post(
            f"{dfastllm_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Count to ten: 1, 2, 3",
                "max_tokens": 50,
                "stop": ["5"],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that it stopped before or at "5"
        assert "5" not in data["choices"][0]["text"] or \
               data["choices"][0]["finish_reason"] == "stop"
    
    def test_streaming_completion(self, dfastllm_url):
        """Test streaming completion."""
        response = requests.post(
            f"{dfastllm_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Tell me a story",
                "max_tokens": 50,
                "stream": True,
            },
            stream=True,
        )
        
        assert response.status_code == 200
        
        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line[6:])
        
        assert len(chunks) > 0


class TestChatCompletionEndpoint:
    """Test chat completion endpoint."""
    
    def test_basic_chat(self, dfastllm_url):
        """Test basic chat completion."""
        response = requests.post(
            f"{dfastllm_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 20,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
    
    def test_chat_with_system_message(self, dfastllm_url):
        """Test chat with system message."""
        response = requests.post(
            f"{dfastllm_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "max_tokens": 20,
            }
        )
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_model(self, dfastllm_url):
        """Test request with invalid model."""
        response = requests.post(
            f"{dfastllm_url}/v1/completions",
            json={
                "model": "nonexistent-model-xyz",
                "prompt": "Hello",
                "max_tokens": 10,
            }
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_empty_prompt(self, dfastllm_url):
        """Test request with empty prompt."""
        response = requests.post(
            f"{dfastllm_url}/v1/completions",
            json={
                "model": "default",
                "prompt": "",
                "max_tokens": 10,
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_empty_messages(self, dfastllm_url):
        """Test chat with empty messages."""
        response = requests.post(
            f"{dfastllm_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [],
                "max_tokens": 10,
            }
        )
        
        assert response.status_code == 400


class TestMetrics:
    """Test Prometheus metrics."""
    
    def test_metrics_endpoint(self, dfastllm_url):
        """Test metrics endpoint exists."""
        response = requests.get(f"{dfastllm_url}/metrics")
        
        assert response.status_code == 200
        assert "dfastllm_" in response.text or "http_" in response.text


class TestModels:
    """Test models endpoint."""
    
    def test_list_models(self, dfastllm_url):
        """Test listing available models."""
        response = requests.get(f"{dfastllm_url}/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert isinstance(data["data"], list)


class TestBenchmark:
    """Performance benchmark tests."""
    
    @pytest.mark.slow
    def test_throughput(self, dfastllm_url):
        """Test throughput with various token lengths."""
        results = []
        
        for max_tokens in [16, 32, 64]:
            start = time.time()
            
            response = requests.post(
                f"{dfastllm_url}/v1/completions",
                json={
                    "model": "default",
                    "prompt": "Explain the concept of",
                    "max_tokens": max_tokens,
                }
            )
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
                throughput = tokens / elapsed if elapsed > 0 else 0
                
                results.append({
                    "max_tokens": max_tokens,
                    "latency": elapsed,
                    "throughput": throughput,
                })
        
        # Just verify we got results, don't assert specific values
        assert len(results) > 0
        
        # Print for debugging
        for r in results:
            print(f"Tokens: {r['max_tokens']}, Latency: {r['latency']:.2f}s, "
                  f"Throughput: {r['throughput']:.1f} tok/s")

