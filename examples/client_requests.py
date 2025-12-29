#!/usr/bin/env python
"""Example client using the requests library.

This example demonstrates how to use dfastllm with plain HTTP requests,
useful for understanding the raw API or for languages without an OpenAI SDK.

Requirements:
    pip install requests

Usage:
    # Start the dfastllm server first:
    # python -m dfastllm.entrypoints.openai.api_server --model GSAI-ML/LLaDA-8B-Instruct

    # Then run this script:
    python examples/client_requests.py
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the server is healthy."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()


def get_version():
    """Get server version information."""
    response = requests.get(f"{BASE_URL}/version")
    return response.json()


def list_models():
    """List available models."""
    response = requests.get(f"{BASE_URL}/v1/models")
    return response.json()


def create_completion(prompt: str, max_tokens: int = 100, temperature: float = 0.7):
    """Create a text completion."""
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": "GSAI-ML/LLaDA-8B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def create_chat_completion(
    messages: list, max_tokens: int = 100, temperature: float = 0.7
):
    """Create a chat completion."""
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "GSAI-ML/LLaDA-8B-Instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def get_metrics():
    """Get Prometheus metrics."""
    response = requests.get(f"{BASE_URL}/metrics")
    return response.text


def main():
    print("=" * 60)
    print("dfastllm Requests API Example")
    print("=" * 60)

    # Check health
    print("\n1. Health check:")
    try:
        health = check_health()
        print(f"   Status: {health['status']}")
    except requests.ConnectionError:
        print("   ERROR: Cannot connect to server. Is dfastllm running?")
        return

    # Get version
    print("\n2. Version info:")
    version = get_version()
    print(f"   Version: {version['version']}")
    print(f"   vLLM compat: {version['vllm_compat_version']}")
    print(f"   Model type: {version['model_type']}")

    # List models
    print("\n3. Available models:")
    models = list_models()
    for model in models["data"]:
        print(f"   - {model['id']} (owned by: {model['owned_by']})")

    # Text completion
    print("\n4. Text completion:")
    completion = create_completion(
        prompt="The future of artificial intelligence is",
        max_tokens=50,
    )
    print(f"   Prompt: The future of artificial intelligence is")
    print(f"   Response: {completion['choices'][0]['text']}")
    print(f"   Usage: {completion['usage']}")

    # Chat completion
    print("\n5. Chat completion:")
    chat = create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What can you help me with?"},
        ],
        max_tokens=100,
    )
    print(f"   User: Hello! What can you help me with?")
    print(f"   Assistant: {chat['choices'][0]['message']['content']}")
    print(f"   Usage: {chat['usage']}")

    # Metrics sample
    print("\n6. Metrics (sample):")
    metrics = get_metrics()
    # Show first few lines
    lines = metrics.split("\n")[:5]
    for line in lines:
        print(f"   {line}")
    print("   ...")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
