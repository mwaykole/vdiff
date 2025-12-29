#!/usr/bin/env python
"""Example client using the OpenAI Python SDK.

This example demonstrates how to use dfastllm with the standard OpenAI client,
showing that the API is fully compatible with existing OpenAI tooling.

Requirements:
    pip install openai

Usage:
    # Start the dfastllm server first:
    # python -m dfastllm.entrypoints.openai.api_server --model GSAI-ML/LLaDA-8B-Instruct

    # Then run this script:
    python examples/client_openai.py
"""

from openai import OpenAI


def main():
    # Create client pointing to dfastllm server
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # dfastllm doesn't require API key by default
    )

    print("=" * 60)
    print("dfastllm OpenAI-Compatible API Example")
    print("=" * 60)

    # Example 1: List available models
    print("\n1. Listing available models:")
    models = client.models.list()
    for model in models.data:
        print(f"   - {model.id}")

    # Example 2: Chat completion
    print("\n2. Chat completion:")
    chat_response = client.chat.completions.create(
        model="GSAI-ML/LLaDA-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that explains things clearly.",
            },
            {
                "role": "user",
                "content": "What makes diffusion language models different from regular LLMs?",
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    print(f"   Model: {chat_response.model}")
    print(f"   Response: {chat_response.choices[0].message.content}")
    print(f"   Finish reason: {chat_response.choices[0].finish_reason}")
    print(f"   Usage: {chat_response.usage}")

    # Example 3: Text completion
    print("\n3. Text completion:")
    completion_response = client.completions.create(
        model="GSAI-ML/LLaDA-8B-Instruct",
        prompt="The key advantages of diffusion language models are:",
        max_tokens=100,
        temperature=0.7,
    )

    print(f"   Response: {completion_response.choices[0].text}")
    print(f"   Finish reason: {completion_response.choices[0].finish_reason}")

    # Example 4: Multi-turn conversation
    print("\n4. Multi-turn conversation:")
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I create a Python function?"},
    ]

    # First turn
    response1 = client.chat.completions.create(
        model="GSAI-ML/LLaDA-8B-Instruct",
        messages=messages,
        max_tokens=150,
    )
    assistant_reply = response1.choices[0].message.content
    print(f"   User: How do I create a Python function?")
    print(f"   Assistant: {assistant_reply[:100]}...")

    # Add assistant response and continue
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Can you show me an example?"})

    response2 = client.chat.completions.create(
        model="GSAI-ML/LLaDA-8B-Instruct",
        messages=messages,
        max_tokens=150,
    )
    print(f"   User: Can you show me an example?")
    print(f"   Assistant: {response2.choices[0].message.content[:100]}...")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
