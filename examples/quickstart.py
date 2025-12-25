#!/usr/bin/env python3
"""
🚀 dfastllm Quickstart - Get running in seconds!

Usage:
    python examples/quickstart.py

This script helps you:
1. Check if server is running
2. Test basic functionality
3. Show example usage
"""

import sys
import time

try:
    import requests
except ImportError:
    print("❌ Please install requests: pip install requests")
    sys.exit(1)


def colored(text, color):
    """Simple colored output."""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def check_server(base_url="http://localhost:8000"):
    """Check if server is running."""
    print(colored("\n🔍 Checking server...", 'blue'))
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(colored("✅ Server is running!", 'green'))
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Device: {data.get('device', 'unknown')}")
            return True
    except requests.exceptions.ConnectionError:
        print(colored("❌ Server not running!", 'red'))
        print("\n💡 To start the server:")
        print(colored("   python -m dfastllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0", 'yellow'))
        return False
    except Exception as e:
        print(colored(f"❌ Error: {e}", 'red'))
        return False


def test_completion(base_url="http://localhost:8000"):
    """Test text completion."""
    print(colored("\n📝 Testing text completion...", 'blue'))
    
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "default",
            "prompt": "Hello, I am",
            "max_tokens": 20,
            "temperature": 0.7
        },
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        text = data['choices'][0]['text']
        print(colored("✅ Completion works!", 'green'))
        print(f"   Prompt: \"Hello, I am\"")
        print(f"   Response: \"{text}\"")
        return True
    else:
        print(colored(f"❌ Error: {resp.text}", 'red'))
        return False


def test_chat(base_url="http://localhost:8000"):
    """Test chat completion."""
    print(colored("\n💬 Testing chat completion...", 'blue'))
    
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "Say hello!"}
            ],
            "max_tokens": 30
        },
        timeout=30
    )
    
    if resp.status_code == 200:
        data = resp.json()
        content = data['choices'][0]['message']['content']
        print(colored("✅ Chat works!", 'green'))
        print(f"   User: \"Say hello!\"")
        print(f"   Assistant: \"{content}\"")
        return True
    else:
        print(colored(f"❌ Error: {resp.text}", 'red'))
        return False


def show_examples():
    """Show code examples."""
    print(colored("\n📚 Code Examples", 'bold'))
    print("=" * 50)
    
    print(colored("\n1. Using OpenAI Python SDK:", 'blue'))
    print("""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
""")

    print(colored("\n2. Using curl:", 'blue'))
    print("""
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
""")

    print(colored("\n3. Streaming:", 'blue'))
    print("""
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=200,
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
""")


def main():
    print(colored("=" * 50, 'bold'))
    print(colored("  🚀 dfastllm Quickstart", 'bold'))
    print(colored("=" * 50, 'bold'))
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"\n📡 Server URL: {base_url}")
    
    # Check server
    if not check_server(base_url):
        show_examples()
        sys.exit(1)
    
    # Run tests
    success = True
    success &= test_completion(base_url)
    success &= test_chat(base_url)
    
    # Show examples
    show_examples()
    
    # Summary
    print(colored("\n" + "=" * 50, 'bold'))
    if success:
        print(colored("🎉 All tests passed! You're ready to go!", 'green'))
    else:
        print(colored("⚠️  Some tests failed. Check the errors above.", 'yellow'))
    print(colored("=" * 50, 'bold'))
    
    print("\n📖 Next steps:")
    print("   • Read the docs: docs/QUICK_START.md")
    print("   • Try examples: python examples/client_openai.py")
    print("   • API docs: http://localhost:8000/docs")
    print("")


if __name__ == "__main__":
    main()

