#!/usr/bin/env python3
"""
dfastllm CLI - Simple command-line interface.

Usage:
    dfastllm serve --model <model>
    dfastllm check [--url <url>]
    dfastllm test [--url <url>]
"""

import argparse
import sys
import os


def cmd_serve(args):
    """Start the dfastllm server."""
    # Build command
    cmd_args = [
        sys.executable, "-m", "dfastllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    if args.trust_remote_code:
        cmd_args.append("--trust-remote-code")
    
    if args.enable_apd:
        cmd_args.append("--enable-apd")
    
    print(f"🚀 Starting dfastllm server...")
    print(f"   Model: {args.model}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL: http://{args.host}:{args.port}")
    print("")
    
    os.execvp(sys.executable, cmd_args)


def cmd_check(args):
    """Check if server is running."""
    try:
        import requests
    except ImportError:
        print("❌ Please install requests: pip install requests")
        sys.exit(1)
    
    url = args.url.rstrip('/')
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Server is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model: {data.get('model_loaded')}")
            print(f"   Device: {data.get('device')}")
            print(f"   GPU Memory: {data.get('gpu_memory', {}).get('used_mb', 'N/A')} MB")
            sys.exit(0)
        else:
            print(f"❌ Server unhealthy: {resp.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {url}")
        print("\n💡 Start the server with:")
        print("   dfastllm serve --model <your-model>")
        sys.exit(1)


def cmd_test(args):
    """Run quick test against server."""
    try:
        import requests
    except ImportError:
        print("❌ Please install requests: pip install requests")
        sys.exit(1)
    
    url = args.url.rstrip('/')
    
    print(f"🧪 Testing server at {url}...")
    
    # Test completion
    try:
        resp = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "default",
                "prompt": "Hello",
                "max_tokens": 10
            },
            timeout=30
        )
        if resp.status_code == 200:
            print("✅ Completion API: Working")
        else:
            print(f"❌ Completion API: {resp.status_code}")
    except Exception as e:
        print(f"❌ Completion API: {e}")
    
    # Test chat
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            },
            timeout=30
        )
        if resp.status_code == 200:
            print("✅ Chat API: Working")
        else:
            print(f"❌ Chat API: {resp.status_code}")
    except Exception as e:
        print(f"❌ Chat API: {e}")
    
    print("\n🎉 Test complete!")


def main():
    parser = argparse.ArgumentParser(
        description="dfastllm - Fast inference for Diffusion LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dfastllm serve --model microsoft/phi-2
  dfastllm serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --enable-apd
  dfastllm check
  dfastllm test

For more info: https://github.com/mwaykole/dfastllm
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the server")
    serve_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    serve_parser.add_argument("--host", "-H", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    serve_parser.add_argument("--enable-apd", action="store_true", help="Enable APD")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check server health")
    check_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test server APIs")
    test_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

