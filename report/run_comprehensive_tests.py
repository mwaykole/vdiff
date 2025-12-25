#!/usr/bin/env python3
"""
Comprehensive dfastllm Benchmark Suite
Tests: Accuracy, Throughput, Latency, API Compatibility, Metrics
"""

import json
import time
import statistics
import sys
import os
from datetime import datetime
from typing import Dict, Any

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

BASE_URL = os.environ.get("DFASTLLM_URL", "http://localhost:8000")
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_result(name: str, status: bool, details: str = ""):
    icon = "✅ PASS" if status else "❌ FAIL"
    print(f"  {name}: {icon} {details}")

# Accuracy Tests
ACCURACY_TESTS = [
    {"name": "Language Completion", "prompt": "The quick brown fox jumps over the", "expected": ["lazy", "dog"], "tokens": 20},
    {"name": "Sentence Generation", "prompt": "Hello, my name is", "expected": ["i", "am", "my"], "tokens": 20},
    {"name": "Story Continuation", "prompt": "Once upon a time, there was a", "expected": ["king", "queen", "princess", "prince", "dragon", "castle", "village", "man", "woman", "boy", "girl"], "tokens": 30},
    {"name": "Code-like Pattern", "prompt": "def hello_world():", "expected": ["print", "return", "hello", "world"], "tokens": 30},
    {"name": "List Generation", "prompt": "Colors of the rainbow:", "expected": ["red", "orange", "yellow", "green", "blue", "purple", "violet", "indigo"], "tokens": 40},
]

def test_accuracy() -> Dict[str, Any]:
    results = {"total": len(ACCURACY_TESTS), "passed": 0, "details": []}
    
    for test in ACCURACY_TESTS:
        try:
            resp = httpx.post(f"{BASE_URL}/v1/completions", json={
                "model": "test", "prompt": test["prompt"], "max_tokens": test["tokens"], "temperature": 0.7
            }, timeout=60)
            data = resp.json()
            text = data.get("choices", [{}])[0].get("text", "").lower()
            matches = sum(1 for kw in test["expected"] if kw.lower() in text)
            passed = matches >= 1
            if passed: results["passed"] += 1
            results["details"].append({"name": test["name"], "passed": passed, "response": text[:100], "matches": matches})
        except Exception as e:
            results["details"].append({"name": test["name"], "passed": False, "error": str(e)})
    
    results["accuracy_pct"] = (results["passed"] / results["total"]) * 100
    return results

def test_throughput(num_requests: int, max_tokens: int, concurrency: int = 1) -> Dict[str, Any]:
    latencies = []
    tokens_list = []
    
    # Simple sequential test
    start_time = time.perf_counter()
    for _ in range(num_requests):
        try:
            req_start = time.perf_counter()
            resp = httpx.post(f"{BASE_URL}/v1/completions", json={
                "model": "test", "prompt": "Once upon a time", "max_tokens": max_tokens
            }, timeout=120)
            latencies.append((time.perf_counter() - req_start) * 1000)
            tokens_list.append(resp.json().get("usage", {}).get("completion_tokens", 0))
        except:
            pass
    total_time = time.perf_counter() - start_time
    
    if not latencies:
        return {"error": "All requests failed"}
    
    latencies.sort()
    n = len(latencies)
    total_tokens = sum(tokens_list)
    
    return {
        "config": {"num_requests": num_requests, "max_tokens": max_tokens, "concurrency": concurrency},
        "results": {
            "successful": n, "failed": num_requests - n,
            "success_rate": round((n / num_requests) * 100, 1),
            "total_time_s": round(total_time, 2),
            "total_tokens": total_tokens,
            "throughput_tok_s": round(total_tokens / total_time, 2)
        },
        "latency": {
            "min_ms": round(min(latencies), 2), "max_ms": round(max(latencies), 2),
            "avg_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)] if n >= 20 else latencies[-1], 2)
        }
    }

def test_api() -> Dict[str, Any]:
    results = {"tests": [], "passed": 0, "total": 0}
    tests = [
        ("GET /health", lambda: httpx.get(f"{BASE_URL}/health", timeout=10)),
        ("GET /v1/models", lambda: httpx.get(f"{BASE_URL}/v1/models", timeout=10)),
        ("GET /metrics", lambda: httpx.get(f"{BASE_URL}/metrics", timeout=10)),
        ("POST /v1/completions", lambda: httpx.post(f"{BASE_URL}/v1/completions", json={"model": "t", "prompt": "Hi", "max_tokens": 5}, timeout=30)),
        ("POST /v1/chat/completions", lambda: httpx.post(f"{BASE_URL}/v1/chat/completions", json={"model": "t", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}, timeout=30)),
    ]
    for name, fn in tests:
        results["total"] += 1
        try:
            r = fn()
            passed = r.status_code == 200
            if passed: results["passed"] += 1
            results["tests"].append({"name": name, "passed": passed, "status": r.status_code})
        except Exception as e:
            results["tests"].append({"name": name, "passed": False, "error": str(e)})
    return results

def get_system_info() -> Dict[str, Any]:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=10).json()
        return {"status": r.get("status"), "device": r.get("device"), "model_loaded": r.get("model_loaded"),
                "gpu_memory_mb": r.get("gpu_memory", {}).get("used_mb"), "uptime_s": r.get("uptime_seconds")}
    except Exception as e:
        return {"error": str(e)}

def main():
    global BASE_URL
    if len(sys.argv) > 1: BASE_URL = sys.argv[1]
    
    print_header("dfastllm Comprehensive Benchmark Suite")
    print(f"Endpoint: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {"timestamp": datetime.now().isoformat(), "endpoint": BASE_URL}
    
    # System
    print_header("System Information")
    results["system"] = get_system_info()
    for k, v in results["system"].items(): print(f"  {k}: {v}")
    
    # API
    print_header("API Compatibility Tests")
    results["api"] = test_api()
    for t in results["api"]["tests"]: print_result(t["name"], t["passed"])
    print(f"\n  Total: {results['api']['passed']}/{results['api']['total']}")
    
    # Accuracy
    print_header("Accuracy Tests")
    results["accuracy"] = test_accuracy()
    for d in results["accuracy"]["details"]: print_result(d["name"], d["passed"], f"(matches: {d.get('matches', 0)})")
    print(f"\n  Accuracy: {results['accuracy']['accuracy_pct']:.1f}%")
    
    # Throughput
    print_header("Throughput Tests")
    results["throughput"] = []
    configs = [(20, 32), (20, 64), (20, 128), (30, 64)]
    for num_req, tokens in configs:
        print(f"\n  Testing: {num_req} requests, {tokens} tokens...")
        r = test_throughput(num_req, tokens)
        results["throughput"].append(r)
        if "error" not in r:
            print(f"    Throughput: {r['results']['throughput_tok_s']} tok/s")
            print(f"    Avg Latency: {r['latency']['avg_ms']} ms")
            print(f"    Success Rate: {r['results']['success_rate']}%")
    
    # Summary
    print_header("Summary")
    api_pass = results["api"]["passed"] == results["api"]["total"]
    acc_pass = results["accuracy"]["accuracy_pct"] >= 60
    thr_pass = all(t.get("results", {}).get("success_rate", 0) >= 90 for t in results["throughput"])
    
    print_result("API Compatibility (5/5)", api_pass)
    print_result("Accuracy (≥60%)", acc_pass, f"({results['accuracy']['accuracy_pct']:.1f}%)")
    print_result("Throughput (≥90% success)", thr_pass)
    overall = api_pass and acc_pass and thr_pass
    print(f"\n  OVERALL: {'✅ PASS' if overall else '❌ FAIL'}")
    
    # Save
    with open(os.path.join(REPORT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: report/benchmark_results.json")
    
    return results

if __name__ == "__main__":
    main()
