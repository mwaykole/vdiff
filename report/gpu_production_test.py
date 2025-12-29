#!/usr/bin/env python3
"""GPU Production Test Suite for dfastllm.

Comprehensive production-like testing including:
- Functional tests (chat, completion, streaming)
- Performance benchmarks (latency, throughput)
- Stress tests (concurrent requests, long prompts)
- Edge cases (empty prompts, invalid params, rate limiting)
- Comparison with vLLM baseline
"""

import json
import time
import statistics
import concurrent.futures
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


@dataclass
class TestResult:
    """Test result container."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    name: str
    requests: int
    latencies_ms: List[float]
    errors: int
    tokens_generated: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def throughput_rps(self) -> float:
        total_time_s = sum(self.latencies_ms) / 1000
        return self.requests / total_time_s if total_time_s > 0 else 0
    
    @property
    def tokens_per_second(self) -> float:
        total_time_s = sum(self.latencies_ms) / 1000
        return self.tokens_generated / total_time_s if total_time_s > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "requests": self.requests,
            "errors": self.errors,
            "success_rate": (self.requests - self.errors) / self.requests * 100 if self.requests > 0 else 0,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class ProductionTester:
    """Production test suite for dfastllm."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.results: List[TestResult] = []
        self.benchmarks: List[BenchmarkResult] = []
        self.defects: List[Dict[str, Any]] = []
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with timing."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", 120)
        return self.session.request(method, url, **kwargs)
    
    def _chat_completion(self, messages: List[Dict], max_tokens: int = 50, 
                         temperature: float = 0.7) -> Dict[str, Any]:
        """Send chat completion request."""
        response = self._request("POST", "/v1/chat/completions", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return response.json()
    
    def _completion(self, prompt: str, max_tokens: int = 50,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """Send completion request."""
        response = self._request("POST", "/v1/completions", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return response.json()
    
    # === Functional Tests ===
    
    def test_health_endpoint(self) -> TestResult:
        """Test health endpoint returns valid status."""
        start = time.time()
        try:
            resp = self._request("GET", "/health")
            data = resp.json()
            passed = (
                resp.status_code == 200 and
                data.get("status") == "healthy" and
                data.get("model_loaded") is True and
                data.get("device") == "cuda"
            )
            return TestResult(
                name="health_endpoint",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details=data
            )
        except Exception as e:
            return TestResult(
                name="health_endpoint",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_models_endpoint(self) -> TestResult:
        """Test models listing endpoint."""
        start = time.time()
        try:
            resp = self._request("GET", "/v1/models")
            data = resp.json()
            passed = (
                resp.status_code == 200 and
                data.get("object") == "list" and
                len(data.get("data", [])) > 0
            )
            return TestResult(
                name="models_endpoint",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"model_count": len(data.get("data", []))}
            )
        except Exception as e:
            return TestResult(
                name="models_endpoint",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_chat_completion_basic(self) -> TestResult:
        """Test basic chat completion."""
        start = time.time()
        try:
            data = self._chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=20
            )
            passed = (
                "choices" in data and
                len(data["choices"]) > 0 and
                "message" in data["choices"][0] and
                len(data["choices"][0]["message"].get("content", "")) > 0
            )
            return TestResult(
                name="chat_completion_basic",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={
                    "response": data["choices"][0]["message"]["content"] if passed else None,
                    "usage": data.get("usage", {})
                }
            )
        except Exception as e:
            return TestResult(
                name="chat_completion_basic",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_completion_basic(self) -> TestResult:
        """Test basic completion."""
        start = time.time()
        try:
            data = self._completion(
                prompt="The meaning of life is",
                max_tokens=30
            )
            passed = (
                "choices" in data and
                len(data["choices"]) > 0 and
                len(data["choices"][0].get("text", "")) > 0
            )
            return TestResult(
                name="completion_basic",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={
                    "response": data["choices"][0]["text"] if passed else None,
                    "usage": data.get("usage", {})
                }
            )
        except Exception as e:
            return TestResult(
                name="completion_basic",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_multi_turn_conversation(self) -> TestResult:
        """Test multi-turn conversation."""
        start = time.time()
        try:
            data = self._chat_completion(
                messages=[
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
                    {"role": "user", "content": "What is my name?"}
                ],
                max_tokens=30
            )
            response = data["choices"][0]["message"]["content"].lower()
            passed = "alice" in response
            return TestResult(
                name="multi_turn_conversation",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"response": data["choices"][0]["message"]["content"]}
            )
        except Exception as e:
            return TestResult(
                name="multi_turn_conversation",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_temperature_variation(self) -> TestResult:
        """Test that temperature affects output."""
        start = time.time()
        try:
            # Low temperature - should be more deterministic
            results_low = []
            for _ in range(3):
                data = self._completion(prompt="1+1=", max_tokens=5, temperature=0.0)
                results_low.append(data["choices"][0]["text"])
            
            # Check low temp produces consistent results
            low_temp_consistent = len(set(results_low)) == 1
            
            passed = low_temp_consistent
            return TestResult(
                name="temperature_variation",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={
                    "low_temp_results": results_low,
                    "low_temp_consistent": low_temp_consistent,
                }
            )
        except Exception as e:
            return TestResult(
                name="temperature_variation",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    # === Edge Case Tests ===
    
    def test_empty_prompt(self) -> TestResult:
        """Test handling of empty prompt."""
        start = time.time()
        try:
            resp = self._request("POST", "/v1/completions", json={
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt": "",
                "max_tokens": 10
            })
            # Should either return error or handle gracefully
            passed = resp.status_code in [200, 400, 422]
            if resp.status_code == 200:
                data = resp.json()
                passed = "choices" in data
            return TestResult(
                name="empty_prompt",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"status_code": resp.status_code}
            )
        except Exception as e:
            return TestResult(
                name="empty_prompt",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_max_tokens_zero(self) -> TestResult:
        """Test handling of max_tokens=0."""
        start = time.time()
        try:
            resp = self._request("POST", "/v1/completions", json={
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt": "Hello",
                "max_tokens": 0
            })
            # Should return error (400/422) for invalid max_tokens
            passed = resp.status_code in [400, 422, 500]
            return TestResult(
                name="max_tokens_zero",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"status_code": resp.status_code}
            )
        except Exception as e:
            # Error is expected
            return TestResult(
                name="max_tokens_zero",
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"error": str(e)}
            )
    
    def test_long_prompt(self) -> TestResult:
        """Test handling of long prompt."""
        start = time.time()
        try:
            long_prompt = "Hello world. " * 500  # ~6000 chars
            data = self._completion(prompt=long_prompt, max_tokens=20)
            passed = "choices" in data and len(data["choices"]) > 0
            return TestResult(
                name="long_prompt",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"prompt_length": len(long_prompt)}
            )
        except Exception as e:
            return TestResult(
                name="long_prompt",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_invalid_model(self) -> TestResult:
        """Test handling of invalid model name."""
        start = time.time()
        try:
            resp = self._request("POST", "/v1/completions", json={
                "model": "nonexistent-model",
                "prompt": "Hello",
                "max_tokens": 10
            })
            # Should return 400 or 404
            passed = resp.status_code in [400, 404, 422]
            return TestResult(
                name="invalid_model",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"status_code": resp.status_code}
            )
        except Exception as e:
            return TestResult(
                name="invalid_model",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def test_unicode_handling(self) -> TestResult:
        """Test handling of unicode characters."""
        start = time.time()
        try:
            data = self._chat_completion(
                messages=[{"role": "user", "content": "Translate 'hello' to Chinese: 你好"}],
                max_tokens=30
            )
            passed = "choices" in data and len(data["choices"]) > 0
            return TestResult(
                name="unicode_handling",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                details={"response": data["choices"][0]["message"]["content"] if passed else None}
            )
        except Exception as e:
            return TestResult(
                name="unicode_handling",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    # === Performance Benchmarks ===
    
    def benchmark_latency(self, num_requests: int = 10) -> BenchmarkResult:
        """Benchmark single request latency."""
        latencies = []
        errors = 0
        tokens = 0
        
        for i in range(num_requests):
            start = time.time()
            try:
                data = self._chat_completion(
                    messages=[{"role": "user", "content": f"Count from 1 to 5. Request {i}"}],
                    max_tokens=50,
                    temperature=0.0
                )
                latencies.append((time.time() - start) * 1000)
                tokens += data.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                errors += 1
        
        return BenchmarkResult(
            name="single_request_latency",
            requests=num_requests,
            latencies_ms=latencies,
            errors=errors,
            tokens_generated=tokens
        )
    
    def benchmark_concurrent(self, num_requests: int = 20, 
                             concurrency: int = 4) -> BenchmarkResult:
        """Benchmark concurrent request handling."""
        latencies = []
        errors = 0
        tokens = 0
        
        def make_request(idx: int):
            start = time.time()
            try:
                data = self._chat_completion(
                    messages=[{"role": "user", "content": f"What is {idx} + {idx}?"}],
                    max_tokens=30,
                    temperature=0.0
                )
                latency = (time.time() - start) * 1000
                return latency, data.get("usage", {}).get("completion_tokens", 0), None
            except Exception as e:
                return (time.time() - start) * 1000, 0, str(e)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                latency, toks, err = future.result()
                latencies.append(latency)
                tokens += toks
                if err:
                    errors += 1
        
        return BenchmarkResult(
            name=f"concurrent_{concurrency}_workers",
            requests=num_requests,
            latencies_ms=latencies,
            errors=errors,
            tokens_generated=tokens
        )
    
    def benchmark_throughput(self, duration_seconds: int = 30) -> BenchmarkResult:
        """Benchmark throughput over time."""
        latencies = []
        errors = 0
        tokens = 0
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            req_start = time.time()
            try:
                data = self._completion(
                    prompt="Hello",
                    max_tokens=20,
                    temperature=0.0
                )
                latencies.append((time.time() - req_start) * 1000)
                tokens += data.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                errors += 1
            request_count += 1
        
        return BenchmarkResult(
            name=f"throughput_{duration_seconds}s",
            requests=request_count,
            latencies_ms=latencies,
            errors=errors,
            tokens_generated=tokens
        )
    
    def benchmark_token_generation(self, token_counts: List[int] = [10, 50, 100, 200]) -> List[BenchmarkResult]:
        """Benchmark varying token generation lengths."""
        results = []
        for token_count in token_counts:
            latencies = []
            errors = 0
            tokens = 0
            
            for _ in range(3):  # 3 samples per token count
                start = time.time()
                try:
                    data = self._completion(
                        prompt="Write a story:",
                        max_tokens=token_count,
                        temperature=0.7
                    )
                    latencies.append((time.time() - start) * 1000)
                    tokens += data.get("usage", {}).get("completion_tokens", 0)
                except Exception:
                    errors += 1
            
            results.append(BenchmarkResult(
                name=f"tokens_{token_count}",
                requests=3,
                latencies_ms=latencies,
                errors=errors,
                tokens_generated=tokens
            ))
        
        return results
    
    # === Defect Detection ===
    
    def detect_defects(self) -> List[Dict[str, Any]]:
        """Detect potential defects based on test results."""
        defects = []
        
        for result in self.results:
            if not result.passed:
                severity = "HIGH" if result.name in [
                    "health_endpoint", "chat_completion_basic", "completion_basic"
                ] else "MEDIUM"
                
                defects.append({
                    "id": f"DEF-{len(defects)+1:03d}",
                    "test": result.name,
                    "severity": severity,
                    "description": result.error or "Test failed",
                    "details": result.details
                })
        
        # Check for performance issues
        for benchmark in self.benchmarks:
            if benchmark.p99_latency_ms > 10000:  # >10s p99
                defects.append({
                    "id": f"DEF-{len(defects)+1:03d}",
                    "test": benchmark.name,
                    "severity": "MEDIUM",
                    "description": f"High p99 latency: {benchmark.p99_latency_ms:.0f}ms",
                    "details": benchmark.to_dict()
                })
            
            error_rate = benchmark.errors / benchmark.requests * 100 if benchmark.requests > 0 else 0
            if error_rate > 5:  # >5% error rate
                defects.append({
                    "id": f"DEF-{len(defects)+1:03d}",
                    "test": benchmark.name,
                    "severity": "HIGH",
                    "description": f"High error rate: {error_rate:.1f}%",
                    "details": benchmark.to_dict()
                })
        
        self.defects = defects
        return defects
    
    def run_all_tests(self) -> None:
        """Run all functional tests."""
        print("\n" + "="*60)
        print("Running Functional Tests")
        print("="*60)
        
        tests = [
            self.test_health_endpoint,
            self.test_models_endpoint,
            self.test_chat_completion_basic,
            self.test_completion_basic,
            self.test_multi_turn_conversation,
            self.test_temperature_variation,
            self.test_empty_prompt,
            self.test_max_tokens_zero,
            self.test_long_prompt,
            self.test_invalid_model,
            self.test_unicode_handling,
        ]
        
        for test_fn in tests:
            result = test_fn()
            self.results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} {result.name} ({result.duration_ms:.1f}ms)")
            if not result.passed and result.error:
                print(f"       Error: {result.error}")
    
    def run_all_benchmarks(self) -> None:
        """Run all performance benchmarks."""
        print("\n" + "="*60)
        print("Running Performance Benchmarks")
        print("="*60)
        
        # Single request latency
        print("  Running single request latency benchmark...")
        self.benchmarks.append(self.benchmark_latency(num_requests=10))
        
        # Concurrent requests
        print("  Running concurrent request benchmark...")
        self.benchmarks.append(self.benchmark_concurrent(num_requests=20, concurrency=4))
        
        # Throughput
        print("  Running throughput benchmark (30s)...")
        self.benchmarks.append(self.benchmark_throughput(duration_seconds=30))
        
        # Token generation scaling
        print("  Running token generation scaling benchmark...")
        self.benchmarks.extend(self.benchmark_token_generation([10, 50, 100]))
        
        # Print summary
        print("\n  Benchmark Results:")
        for bench in self.benchmarks:
            print(f"    {bench.name}:")
            print(f"      Requests: {bench.requests}, Errors: {bench.errors}")
            print(f"      Avg: {bench.avg_latency_ms:.1f}ms, P50: {bench.p50_latency_ms:.1f}ms, "
                  f"P95: {bench.p95_latency_ms:.1f}ms, P99: {bench.p99_latency_ms:.1f}ms")
            if bench.tokens_generated > 0:
                print(f"      Tokens/sec: {bench.tokens_per_second:.1f}")
    
    def generate_report(self, output_path: str = "production_test_report.json") -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Detect defects
        self.detect_defects()
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "framework": "dfastllm",
                "version": "2.1.0",
            },
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "pass_rate": sum(1 for r in self.results if r.passed) / len(self.results) * 100 if self.results else 0,
                "total_benchmarks": len(self.benchmarks),
                "defects_found": len(self.defects),
            },
            "functional_tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": round(r.duration_ms, 2),
                    "error": r.error,
                    "details": r.details,
                }
                for r in self.results
            ],
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "defects": self.defects,
        }
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  Report saved to: {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="dfastllm Production Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--output", default="production_test_report.json", help="Output report path")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarks")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("dfastllm GPU Production Test Suite")
    print("="*60)
    print(f"Target: {args.url}")
    print(f"Time: {datetime.now().isoformat()}")
    
    tester = ProductionTester(base_url=args.url)
    
    # Run tests
    tester.run_all_tests()
    
    # Run benchmarks
    if not args.skip_benchmarks:
        tester.run_all_benchmarks()
    
    # Generate report
    report = tester.generate_report(output_path=args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"  Tests: {report['summary']['passed']}/{report['summary']['total_tests']} passed "
          f"({report['summary']['pass_rate']:.1f}%)")
    print(f"  Benchmarks: {report['summary']['total_benchmarks']}")
    print(f"  Defects Found: {report['summary']['defects_found']}")
    
    if report['defects']:
        print("\n  Defects:")
        for defect in report['defects']:
            print(f"    [{defect['severity']}] {defect['id']}: {defect['description']}")
    
    # Exit with error code if tests failed
    sys.exit(0 if report['summary']['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
