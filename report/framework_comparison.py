#!/usr/bin/env python3
"""Performance Comparison: dfastllm vs vLLM and other frameworks.

Comprehensive benchmark comparing:
- dfastllm (Diffusion LLM inference)
- vLLM (Standard autoregressive inference)

Metrics compared:
- Time to First Token (TTFT)
- Total latency
- Throughput (tokens/sec)
- Concurrent request handling
- Memory efficiency
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
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    warmup_requests: int = 3
    latency_requests: int = 20
    throughput_duration: int = 60
    concurrent_requests: int = 20
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    token_counts: List[int] = field(default_factory=lambda: [10, 50, 100, 200])
    prompt: str = "Hello, how are you today?"
    temperature: float = 0.7


@dataclass
class FrameworkResult:
    """Results for a single framework."""
    name: str
    url: str
    model: str
    available: bool
    error: Optional[str] = None
    
    # Latency metrics (ms)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # TTFT metrics (ms)
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Concurrent performance
    concurrent_results: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Token scaling
    token_scaling: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Memory
    gpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "model": self.model,
            "available": self.available,
            "error": self.error,
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
                "min_ms": round(self.min_latency_ms, 2),
                "max_ms": round(self.max_latency_ms, 2),
            },
            "ttft": {
                "avg_ms": round(self.avg_ttft_ms, 2),
                "p50_ms": round(self.p50_ttft_ms, 2),
                "p95_ms": round(self.p95_ttft_ms, 2),
            },
            "throughput": {
                "tokens_per_second": round(self.tokens_per_second, 2),
                "requests_per_second": round(self.requests_per_second, 2),
            },
            "concurrent_performance": self.concurrent_results,
            "token_scaling": self.token_scaling,
            "gpu_memory_mb": round(self.gpu_memory_mb, 2),
        }


class FrameworkBenchmark:
    """Benchmark runner for a single framework."""
    
    def __init__(self, name: str, url: str, model: Optional[str] = None):
        self.name = name
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.result = FrameworkResult(name=name, url=url, model=model or "unknown", available=False)
    
    def check_health(self) -> bool:
        """Check if framework is available."""
        try:
            resp = self.session.get(f"{self.url}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.result.available = True
                self.result.gpu_memory_mb = data.get("gpu_memory", {}).get("used_mb", 0)
                
                # Try to get model name
                try:
                    models_resp = self.session.get(f"{self.url}/v1/models", timeout=10)
                    if models_resp.status_code == 200:
                        models_data = models_resp.json()
                        if models_data.get("data"):
                            self.model = models_data["data"][0]["id"]
                            self.result.model = self.model
                except Exception:
                    pass
                
                return True
        except Exception as e:
            self.result.error = str(e)
        return False
    
    def _chat_completion(self, prompt: str, max_tokens: int = 50, 
                         temperature: float = 0.7) -> Dict[str, Any]:
        """Send chat completion request."""
        resp = self.session.post(
            f"{self.url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120
        )
        return resp.json()
    
    def warmup(self, config: BenchmarkConfig) -> None:
        """Warm up the model."""
        print(f"    Warming up {self.name}...")
        for _ in range(config.warmup_requests):
            try:
                self._chat_completion(config.prompt, max_tokens=10, temperature=0.0)
            except Exception:
                pass
    
    def benchmark_latency(self, config: BenchmarkConfig) -> None:
        """Benchmark single request latency."""
        print(f"    Running latency benchmark for {self.name}...")
        latencies = []
        
        for _ in range(config.latency_requests):
            start = time.time()
            try:
                self._chat_completion(config.prompt, max_tokens=50, temperature=config.temperature)
                latencies.append((time.time() - start) * 1000)
            except Exception:
                pass
        
        if latencies:
            self.result.avg_latency_ms = statistics.mean(latencies)
            self.result.p50_latency_ms = statistics.median(latencies)
            sorted_latencies = sorted(latencies)
            self.result.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.result.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            self.result.min_latency_ms = min(latencies)
            self.result.max_latency_ms = max(latencies)
    
    def benchmark_concurrent(self, config: BenchmarkConfig) -> None:
        """Benchmark concurrent request handling."""
        print(f"    Running concurrent benchmarks for {self.name}...")
        
        for concurrency in config.concurrency_levels:
            latencies = []
            errors = 0
            
            def make_request(idx: int):
                start = time.time()
                try:
                    self._chat_completion(f"Request {idx}: " + config.prompt, max_tokens=30)
                    return (time.time() - start) * 1000, None
                except Exception as e:
                    return (time.time() - start) * 1000, str(e)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(make_request, i) for i in range(config.concurrent_requests)]
                for future in concurrent.futures.as_completed(futures):
                    latency, err = future.result()
                    latencies.append(latency)
                    if err:
                        errors += 1
            
            if latencies:
                self.result.concurrent_results[concurrency] = {
                    "avg_latency_ms": round(statistics.mean(latencies), 2),
                    "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
                    "errors": errors,
                    "throughput_rps": round(len(latencies) / (sum(latencies) / 1000), 2),
                }
    
    def benchmark_throughput(self, config: BenchmarkConfig) -> None:
        """Benchmark sustained throughput."""
        print(f"    Running throughput benchmark for {self.name}...")
        
        start_time = time.time()
        total_requests = 0
        total_tokens = 0
        total_time = 0
        
        while time.time() - start_time < config.throughput_duration:
            req_start = time.time()
            try:
                data = self._chat_completion(config.prompt, max_tokens=30, temperature=0.0)
                total_time += (time.time() - req_start)
                total_requests += 1
                total_tokens += data.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                pass
        
        if total_time > 0:
            self.result.tokens_per_second = total_tokens / total_time
            self.result.requests_per_second = total_requests / total_time
    
    def benchmark_token_scaling(self, config: BenchmarkConfig) -> None:
        """Benchmark how latency scales with token count."""
        print(f"    Running token scaling benchmark for {self.name}...")
        
        for token_count in config.token_counts:
            latencies = []
            tokens = []
            
            for _ in range(3):  # 3 samples per token count
                start = time.time()
                try:
                    data = self._chat_completion("Tell me a story:", max_tokens=token_count)
                    latencies.append((time.time() - start) * 1000)
                    tokens.append(data.get("usage", {}).get("completion_tokens", 0))
                except Exception:
                    pass
            
            if latencies:
                self.result.token_scaling[token_count] = {
                    "avg_latency_ms": round(statistics.mean(latencies), 2),
                    "avg_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
                    "ms_per_token": round(statistics.mean(latencies) / (statistics.mean(tokens) or 1), 2),
                }
    
    def run_all_benchmarks(self, config: BenchmarkConfig) -> FrameworkResult:
        """Run all benchmarks."""
        if not self.check_health():
            print(f"  ✗ {self.name} is not available: {self.result.error}")
            return self.result
        
        print(f"  ✓ {self.name} is available")
        
        self.warmup(config)
        self.benchmark_latency(config)
        self.benchmark_concurrent(config)
        self.benchmark_throughput(config)
        self.benchmark_token_scaling(config)
        
        return self.result


def compare_frameworks(results: List[FrameworkResult]) -> Dict[str, Any]:
    """Generate comparison analysis between frameworks."""
    available = [r for r in results if r.available]
    
    if len(available) < 2:
        return {"error": "Not enough frameworks available for comparison"}
    
    # Find baseline (vLLM if available, otherwise first framework)
    baseline = next((r for r in available if "vllm" in r.name.lower()), available[0])
    comparison_target = next((r for r in available if r != baseline), available[1])
    
    def calc_diff(target_val: float, baseline_val: float) -> Dict[str, Any]:
        if baseline_val == 0:
            return {"value": target_val, "diff_pct": None}
        diff_pct = ((target_val - baseline_val) / baseline_val) * 100
        return {
            "value": target_val,
            "baseline": baseline_val,
            "diff_pct": round(diff_pct, 2),
            "better": diff_pct < 0 if baseline_val > 0 else diff_pct > 0,
        }
    
    return {
        "baseline": baseline.name,
        "compared_to": comparison_target.name,
        "latency_comparison": {
            "avg_ms": calc_diff(comparison_target.avg_latency_ms, baseline.avg_latency_ms),
            "p95_ms": calc_diff(comparison_target.p95_latency_ms, baseline.p95_latency_ms),
            "p99_ms": calc_diff(comparison_target.p99_latency_ms, baseline.p99_latency_ms),
        },
        "throughput_comparison": {
            "tokens_per_second": calc_diff(comparison_target.tokens_per_second, baseline.tokens_per_second),
            "requests_per_second": calc_diff(comparison_target.requests_per_second, baseline.requests_per_second),
        },
        "memory_comparison": {
            "gpu_memory_mb": calc_diff(comparison_target.gpu_memory_mb, baseline.gpu_memory_mb),
        },
    }


def generate_report(results: List[FrameworkResult], config: BenchmarkConfig) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    comparison = compare_frameworks(results)
    
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_config": {
                "warmup_requests": config.warmup_requests,
                "latency_requests": config.latency_requests,
                "throughput_duration": config.throughput_duration,
                "concurrent_requests": config.concurrent_requests,
                "concurrency_levels": config.concurrency_levels,
                "token_counts": config.token_counts,
            },
        },
        "summary": {
            "frameworks_tested": len(results),
            "frameworks_available": sum(1 for r in results if r.available),
            "fastest_avg_latency": min((r.name for r in results if r.available), 
                                       key=lambda n: next(r.avg_latency_ms for r in results if r.name == n), 
                                       default="N/A"),
            "highest_throughput": max((r.name for r in results if r.available),
                                      key=lambda n: next(r.tokens_per_second for r in results if r.name == n),
                                      default="N/A"),
        },
        "framework_results": [r.to_dict() for r in results],
        "comparison": comparison,
        "recommendations": generate_recommendations(results, comparison),
    }


def generate_recommendations(results: List[FrameworkResult], comparison: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on benchmark results."""
    recommendations = []
    
    available = [r for r in results if r.available]
    
    if not available:
        return ["No frameworks available for testing"]
    
    # Find best performer
    best_latency = min(available, key=lambda r: r.avg_latency_ms)
    best_throughput = max(available, key=lambda r: r.tokens_per_second)
    
    recommendations.append(
        f"For lowest latency: Use {best_latency.name} "
        f"(avg {best_latency.avg_latency_ms:.0f}ms)"
    )
    
    recommendations.append(
        f"For highest throughput: Use {best_throughput.name} "
        f"({best_throughput.tokens_per_second:.1f} tokens/sec)"
    )
    
    # Check for high concurrency performance
    for r in available:
        if r.concurrent_results:
            max_conc = max(r.concurrent_results.keys())
            if r.concurrent_results[max_conc].get("errors", 0) == 0:
                recommendations.append(
                    f"{r.name} handles {max_conc} concurrent requests with 0 errors"
                )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Framework Performance Comparison")
    parser.add_argument("--dfastllm-url", default="http://localhost:8000", 
                        help="dfastllm server URL")
    parser.add_argument("--vllm-url", default="http://localhost:8001",
                        help="vLLM server URL (optional)")
    parser.add_argument("--output", default="framework_comparison_report.json",
                        help="Output report path")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmarks (fewer requests)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Framework Performance Comparison")
    print("="*60)
    print(f"Time: {datetime.now().isoformat()}")
    
    # Configure benchmarks
    if args.quick:
        config = BenchmarkConfig(
            warmup_requests=1,
            latency_requests=5,
            throughput_duration=15,
            concurrent_requests=10,
            concurrency_levels=[1, 2, 4],
            token_counts=[10, 50],
        )
    else:
        config = BenchmarkConfig()
    
    # Define frameworks to test
    frameworks = [
        FrameworkBenchmark("dfastllm", args.dfastllm_url),
        FrameworkBenchmark("vLLM", args.vllm_url),
    ]
    
    print(f"\nTesting {len(frameworks)} frameworks...")
    
    # Run benchmarks
    results = []
    for framework in frameworks:
        print(f"\nBenchmarking {framework.name}...")
        result = framework.run_all_benchmarks(config)
        results.append(result)
    
    # Generate report
    report = generate_report(results, config)
    
    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for result in results:
        if result.available:
            print(f"\n{result.name} ({result.model}):")
            print(f"  Avg Latency: {result.avg_latency_ms:.0f}ms")
            print(f"  P95 Latency: {result.p95_latency_ms:.0f}ms")
            print(f"  Throughput: {result.tokens_per_second:.1f} tok/s")
            print(f"  GPU Memory: {result.gpu_memory_mb:.0f} MB")
        else:
            print(f"\n{result.name}: Not available")
    
    if report.get("comparison") and "error" not in report["comparison"]:
        print("\n" + "-"*40)
        print("Comparison:")
        comp = report["comparison"]
        print(f"  {comp['compared_to']} vs {comp['baseline']}:")
        lat = comp['latency_comparison']['avg_ms']
        if lat.get('diff_pct') is not None:
            direction = "faster" if lat['diff_pct'] < 0 else "slower"
            print(f"    Latency: {abs(lat['diff_pct']):.1f}% {direction}")
        tp = comp['throughput_comparison']['tokens_per_second']
        if tp.get('diff_pct') is not None:
            direction = "higher" if tp['diff_pct'] > 0 else "lower"
            print(f"    Throughput: {abs(tp['diff_pct']):.1f}% {direction}")
    
    print("\n" + "="*60)
    print("Recommendations:")
    for rec in report.get("recommendations", []):
        print(f"  • {rec}")


if __name__ == "__main__":
    main()
