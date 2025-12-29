#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for dfastllm

Run as:
    python tests/run_qa_tests.py

Generates a comprehensive test report for production readiness.
"""

import sys
import time
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Results for a test suite."""
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors
    
    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total * 100


@dataclass
class QAReport:
    """Complete QA test report."""
    timestamp: str
    version: str
    environment: str
    total_passed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0.0
    suites: List[TestSuiteResult] = field(default_factory=list)
    
    @property
    def overall_pass_rate(self) -> float:
        total = self.total_passed + self.total_failed
        if total == 0:
            return 0.0
        return self.total_passed / total * 100


def run_import_tests() -> TestSuiteResult:
    """Test all critical imports."""
    suite = TestSuiteResult(name="Import Tests")
    start = time.time()
    
    imports_to_test = [
        ("dfastllm", "Main package"),
        ("dfastllm.config", "Configuration"),
        ("dfastllm.engine", "Engine module"),
        ("dfastllm.engine.base", "SOLID base classes"),
        ("dfastllm.engine.hybrid_engine", "Hybrid engine"),
        ("dfastllm.engine.continuous_batching", "Continuous batching"),
        ("dfastllm.engine.entropy_controller", "Entropy controller"),
        ("dfastllm.engine.mor_decoder", "MoR decoder"),
        ("dfastllm.engine.sampling_params", "Sampling params"),
        ("dfastllm.engine.outputs", "Output types"),
        ("dfastllm.engine.apd", "APD module"),
        ("dfastllm.engine.dfastllm_engine", "Main engine"),
        ("dfastllm.entrypoints.openai.protocol", "OpenAI protocol"),
        ("dfastllm.cli", "CLI module"),
    ]
    
    for module_name, description in imports_to_test:
        test_start = time.time()
        try:
            __import__(module_name)
            duration = (time.time() - test_start) * 1000
            suite.tests.append(TestResult(
                name=f"Import {description} ({module_name})",
                status="PASS",
                duration_ms=duration
            ))
            suite.passed += 1
        except Exception as e:
            duration = (time.time() - test_start) * 1000
            suite.tests.append(TestResult(
                name=f"Import {description} ({module_name})",
                status="FAIL",
                duration_ms=duration,
                error_message=str(e)
            ))
            suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_solid_tests() -> TestSuiteResult:
    """Test SOLID architecture compliance."""
    suite = TestSuiteResult(name="SOLID Architecture Tests")
    start = time.time()
    
    # Test BaseStats inheritance
    test_start = time.time()
    try:
        from dfastllm.engine.base import BaseStats
        from dfastllm.engine.hybrid_engine import HybridStats
        from dfastllm.engine.continuous_batching import BatcherStats
        from dfastllm.engine.entropy_controller import EntropyStats
        
        hs = HybridStats()
        bs = BatcherStats()
        es = EntropyStats()
        
        assert isinstance(hs, BaseStats), "HybridStats should inherit BaseStats"
        assert isinstance(bs, BaseStats), "BatcherStats should inherit BaseStats"
        assert isinstance(es, BaseStats), "EntropyStats should inherit BaseStats"
        
        # Test to_dict method (Liskov Substitution)
        assert hasattr(hs, 'to_dict')
        assert hasattr(bs, 'to_dict')
        assert hasattr(es, 'to_dict')
        
        suite.tests.append(TestResult(
            name="BaseStats Inheritance (Liskov)",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="BaseStats Inheritance (Liskov)",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test BaseConfig inheritance
    test_start = time.time()
    try:
        from dfastllm.engine.base import BaseConfig
        from dfastllm.engine.hybrid_engine import HybridConfig
        from dfastllm.engine.continuous_batching import BatcherConfig
        from dfastllm.engine.entropy_controller import EntropyConfig
        
        hc = HybridConfig()
        bc = BatcherConfig()
        ec = EntropyConfig()
        
        assert isinstance(hc, BaseConfig), "HybridConfig should inherit BaseConfig"
        assert isinstance(bc, BaseConfig), "BatcherConfig should inherit BaseConfig"
        assert isinstance(ec, BaseConfig), "EntropyConfig should inherit BaseConfig"
        
        suite.tests.append(TestResult(
            name="BaseConfig Inheritance (Open/Closed)",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="BaseConfig Inheritance (Open/Closed)",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test BaseCache inheritance
    test_start = time.time()
    try:
        from dfastllm.engine.base import BaseCache
        from dfastllm.engine.continuous_batching import PrefixCache
        
        pc = PrefixCache(max_cache_size=10)
        
        assert isinstance(pc, BaseCache), "PrefixCache should inherit BaseCache"
        assert hasattr(pc, 'get')
        assert hasattr(pc, 'put')
        assert hasattr(pc, 'clear')
        assert hasattr(pc, 'hit_rate')
        
        suite.tests.append(TestResult(
            name="BaseCache Inheritance (Interface Segregation)",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="BaseCache Inheritance (Interface Segregation)",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test EntropyComputer (Single Responsibility)
    test_start = time.time()
    try:
        from dfastllm.engine.base import EntropyComputer
        
        assert hasattr(EntropyComputer, 'compute')
        assert hasattr(EntropyComputer, 'compute_normalized')
        assert hasattr(EntropyComputer, 'compute_top_k')
        
        suite.tests.append(TestResult(
            name="EntropyComputer Utilities (Single Responsibility)",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="EntropyComputer Utilities (Single Responsibility)",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_hybrid_engine_tests() -> TestSuiteResult:
    """Test hybrid engine components."""
    suite = TestSuiteResult(name="Hybrid Engine Tests")
    start = time.time()
    
    # Test HybridConfig
    test_start = time.time()
    try:
        from dfastllm.engine.hybrid_engine import HybridConfig, HybridMode
        
        config = HybridConfig(mode=HybridMode.DEER)
        assert config.mode == HybridMode.DEER
        assert config.enabled is True
        assert config.draft_block_size == 8
        
        suite.tests.append(TestResult(
            name="HybridConfig Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="HybridConfig Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test HybridStats
    test_start = time.time()
    try:
        from dfastllm.engine.hybrid_engine import HybridStats
        
        stats = HybridStats()
        stats.update(drafted=8, accepted=6, diffusion_time=0.01, ar_time=0.005)
        
        assert stats.total_drafts == 1
        assert stats.tokens_accepted == 6
        assert stats.tokens_rejected == 2
        
        suite.tests.append(TestResult(
            name="HybridStats Tracking",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="HybridStats Tracking",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test HybridMode enum
    test_start = time.time()
    try:
        from dfastllm.engine.hybrid_engine import HybridMode
        
        assert HybridMode.DEER.value == "deer"
        assert HybridMode.SPEC_DIFF.value == "spec_diff"
        assert HybridMode.SEMI_AR.value == "semi_ar"
        assert HybridMode.ADAPTIVE.value == "adaptive"
        
        suite.tests.append(TestResult(
            name="HybridMode Enum Values",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="HybridMode Enum Values",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_continuous_batching_tests() -> TestSuiteResult:
    """Test continuous batching components."""
    suite = TestSuiteResult(name="Continuous Batching Tests")
    start = time.time()
    
    # Test BatcherConfig
    test_start = time.time()
    try:
        from dfastllm.engine.continuous_batching import BatcherConfig
        
        config = BatcherConfig(max_batch_size=16, max_wait_time_ms=100.0)
        assert config.max_batch_size == 16
        assert config.max_wait_time_ms == 100.0
        
        suite.tests.append(TestResult(
            name="BatcherConfig Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="BatcherConfig Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test PrefixCache
    test_start = time.time()
    try:
        from dfastllm.engine.continuous_batching import PrefixCache
        
        cache = PrefixCache(max_cache_size=10, min_prefix_length=4)
        
        # Test put and get
        tokens = list(range(20))
        cache.put(tokens, {"kv": "test"})
        result = cache.get(tokens)
        
        assert result is not None
        assert result["kv"] == "test"
        
        # Test hit rate
        assert cache.hit_rate > 0
        
        suite.tests.append(TestResult(
            name="PrefixCache Operations",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="PrefixCache Operations",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test cache eviction
    test_start = time.time()
    try:
        from dfastllm.engine.continuous_batching import PrefixCache
        
        cache = PrefixCache(max_cache_size=3, min_prefix_length=4)
        
        for i in range(5):
            tokens = list(range(i * 10, i * 10 + 20))
            cache.put(tokens, {"index": i})
        
        stats = cache.get_stats()
        assert stats["size"] <= 3
        
        suite.tests.append(TestResult(
            name="PrefixCache LRU Eviction",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="PrefixCache LRU Eviction",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_entropy_controller_tests() -> TestSuiteResult:
    """Test entropy controller components."""
    suite = TestSuiteResult(name="Entropy Controller Tests")
    start = time.time()
    
    # Test EntropyConfig
    test_start = time.time()
    try:
        from dfastllm.engine.entropy_controller import EntropyConfig, AdaptationStrategy
        
        config = EntropyConfig(strategy=AdaptationStrategy.COMBINED)
        assert config.enabled is True
        assert config.strategy == AdaptationStrategy.COMBINED
        
        suite.tests.append(TestResult(
            name="EntropyConfig Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="EntropyConfig Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test EntropyStats
    test_start = time.time()
    try:
        from dfastllm.engine.entropy_controller import EntropyStats
        
        stats = EntropyStats()
        stats.total_predictions = 100
        stats.high_entropy_count = 20
        stats.low_entropy_count = 60
        
        result = stats.to_dict()
        assert "high_entropy_pct" in result
        assert result["high_entropy_pct"] == 20.0
        
        suite.tests.append(TestResult(
            name="EntropyStats Calculation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="EntropyStats Calculation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_config_tests() -> TestSuiteResult:
    """Test configuration components."""
    suite = TestSuiteResult(name="Configuration Tests")
    start = time.time()
    
    # Test DFastLLMConfig
    test_start = time.time()
    try:
        from dfastllm.config import DFastLLMConfig
        
        config = DFastLLMConfig(model="test-model")
        assert config.model == "test-model"
        assert hasattr(config, 'enable_hybrid')
        assert hasattr(config, 'compile_model')
        
        suite.tests.append(TestResult(
            name="DFastLLMConfig Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="DFastLLMConfig Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test from_env
    test_start = time.time()
    try:
        import os
        from dfastllm.config import DFastLLMConfig
        
        os.environ["VDIFF_MODEL"] = "test-env-model"
        config = DFastLLMConfig.from_env()
        
        assert config.model == "test-env-model"
        
        suite.tests.append(TestResult(
            name="DFastLLMConfig from_env",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="DFastLLMConfig from_env",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def run_protocol_tests() -> TestSuiteResult:
    """Test OpenAI protocol components."""
    suite = TestSuiteResult(name="OpenAI Protocol Tests")
    start = time.time()
    
    # Test CompletionRequest
    test_start = time.time()
    try:
        from dfastllm.entrypoints.openai.protocol import CompletionRequest
        
        request = CompletionRequest(model="test", prompt="Hello")
        assert request.model == "test"
        assert request.prompt == "Hello"
        
        suite.tests.append(TestResult(
            name="CompletionRequest Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="CompletionRequest Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test ChatCompletionRequest
    test_start = time.time()
    try:
        from dfastllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatMessage
        
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(model="test", messages=messages)
        
        assert request.model == "test"
        assert len(request.messages) == 1
        
        suite.tests.append(TestResult(
            name="ChatCompletionRequest Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="ChatCompletionRequest Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    # Test HealthResponse
    test_start = time.time()
    try:
        from dfastllm.entrypoints.openai.protocol import HealthResponse
        
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"
        
        suite.tests.append(TestResult(
            name="HealthResponse Creation",
            status="PASS",
            duration_ms=(time.time() - test_start) * 1000
        ))
        suite.passed += 1
    except Exception as e:
        suite.tests.append(TestResult(
            name="HealthResponse Creation",
            status="FAIL",
            duration_ms=(time.time() - test_start) * 1000,
            error_message=str(e)
        ))
        suite.failed += 1
    
    suite.duration_ms = (time.time() - start) * 1000
    return suite


def generate_report() -> QAReport:
    """Generate comprehensive QA report."""
    report = QAReport(
        timestamp=datetime.now().isoformat(),
        version="2.3.0",
        environment=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    
    # Run all test suites
    suites = [
        run_import_tests(),
        run_solid_tests(),
        run_hybrid_engine_tests(),
        run_continuous_batching_tests(),
        run_entropy_controller_tests(),
        run_config_tests(),
        run_protocol_tests(),
    ]
    
    for suite in suites:
        report.suites.append(suite)
        report.total_passed += suite.passed
        report.total_failed += suite.failed
        report.total_skipped += suite.skipped
        report.total_errors += suite.errors
        report.total_duration_ms += suite.duration_ms
    
    return report


def print_report(report: QAReport):
    """Print formatted report."""
    print("=" * 80)
    print("                    DFASTLLM QA TEST REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print(f"Version: {report.version}")
    print(f"Environment: {report.environment}")
    print("-" * 80)
    print()
    
    for suite in report.suites:
        print(f"📦 {suite.name}")
        print(f"   Passed: {suite.passed} | Failed: {suite.failed} | "
              f"Total: {suite.total} | Pass Rate: {suite.pass_rate:.1f}%")
        
        for test in suite.tests:
            status_icon = "✅" if test.status == "PASS" else "❌"
            print(f"   {status_icon} {test.name} ({test.duration_ms:.1f}ms)")
            if test.error_message:
                print(f"      Error: {test.error_message[:80]}")
        print()
    
    print("=" * 80)
    print("                           SUMMARY")
    print("=" * 80)
    print(f"Total Passed:  {report.total_passed}")
    print(f"Total Failed:  {report.total_failed}")
    print(f"Total Skipped: {report.total_skipped}")
    print(f"Total Errors:  {report.total_errors}")
    print(f"Pass Rate:     {report.overall_pass_rate:.1f}%")
    print(f"Duration:      {report.total_duration_ms:.1f}ms")
    print("=" * 80)
    
    if report.total_failed == 0 and report.total_errors == 0:
        print("🎉 ALL TESTS PASSED - PRODUCTION READY!")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 80)


def main():
    """Main entry point."""
    print("Starting dfastllm QA Test Suite...")
    print()
    
    report = generate_report()
    print_report(report)
    
    # Save JSON report
    report_path = "report/QA_TEST_REPORT.json"
    try:
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "version": report.version,
                "environment": report.environment,
                "total_passed": report.total_passed,
                "total_failed": report.total_failed,
                "total_skipped": report.total_skipped,
                "total_errors": report.total_errors,
                "pass_rate": report.overall_pass_rate,
                "duration_ms": report.total_duration_ms,
                "suites": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "failed": s.failed,
                        "pass_rate": s.pass_rate,
                        "tests": [asdict(t) for t in s.tests]
                    }
                    for s in report.suites
                ]
            }, f, indent=2)
        print(f"\n📄 JSON report saved to: {report_path}")
    except Exception as e:
        print(f"\nWarning: Could not save JSON report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if report.total_failed == 0 else 1)


if __name__ == "__main__":
    main()
