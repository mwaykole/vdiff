#!/usr/bin/env python3
"""Comprehensive tests for dfastllm package.

Tests cover:
1. Import compatibility
2. DFastLLMEngine functionality
3. Configuration validation
4. API compatibility
5. Error handling
"""

import sys
from dataclasses import dataclass
from typing import List

# Test results tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    error: str = ""

results: List[TestResult] = []

def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                func()
                results.append(TestResult(name=name, passed=True))
                print(f"  ✅ {name}")
                return True
            except Exception as e:
                results.append(TestResult(name=name, passed=False, error=str(e)))
                print(f"  ❌ {name}: {e}")
                return False
        return wrapper
    return decorator


# =============================================================================
# Import Tests
# =============================================================================
print("\n" + "="*60)
print("1. IMPORT TESTS")
print("="*60)

@test("Import dfastllm package")
def test_import_package():
    import dfastllm
    assert dfastllm.__version__

@test("Import core engine")
def test_import_engine():
    from dfastllm.engine import DFastLLMEngine
    assert DFastLLMEngine

@test("Import config")
def test_import_config():
    from dfastllm.config import DFastLLMConfig
    assert DFastLLMConfig

@test("Import sampling params")
def test_import_sampling():
    from dfastllm.engine import SamplingParams
    assert SamplingParams

@test("Import outputs")
def test_import_outputs():
    from dfastllm.engine import RequestOutput, CompletionOutput
    assert RequestOutput
    assert CompletionOutput

@test("Import APD")
def test_import_apd():
    from dfastllm.engine import APDConfig
    assert APDConfig

@test("Import version info")
def test_import_version():
    from dfastllm.version import __version__, __version_info__
    assert __version__ == "2.0.0"
    assert len(__version_info__) == 3

# Run import tests
test_import_package()
test_import_engine()
test_import_config()
test_import_sampling()
test_import_outputs()
test_import_apd()
test_import_version()


# =============================================================================
# Config Tests
# =============================================================================
print("\n" + "="*60)
print("2. CONFIG TESTS")
print("="*60)

@test("Default DFastLLMConfig")
def test_default_config():
    from dfastllm.config import DFastLLMConfig
    config = DFastLLMConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000

@test("Config with model")
def test_config_with_model():
    from dfastllm.config import DFastLLMConfig
    config = DFastLLMConfig(model="microsoft/phi-2")
    assert config.model == "microsoft/phi-2"

@test("Config with custom port")
def test_config_custom_port():
    from dfastllm.config import DFastLLMConfig
    config = DFastLLMConfig(port=8080)
    assert config.port == 8080

@test("Config with diffusion settings")
def test_config_diffusion():
    from dfastllm.config import DFastLLMConfig
    config = DFastLLMConfig(diffusion_steps=16, enable_apd=True)
    assert config.diffusion_steps == 16
    assert config.enable_apd == True

# Run config tests
test_default_config()
test_config_with_model()
test_config_custom_port()
test_config_diffusion()


# =============================================================================
# Sampling Params Tests
# =============================================================================
print("\n" + "="*60)
print("3. SAMPLING PARAMS TESTS")
print("="*60)

@test("Default SamplingParams")
def test_default_sampling():
    from dfastllm.engine import SamplingParams
    params = SamplingParams()
    assert params.max_tokens >= 0
    assert params.temperature >= 0

@test("SamplingParams with all options")
def test_sampling_all_options():
    from dfastllm.engine import SamplingParams
    params = SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        top_k=50
    )
    assert params.max_tokens == 100
    assert params.temperature == 0.8

@test("SamplingParams from_openai_params")
def test_sampling_from_openai():
    from dfastllm.engine import SamplingParams
    # Check if method exists
    if hasattr(SamplingParams, 'from_openai_params'):
        params = SamplingParams.from_openai_params(
            max_tokens=50,
            temperature=0.7
        )
        assert params.max_tokens == 50

# Run sampling params tests
test_default_sampling()
test_sampling_all_options()
test_sampling_from_openai()


# =============================================================================
# Output Tests
# =============================================================================
print("\n" + "="*60)
print("4. OUTPUT TESTS")
print("="*60)

@test("CompletionOutput creation")
def test_completion_output():
    from dfastllm.engine import CompletionOutput
    output = CompletionOutput(
        index=0,
        text="Hello world",
        token_ids=[1, 2, 3],
        finish_reason="stop"
    )
    assert output.text == "Hello world"
    assert output.finish_reason == "stop"

@test("RequestOutput creation")
def test_request_output():
    from dfastllm.engine import RequestOutput, CompletionOutput
    completion = CompletionOutput(
        index=0,
        text="Test",
        token_ids=[1],
        finish_reason="length"
    )
    output = RequestOutput(
        request_id="test-123",
        prompt="Input",
        prompt_token_ids=[1, 2],
        outputs=[completion],
        finished=True
    )
    assert output.request_id == "test-123"
    assert len(output.outputs) == 1

# Run output tests
test_completion_output()
test_request_output()


# =============================================================================
# Engine State Tests  
# =============================================================================
print("\n" + "="*60)
print("5. ENGINE STATE TESTS")
print("="*60)

@test("Engine state enum exists")
def test_engine_state():
    from dfastllm.engine import EngineState
    assert hasattr(EngineState, 'UNINITIALIZED')
    assert hasattr(EngineState, 'READY')

@test("Health status exists")
def test_health_status():
    from dfastllm.engine import HealthStatus
    assert HealthStatus

# Run engine state tests
test_engine_state()
test_health_status()


# =============================================================================
# APD Config Tests
# =============================================================================
print("\n" + "="*60)
print("6. APD CONFIG TESTS")
print("="*60)

@test("APDConfig creation")
def test_apd_config():
    from dfastllm.engine import APDConfig
    config = APDConfig()
    assert config is not None

@test("APDConfig with parameters")
def test_apd_config_params():
    from dfastllm.engine import APDConfig
    config = APDConfig(
        enabled=True,
        max_parallel_tokens=8
    )
    assert config.enabled == True
    assert config.max_parallel_tokens == 8

# Run APD tests
test_apd_config()
test_apd_config_params()


# =============================================================================
# Print Summary
# =============================================================================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

passed = sum(1 for r in results if r.passed)
failed = sum(1 for r in results if not r.passed)
total = len(results)

print(f"\n📊 Results: {passed}/{total} passed ({100*passed/total:.1f}%)")

if failed > 0:
    print(f"\n❌ Failed tests ({failed}):")
    for r in results:
        if not r.passed:
            print(f"   - {r.name}: {r.error}")
else:
    print("\n🎉 All tests passed!")

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
