# vdiff Roadmap: Production Optimization

This document outlines the path to make vdiff production-ready at scale.

## 1. Performance Optimizations

### Priority 1: torch.compile (Easy, High Impact)

Add PyTorch 2.0+ compilation for 2-4x speedup with zero code changes:

```python
# In vdiff_engine.py _load_torch_model()
if torch.__version__ >= "2.0" and self._device == "cuda":
    self._model = torch.compile(self._model, mode="reduce-overhead")
```

**Implementation:**
```python
# Add to VDiffConfig
compile_model: bool = True
compile_mode: str = "reduce-overhead"  # or "max-autotune"

# Add CLI args
--compile / --no-compile
--compile-mode reduce-overhead|max-autotune
```

### Priority 2: Flash Attention

Use Flash Attention for memory-efficient attention:

```python
# In model loading
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Add this
    torch_dtype=torch.float16,
)
```

**Requirements:**
- `pip install flash-attn --no-build-isolation`
- CUDA 11.6+, Ampere GPU or newer

### Priority 3: KV Cache Optimization for Diffusion

Unlike autoregressive models, diffusion models can reuse KV cache for **unchanged tokens**:

```python
class DiffusionKVCache:
    """Cache attention states for tokens that haven't changed between steps."""
    
    def __init__(self, max_batch_size: int, max_seq_len: int, num_layers: int):
        self.cache = {}
        self.token_versions = {}  # Track which tokens changed
    
    def get_or_compute(self, layer_idx: int, tokens: torch.Tensor, compute_fn):
        # Only recompute attention for changed tokens
        changed_mask = self._get_changed_mask(tokens)
        
        if changed_mask.any():
            new_kv = compute_fn(tokens[changed_mask])
            self.cache[layer_idx][changed_mask] = new_kv
        
        return self.cache[layer_idx]
```

### Priority 4: CUDA Kernels (Advanced)

Write custom CUDA kernels for diffusion-specific operations:

```python
# Example: Fused confidence + sampling kernel
@torch.jit.script
def fused_confidence_sample(
    logits: torch.Tensor,
    temperature: float,
    mask_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused confidence calculation and Gumbel sampling."""
    probs = F.softmax(logits / temperature, dim=-1)
    confidence = probs.max(dim=-1).values
    
    # Gumbel-max trick
    gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
    tokens = (probs.log() + gumbel).argmax(dim=-1)
    
    return tokens, confidence
```

### Priority 5: Memory Optimization

```python
# Add to config
class VDiffConfig:
    # Memory settings
    max_memory_gb: Optional[float] = None  # Limit GPU memory
    offload_to_cpu: bool = False  # Offload layers to CPU
    use_8bit: bool = False  # 8-bit quantization
    use_4bit: bool = False  # 4-bit quantization

# In engine
if config.use_8bit:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
    )
```

---

## 2. True Continuous Batching

### Current State
- Request queue with concurrency limits
- Each request processed independently
- No batching of forward passes

### Goal
- Batch multiple requests in single GPU forward pass
- Dynamic batching as requests arrive
- Padding/packing strategies

### Implementation Plan

#### Step 1: Request Batcher

```python
class RequestBatcher:
    """Collect requests and batch them for efficient processing."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 50.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.pending_requests: List[Request] = []
        self.batch_ready = asyncio.Event()
    
    async def add_request(self, request: Request) -> None:
        self.pending_requests.append(request)
        
        if len(self.pending_requests) >= self.max_batch_size:
            self.batch_ready.set()
    
    async def get_batch(self) -> List[Request]:
        # Wait for batch to fill or timeout
        try:
            await asyncio.wait_for(
                self.batch_ready.wait(),
                timeout=self.max_wait_time_ms / 1000,
            )
        except asyncio.TimeoutError:
            pass
        
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        self.batch_ready.clear()
        
        return batch
```

#### Step 2: Batched Diffusion Generation

```python
def batched_diffusion_generate(
    model: nn.Module,
    prompts: List[torch.Tensor],  # List of prompt tensors
    gen_length: int,
    steps: int,
    mask_id: int,
) -> List[torch.Tensor]:
    """Generate for multiple prompts in single forward pass."""
    
    # Pad prompts to same length
    max_prompt_len = max(p.shape[1] for p in prompts)
    batch_size = len(prompts)
    
    # Create batched input [B, seq_len]
    batched_input = torch.full(
        (batch_size, max_prompt_len + gen_length),
        mask_id,
        device=prompts[0].device,
    )
    
    # Fill in prompts
    for i, prompt in enumerate(prompts):
        batched_input[i, :prompt.shape[1]] = prompt[0]
    
    # Create attention mask for padding
    attention_mask = (batched_input != pad_token_id).long()
    
    # Run diffusion steps on batch
    for step in range(steps):
        with torch.no_grad():
            outputs = model(batched_input, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Process all sequences in parallel
        # ... (confidence, sampling, unmasking)
    
    return [batched_input[i:i+1] for i in range(batch_size)]
```

#### Step 3: Scheduler

```python
class DiffusionScheduler:
    """Schedule batched generation with iteration-level batching."""
    
    def __init__(self, engine: VDiffEngine, max_batch_size: int = 8):
        self.engine = engine
        self.batcher = RequestBatcher(max_batch_size)
        self._running = False
    
    async def start(self):
        self._running = True
        asyncio.create_task(self._batch_loop())
    
    async def _batch_loop(self):
        while self._running:
            batch = await self.batcher.get_batch()
            if batch:
                results = await self._process_batch(batch)
                for request, result in zip(batch, results):
                    request.future.set_result(result)
    
    async def generate(self, prompt: str, params: SamplingParams) -> RequestOutput:
        request = Request(prompt=prompt, params=params)
        await self.batcher.add_request(request)
        return await request.future
```

---

## 3. Real-World Testing

### Benchmarking Suite

Create `benchmarks/` directory:

```
benchmarks/
├── run_benchmark.py      # Main benchmark script
├── datasets/
│   ├── prompts_short.txt    # 10-50 token prompts
│   ├── prompts_medium.txt   # 50-200 token prompts
│   └── prompts_long.txt     # 200-500 token prompts
├── results/
│   └── .gitkeep
└── compare_vllm.py       # Compare with vLLM (for AR models)
```

#### Benchmark Script

```python
#!/usr/bin/env python3
"""vdiff Benchmark Suite"""

import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import List
import httpx

@dataclass
class BenchmarkResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    tokens_per_second: float

async def benchmark(
    url: str = "http://localhost:8000/v1/completions",
    num_requests: int = 100,
    concurrency: int = 10,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 64,
) -> BenchmarkResult:
    """Run benchmark against vdiff server."""
    
    latencies: List[float] = []
    tokens_generated: int = 0
    failures: int = 0
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def make_request(client: httpx.AsyncClient) -> float:
        nonlocal tokens_generated, failures
        
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    json={
                        "model": "default",
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                    },
                    timeout=60.0,
                )
                elapsed = (time.perf_counter() - start) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    tokens_generated += data.get("usage", {}).get("completion_tokens", 0)
                    return elapsed
                else:
                    failures += 1
                    return -1
            except Exception:
                failures += 1
                return -1
    
    start_time = time.perf_counter()
    
    async with httpx.AsyncClient() as client:
        tasks = [make_request(client) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    latencies = [r for r in results if r > 0]
    
    if not latencies:
        raise RuntimeError("All requests failed")
    
    latencies.sort()
    
    return BenchmarkResult(
        total_requests=num_requests,
        successful_requests=len(latencies),
        failed_requests=failures,
        total_time_s=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies[len(latencies) // 2],
        p95_latency_ms=latencies[int(len(latencies) * 0.95)],
        p99_latency_ms=latencies[int(len(latencies) * 0.99)],
        throughput_rps=len(latencies) / total_time,
        tokens_per_second=tokens_generated / total_time,
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    
    result = asyncio.run(benchmark(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    ))
    
    print(f"\n{'='*50}")
    print("vdiff Benchmark Results")
    print(f"{'='*50}")
    print(f"Total Requests:     {result.total_requests}")
    print(f"Successful:         {result.successful_requests}")
    print(f"Failed:             {result.failed_requests}")
    print(f"Total Time:         {result.total_time_s:.2f}s")
    print(f"Throughput:         {result.throughput_rps:.2f} req/s")
    print(f"Tokens/sec:         {result.tokens_per_second:.2f}")
    print(f"\nLatency:")
    print(f"  Average:          {result.avg_latency_ms:.2f}ms")
    print(f"  P50:              {result.p50_latency_ms:.2f}ms")
    print(f"  P95:              {result.p95_latency_ms:.2f}ms")
    print(f"  P99:              {result.p99_latency_ms:.2f}ms")
    print(f"{'='*50}")
```

### Load Testing with Locust

```python
# benchmarks/locustfile.py
from locust import HttpUser, task, between

class VDiffUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task(3)
    def chat_completion(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50,
            },
        )
    
    @task(1)
    def completion(self):
        self.client.post(
            "/v1/completions",
            json={
                "model": "default",
                "prompt": "Once upon a time",
                "max_tokens": 100,
            },
        )
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
```

Run with:
```bash
pip install locust
locust -f benchmarks/locustfile.py --host http://localhost:8000
```

---

## 4. Community Adoption

### GitHub Repository Setup

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=vdiff

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff black mypy
      - run: ruff check vdiff/
      - run: black --check vdiff/

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t vdiff:test --build-arg USE_CUDA=0 .
```

### Issue Templates

```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: Bug Report
about: Report a bug in vdiff
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
```bash
vdiff --model ... --port 8000
curl http://localhost:8000/v1/completions ...
```

**Expected behavior**
What you expected to happen.

**Environment**
- vdiff version:
- Python version:
- PyTorch version:
- CUDA version:
- GPU:
- OS:

**Logs**
```
Paste relevant logs here
```
```

### Contributing Guide

```markdown
<!-- CONTRIBUTING.md -->
# Contributing to vdiff

## Development Setup

```bash
git clone https://github.com/your-org/vdiff
cd vdiff
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
make test           # All tests
make test-unit      # Unit tests only
make test-coverage  # With coverage report
```

## Code Style

- Use `ruff` for linting
- Use `black` for formatting
- Add type hints to all functions
- Write docstrings for public APIs

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `make check`
4. Update documentation if needed
5. Submit PR with clear description
```

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | torch.compile | Low | High |
| 2 | Flash Attention | Low | Medium |
| 3 | Benchmark suite | Medium | High |
| 4 | GitHub CI/CD | Low | High |
| 5 | Request batching | High | High |
| 6 | 8-bit quantization | Low | Medium |
| 7 | KV cache optimization | High | Medium |
| 8 | CUDA kernels | Very High | High |

## Recommended Next Steps

1. **Week 1**: Add torch.compile + Flash Attention + benchmarks
2. **Week 2**: Setup GitHub CI/CD + issue templates
3. **Week 3-4**: Implement request batching
4. **Week 5+**: Performance tuning based on benchmark results

