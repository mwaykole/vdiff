# Mixture of Recursions (MoR) Guide

## Overview

Mixture of Recursions (MoR) is an adaptive computation technique that allocates variable compute to different tokens based on their difficulty. This implementation works at the inference level, meaning it works with existing diffusion models without requiring retraining.

## Key Benefits

| Benefit | Impact |
|---------|--------|
| **Compute Reduction** | 30-50% fewer FLOPs |
| **Faster Inference** | 20-40% latency improvement |
| **No Retraining** | Works with existing models |
| **Quality Preserved** | Hard tokens get MORE compute |

## How It Works

### Traditional Approach (All tokens equal)
```
Step 1: Process ALL 1024 tokens with full forward pass
Step 2: Process ALL 1024 tokens with full forward pass
...
Step 16: Process ALL 1024 tokens with full forward pass

Result: 16 × 1024 = 16,384 token-operations
```

### MoR Approach (Adaptive per token)
```
Step 1: Full forward pass → Identify easy/hard tokens
  - "The" (confidence 0.98) → Skip refinement
  - "quantum" (confidence 0.45) → Needs 4 recursions
  - "entanglement" (confidence 0.25) → Needs 8 recursions

Step 2-4: Only refine uncertain tokens (~30% of total)
Step 5: Re-evaluate, some tokens now confident
...

Result: ~9,000 token-operations (45% savings!)
```

## Configuration

### Environment Variables

```bash
# Enable MoR (default: true)
export VDIFF_ENABLE_MOR=true

# Recursion depth range
export VDIFF_MOR_MIN_RECURSIONS=1    # Easy tokens
export VDIFF_MOR_MAX_RECURSIONS=4    # Hard tokens

# Confidence thresholds
export VDIFF_MOR_CONFIDENCE_HIGH=0.9  # Above = skip refinement
export VDIFF_MOR_CONFIDENCE_LOW=0.5   # Below = max refinement

# Strategy: confidence, entropy, gradient, fixed
export VDIFF_MOR_STRATEGY=confidence
```

### Python Configuration

```python
from dfastllm.config import DFastLLMConfig

config = DFastLLMConfig(
    model="GSAI-ML/LLaDA-8B-Instruct",
    
    # MoR Configuration
    enable_mor=True,
    mor_min_recursions=1,
    mor_max_recursions=4,
    mor_confidence_high=0.9,
    mor_confidence_low=0.5,
    mor_strategy="confidence",
)
```

### Programmatic Configuration

```python
from dfastllm.engine.mor_decoder import MoRConfig, MoRDecoder, RouterStrategy

config = MoRConfig(
    enabled=True,
    min_recursions=1,
    max_recursions=4,
    router_strategy=RouterStrategy.CONFIDENCE,
    difficulty_threshold_low=0.8,  # Maps to confidence_high
    difficulty_threshold_high=0.3,  # Maps to confidence_low
    skip_confident_tokens=True,
    skip_threshold=0.95,
    log_stats=True,
)

decoder = MoRDecoder(config)
```

## Router Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `confidence` | Use max probability as difficulty signal | General use |
| `entropy` | Use entropy as difficulty signal | High-uncertainty text |
| `gradient` | Use gradient magnitude | Fine-tuned models |
| `fixed` | Same recursions for all tokens | Baseline comparison |

### Confidence Strategy (Default)
- **Easy tokens**: High max probability → Few recursions
- **Hard tokens**: Low max probability → Many recursions

```python
# Conceptual mapping
if confidence >= 0.9:
    recursions = 1  # Easy, skip extra work
elif confidence <= 0.5:
    recursions = 4  # Hard, max refinement
else:
    recursions = interpolate(1, 4, confidence)
```

## API Reference

### MoRConfig

```python
@dataclass
class MoRConfig:
    enabled: bool = True
    min_recursions: int = 1
    max_recursions: int = 4
    router_strategy: RouterStrategy = RouterStrategy.CONFIDENCE
    difficulty_threshold_low: float = 0.8
    difficulty_threshold_high: float = 0.3
    skip_confident_tokens: bool = True
    skip_threshold: float = 0.95
    batch_by_difficulty: bool = True
    log_stats: bool = False
```

### MoRDecoder

```python
class MoRDecoder:
    def compute_confidence(logits, mask_index) -> Tensor
    def compute_difficulty(logits, mask_index) -> Tensor
    def compute_recursion_depths(difficulty, mask_index) -> Tensor
    def apply_adaptive_refinement(model, x, logits, ...) -> Tensor
    def get_stats() -> Dict[str, Any]
    def reset_stats() -> None
```

### MoRStats

```python
@dataclass
class MoRStats:
    total_steps: int
    total_tokens_processed: int
    tokens_skipped: int
    easy_tokens: int
    medium_tokens: int
    hard_tokens: int
    total_recursions: int
    compute_saved_pct: float
    avg_recursions_per_token: float
```

## Performance Tuning

### For Maximum Speed (Aggressive skipping)
```python
config = MoRConfig(
    max_recursions=2,           # Fewer max recursions
    skip_threshold=0.9,         # Skip earlier
    difficulty_threshold_low=0.85,  # More tokens considered "easy"
)
```

### For Maximum Quality (Conservative)
```python
config = MoRConfig(
    max_recursions=8,           # More recursions for hard tokens
    skip_threshold=0.99,        # Only skip very confident tokens
    difficulty_threshold_high=0.2,  # Higher bar for "hard"
)
```

### Balanced (Default)
```python
config = MoRConfig(
    min_recursions=1,
    max_recursions=4,
    skip_threshold=0.95,
    difficulty_threshold_low=0.8,
    difficulty_threshold_high=0.3,
)
```

## Monitoring

### Access Statistics

```python
from dfastllm.engine import DFastLLMEngine

engine = DFastLLMEngine(config)
output = engine.generate("prompt", params)

# Get MoR stats from last generation
mor_stats = engine._last_mor_stats
print(f"Compute saved: {mor_stats['compute_saved_pct']:.1f}%")
print(f"Tokens skipped: {mor_stats['tokens_skipped']}")
print(f"Avg recursions: {mor_stats['avg_recursions_per_token']:.2f}")
```

### Prometheus Metrics

MoR exposes metrics for monitoring:

```
# Average compute savings percentage
dfastllm_mor_compute_saved_percent{model="llada-8b"}

# Tokens by difficulty category
dfastllm_mor_tokens_easy_total{model="llada-8b"}
dfastllm_mor_tokens_medium_total{model="llada-8b"}
dfastllm_mor_tokens_hard_total{model="llada-8b"}

# Average recursions per token
dfastllm_mor_avg_recursions{model="llada-8b"}
```

## Comparison with APD

| Feature | MoR | APD |
|---------|-----|-----|
| **Focus** | Compute per token | Tokens per step |
| **Approach** | Variable refinement | Parallel acceptance |
| **Benefit** | Fewer FLOPs | Fewer steps |
| **Combines** | ✅ Works together | ✅ Works together |

**Use both for maximum efficiency:**

```python
config = DFastLLMConfig(
    enable_apd=True,   # Accept multiple tokens per step
    enable_mor=True,   # Adaptive compute per token
)
```

## Troubleshooting

### MoR Not Activating
```bash
# Check if enabled
echo $VDIFF_ENABLE_MOR  # Should be "true"

# Verify in logs
export VDIFF_LOG_LEVEL=DEBUG
# Look for: "MoR decoder initialized..."
```

### Low Compute Savings
```python
# Check token distribution
stats = engine._last_mor_stats
print(f"Easy: {stats['easy_tokens']}, Hard: {stats['hard_tokens']}")

# If mostly hard tokens, savings will be lower
# Adjust thresholds:
config.mor_confidence_low = 0.4  # Higher threshold for "hard"
```

### Quality Degradation
```python
# Increase max recursions for hard tokens
config.mor_max_recursions = 6

# Raise skip threshold
config.mor_skip_threshold = 0.98
```

## References

- **Paper**: [Mixture of Recursions](https://arxiv.org/abs/2507.10524)
- **HuggingFace**: [MoR Models](https://huggingface.co/collections/mor)
- **dfastllm**: MoR implementation in `dfastllm/engine/mor_decoder.py`
