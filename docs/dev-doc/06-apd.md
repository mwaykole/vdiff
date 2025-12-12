# APD: Adaptive Parallel Decoding

This document explains APD - an optimization that makes diffusion generation **2-4x faster**.

## What is APD?

**APD** = Adaptive Parallel Decoding

Standard diffusion unmasks a fixed number of tokens per step. APD is smarter:
- Unmask **more tokens** when confident
- Unmask **fewer tokens** when uncertain

```mermaid
flowchart LR
    subgraph Standard["Standard Diffusion"]
        S1["Always unmask K tokens per step"]
        S2["K is fixed (e.g., 1 or 2)"]
    end
    
    subgraph APD["APD"]
        A1["Unmask N tokens where N varies"]
        A2["High confidence → more tokens"]
        A3["Low confidence → fewer tokens"]
    end
    
    Standard --> SLOW["Slower"]
    APD --> FAST["2-4x Faster"]
```

## Visual Comparison

### Standard Diffusion (4 steps for 4 tokens)

```mermaid
flowchart TB
    subgraph Standard["Standard: 4 Steps"]
        S1["[M] [M] [M] [M]"]
        S2["The [M] [M] [M]"]
        S3["The cat [M] [M]"]
        S4["The cat sat [M]"]
        S5["The cat sat down"]
    end
    
    S1 -->|"Step 1"| S2
    S2 -->|"Step 2"| S3
    S3 -->|"Step 3"| S4
    S4 -->|"Step 4"| S5
```

### APD (2 steps for 4 tokens)

```mermaid
flowchart TB
    subgraph APD["APD: 2 Steps"]
        A1["[M] [M] [M] [M]"]
        A2["The cat [M] [M]"]
        A3["The cat sat down"]
    end
    
    A1 -->|"Step 1: Unmask 2"| A2
    A2 -->|"Step 2: Unmask 2"| A3
```

**Result: Same output, half the steps!**

## How APD Works

```mermaid
flowchart TB
    subgraph APD["APD Algorithm"]
        subgraph Step1["Step 1: Generate Candidates"]
            C1["Get predictions for ALL mask positions"]
            C2["Calculate confidence for each"]
        end
        
        subgraph Step2["Step 2: Adaptive Selection"]
            C3["Sort by confidence"]
            C4["Accept all above threshold"]
            C5["Adjust threshold adaptively"]
        end
        
        subgraph Step3["Step 3: Parallel Unmask"]
            C6["Unmask all accepted tokens<br/>in ONE step"]
        end
    end
    
    Step1 --> Step2 --> Step3
```

## The Confidence Threshold

APD uses a **threshold** to decide which tokens to accept:

```mermaid
flowchart TB
    subgraph Threshold["Threshold Decision"]
        CONF["Token confidence: 0.75"]
        THRESH["Threshold: 0.50"]
        
        COMPARE{"confidence ><br/>threshold?"}
        
        ACCEPT["✓ Accept and unmask"]
        REJECT["✗ Keep masked for later"]
    end
    
    CONF --> COMPARE
    THRESH --> COMPARE
    COMPARE -->|"0.75 > 0.50"| ACCEPT
    COMPARE -->|"No"| REJECT
```

### Adaptive Threshold

The threshold adapts based on results:

```mermaid
flowchart TB
    subgraph Adaptive["Adaptive Threshold"]
        HIGH["Too many accepted?"]
        LOW["Too few accepted?"]
        
        HIGH -->|"Yes"| RAISE["Raise threshold<br/>(be stricter)"]
        LOW -->|"Yes"| LOWER["Lower threshold<br/>(be lenient)"]
    end
```

## APD Code Walkthrough

### APDDecoder Class

```mermaid
classDiagram
    class APDDecoder {
        +config: APDConfig
        -_stats: APDStats
        
        +generate(model, prompt, gen_length, steps, mask_id) Tensor
        +_compute_confidence(logits) Tensor
        +_select_parallel_tokens(confidence, threshold, max_parallel) Tensor
        +_update_threshold(accepted_ratio) float
        +get_stats() dict
    }
    
    class APDConfig {
        +enabled: bool
        +max_parallel_tokens: int
        +acceptance_threshold: float
        +min_threshold: float
        +max_threshold: float
        +adaptation_rate: float
        +temperature: float
    }
    
    class APDStats {
        +total_tokens: int
        +total_steps: int
        +avg_tokens_per_step: float
        +acceptance_rates: List
    }
    
    APDDecoder --> APDConfig
    APDDecoder --> APDStats
```

### Main Generate Function

```python
def generate(
    self,
    model,          # The AI model
    prompt,         # Input tokens
    gen_length,     # How many tokens to generate
    steps,          # Max number of steps
    mask_id,        # MASK token ID
    temperature,    # Sampling temperature
) -> torch.Tensor:
    
    # 1. Initialize
    x = self._initialize_sequence(prompt, gen_length, mask_id)
    threshold = self.config.acceptance_threshold
    
    # 2. Main loop
    for step in range(steps):
        # a. Get mask positions
        mask_positions = (x == mask_id)
        if not mask_positions.any():
            break  # Early stop: all done!
        
        # b. Forward pass
        logits = model(x).logits
        
        # c. Calculate confidence
        confidence = self._compute_confidence(logits)
        
        # d. Select tokens above threshold
        selected = self._select_parallel_tokens(
            confidence,
            threshold,
            self.config.max_parallel_tokens,
            mask_positions,
        )
        
        # e. Sample and unmask
        sampled = self._sample_tokens(logits, temperature)
        x[selected] = sampled[selected]
        
        # f. Update threshold adaptively
        accepted_ratio = selected.sum() / mask_positions.sum()
        threshold = self._update_threshold(accepted_ratio, threshold)
        
        # g. Track stats
        self._stats.update(selected.sum().item())
    
    return x
```

## Parallel Token Selection

```mermaid
flowchart TB
    subgraph Selection["Parallel Token Selection"]
        subgraph Input["Input"]
            CONF["Confidence scores:<br/>[0.8, 0.3, 0.9, 0.4, 0.7]"]
            THRESH["Threshold: 0.5"]
            MAX["Max parallel: 3"]
        end
        
        subgraph Process["Process"]
            FILTER["Filter above threshold:<br/>[0.8, 0.9, 0.7]"]
            SORT["Sort descending:<br/>[0.9, 0.8, 0.7]"]
            LIMIT["Limit to max:<br/>[0.9, 0.8, 0.7]"]
        end
        
        subgraph Output["Output"]
            SELECTED["Selected positions: [2, 0, 4]"]
        end
    end
    
    Input --> Process --> Output
```

### Code

```python
def _select_parallel_tokens(
    self,
    confidence: torch.Tensor,     # [seq_len]
    threshold: float,             # e.g., 0.5
    max_parallel: int,            # e.g., 8
    mask_positions: torch.Tensor, # Boolean mask
) -> torch.Tensor:
    
    # Only consider MASK positions
    masked_conf = confidence.clone()
    masked_conf[~mask_positions] = -float('inf')
    
    # Filter by threshold
    above_threshold = masked_conf >= threshold
    
    # Limit to max_parallel
    if above_threshold.sum() > max_parallel:
        # Take top-k by confidence
        _, indices = masked_conf.topk(max_parallel)
        selected = torch.zeros_like(mask_positions)
        selected[indices] = True
    else:
        selected = above_threshold
    
    return selected
```

## Threshold Adaptation

```mermaid
flowchart TB
    subgraph Adaptation["Threshold Adaptation"]
        subgraph Measure["Measure"]
            RATIO["Acceptance ratio = accepted / total_masks"]
        end
        
        subgraph Adjust["Adjust"]
            HIGH{"ratio > 0.7?"}
            LOW{"ratio < 0.3?"}
            
            RAISE["threshold += adaptation_rate"]
            LOWER["threshold -= adaptation_rate"]
            KEEP["Keep threshold same"]
        end
        
        subgraph Clamp["Clamp"]
            MIN["min_threshold = 0.1"]
            MAX["max_threshold = 0.9"]
        end
    end
    
    RATIO --> HIGH
    HIGH -->|Yes| RAISE
    HIGH -->|No| LOW
    LOW -->|Yes| LOWER
    LOW -->|No| KEEP
    
    RAISE --> Clamp
    LOWER --> Clamp
    KEEP --> Clamp
```

### Code

```python
def _update_threshold(
    self,
    accepted_ratio: float,
    current_threshold: float,
) -> float:
    target_ratio = 0.5  # Aim for 50% acceptance
    
    if accepted_ratio > target_ratio + 0.2:
        # Too many accepted, be stricter
        new_threshold = current_threshold + self.config.adaptation_rate
    elif accepted_ratio < target_ratio - 0.2:
        # Too few accepted, be lenient
        new_threshold = current_threshold - self.config.adaptation_rate
    else:
        new_threshold = current_threshold
    
    # Clamp to valid range
    return max(
        self.config.min_threshold,
        min(self.config.max_threshold, new_threshold)
    )
```

## APD Configuration

```mermaid
flowchart LR
    subgraph Config["APDConfig Options"]
        OPT1["enabled: bool<br/>Turn APD on/off"]
        OPT2["max_parallel_tokens: int<br/>Max tokens per step (8)"]
        OPT3["acceptance_threshold: float<br/>Initial threshold (0.3)"]
        OPT4["min_threshold: float<br/>Minimum allowed (0.1)"]
        OPT5["max_threshold: float<br/>Maximum allowed (0.9)"]
        OPT6["adaptation_rate: float<br/>How fast to adapt (0.05)"]
    end
```

### CLI Usage

```bash
# Enable APD (default)
vdiff --model llada --enable-apd

# Disable APD
vdiff --model llada --disable-apd

# Custom APD settings
vdiff --model llada \
    --apd-max-parallel 16 \
    --apd-threshold 0.2
```

## Performance Comparison

```mermaid
flowchart TB
    subgraph Performance["Performance Comparison"]
        subgraph Standard["Standard Diffusion"]
            S1["64 tokens"]
            S2["64 steps"]
            S3["64 forward passes"]
        end
        
        subgraph APD["With APD"]
            A1["64 tokens"]
            A2["~16-20 steps"]
            A3["~16-20 forward passes"]
        end
    end
    
    Standard --> SLOW["Slower"]
    APD --> FAST["2-4x Faster!"]
```

### Actual Numbers

| Metric | Standard | APD | Improvement |
|--------|----------|-----|-------------|
| Steps for 64 tokens | 64 | ~20 | 3.2x fewer |
| Time per generation | 1000ms | ~300ms | 3.3x faster |
| Tokens per step | 1 | ~3.2 | 3.2x more |

## APD Statistics

Track performance with stats:

```mermaid
flowchart LR
    subgraph Stats["APDStats"]
        S1["total_tokens: 1000"]
        S2["total_steps: 312"]
        S3["avg_tokens_per_step: 3.2"]
        S4["acceptance_rates: [0.4, 0.6, ...]"]
    end
```

### Get Stats

```python
# In VDiffEngine
stats = engine.get_stats()

# APD-specific stats
apd_stats = stats.get("apd", {})
print(f"Avg tokens/step: {apd_stats['avg_tokens_per_step']}")
```

## When APD Helps Most

```mermaid
flowchart TB
    subgraph BestCases["APD Works Best When"]
        B1["Long generations<br/>(more tokens = more savings)"]
        B2["High-confidence predictions<br/>(model is sure)"]
        B3["Repetitive patterns<br/>(easy predictions)"]
    end
    
    subgraph WorstCases["APD Helps Less When"]
        W1["Short generations<br/>(overhead not worth it)"]
        W2["Low-confidence predictions<br/>(can't parallelize)"]
        W3["Creative/random text<br/>(hard predictions)"]
    end
```

## Complete APD Flow

```mermaid
sequenceDiagram
    participant Engine
    participant APD as APDDecoder
    participant Model
    
    Engine->>APD: generate(prompt, gen_length)
    APD->>APD: Initialize with MASKs
    
    loop Until all unmasked
        APD->>Model: Forward pass
        Model-->>APD: Logits
        
        APD->>APD: Calculate confidence
        APD->>APD: Select tokens above threshold
        APD->>APD: Unmask selected (parallel)
        APD->>APD: Update threshold
        
        alt All unmasked
            APD->>APD: Early stop
        end
    end
    
    APD-->>Engine: Generated tokens
```

## Summary

```mermaid
flowchart LR
    subgraph APDSummary["APD Summary"]
        A["Adaptive threshold"]
        B["Parallel unmasking"]
        C["2-4x speedup"]
    end
    
    A --> B --> C
```

| Concept | Description |
|---------|-------------|
| **Idea** | Unmask multiple tokens when confident |
| **Threshold** | Minimum confidence to accept |
| **Adaptive** | Threshold adjusts automatically |
| **Max Parallel** | Limit on simultaneous unmasks |
| **Result** | 2-4x faster generation |

## Next Steps

👉 [07-api-server.md](07-api-server.md) - How the API server works

