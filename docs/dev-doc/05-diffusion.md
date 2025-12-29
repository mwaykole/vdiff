# Diffusion Generation Algorithm

This document explains how diffusion language models generate text, step by step.

## The Big Idea

Traditional AI writes **one word at a time**:

```
"The" → "The cat" → "The cat sat" → "The cat sat on" → ...
```

Diffusion AI starts with **blanks** and fills them in:

```
"[?] [?] [?] [?]" → "The [?] sat [?]" → "The cat sat on"
```

## Visual Comparison

```mermaid
flowchart TB
    subgraph AR["Autoregressive (Traditional)"]
        direction LR
        A1["The"] --> A2["cat"] --> A3["sat"] --> A4["on"]
    end
    
    subgraph Diff["Diffusion (dfastllm)"]
        direction TB
        D1["[M] [M] [M] [M]"]
        D2["The [M] sat [M]"]
        D3["The cat sat on"]
        
        D1 -->|Step 1| D2
        D2 -->|Step 2| D3
    end
    
    style AR fill:#e8f5e9
    style Diff fill:#fff3e0
```

## The Algorithm

### Step-by-Step Overview

```mermaid
flowchart TB
    subgraph Algorithm["Diffusion Generation Algorithm"]
        subgraph Step1["Step 1: Initialize"]
            S1["Create sequence with MASK tokens"]
        end
        
        subgraph Step2["Step 2: Repeat"]
            S2A["Forward pass through model"]
            S2B["Get predictions for each position"]
            S2C["Calculate confidence scores"]
            S2D["Select highest confidence positions"]
            S2E["Unmask selected tokens"]
            S2F{"All<br/>unmasked?"}
        end
        
        subgraph Step3["Step 3: Done"]
            S3["Return generated text"]
        end
    end
    
    Step1 --> S2A
    S2A --> S2B --> S2C --> S2D --> S2E --> S2F
    S2F -->|No| S2A
    S2F -->|Yes| Step3
```

## Detailed Walkthrough

Let's generate "How are you doing" from prompt "Hello":

### Step 1: Initialize

```mermaid
flowchart LR
    subgraph Input["Input"]
        PROMPT["Hello"]
        GEN_LEN["Generate 4 tokens"]
    end
    
    subgraph Output["Initial Sequence"]
        SEQ["Hello [M] [M] [M] [M]"]
    end
    
    Input --> Output
```

**What happens:**
- Prompt is tokenized: `"Hello"` → `[15496]`
- Add MASK tokens: `[15496, 126336, 126336, 126336, 126336]`
- `126336` is the special MASK token ID

### Step 2: Forward Pass

```mermaid
flowchart TB
    subgraph Forward["Model Forward Pass"]
        INPUT["Input tokens:<br/>[Hello, M, M, M, M]"]
        
        subgraph Model["Transformer Model"]
            EMB["Embedding Layer"]
            TF1["Transformer Layer 1"]
            TF2["Transformer Layer 2"]
            TFN["... Layer N"]
            HEAD["LM Head"]
        end
        
        OUTPUT["Output logits:<br/>Probability for each vocab word<br/>at each position"]
    end
    
    INPUT --> EMB --> TF1 --> TF2 --> TFN --> HEAD --> OUTPUT
```

**What the model outputs:**

For each position, the model predicts probabilities:

```
Position 1 (MASK): "How"=0.3, "What"=0.2, "Are"=0.1, ...
Position 2 (MASK): "are"=0.4, "is"=0.2, "do"=0.1, ...
Position 3 (MASK): "you"=0.5, "they"=0.2, "we"=0.1, ...
Position 4 (MASK): "doing"=0.3, "today"=0.2, "?"=0.15, ...
```

### Step 3: Calculate Confidence

```mermaid
flowchart LR
    subgraph Logits["Model Output (Logits)"]
        L1["Pos 1: [0.3, 0.2, 0.1, ...]"]
        L2["Pos 2: [0.4, 0.2, 0.1, ...]"]
        L3["Pos 3: [0.5, 0.2, 0.1, ...]"]
        L4["Pos 4: [0.3, 0.2, 0.15, ...]"]
    end
    
    subgraph Confidence["Confidence Scores"]
        C1["Pos 1: 0.30"]
        C2["Pos 2: 0.40"]
        C3["Pos 3: 0.50 ← Highest"]
        C4["Pos 4: 0.30"]
    end
    
    L1 --> C1
    L2 --> C2
    L3 --> C3
    L4 --> C4
```

**Confidence = maximum probability at each position**

### Step 4: Select and Unmask

```mermaid
flowchart TB
    subgraph Selection["Token Selection"]
        CONF["Confidence scores:<br/>Pos1=0.30, Pos2=0.40, Pos3=0.50, Pos4=0.30"]
        
        K["Select top K positions<br/>(K based on step number)"]
        
        SELECTED["Selected: Position 3 (confidence=0.50)"]
    end
    
    subgraph Unmask["Unmask"]
        BEFORE["Hello [M] [M] [M] [M]"]
        AFTER["Hello [M] [M] you [M]"]
    end
    
    CONF --> K --> SELECTED
    SELECTED --> AFTER
```

### Step 5: Repeat

```mermaid
flowchart TB
    subgraph Iteration["Iteration by Iteration"]
        I0["Start: Hello [M] [M] [M] [M]"]
        I1["Step 1: Hello [M] [M] you [M]"]
        I2["Step 2: Hello [M] are you [M]"]
        I3["Step 3: Hello How are you [M]"]
        I4["Step 4: Hello How are you doing"]
    end
    
    I0 --> I1 --> I2 --> I3 --> I4
```

## The Code

### Main Function: diffusion_generate

```python
def diffusion_generate(
    model,           # The AI model
    prompt,          # Input token IDs [1, prompt_len]
    steps,           # Number of diffusion steps
    gen_length,      # How many tokens to generate
    block_length,    # Block size for semi-AR
    temperature,     # Randomness (0 = deterministic)
    mask_id,         # The MASK token ID
):
    # 1. Setup
    device = prompt.device
    batch_size = 1
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    
    # 2. Create initial sequence with MASKs
    x = torch.full((batch_size, total_len), mask_id, device=device)
    x[:, :prompt_len] = prompt  # Keep prompt unchanged
    
    # 3. Calculate how many tokens to unmask per step
    tokens_per_step = gen_length // steps
    
    # 4. Main diffusion loop
    for step in range(steps):
        # a. Forward pass
        with torch.no_grad():
            logits = model(x).logits  # [batch, seq_len, vocab_size]
        
        # b. Only look at MASK positions
        mask_positions = (x == mask_id)
        
        # c. Calculate confidence (max probability)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [batch, seq_len]
        
        # d. Sample tokens using Gumbel trick
        gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
        sampled = (probs.log() + gumbel).argmax(dim=-1)
        
        # e. Select top-k confident MASK positions
        confidence[~mask_positions] = -float('inf')  # Ignore non-MASK
        num_to_unmask = min(tokens_per_step, mask_positions.sum().item())
        
        # Get indices of top-k confident positions
        _, top_indices = confidence.topk(num_to_unmask)
        
        # f. Unmask selected positions
        for idx in top_indices:
            x[0, idx] = sampled[0, idx]
    
    return x
```

### Key Components Explained

#### 1. Gumbel Sampling

```mermaid
flowchart LR
    subgraph Gumbel["Gumbel-Max Trick"]
        PROBS["Probabilities<br/>[0.3, 0.4, 0.2, 0.1]"]
        GUMBEL["Add Gumbel noise"]
        ARGMAX["Take argmax"]
        SAMPLE["Sampled token"]
    end
    
    PROBS --> GUMBEL --> ARGMAX --> SAMPLE
```

**Why Gumbel sampling?**
- Adds controlled randomness
- Higher probability = more likely to be selected
- But low probability tokens can still win sometimes

```python
# Gumbel noise formula
gumbel = -torch.log(-torch.log(torch.rand(...) + 1e-10) + 1e-10)

# Add to log probabilities and take argmax
sampled = (probs.log() + gumbel).argmax(dim=-1)
```

#### 2. Confidence Calculation

```mermaid
flowchart TB
    subgraph Confidence["Confidence = max(probability)"]
        LOGITS["Logits: [2.1, 1.5, 0.8, ...]"]
        SOFTMAX["Softmax: [0.4, 0.3, 0.15, ...]"]
        MAX["Max: 0.4"]
    end
    
    LOGITS --> SOFTMAX --> MAX
```

```python
# Convert logits to probabilities
probs = F.softmax(logits, dim=-1)

# Confidence = highest probability
confidence = probs.max(dim=-1).values
```

#### 3. Remasking Strategy

```mermaid
flowchart TB
    subgraph Strategies["Remasking Strategies"]
        LC["low_confidence<br/>Unmask highest confidence first"]
        RAND["random<br/>Unmask randomly"]
        
        LC --> |"Most common"| DEFAULT["Default in dfastllm"]
    end
```

**"low_confidence" strategy:**
- At each step, unmask the most confident predictions
- Leave uncertain positions for later refinement

## Block-wise Generation

For efficiency, dfastllm generates in blocks:

```mermaid
flowchart LR
    subgraph Blocks["Block-wise Generation"]
        B1["Block 1<br/>[M][M][M][M]"]
        B2["Block 2<br/>[M][M][M][M]"]
        B3["Block 3<br/>[M][M][M][M]"]
    end
    
    B1 -->|"Generate"| B1F["Block 1 done"]
    B1F --> B2
    B2 -->|"Generate"| B2F["Block 2 done"]
    B2F --> B3
    B3 -->|"Generate"| B3F["All done"]
```

**Why blocks?**
- Attention complexity is O(n²)
- Smaller blocks = faster attention
- Each block can reference previous blocks

## Temperature Effect

```mermaid
flowchart TB
    subgraph Temperature["Temperature Effect"]
        T0["temp=0<br/>Always pick highest prob"]
        T05["temp=0.5<br/>Mostly highest, some variation"]
        T1["temp=1.0<br/>Sample according to distribution"]
        T2["temp=2.0<br/>More random, creative"]
    end
```

| Temperature | Effect |
|-------------|--------|
| 0.0 | Deterministic, always picks highest probability |
| 0.5 | Low randomness |
| 1.0 | Normal sampling |
| 2.0 | High randomness, more creative |

## Example Trace

Let's trace through generating 4 tokens with 4 steps:

```mermaid
flowchart TB
    subgraph Trace["Complete Trace"]
        subgraph S0["Initial State"]
            I0["Prompt: 'Once'"]
            I1["Sequence: 'Once [M] [M] [M] [M]'"]
        end
        
        subgraph S1["Step 1"]
            S1A["Forward pass → logits"]
            S1B["Confidence: [0.3, 0.5, 0.2, 0.4]"]
            S1C["Best: position 2 (0.5)"]
            S1D["Sequence: 'Once [M] upon [M] [M]'"]
        end
        
        subgraph S2["Step 2"]
            S2A["Forward pass → logits"]
            S2B["Confidence: [0.4, -, 0.3, 0.6]"]
            S2C["Best: position 4 (0.6)"]
            S2D["Sequence: 'Once [M] upon a [M]'"]
        end
        
        subgraph S3["Step 3"]
            S3A["Forward pass → logits"]
            S3B["Confidence: [0.7, -, -, 0.4]"]
            S3C["Best: position 1 (0.7)"]
            S3D["Sequence: 'Once a upon a [M]'"]
        end
        
        subgraph S4["Step 4"]
            S4A["Forward pass → logits"]
            S4B["Confidence: [-, -, -, 0.8]"]
            S4C["Last mask"]
            S4D["Sequence: 'Once a upon a time'"]
        end
    end
    
    S0 --> S1 --> S2 --> S3 --> S4
```

## DiffusionSampler Class

```mermaid
classDiagram
    class DiffusionSampler {
        +model: nn.Module
        +tokenizer: Tokenizer
        +config: DiffusionSamplerConfig
        
        +generate(prompt, gen_length) Tensor
        +_forward_pass(x) Tensor
        +_calculate_confidence(logits, mask) Tensor
        +_sample_tokens(logits, temperature) Tensor
        +_select_positions(confidence, k) Tensor
    }
    
    class DiffusionSamplerConfig {
        +steps: int
        +block_length: int
        +temperature: float
        +remasking: str
        +mask_id: int
        +use_float32_gumbel: bool
        +enable_early_stopping: bool
    }
    
    DiffusionSampler --> DiffusionSamplerConfig
```

## Early Stopping

dfastllm can stop early if all tokens are unmasked:

```mermaid
flowchart TB
    subgraph EarlyStopping["Early Stopping Optimization"]
        CHECK["Check: Any MASK tokens left?"]
        YES["Continue diffusion loop"]
        NO["Stop early, return result"]
    end
    
    CHECK -->|Yes| YES --> CHECK
    CHECK -->|No| NO
```

```python
# Early stopping check
if not (x == mask_id).any():
    break  # All tokens unmasked, done!
```

## Performance Optimizations

```mermaid
flowchart TB
    subgraph Optimizations["Optimizations in dfastllm"]
        O1["torch.no_grad()<br/>Disable gradient tracking"]
        O2["torch.compile()<br/>JIT compilation"]
        O3["Float16<br/>Half precision"]
        O4["Flash Attention<br/>Memory efficient"]
        O5["Early stopping<br/>Skip unnecessary steps"]
    end
```

## Summary

```mermaid
flowchart LR
    subgraph Summary["Diffusion Generation Summary"]
        A["Initialize with MASKs"]
        B["Forward pass"]
        C["Calculate confidence"]
        D["Unmask top-k"]
        E["Repeat until done"]
    end
    
    A --> B --> C --> D --> E
    E -->|"Loop"| B
```

| Step | Action |
|------|--------|
| 1 | Create sequence: prompt + MASK tokens |
| 2 | Run model to get predictions |
| 3 | Calculate confidence for each MASK |
| 4 | Unmask highest confidence positions |
| 5 | Repeat until all unmasked |

## Next Steps

👉 [06-apd.md](06-apd.md) - APD optimization for faster generation

