# Basic Concepts

This document explains fundamental concepts you need to understand dfastllm. **No prior knowledge required.**

## What is an AI Language Model?

An AI Language Model is a computer program that can:
1. **Understand text** - Know what words mean
2. **Generate text** - Write new sentences

```mermaid
flowchart LR
    INPUT["Input: 'The sky is'"] --> MODEL[AI Model]
    MODEL --> OUTPUT["Output: 'blue and beautiful'"]
```

### Real Example

```
You type:    "Write a poem about cats"
AI responds: "Soft paws padding through the night,
              Whiskers twitching in moonlight..."
```

## How Does Text Become Numbers?

Computers don't understand text - they only understand numbers. So we need to convert:

```mermaid
flowchart LR
    subgraph Step1["Step 1: Text"]
        TEXT["Hello world"]
    end
    
    subgraph Step2["Step 2: Tokens"]
        TOK["['Hello', ' world']"]
    end
    
    subgraph Step3["Step 3: Numbers"]
        NUM["[15496, 995]"]
    end
    
    TEXT --> TOK --> NUM
```

### What is a Token?

A **token** is a piece of text. It can be:
- A whole word: `"hello"` → 1 token
- Part of a word: `"running"` → `["run", "ning"]` → 2 tokens
- A symbol: `"!"` → 1 token

```mermaid
flowchart TB
    subgraph Example["Tokenization Example"]
        SENTENCE["I love programming!"]
        T1["I"]
        T2["love"]
        T3["program"]
        T4["ming"]
        T5["!"]
    end
    
    SENTENCE --> T1
    SENTENCE --> T2
    SENTENCE --> T3
    SENTENCE --> T4
    SENTENCE --> T5
```

Each token has a unique number (ID):
- `"I"` = 40
- `"love"` = 2751
- `"program"` = 4923
- `"ming"` = 1723
- `"!"` = 0

## Two Types of Language Models

### Type 1: Autoregressive Models (Traditional)

These generate text **one word at a time**, from left to right.

```mermaid
flowchart LR
    subgraph AR["Autoregressive Generation"]
        direction LR
        S1["The"] 
        S2["The cat"]
        S3["The cat sat"]
        S4["The cat sat on"]
        S5["The cat sat on the"]
        S6["The cat sat on the mat"]
    end
    
    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

**Examples**: GPT-4, LLaMA, Mistral, Claude

**How it works**:
1. Look at all previous words
2. Predict the next word
3. Add that word
4. Repeat

### Type 2: Diffusion Models (New)

These generate text by **filling in blanks** simultaneously.

```mermaid
flowchart TB
    subgraph Diffusion["Diffusion Generation"]
        D1["[M] [M] [M] [M] [M] [M]"]
        D2["The [M] [M] on [M] [M]"]
        D3["The cat sat on the [M]"]
        D4["The cat sat on the mat"]
    end
    
    D1 -->|"Step 1"| D2
    D2 -->|"Step 2"| D3
    D3 -->|"Step 3"| D4
```

**Examples**: LLaDA, Dream

**How it works**:
1. Start with all `[MASK]` tokens (blanks)
2. Look at everything, predict what each blank should be
3. Fill in the most confident predictions
4. Repeat until no blanks left

## Why Diffusion Models?

```mermaid
flowchart TB
    subgraph Benefits["Diffusion Model Benefits"]
        B1["🚀 Can fill multiple blanks at once"]
        B2["🎯 Can change earlier words based on later context"]
        B3["📝 Better for some creative tasks"]
    end
```

| Aspect | Autoregressive | Diffusion |
|--------|---------------|-----------|
| Generation | One token at a time | Multiple tokens at once |
| Speed | Slower | Can be faster with APD |
| Flexibility | Can't go back | Can revise any position |
| Examples | GPT, LLaMA | LLaDA, Dream |

## What is a Server?

A **server** is a program that:
1. **Listens** for requests from other programs
2. **Processes** those requests
3. **Responds** with results

```mermaid
flowchart LR
    subgraph Clients["Clients (Your Apps)"]
        C1[Website]
        C2[Mobile App]
        C3[Python Script]
    end
    
    subgraph Server["dfastllm Server"]
        API[Receive Request]
        PROCESS[Run AI Model]
        RESPOND[Send Response]
    end
    
    C1 --> API
    C2 --> API
    C3 --> API
    API --> PROCESS --> RESPOND
    RESPOND --> C1
    RESPOND --> C2
    RESPOND --> C3
```

## What is an API?

**API** = Application Programming Interface

It's a set of rules for how programs talk to each other.

```mermaid
flowchart TB
    subgraph API["API Endpoint"]
        URL["POST /v1/completions"]
        INPUT["Input: JSON with prompt"]
        OUTPUT["Output: JSON with response"]
    end
    
    REQUEST["Your Request"] --> URL
    URL --> INPUT
    INPUT --> OUTPUT
    OUTPUT --> RESPONSE["AI Response"]
```

### Example API Call

**You send:**
```json
{
  "model": "llada-8b",
  "prompt": "Hello, how are",
  "max_tokens": 10
}
```

**Server responds:**
```json
{
  "choices": [
    {
      "text": " you doing today?"
    }
  ]
}
```

## What is OpenAI Compatibility?

OpenAI (creators of ChatGPT) defined a standard API format. Many tools use it:

```mermaid
flowchart TB
    subgraph Standard["OpenAI API Standard"]
        E1["/v1/completions"]
        E2["/v1/chat/completions"]
        E3["/v1/models"]
    end
    
    subgraph Servers["Servers that support it"]
        S1["OpenAI"]
        S2["vLLM"]
        S3["dfastllm ✓"]
        S4["TGI"]
    end
    
    Standard --> S1
    Standard --> S2
    Standard --> S3
    Standard --> S4
```

**Why it matters**: If your app works with OpenAI, it works with dfastllm!

## What is GPU vs CPU?

```mermaid
flowchart TB
    subgraph CPU["CPU (Central Processing Unit)"]
        C1["Good at complex tasks"]
        C2["Few cores (4-16)"]
        C3["Sequential processing"]
    end
    
    subgraph GPU["GPU (Graphics Processing Unit)"]
        G1["Good at simple parallel tasks"]
        G2["Many cores (1000s)"]
        G3["Parallel processing"]
    end
    
    subgraph AI["AI Models need"]
        A1["Matrix multiplication"]
        A2["Same operation on many numbers"]
        A3["→ GPU is 10-100x faster!"]
    end
```

| | CPU | GPU |
|--|-----|-----|
| Speed for AI | Slow | Fast |
| Cost | Included | Extra hardware |
| Power | Low | High |
| dfastllm support | ✅ Yes | ✅ Yes |

## What is Kubernetes?

Kubernetes (K8s) is a system that manages containers (packaged applications):

```mermaid
flowchart TB
    subgraph K8s["Kubernetes Cluster"]
        subgraph Node1["Server 1"]
            P1[dfastllm Pod]
            P2[dfastllm Pod]
        end
        
        subgraph Node2["Server 2"]
            P3[dfastllm Pod]
        end
        
        LB[Load Balancer]
    end
    
    USER[Users] --> LB
    LB --> P1
    LB --> P2
    LB --> P3
```

**Why use it?**
- Automatic scaling
- Self-healing (restart crashed apps)
- Load balancing

## What is KServe?

**KServe** is a standard for serving ML models on Kubernetes.

It provides:
- **ServingRuntime**: Defines how to run inference servers
- **InferenceService**: Deploys a specific model

```mermaid
flowchart TB
    subgraph KServe["KServe (Model Serving)"]
        RT[ServingRuntime]
        IS[InferenceService]
        
        subgraph Models["Your Models"]
            M1["LLaDA-8B"]
            M2["Dream"]
        end
    end
    
    RT --> IS
    IS --> M1
    IS --> M2
```

## Summary

```mermaid
flowchart TB
    subgraph Concepts["Key Concepts"]
        LLM["LLM = AI that generates text"]
        TOKEN["Token = Piece of text as number"]
        DIFF["Diffusion = Fill blanks approach"]
        API["API = How programs communicate"]
        GPU["GPU = Fast AI hardware"]
    end
    
    subgraph dfastllm["dfastllm combines all"]
        V["Server for Diffusion LLMs<br/>with OpenAI API<br/>on GPU/CPU"]
    end
    
    LLM --> V
    TOKEN --> V
    DIFF --> V
    API --> V
    GPU --> V
```

## Next Steps

Now that you understand the basics, continue to:

👉 [02-architecture.md](02-architecture.md) - See how dfastllm is built

