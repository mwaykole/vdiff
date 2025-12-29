# dfastllm Developer Documentation

Welcome to the dfastllm developer documentation. This guide explains **everything** about the dfastllm codebase, from basic concepts to implementation details.

## What is dfastllm?

**dfastllm** is a server that runs AI language models (like ChatGPT) and lets other programs talk to them through a web API.

```mermaid
flowchart LR
    subgraph You["Your Application"]
        APP[Your Code]
    end
    
    subgraph dfastllm["dfastllm Server"]
        API[API Server]
        ENGINE[Engine]
        MODEL[AI Model]
    end
    
    APP -->|"Send: 'Hello!'"| API
    API --> ENGINE
    ENGINE --> MODEL
    MODEL -->|"Generate text"| ENGINE
    ENGINE --> API
    API -->|"Response: 'Hi there!'"| APP
```

### What makes dfastllm special?

dfastllm is designed for **Diffusion Language Models** - a new type of AI that generates text differently than traditional models.

```mermaid
flowchart TB
    subgraph Traditional["Traditional AI (GPT, LLaMA)"]
        direction LR
        T1["The"] --> T2["The cat"] --> T3["The cat sat"] --> T4["The cat sat on"]
    end
    
    subgraph Diffusion["Diffusion AI (LLaDA, Dream)"]
        direction LR
        D1["[?] [?] [?] [?]"] --> D2["The [?] sat [?]"] --> D3["The cat sat on"]
    end
    
    style Traditional fill:#e8f5e9
    style Diffusion fill:#fff3e0
```

**Traditional AI**: Writes one word at a time, left to right.
**Diffusion AI**: Starts with blanks, fills in multiple words at once.

## Documentation Structure

```
docs/dev-doc/
├── index.md              ← You are here
├── 01-concepts.md        ← Basic concepts (start here if new)
├── 02-architecture.md    ← System overview
├── 03-project-structure.md ← File/folder organization
├── 04-engine.md          ← Core engine explained
├── 05-diffusion.md       ← Diffusion algorithm
├── 06-apd.md             ← APD optimization
├── 07-api-server.md      ← API endpoints
├── 08-config.md          ← Configuration
├── 09-deployment.md      ← How to deploy
└── 10-code-walkthrough.md ← Line-by-line code
```

## Quick Start Reading Guide

### If you're completely new:
1. Start with [01-concepts.md](01-concepts.md) - Understand what LLMs are
2. Then [02-architecture.md](02-architecture.md) - See the big picture
3. Then [03-project-structure.md](03-project-structure.md) - Know where things are

### If you want to understand the code:
1. [04-engine.md](04-engine.md) - The brain of dfastllm
2. [05-diffusion.md](05-diffusion.md) - How text is generated
3. [10-code-walkthrough.md](10-code-walkthrough.md) - Code explanations

### If you want to deploy:
1. [08-config.md](08-config.md) - Configuration options
2. [09-deployment.md](09-deployment.md) - Deployment guides

## The Complete Picture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        USER[User/Application]
        SDK[OpenAI SDK / curl]
    end
    
    subgraph Server["API Server Layer"]
        FASTAPI[FastAPI Server]
        MIDDLEWARE[Middleware<br/>Rate Limit, Auth, Logging]
        ROUTES[API Routes<br/>/v1/completions<br/>/v1/chat/completions]
    end
    
    subgraph Engine["Engine Layer"]
        VDIFF[DFastLLMEngine]
        QUEUE[Request Queue]
        TOKENIZER[Tokenizer]
    end
    
    subgraph Generation["Generation Layer"]
        DIFFUSION[Diffusion Sampler]
        APD[APD Decoder]
    end
    
    subgraph Model["Model Layer"]
        HF[HuggingFace Model]
        GPU[GPU/CPU]
    end
    
    subgraph Output["Output Layer"]
        DECODE[Decode Tokens]
        RESPONSE[Format Response]
    end
    
    USER --> SDK
    SDK --> FASTAPI
    FASTAPI --> MIDDLEWARE
    MIDDLEWARE --> ROUTES
    ROUTES --> VDIFF
    VDIFF --> QUEUE
    QUEUE --> TOKENIZER
    TOKENIZER --> DIFFUSION
    TOKENIZER --> APD
    DIFFUSION --> HF
    APD --> HF
    HF --> GPU
    GPU --> DECODE
    DECODE --> RESPONSE
    RESPONSE --> USER
    
    style Client fill:#e3f2fd
    style Server fill:#f3e5f5
    style Engine fill:#fff3e0
    style Generation fill:#e8f5e9
    style Model fill:#fce4ec
    style Output fill:#e0f2f1
```

## Key Terms Glossary

| Term | Meaning |
|------|---------|
| **LLM** | Large Language Model - An AI that understands and generates text |
| **Token** | A piece of text (word or part of word) that the AI processes |
| **Prompt** | The input text you give to the AI |
| **Completion** | The text the AI generates in response |
| **Diffusion** | A method of generating by starting with noise and refining |
| **Mask** | A placeholder token `[MASK]` that gets replaced with real words |
| **APD** | Adaptive Parallel Decoding - Generates multiple tokens at once |
| **Inference** | The process of running the AI model to generate output |
| **Endpoint** | A URL path like `/v1/completions` that accepts requests |
| **Engine** | The core component that manages model and generation |

## Next Steps

👉 **New to AI/ML?** Start with [01-concepts.md](01-concepts.md)

👉 **Know AI basics?** Jump to [02-architecture.md](02-architecture.md)

👉 **Want to dive into code?** Go to [10-code-walkthrough.md](10-code-walkthrough.md)

