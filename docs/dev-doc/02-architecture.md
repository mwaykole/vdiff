# System Architecture

This document explains how dfastllm is built and how all the pieces fit together.

## High-Level Overview

dfastllm has **5 main layers**:

```mermaid
flowchart TB
    subgraph L1["Layer 1: Client"]
        CLIENT[Your Application]
    end
    
    subgraph L2["Layer 2: API Server"]
        FASTAPI[FastAPI Application]
    end
    
    subgraph L3["Layer 3: Engine"]
        ENGINE[DFastLLMEngine]
    end
    
    subgraph L4["Layer 4: Generation"]
        GEN[Diffusion / APD]
    end
    
    subgraph L5["Layer 5: Model"]
        MODEL[HuggingFace Model]
    end
    
    L1 --> L2 --> L3 --> L4 --> L5
    L5 --> L4 --> L3 --> L2 --> L1
    
    style L1 fill:#e3f2fd
    style L2 fill:#f3e5f5
    style L3 fill:#fff3e0
    style L4 fill:#e8f5e9
    style L5 fill:#fce4ec
```

## Detailed Architecture

```mermaid
flowchart TB
    subgraph Clients["Client Layer"]
        C1["Python App<br/>(OpenAI SDK)"]
        C2["curl / HTTP"]
        C3["Web Browser"]
    end
    
    subgraph APIServer["API Server Layer"]
        subgraph Middleware["Middleware Stack"]
            MW1["Request ID"]
            MW2["Logging"]
            MW3["Security Headers"]
            MW4["CORS"]
            MW5["Rate Limiter"]
            MW6["Auth"]
        end
        
        subgraph Routes["API Routes"]
            R1["/health"]
            R2["/v1/models"]
            R3["/v1/completions"]
            R4["/v1/chat/completions"]
            R5["/metrics"]
        end
        
        subgraph Serving["Serving Components"]
            SC["OpenAIServingCompletion"]
            SH["OpenAIServingChat"]
        end
    end
    
    subgraph EngineLayer["Engine Layer"]
        subgraph DFastLLMEngine["DFastLLMEngine"]
            STATE["Engine State"]
            QUEUE["Request Queue"]
            STATS["Statistics"]
        end
        
        TOK["TokenizerWrapper"]
        CONFIG["DFastLLMConfig"]
    end
    
    subgraph GenerationLayer["Generation Layer"]
        DIFF["DiffusionSampler"]
        APD["APDDecoder"]
    end
    
    subgraph ModelLayer["Model Layer"]
        HF["HuggingFace<br/>AutoModelForCausalLM"]
        GPU["GPU / CPU"]
    end
    
    C1 --> MW1
    C2 --> MW1
    C3 --> MW1
    MW1 --> MW2 --> MW3 --> MW4 --> MW5 --> MW6
    MW6 --> Routes
    R3 --> SC
    R4 --> SH
    SC --> DFastLLMEngine
    SH --> DFastLLMEngine
    DFastLLMEngine --> TOK
    TOK --> DIFF
    TOK --> APD
    DIFF --> HF
    APD --> HF
    HF --> GPU
```

## Layer 1: Client Layer

The **Client Layer** is where requests come from.

```mermaid
flowchart LR
    subgraph Clients["Different ways to call dfastllm"]
        subgraph Python["Python"]
            PY["from openai import OpenAI<br/>client = OpenAI(base_url='...')"]
        end
        
        subgraph Curl["Command Line"]
            CU["curl -X POST .../v1/completions"]
        end
        
        subgraph Browser["Browser"]
            BR["fetch('/v1/completions', {...})"]
        end
    end
    
    Python --> API[dfastllm API]
    Curl --> API
    Browser --> API
```

### Python Example

```python
from openai import OpenAI

# Create client pointing to dfastllm server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Make a request
response = client.chat.completions.create(
    model="llada-8b",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Layer 2: API Server Layer

The **API Server** receives HTTP requests and returns responses.

```mermaid
flowchart TB
    subgraph Request["Incoming Request"]
        REQ["POST /v1/completions<br/>Body: {prompt: 'Hello'}"]
    end
    
    subgraph Pipeline["Processing Pipeline"]
        direction TB
        M1["1️⃣ Add Request ID"]
        M2["2️⃣ Log Request"]
        M3["3️⃣ Add Security Headers"]
        M4["4️⃣ Check CORS"]
        M5["5️⃣ Check Rate Limit"]
        M6["6️⃣ Verify API Key"]
        M7["7️⃣ Route to Handler"]
        M8["8️⃣ Process Request"]
        M9["9️⃣ Format Response"]
    end
    
    subgraph Response["Outgoing Response"]
        RES["200 OK<br/>Body: {text: '...'}"]
    end
    
    Request --> M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8 --> M9 --> Response
```

### Key Files

| File | Purpose |
|------|---------|
| `api_server.py` | Main FastAPI app, routes, middleware |
| `protocol.py` | Request/response data structures |
| `serving_completion.py` | Handles `/v1/completions` |
| `serving_chat.py` | Handles `/v1/chat/completions` |

### Middleware Explained

```mermaid
flowchart LR
    subgraph Middleware["Middleware (runs on every request)"]
        subgraph RequestID["Request ID Middleware"]
            RI["Adds unique ID to track request"]
        end
        
        subgraph Logging["Logging Middleware"]
            LO["Logs request start/end time"]
        end
        
        subgraph Security["Security Middleware"]
            SE["Adds security headers"]
        end
        
        subgraph RateLimit["Rate Limiter"]
            RL["Blocks too many requests"]
        end
        
        subgraph Auth["Auth"]
            AU["Checks API key"]
        end
    end
```

## Layer 3: Engine Layer

The **Engine** is the brain of dfastllm. It manages everything.

```mermaid
flowchart TB
    subgraph DFastLLMEngine["DFastLLMEngine Class"]
        subgraph State["State Management"]
            S1["UNINITIALIZED"]
            S2["LOADING"]
            S3["READY"]
            S4["BUSY"]
            S5["DRAINING"]
            S6["SHUTDOWN"]
            S7["ERROR"]
        end
        
        subgraph Components["Components"]
            MODEL["_model<br/>(HuggingFace model)"]
            TOKENIZER["_tokenizer<br/>(TokenizerWrapper)"]
            DIFFUSION["_diffusion_sampler<br/>(DiffusionSampler)"]
            APD["_apd_decoder<br/>(APDDecoder)"]
        end
        
        subgraph Methods["Key Methods"]
            INIT["__init__()"]
            LOAD["_load_model()"]
            GEN["generate()"]
            GENASYNC["generate_async()"]
            SHUTDOWN["shutdown()"]
        end
    end
    
    INIT --> LOAD
    LOAD --> S3
    S3 --> GEN
    GEN --> DIFFUSION
    GEN --> APD
```

### Engine Lifecycle

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED
    UNINITIALIZED --> LOADING: __init__()
    LOADING --> READY: Model loaded
    LOADING --> ERROR: Load failed
    READY --> BUSY: Processing request
    BUSY --> READY: Request done
    READY --> DRAINING: shutdown() called
    DRAINING --> SHUTDOWN: Requests drained
    SHUTDOWN --> [*]
    ERROR --> [*]
```

### Key Methods

```mermaid
flowchart TB
    subgraph Generate["generate() method"]
        G1["1. Validate input"]
        G2["2. Tokenize prompt"]
        G3["3. Check model type"]
        G4{"Diffusion<br/>model?"}
        G5["Use diffusion_generate()"]
        G6["Use standard_generate()"]
        G7["Decode output"]
        G8["Return response"]
    end
    
    G1 --> G2 --> G3 --> G4
    G4 -->|Yes| G5
    G4 -->|No| G6
    G5 --> G7
    G6 --> G7
    G7 --> G8
```

## Layer 4: Generation Layer

The **Generation Layer** actually produces the text.

### Diffusion Generation

```mermaid
flowchart TB
    subgraph Input["Input"]
        PROMPT["Prompt: 'Hello'"]
        MASKS["Masks: [M] [M] [M] [M]"]
    end
    
    subgraph Process["Diffusion Process"]
        COMBINE["Combine: 'Hello [M] [M] [M] [M]'"]
        
        subgraph Loop["For each step"]
            FORWARD["1. Forward pass → logits"]
            CONF["2. Calculate confidence"]
            SELECT["3. Select high confidence"]
            UNMASK["4. Unmask selected tokens"]
        end
        
        CHECK{"All<br/>unmasked?"}
    end
    
    subgraph Output["Output"]
        RESULT["'Hello how are you today'"]
    end
    
    PROMPT --> COMBINE
    MASKS --> COMBINE
    COMBINE --> FORWARD
    FORWARD --> CONF --> SELECT --> UNMASK
    UNMASK --> CHECK
    CHECK -->|No| FORWARD
    CHECK -->|Yes| RESULT
```

### APD (Adaptive Parallel Decoding)

```mermaid
flowchart TB
    subgraph APD["APD Process"]
        subgraph Step1["Step 1: Generate Candidates"]
            C1["Generate multiple tokens<br/>in parallel"]
        end
        
        subgraph Step2["Step 2: Calculate Confidence"]
            C2["Score each candidate<br/>0.0 - 1.0"]
        end
        
        subgraph Step3["Step 3: Accept/Reject"]
            C3{"Confidence ><br/>threshold?"}
            ACC["Accept token"]
            REJ["Reject token"]
        end
        
        subgraph Step4["Step 4: Continue"]
            C4["Repeat for remaining"]
        end
    end
    
    Step1 --> Step2 --> C3
    C3 -->|Yes| ACC
    C3 -->|No| REJ
    ACC --> Step4
    REJ --> Step4
```

## Layer 5: Model Layer

The **Model Layer** is the actual AI model from HuggingFace.

```mermaid
flowchart TB
    subgraph ModelLoading["Model Loading"]
        PATH["Model path/name"]
        HF["HuggingFace Hub"]
        DOWNLOAD["Download weights"]
        LOAD["Load into memory"]
        DEVICE{"Which<br/>device?"}
        GPU["Move to GPU"]
        CPU["Keep on CPU"]
    end
    
    PATH --> HF --> DOWNLOAD --> LOAD --> DEVICE
    DEVICE -->|CUDA available| GPU
    DEVICE -->|No GPU| CPU
```

### Model Inference

```mermaid
flowchart LR
    subgraph Input["Input"]
        TOKENS["Token IDs<br/>[15496, 995, 126336, 126336]"]
    end
    
    subgraph Model["Model"]
        EMB["Embedding Layer"]
        TF["Transformer Layers<br/>(32 layers)"]
        HEAD["LM Head"]
    end
    
    subgraph Output["Output"]
        LOGITS["Logits<br/>[vocab_size] per position"]
    end
    
    TOKENS --> EMB --> TF --> HEAD --> LOGITS
```

## Data Flow Example

Let's trace a complete request:

```mermaid
sequenceDiagram
    participant User
    participant API as API Server
    participant MW as Middleware
    participant SC as ServingCompletion
    participant Engine as DFastLLMEngine
    participant Diff as DiffusionSampler
    participant Model as HF Model
    
    User->>API: POST /v1/completions
    API->>MW: Process request
    MW->>MW: Add request ID
    MW->>MW: Check rate limit
    MW->>MW: Verify auth
    MW->>SC: Route to handler
    SC->>SC: Parse request
    SC->>Engine: generate_async()
    Engine->>Engine: Tokenize prompt
    Engine->>Diff: diffusion_generate()
    
    loop For each step
        Diff->>Model: Forward pass
        Model-->>Diff: Logits
        Diff->>Diff: Calculate confidence
        Diff->>Diff: Unmask tokens
    end
    
    Diff-->>Engine: Generated tokens
    Engine->>Engine: Decode tokens
    Engine-->>SC: RequestOutput
    SC->>SC: Format response
    SC-->>API: CompletionResponse
    API-->>User: JSON Response
```

## Component Interactions

```mermaid
flowchart TB
    subgraph Config["Configuration"]
        VC[DFastLLMConfig]
        MC[ModelConfig]
    end
    
    subgraph Server["Server"]
        SS[ServerState]
        APP[FastAPI App]
    end
    
    subgraph Engine["Engine"]
        VE[DFastLLMEngine]
        TW[TokenizerWrapper]
    end
    
    subgraph Generation["Generation"]
        DS[DiffusionSampler]
        AD[APDDecoder]
    end
    
    subgraph Model["Model"]
        HF[HuggingFace Model]
    end
    
    VC --> VE
    MC --> VE
    SS --> APP
    APP --> VE
    VE --> TW
    VE --> DS
    VE --> AD
    DS --> HF
    AD --> HF
```

## Error Handling Flow

```mermaid
flowchart TB
    subgraph Errors["Error Types"]
        E1["EngineError<br/>Base error"]
        E2["ModelLoadError<br/>Can't load model"]
        E3["GenerationError<br/>Generation failed"]
        E4["TimeoutError<br/>Request too slow"]
        E5["QueueFullError<br/>Too many requests"]
    end
    
    subgraph Handling["Error Handling"]
        H1["Catch in engine"]
        H2["Log error"]
        H3["Update stats"]
        H4["Return HTTP error"]
    end
    
    E1 --> H1
    E2 --> H1
    E3 --> H1
    E4 --> H1
    E5 --> H1
    H1 --> H2 --> H3 --> H4
```

## Summary

```mermaid
flowchart LR
    subgraph Summary["dfastllm Architecture Summary"]
        A["Client sends request"]
        B["API Server receives"]
        C["Middleware processes"]
        D["Engine generates"]
        E["Model runs"]
        F["Response returns"]
    end
    
    A --> B --> C --> D --> E --> F
```

## Next Steps

👉 [03-project-structure.md](03-project-structure.md) - See the file organization

