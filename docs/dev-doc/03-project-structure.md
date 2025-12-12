# Project Structure

This document explains every file and folder in the vdiff codebase.

## Directory Tree

```
vdiff/
├── 📁 vdiff/                    # Main Python package
│   ├── 📄 __init__.py           # Package initialization
│   ├── 📄 version.py            # Version information
│   ├── 📄 config.py             # Configuration classes
│   │
│   ├── 📁 engine/               # Core inference engine
│   │   ├── 📄 __init__.py       # Engine exports
│   │   ├── 📄 vdiff_engine.py   # Main engine class
│   │   ├── 📄 diffusion_sampler.py  # Diffusion algorithm
│   │   ├── 📄 apd.py            # APD optimization
│   │   ├── 📄 sampling_params.py    # Generation parameters
│   │   ├── 📄 outputs.py        # Output data structures
│   │   ├── 📄 tokenizer.py      # Tokenizer wrapper
│   │   └── 📄 request_queue.py  # Request queue
│   │
│   ├── 📁 entrypoints/          # API server
│   │   ├── 📄 __init__.py
│   │   ├── 📄 launcher.py       # Server launcher
│   │   └── 📁 openai/           # OpenAI-compatible API
│   │       ├── 📄 __init__.py
│   │       ├── 📄 api_server.py     # FastAPI application
│   │       ├── 📄 protocol.py       # Request/response types
│   │       ├── 📄 serving_completion.py  # Completion handler
│   │       └── 📄 serving_chat.py   # Chat handler
│   │
│   └── 📁 metrics/              # Monitoring
│       ├── 📄 __init__.py
│       └── 📄 prometheus.py     # Prometheus metrics
│
├── 📁 deploy/                   # Deployment files
│   ├── 📁 docker/               # Docker files
│   └── 📁 kubernetes/           # K8s manifests
│       ├── 📁 kserve/           # KServe/RHOAI
│       ├── 📁 llmd/             # llm-d integration
│       └── 📁 standalone/       # Standalone K8s
│
├── 📁 tests/                    # Test suite
│   ├── 📁 unit/                 # Unit tests
│   ├── 📁 integration/          # Integration tests
│   └── 📁 compatibility/        # Compatibility tests
│
├── 📁 benchmarks/               # Performance benchmarks
│   └── 📄 run_benchmark.py
│
├── 📁 examples/                 # Usage examples
├── 📁 scripts/                  # Utility scripts
├── 📁 docs/                     # Documentation
│
├── 📄 Dockerfile                # Container build
├── 📄 docker-compose.yml        # Container orchestration
├── 📄 pyproject.toml            # Python package config
├── 📄 Makefile                  # Build automation
├── 📄 README.md                 # Project readme
├── 📄 LICENSE                   # Apache 2.0 license
├── 📄 CHANGELOG.md              # Version history
├── 📄 requirements.txt          # Dependencies
└── 📄 requirements-dev.txt      # Dev dependencies
```

## Visual Structure

```mermaid
flowchart TB
    subgraph Root["vdiff/ (Root)"]
        subgraph Package["vdiff/ (Python Package)"]
            INIT["__init__.py"]
            VERSION["version.py"]
            CONFIG["config.py"]
            
            subgraph Engine["engine/"]
                ENG["vdiff_engine.py"]
                DIFF["diffusion_sampler.py"]
                APD["apd.py"]
                SP["sampling_params.py"]
                OUT["outputs.py"]
                TOK["tokenizer.py"]
            end
            
            subgraph Entry["entrypoints/openai/"]
                API["api_server.py"]
                PROTO["protocol.py"]
                SERVC["serving_completion.py"]
                SERVCH["serving_chat.py"]
            end
            
            subgraph Metrics["metrics/"]
                PROM["prometheus.py"]
            end
        end
        
        subgraph Deploy["deploy/"]
            DOCKER["docker/"]
            K8S["kubernetes/"]
        end
        
        subgraph Tests["tests/"]
            UNIT["unit/"]
            INT["integration/"]
        end
        
        subgraph Config["Config Files"]
            PYPROJ["pyproject.toml"]
            MAKE["Makefile"]
            DFILE["Dockerfile"]
        end
    end
```

## File-by-File Explanation

### Root Files

```mermaid
flowchart LR
    subgraph RootFiles["Root Configuration Files"]
        PT["pyproject.toml<br/>Package metadata"]
        MK["Makefile<br/>Build commands"]
        DF["Dockerfile<br/>Container build"]
        DC["docker-compose.yml<br/>Container orchestration"]
        RQ["requirements.txt<br/>Dependencies"]
        RM["README.md<br/>Documentation"]
    end
```

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python package configuration (name, version, dependencies) |
| `Makefile` | Shortcuts for common commands (`make test`, `make lint`) |
| `Dockerfile` | Instructions to build Docker container |
| `docker-compose.yml` | Multi-container setup for development |
| `requirements.txt` | List of Python dependencies |
| `README.md` | Project documentation and quick start |

---

### vdiff/__init__.py

**Purpose**: Package initialization - what gets exported when you `import vdiff`

```python
# What it contains:
from vdiff.version import __version__
from vdiff.config import VDiffConfig, ModelConfig

__all__ = ["__version__", "VDiffConfig", "ModelConfig"]
```

```mermaid
flowchart LR
    IMPORT["from vdiff import ..."]
    VERSION["__version__"]
    CONFIG["VDiffConfig"]
    MODEL["ModelConfig"]
    
    IMPORT --> VERSION
    IMPORT --> CONFIG
    IMPORT --> MODEL
```

---

### vdiff/version.py

**Purpose**: Version information

```python
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
VLLM_COMPAT_VERSION = "0.4.0"  # Compatible vLLM version
```

---

### vdiff/config.py

**Purpose**: Configuration classes for the engine and server

```mermaid
flowchart TB
    subgraph VDiffConfig["VDiffConfig"]
        subgraph Model["Model Config"]
            M1["model: str"]
            M2["tokenizer: str"]
            M3["dtype: str"]
        end
        
        subgraph Server["Server Config"]
            S1["host: str"]
            S2["port: int"]
            S3["api_key: str"]
        end
        
        subgraph Diffusion["Diffusion Config"]
            D1["diffusion_steps: int"]
            D2["block_size: int"]
            D3["enable_apd: bool"]
        end
        
        subgraph Production["Production Config"]
            P1["max_concurrent: int"]
            P2["rate_limit: int"]
            P3["timeout: float"]
        end
    end
```

---

### vdiff/engine/__init__.py

**Purpose**: Export engine classes

```mermaid
flowchart LR
    subgraph Exports["Engine Exports"]
        VE["VDiffEngine"]
        AVE["AsyncVDiffEngine"]
        SP["SamplingParams"]
        CO["CompletionOutput"]
        RO["RequestOutput"]
        DS["DiffusionSampler"]
        APD["APDDecoder"]
    end
```

---

### vdiff/engine/vdiff_engine.py

**Purpose**: The main engine class - the brain of vdiff

```mermaid
classDiagram
    class VDiffEngine {
        -config: VDiffConfig
        -_model: AutoModelForCausalLM
        -_tokenizer: TokenizerWrapper
        -_state: EngineState
        -_stats: EngineStats
        
        +__init__(config)
        +generate(prompt, params) RequestOutput
        +generate_async(prompt, params) RequestOutput
        +get_health() HealthStatus
        +get_stats() dict
        +shutdown()
    }
    
    class EngineState {
        <<enumeration>>
        UNINITIALIZED
        LOADING
        READY
        BUSY
        DRAINING
        SHUTDOWN
        ERROR
    }
    
    VDiffEngine --> EngineState
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize engine, load model |
| `generate()` | Synchronous text generation |
| `generate_async()` | Async text generation |
| `get_health()` | Get health status |
| `shutdown()` | Graceful shutdown |

---

### vdiff/engine/diffusion_sampler.py

**Purpose**: Implements the diffusion generation algorithm

```mermaid
flowchart TB
    subgraph DiffusionSampler["DiffusionSampler Class"]
        INIT["__init__(model, tokenizer, config)"]
        
        subgraph Methods["Key Methods"]
            GEN["generate()"]
            STEP["diffusion_step()"]
            CONF["calculate_confidence()"]
            UNMASK["unmask_tokens()"]
        end
    end
    
    subgraph Config["DiffusionSamplerConfig"]
        C1["steps: int"]
        C2["block_length: int"]
        C3["temperature: float"]
        C4["remasking: str"]
        C5["mask_id: int"]
    end
    
    Config --> DiffusionSampler
```

---

### vdiff/engine/apd.py

**Purpose**: Adaptive Parallel Decoding optimization

```mermaid
flowchart TB
    subgraph APDDecoder["APDDecoder Class"]
        CONFIG["APDConfig"]
        
        subgraph Methods["Methods"]
            GEN["generate()"]
            PARALLEL["parallel_decode_step()"]
            ACCEPT["should_accept()"]
        end
        
        subgraph Stats["Statistics"]
            S1["tokens_per_step"]
            S2["acceptance_rate"]
        end
    end
```

---

### vdiff/engine/sampling_params.py

**Purpose**: Parameters for text generation

```mermaid
flowchart LR
    subgraph SamplingParams["SamplingParams"]
        N["n: int = 1"]
        MAX["max_tokens: int = 16"]
        TEMP["temperature: float = 1.0"]
        TOPP["top_p: float = 1.0"]
        TOPK["top_k: int = -1"]
        STOP["stop: List[str]"]
    end
```

---

### vdiff/engine/outputs.py

**Purpose**: Output data structures

```mermaid
classDiagram
    class CompletionOutput {
        +index: int
        +text: str
        +token_ids: List[int]
        +finish_reason: str
    }
    
    class RequestOutput {
        +request_id: str
        +prompt: str
        +outputs: List[CompletionOutput]
        +finished: bool
        +metrics: RequestMetrics
    }
    
    class RequestMetrics {
        +arrival_time: float
        +first_token_time: float
        +finished_time: float
        +prompt_tokens: int
        +generated_tokens: int
    }
    
    RequestOutput --> CompletionOutput
    RequestOutput --> RequestMetrics
```

---

### vdiff/engine/tokenizer.py

**Purpose**: Wrapper around HuggingFace tokenizer

```mermaid
flowchart LR
    subgraph TokenizerWrapper["TokenizerWrapper"]
        INIT["Load from HuggingFace"]
        ENC["encode(text) → tokens"]
        DEC["decode(tokens) → text"]
        CHAT["apply_chat_template()"]
    end
    
    TEXT["Hello world"] --> ENC --> TOKENS["[15496, 995]"]
    TOKENS --> DEC --> TEXT2["Hello world"]
```

---

### vdiff/entrypoints/openai/api_server.py

**Purpose**: Main FastAPI application

```mermaid
flowchart TB
    subgraph APIServer["api_server.py"]
        subgraph Functions["Functions"]
            CREATE["create_app()"]
            REGISTER["register_routes()"]
            PARSE["parse_args()"]
            RUN["run_server()"]
            MAIN["main()"]
        end
        
        subgraph Classes["Classes"]
            STATE["ServerState"]
            RATE["RateLimiter"]
            REQID["RequestIDMiddleware"]
            LOG["LoggingMiddleware"]
        end
        
        subgraph Routes["Routes"]
            R1["/health"]
            R2["/v1/models"]
            R3["/v1/completions"]
            R4["/v1/chat/completions"]
            R5["/metrics"]
        end
    end
```

---

### vdiff/entrypoints/openai/protocol.py

**Purpose**: Request and response data types

```mermaid
classDiagram
    class CompletionRequest {
        +model: str
        +prompt: str
        +max_tokens: int
        +temperature: float
        +stream: bool
    }
    
    class CompletionResponse {
        +id: str
        +model: str
        +choices: List[Choice]
        +usage: UsageInfo
    }
    
    class ChatCompletionRequest {
        +model: str
        +messages: List[ChatMessage]
        +max_tokens: int
    }
    
    class ChatMessage {
        +role: str
        +content: str
    }
```

---

### vdiff/metrics/prometheus.py

**Purpose**: Prometheus metrics for monitoring

```mermaid
flowchart LR
    subgraph Metrics["Prometheus Metrics"]
        REQ["vdiff_requests_total"]
        LAT["vdiff_request_latency_seconds"]
        TOK["vdiff_tokens_generated_total"]
        ERR["vdiff_errors_total"]
    end
    
    subgraph Endpoint["Endpoint"]
        EP["/metrics"]
    end
    
    Metrics --> EP
```

---

## Deployment Files

### deploy/kubernetes/kserve/

```mermaid
flowchart TB
    subgraph KServe["KServe Deployment"]
        SR["serving-runtime.yaml<br/>Define vdiff runtime"]
        IS["inference-service.yaml<br/>Deploy model"]
    end
    
    SR --> IS
```

| File | Purpose |
|------|---------|
| `serving-runtime.yaml` | Defines vdiff as a ServingRuntime |
| `inference-service.yaml` | Deploys a model using vdiff |

---

## Test Files

```mermaid
flowchart TB
    subgraph Tests["Test Organization"]
        subgraph Unit["tests/unit/"]
            U1["test_engine.py"]
            U2["test_sampling_params.py"]
            U3["test_diffusion_sampler.py"]
            U4["test_protocol.py"]
        end
        
        subgraph Integration["tests/integration/"]
            I1["test_api_completion.py"]
            I2["test_api_chat.py"]
            I3["test_metrics.py"]
        end
    end
```

---

## Import Graph

```mermaid
flowchart TB
    subgraph External["External Libraries"]
        TORCH["torch"]
        TRANSFORMERS["transformers"]
        FASTAPI["fastapi"]
        PYDANTIC["pydantic"]
    end
    
    subgraph VDiff["vdiff Package"]
        CONFIG["config.py"]
        ENGINE["engine/"]
        ENTRY["entrypoints/"]
        METRICS["metrics/"]
    end
    
    TORCH --> ENGINE
    TRANSFORMERS --> ENGINE
    FASTAPI --> ENTRY
    PYDANTIC --> ENTRY
    PYDANTIC --> CONFIG
    
    CONFIG --> ENGINE
    ENGINE --> ENTRY
    METRICS --> ENTRY
```

## Summary

```mermaid
flowchart TB
    subgraph Summary["Project Structure Summary"]
        CODE["vdiff/<br/>Main code"]
        DEPLOY["deploy/<br/>Deployment"]
        TESTS["tests/<br/>Testing"]
        DOCS["docs/<br/>Documentation"]
        CONFIG["Config files<br/>pyproject.toml, etc."]
    end
```

| Directory | Contains |
|-----------|----------|
| `vdiff/` | All Python source code |
| `deploy/` | Docker, Kubernetes files |
| `tests/` | Unit and integration tests |
| `docs/` | Documentation |
| `benchmarks/` | Performance tests |
| Root files | Configuration and metadata |

## Next Steps

👉 [04-engine.md](04-engine.md) - Deep dive into the engine

