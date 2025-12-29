# DFastLLMEngine Deep Dive

This document explains the `DFastLLMEngine` class - the heart of dfastllm.

## What is the Engine?

The **Engine** is the central component that:
1. Loads AI models
2. Manages requests
3. Generates text
4. Handles errors

```mermaid
flowchart TB
    subgraph Engine["DFastLLMEngine"]
        LOAD["Load Model"]
        QUEUE["Manage Queue"]
        GEN["Generate Text"]
        ERROR["Handle Errors"]
        STATS["Track Statistics"]
    end
    
    INPUT[Request] --> Engine
    Engine --> OUTPUT[Response]
```

## Engine Class Overview

```mermaid
classDiagram
    class DFastLLMEngine {
        %% Configuration
        +config: DFastLLMConfig
        
        %% State
        -_state: EngineState
        -_stats: EngineStats
        -_is_ready: bool
        
        %% Model Components
        -_model: AutoModelForCausalLM
        -_tokenizer: TokenizerWrapper
        -_model_config: ModelConfig
        
        %% Diffusion Components
        -_diffusion_sampler: DiffusionSampler
        -_apd_decoder: APDDecoder
        -_is_diffusion_model: bool
        -_mask_id: int
        
        %% Concurrency
        -_executor: ThreadPoolExecutor
        -_request_semaphore: Semaphore
        
        %% Methods
        +__init__(config)
        +generate(prompt, params) RequestOutput
        +generate_async(prompt, params) RequestOutput
        +generate_stream(prompt, params) AsyncIterator
        +get_health() HealthStatus
        +get_stats() dict
        +shutdown()
    }
```

## Engine States

The engine goes through different **states** during its lifecycle:

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED: Created
    
    UNINITIALIZED --> LOADING: __init__() called
    
    LOADING --> READY: Model loaded successfully
    LOADING --> ERROR: Model loading failed
    
    READY --> READY: Processing requests
    
    READY --> DRAINING: shutdown() called
    DRAINING --> SHUTDOWN: All requests completed
    
    SHUTDOWN --> [*]: Cleanup complete
    ERROR --> [*]: Engine failed
```

### State Meanings

| State | Meaning |
|-------|---------|
| `UNINITIALIZED` | Engine created but not started |
| `LOADING` | Loading model and tokenizer |
| `READY` | Ready to accept requests |
| `DRAINING` | Finishing pending requests before shutdown |
| `SHUTDOWN` | Engine has stopped |
| `ERROR` | Something went wrong |

## Initialization Flow

When you create an engine, here's what happens:

```mermaid
flowchart TB
    subgraph Init["__init__(config)"]
        I1["1. Save config"]
        I2["2. Set initial state"]
        I3["3. Detect device (GPU/CPU)"]
        I4["4. Create thread pool"]
        I5["5. Call _load_model()"]
    end
    
    subgraph LoadModel["_load_model()"]
        L1["1. Set state to LOADING"]
        L2["2. Load tokenizer"]
        L3["3. Load model config"]
        L4["4. Load PyTorch model"]
        L5["5. Check if diffusion model"]
        L6["6. Setup diffusion components"]
        L7["7. Set state to READY"]
    end
    
    subgraph LoadTorch["_load_torch_model()"]
        T1["1. Determine dtype"]
        T2["2. Clear GPU cache"]
        T3["3. Load from HuggingFace"]
        T4["4. Move to device"]
        T5["5. Set to eval mode"]
        T6["6. Apply torch.compile"]
    end
    
    I1 --> I2 --> I3 --> I4 --> I5
    I5 --> L1
    L1 --> L2 --> L3 --> L4
    L4 --> T1
    T1 --> T2 --> T3 --> T4 --> T5 --> T6
    T6 --> L5 --> L6 --> L7
```

### Code Walkthrough: __init__

```python
def __init__(
    self,
    config: DFastLLMConfig,                    # Configuration object
    max_queue_size: Optional[int] = None,   # Max pending requests
    max_concurrent: Optional[int] = None,   # Max parallel generations
    default_timeout: Optional[float] = None, # Request timeout
):
    # 1. Save configuration
    self.config = config
    self._max_queue_size = max_queue_size or 256
    self._max_concurrent = max_concurrent or 4
    
    # 2. Set initial state
    self._state = EngineState.UNINITIALIZED
    
    # 3. Initialize components (empty for now)
    self._model = None
    self._tokenizer = None
    
    # 4. Detect device
    self._device = self._get_device()  # "cuda", "mps", or "cpu"
    
    # 5. Create thread pool for async
    self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent)
    
    # 6. Load the model
    self._load_model()
```

## Device Detection

```mermaid
flowchart TB
    subgraph GetDevice["_get_device()"]
        CHECK1{"CUDA<br/>available?"}
        CHECK2{"MPS<br/>available?"}
        
        CUDA["Return 'cuda'<br/>(NVIDIA GPU)"]
        MPS["Return 'mps'<br/>(Apple Silicon)"]
        CPU["Return 'cpu'"]
    end
    
    CHECK1 -->|Yes| CUDA
    CHECK1 -->|No| CHECK2
    CHECK2 -->|Yes| MPS
    CHECK2 -->|No| CPU
```

### Code

```python
def _get_device(self) -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

## Model Loading

```mermaid
flowchart TB
    subgraph ModelLoading["Model Loading Process"]
        subgraph Input["Input"]
            PATH["Model name/path<br/>'GSAI-ML/LLaDA-8B-Instruct'"]
        end
        
        subgraph Process["Process"]
            HF["HuggingFace Hub"]
            DOWNLOAD["Download weights<br/>(if not cached)"]
            LOAD["Load into memory"]
            DEVICE["Move to GPU/CPU"]
            EVAL["Set eval mode"]
            COMPILE["torch.compile()"]
        end
        
        subgraph Output["Output"]
            MODEL["Ready model"]
        end
    end
    
    PATH --> HF --> DOWNLOAD --> LOAD --> DEVICE --> EVAL --> COMPILE --> MODEL
```

### Code Walkthrough: _load_torch_model

```python
def _load_torch_model(self) -> None:
    from transformers import AutoModelForCausalLM
    
    # 1. Determine data type
    if self.config.dtype == "auto":
        dtype = torch.float16 if self._device == "cuda" else torch.float32
    elif self.config.dtype == "float16":
        dtype = torch.float16
    # ...
    
    # 2. Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Load model from HuggingFace
    self._model = AutoModelForCausalLM.from_pretrained(
        self.config.model,                    # Model name
        torch_dtype=dtype,                    # Data type
        device_map="auto" if cuda else None,  # Auto GPU placement
        trust_remote_code=True,               # Allow custom code
    )
    
    # 4. Move to device if needed
    if self._device != "cuda":
        self._model = self._model.to(self._device)
    
    # 5. Set to evaluation mode (no training)
    self._model.eval()
    
    # 6. Apply torch.compile for speed
    if self.config.compile_model and self._device == "cuda":
        self._model = torch.compile(self._model, mode="reduce-overhead")
```

## Generation Flow

This is the main function that generates text:

```mermaid
flowchart TB
    subgraph Generate["generate(prompt, sampling_params)"]
        V["1. Validate request"]
        T["2. Tokenize prompt"]
        
        CHECK{"Diffusion<br/>model?"}
        
        subgraph Diffusion["Diffusion Path"]
            APD_CHECK{"APD<br/>enabled?"}
            APD_GEN["APD generate"]
            DIFF_GEN["Diffusion generate"]
        end
        
        subgraph Standard["Standard Path"]
            STD_GEN["Standard generate"]
        end
        
        D["3. Decode tokens"]
        F["4. Format response"]
        R["5. Return output"]
    end
    
    V --> T --> CHECK
    CHECK -->|Yes| APD_CHECK
    CHECK -->|No| STD_GEN
    APD_CHECK -->|Yes| APD_GEN
    APD_CHECK -->|No| DIFF_GEN
    APD_GEN --> D
    DIFF_GEN --> D
    STD_GEN --> D
    D --> F --> R
```

### Code Walkthrough: generate

```python
def generate(
    self,
    prompt: str,                      # Input text
    sampling_params: SamplingParams,  # Generation parameters
    request_id: Optional[str] = None, # Unique ID
    timeout: Optional[float] = None,  # Timeout in seconds
) -> RequestOutput:
    
    # 1. Check engine is ready
    if self._state != EngineState.READY:
        raise EngineError("Engine not ready")
    
    # 2. Validate input
    self._validate_request(prompt, sampling_params)
    
    # 3. Generate request ID if not provided
    request_id = request_id or str(uuid.uuid4())
    
    # 4. Track metrics
    metrics = RequestMetrics(arrival_time=time.time())
    
    # 5. Tokenize the prompt
    input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(self._device)
    
    # 6. Generate based on model type
    if self._is_diffusion_model:
        if self.config.enable_apd and self._apd_decoder:
            output_ids = self._apd_generate(input_ids, sampling_params)
        else:
            output_ids = self._diffusion_generate(input_ids, sampling_params)
    else:
        output_ids = self._standard_generate(input_ids, sampling_params)
    
    # 7. Decode output tokens to text
    generated_text = self._tokenizer.decode(output_ids)
    
    # 8. Remove prompt from output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    # 9. Build response
    output = RequestOutput(
        request_id=request_id,
        prompt=prompt,
        outputs=[CompletionOutput(text=generated_text, ...)],
        finished=True,
        metrics=metrics,
    )
    
    return output
```

## Async Generation

For handling multiple requests concurrently:

```mermaid
flowchart TB
    subgraph Async["generate_async()"]
        SEM["Acquire semaphore<br/>(limit concurrency)"]
        EXEC["Run in thread pool"]
        WAIT["Wait with timeout"]
        REL["Release semaphore"]
    end
    
    REQ[Request] --> SEM --> EXEC --> WAIT --> REL --> RES[Response]
```

### Code

```python
async def generate_async(
    self,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> RequestOutput:
    
    # Limit concurrent requests
    async with self._request_semaphore:
        # Run sync generate() in thread pool
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                self._executor,          # Thread pool
                self.generate,           # Function to run
                prompt,                  # Arguments...
                sampling_params,
                request_id,
                timeout,
            ),
            timeout=timeout,
        )
        return result
```

## Health Status

```mermaid
flowchart TB
    subgraph Health["get_health()"]
        STATE["Check engine state"]
        GPU["Check GPU memory"]
        QUEUE["Check queue size"]
        
        CALC{"Calculate<br/>status"}
        
        HEALTHY["'healthy'"]
        DEGRADED["'degraded'"]
        UNHEALTHY["'unhealthy'"]
    end
    
    STATE --> CALC
    GPU --> CALC
    QUEUE --> CALC
    
    CALC -->|All good| HEALTHY
    CALC -->|Some issues| DEGRADED
    CALC -->|Errors| UNHEALTHY
```

### HealthStatus Object

```mermaid
classDiagram
    class HealthStatus {
        +status: str
        +state: EngineState
        +model_loaded: bool
        +device: str
        +gpu_memory_used_mb: float
        +gpu_memory_total_mb: float
        +queue_size: int
        +queue_capacity: int
        +uptime_seconds: float
        +last_error: str
        
        +to_dict() dict
    }
```

## Statistics

```mermaid
classDiagram
    class EngineStats {
        +requests_processed: int
        +requests_failed: int
        +requests_timeout: int
        +tokens_generated: int
        +total_latency_ms: float
        +avg_tokens_per_step: float
        +peak_memory_mb: float
        +current_queue_size: int
        +uptime_seconds: float
        +start_time: float
        
        +to_dict() dict
    }
```

## Shutdown Flow

```mermaid
flowchart TB
    subgraph Shutdown["shutdown()"]
        SET["Set state to DRAINING"]
        WAIT["Wait for pending requests<br/>(max 30 seconds)"]
        STOP["Stop thread pool"]
        FREE["Free model memory"]
        GC["Garbage collect"]
        DONE["Set state to SHUTDOWN"]
    end
    
    SET --> WAIT --> STOP --> FREE --> GC --> DONE
```

### Code

```python
async def shutdown(self, timeout: float = 30) -> None:
    # 1. Start draining
    self._set_state(EngineState.DRAINING)
    
    # 2. Wait for pending requests
    drain_start = time.time()
    while self._stats.current_queue_size > 0:
        if time.time() - drain_start > timeout:
            logger.warning("Shutdown timeout, forcing...")
            break
        await asyncio.sleep(0.1)
    
    # 3. Set final state
    self._set_state(EngineState.SHUTDOWN)
    
    # 4. Stop thread pool
    self._executor.shutdown(wait=False)
    
    # 5. Free model memory
    if self._model is not None:
        del self._model
        self._model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 6. Force garbage collection
    gc.collect()
```

## Error Handling

```mermaid
flowchart TB
    subgraph Errors["Error Hierarchy"]
        BASE["EngineError<br/>(base class)"]
        LOAD["ModelLoadError<br/>Can't load model"]
        GEN["GenerationError<br/>Generation failed"]
        TIMEOUT["TimeoutError<br/>Request took too long"]
        QUEUE["QueueFullError<br/>Too many requests"]
    end
    
    BASE --> LOAD
    BASE --> GEN
    BASE --> TIMEOUT
    BASE --> QUEUE
```

### Error Handling in generate()

```python
try:
    # ... generation code ...
except TimeoutError:
    self._stats.requests_timeout += 1
    raise
except Exception as e:
    self._stats.requests_failed += 1
    self._last_error = str(e)
    raise GenerationError(f"Generation failed: {e}")
finally:
    # Always update queue size
    self._stats.current_queue_size -= 1
```

## Memory Management

```mermaid
flowchart TB
    subgraph Memory["Memory Management"]
        CHECK["Check memory periodically"]
        LOW{"Memory<br/>low?"}
        CACHE["Clear CUDA cache"]
        GC["Run garbage collection"]
        WARN["Log warning"]
    end
    
    CHECK --> LOW
    LOW -->|Yes| CACHE --> GC --> WARN
    LOW -->|No| CHECK
```

## Complete Request Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Engine as DFastLLMEngine
    participant Queue as Request Queue
    participant Tokenizer
    participant Diffusion as DiffusionSampler
    participant Model as HF Model
    participant GPU
    
    Client->>Engine: generate_async(prompt)
    Engine->>Engine: Validate request
    Engine->>Queue: Add to queue
    Queue->>Engine: Semaphore acquired
    
    Engine->>Tokenizer: encode(prompt)
    Tokenizer-->>Engine: token_ids
    
    Engine->>Diffusion: diffusion_generate(token_ids)
    
    loop For each step
        Diffusion->>Model: forward(tokens)
        Model->>GPU: Matrix operations
        GPU-->>Model: Logits
        Model-->>Diffusion: Logits
        Diffusion->>Diffusion: Calculate confidence
        Diffusion->>Diffusion: Unmask tokens
    end
    
    Diffusion-->>Engine: output_ids
    Engine->>Tokenizer: decode(output_ids)
    Tokenizer-->>Engine: text
    Engine->>Engine: Build RequestOutput
    Engine->>Queue: Release semaphore
    Engine-->>Client: RequestOutput
```

## Summary

```mermaid
flowchart LR
    subgraph Engine["DFastLLMEngine Summary"]
        INIT["Initialize"]
        LOAD["Load Model"]
        READY["Ready"]
        GEN["Generate"]
        SHUT["Shutdown"]
    end
    
    INIT --> LOAD --> READY --> GEN --> SHUT
    GEN --> GEN
```

| Component | Purpose |
|-----------|---------|
| State Management | Track engine lifecycle |
| Model Loading | Load HuggingFace models |
| Request Queue | Handle concurrent requests |
| Generation | Produce text output |
| Health/Stats | Monitor engine status |
| Shutdown | Graceful cleanup |

## Next Steps

👉 [05-diffusion.md](05-diffusion.md) - How diffusion generation works

