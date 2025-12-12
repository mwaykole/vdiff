# API Server Deep Dive

This document explains how the FastAPI server works.

## What is the API Server?

The **API Server** is the part of vdiff that:
1. Listens for HTTP requests
2. Processes them
3. Returns responses

```mermaid
flowchart LR
    subgraph Clients["Clients"]
        C1["Python App"]
        C2["curl"]
        C3["Browser"]
    end
    
    subgraph Server["vdiff API Server"]
        API["FastAPI"]
    end
    
    C1 --> API
    C2 --> API
    C3 --> API
    API --> C1
    API --> C2
    API --> C3
```

## Server Architecture

```mermaid
flowchart TB
    subgraph Server["API Server Components"]
        subgraph FastAPI["FastAPI Application"]
            APP["FastAPI()"]
        end
        
        subgraph Middleware["Middleware Stack"]
            MW1["RequestIDMiddleware"]
            MW2["LoggingMiddleware"]
            MW3["SecurityHeadersMiddleware"]
            MW4["CORSMiddleware"]
        end
        
        subgraph Routes["Route Handlers"]
            R1["/health"]
            R2["/v1/models"]
            R3["/v1/completions"]
            R4["/v1/chat/completions"]
            R5["/metrics"]
        end
        
        subgraph Auth["Authentication"]
            RATE["RateLimiter"]
            KEY["API Key Verification"]
        end
        
        subgraph State["Server State"]
            SS["ServerState"]
            ENG["VDiffEngine"]
        end
    end
    
    APP --> Middleware
    Middleware --> Auth
    Auth --> Routes
    Routes --> State
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Middleware
    participant Auth
    participant Route
    participant Serving
    participant Engine
    
    Client->>FastAPI: POST /v1/completions
    FastAPI->>Middleware: Pass request
    
    Note over Middleware: RequestIDMiddleware
    Middleware->>Middleware: Add X-Request-ID
    
    Note over Middleware: LoggingMiddleware
    Middleware->>Middleware: Log request start
    
    Note over Middleware: SecurityHeaders
    Middleware->>Middleware: Add security headers
    
    Note over Middleware: CORS
    Middleware->>Middleware: Check origin
    
    Middleware->>Auth: Check auth
    Auth->>Auth: Verify API key
    Auth->>Auth: Check rate limit
    
    Auth->>Route: /v1/completions handler
    Route->>Serving: create_completion()
    Serving->>Engine: generate_async()
    Engine-->>Serving: RequestOutput
    Serving-->>Route: CompletionResponse
    Route-->>FastAPI: JSON response
    FastAPI-->>Client: HTTP 200
```

## Middleware Explained

### 1. Request ID Middleware

Adds a unique ID to every request for tracking.

```mermaid
flowchart LR
    REQ["Request"] --> MW["RequestIDMiddleware"]
    MW --> ADD["Add X-Request-ID header"]
    ADD --> NEXT["Next middleware"]
```

```python
class RequestIDMiddleware:
    async def dispatch(self, request, call_next):
        # Get or create request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Store in request state
        request.state.request_id = request_id
        
        # Call next middleware
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
```

### 2. Logging Middleware

Logs every request and response.

```mermaid
flowchart TB
    subgraph Logging["LoggingMiddleware"]
        START["Log: Request started"]
        PROCESS["Process request"]
        END["Log: Request finished"]
    end
    
    REQ[Request] --> START --> PROCESS --> END --> RES[Response]
```

```python
class LoggingMiddleware:
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = (time.time() - start_time) * 1000
        logger.info(f"Response: {response.status_code} in {duration:.2f}ms")
        
        return response
```

### 3. Security Headers Middleware

Adds security headers to prevent attacks.

```mermaid
flowchart LR
    subgraph Security["Security Headers"]
        H1["X-Content-Type-Options: nosniff"]
        H2["X-Frame-Options: DENY"]
        H3["X-XSS-Protection: 1"]
        H4["Strict-Transport-Security"]
    end
```

### 4. CORS Middleware

Controls which websites can call the API.

```mermaid
flowchart TB
    subgraph CORS["CORS Check"]
        CHECK{"Origin<br/>allowed?"}
        ALLOW["Allow request"]
        DENY["Block request"]
    end
    
    REQ[Request] --> CHECK
    CHECK -->|"In allowed list"| ALLOW
    CHECK -->|"Not allowed"| DENY
```

## Rate Limiting

Prevents too many requests from one client.

```mermaid
flowchart TB
    subgraph RateLimiter["Rate Limiter"]
        subgraph Window["Sliding Window (60 seconds)"]
            W1["Request 1: t=0s"]
            W2["Request 2: t=5s"]
            W3["Request 3: t=10s"]
            WN["..."]
            W100["Request 100: t=55s"]
        end
        
        CHECK{"Count < Limit?"}
        ALLOW["Allow request"]
        DENY["HTTP 429: Too Many Requests"]
    end
    
    W100 --> CHECK
    CHECK -->|"< 100"| ALLOW
    CHECK -->|">= 100"| DENY
```

```python
class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # client_id -> [timestamps]
    
    async def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Remove old requests
        requests = [t for t in self._requests.get(client_id, []) if t > cutoff]
        
        # Check limit
        if len(requests) >= self.max_requests:
            return False
        
        # Add current request
        requests.append(now)
        self._requests[client_id] = requests
        return True
```

## API Endpoints

### All Endpoints

```mermaid
flowchart TB
    subgraph Endpoints["API Endpoints"]
        subgraph Health["Health Checks"]
            E1["GET /health<br/>Detailed health"]
            E2["GET /health/live<br/>Liveness probe"]
            E3["GET /health/ready<br/>Readiness probe"]
        end
        
        subgraph Models["Model Info"]
            E4["GET /v1/models<br/>List models"]
        end
        
        subgraph Generation["Text Generation"]
            E5["POST /v1/completions<br/>Text completion"]
            E6["POST /v1/chat/completions<br/>Chat completion"]
        end
        
        subgraph Monitoring["Monitoring"]
            E7["GET /metrics<br/>Prometheus metrics"]
            E8["GET /v1/engine/stats<br/>Engine statistics"]
        end
    end
```

### /health Endpoint

```mermaid
flowchart TB
    subgraph Health["/health"]
        CHECK["Check engine health"]
        
        subgraph Response["Response"]
            STATUS["status: healthy|degraded|unhealthy"]
            STATE["state: ready|loading|error"]
            GPU["gpu_memory: used/total"]
            QUEUE["queue: size/capacity"]
        end
    end
    
    CHECK --> Response
```

### /v1/completions Endpoint

```mermaid
flowchart TB
    subgraph Completions["/v1/completions"]
        subgraph Request["Request Body"]
            R1["model: 'llada-8b'"]
            R2["prompt: 'Once upon a time'"]
            R3["max_tokens: 100"]
            R4["temperature: 0.7"]
            R5["stream: false"]
        end
        
        subgraph Process["Processing"]
            P1["Parse request"]
            P2["Validate parameters"]
            P3["Call engine.generate()"]
            P4["Format response"]
        end
        
        subgraph Response["Response Body"]
            S1["id: 'cmpl-xxx'"]
            S2["choices: [{text: '...'}]"]
            S3["usage: {tokens: ...}"]
        end
    end
    
    Request --> Process --> Response
```

### /v1/chat/completions Endpoint

```mermaid
flowchart TB
    subgraph Chat["/v1/chat/completions"]
        subgraph Request["Request Body"]
            R1["model: 'llada-8b'"]
            R2["messages: [<br/>  {role: 'user', content: 'Hi'}<br/>]"]
            R3["max_tokens: 100"]
        end
        
        subgraph Process["Processing"]
            P1["Format messages to prompt"]
            P2["Call engine.generate()"]
            P3["Format as chat response"]
        end
        
        subgraph Response["Response Body"]
            S1["id: 'chatcmpl-xxx'"]
            S2["choices: [{message: ...}]"]
            S3["usage: {tokens: ...}"]
        end
    end
    
    Request --> Process --> Response
```

## Streaming

For long generations, stream responses:

```mermaid
sequenceDiagram
    participant Client
    participant Server
    
    Client->>Server: POST /v1/completions (stream=true)
    
    Server-->>Client: data: {"text": "The"}
    Server-->>Client: data: {"text": "The cat"}
    Server-->>Client: data: {"text": "The cat sat"}
    Server-->>Client: data: [DONE]
```

## Server State

```mermaid
classDiagram
    class ServerState {
        +engine: VDiffEngine
        +completion_serving: OpenAIServingCompletion
        +chat_serving: OpenAIServingChat
        +config: VDiffConfig
        +rate_limiter: RateLimiter
        +start_time: float
        +request_count: int
        +active_requests: int
        
        +is_ready: bool
        +uptime_seconds: float
    }
```

## Error Handling

```mermaid
flowchart TB
    subgraph Errors["Error Handling"]
        subgraph ErrorTypes["Error Types"]
            E1["EngineError → 500"]
            E2["GenerationError → 500"]
            E3["TimeoutError → 504"]
            E4["QueueFullError → 503"]
            E5["ValidationError → 400"]
            E6["RateLimitError → 429"]
            E7["AuthError → 401/403"]
        end
        
        subgraph Response["Error Response"]
            ER["error: {<br/>  message: '...',<br/>  type: '...',<br/>  code: 500,<br/>  request_id: '...'<br/>}"]
        end
    end
    
    ErrorTypes --> Response
```

## Startup and Shutdown

```mermaid
flowchart TB
    subgraph Lifecycle["Server Lifecycle"]
        subgraph Startup["Startup (lifespan)"]
            S1["Initialize rate limiter"]
            S2["Load engine"]
            S3["Wait for engine ready"]
            S4["Initialize serving components"]
            S5["Setup metrics"]
            S6["Log ready message"]
        end
        
        subgraph Running["Running"]
            R1["Accept requests"]
        end
        
        subgraph Shutdown["Shutdown"]
            D1["Set shutdown event"]
            D2["Wait for active requests"]
            D3["Shutdown engine"]
            D4["Log shutdown complete"]
        end
    end
    
    Startup --> Running --> Shutdown
```

## Configuration

### CLI Arguments

```mermaid
flowchart LR
    subgraph CLI["CLI Arguments"]
        subgraph Server["Server"]
            A1["--host 0.0.0.0"]
            A2["--port 8000"]
            A3["--api-key SECRET"]
        end
        
        subgraph Model["Model"]
            A4["--model llada"]
            A5["--dtype float16"]
        end
        
        subgraph Production["Production"]
            A6["--rate-limit-requests 100"]
            A7["--max-concurrent-requests 4"]
        end
    end
```

### Environment Variables

```mermaid
flowchart LR
    subgraph Env["Environment Variables"]
        E1["VDIFF_MODEL"]
        E2["VDIFF_PORT"]
        E3["VDIFF_API_KEY"]
        E4["VDIFF_RATE_LIMIT_REQUESTS"]
    end
```

## Serving Components

### OpenAIServingCompletion

```mermaid
classDiagram
    class OpenAIServingCompletion {
        +engine: VDiffEngine
        +model_name: str
        +served_model_names: List
        
        +create_completion(request) Response
        +_generate_completion() Response
        +_stream_completion() AsyncIterator
        +show_available_models() ModelList
    }
```

### OpenAIServingChat

```mermaid
classDiagram
    class OpenAIServingChat {
        +engine: VDiffEngine
        +model_name: str
        
        +create_chat_completion(request) Response
        +_format_messages(messages) str
        +_generate_chat_completion() Response
        +_stream_chat_completion() AsyncIterator
    }
```

## Message Formatting

Chat messages are formatted into a prompt:

```mermaid
flowchart TB
    subgraph Input["Input Messages"]
        M1["system: 'You are helpful'"]
        M2["user: 'Hello!'"]
        M3["assistant: 'Hi there!'"]
        M4["user: 'How are you?'"]
    end
    
    subgraph Formatted["Formatted Prompt"]
        F["System: You are helpful<br/><br/>User: Hello!<br/><br/>Assistant: Hi there!<br/><br/>User: How are you?<br/><br/>Assistant:"]
    end
    
    Input --> Formatted
```

## Summary

```mermaid
flowchart LR
    subgraph Summary["API Server Summary"]
        A["FastAPI app"]
        B["Middleware stack"]
        C["Route handlers"]
        D["Engine calls"]
        E["JSON responses"]
    end
    
    A --> B --> C --> D --> E
```

| Component | Purpose |
|-----------|---------|
| FastAPI | Web framework |
| Middleware | Request processing |
| Routes | Endpoint handlers |
| Serving | OpenAI formatting |
| State | Global engine reference |

## Next Steps

👉 [08-config.md](08-config.md) - Configuration options explained

