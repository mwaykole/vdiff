"""Production-Ready OpenAI-Compatible API Server for dfastllm.

Enterprise-grade FastAPI application matching vLLM's API server interface with:
- Rate limiting and request throttling
- API key authentication
- Request ID tracking and correlation
- Structured error responses
- Security headers
- Graceful shutdown
- Health checks with detailed status
- Prometheus metrics
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator, Dict, Any, Callable

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from dfastllm.config import DFastLLMConfig
from dfastllm.version import __version__, VLLM_COMPAT_VERSION
from dfastllm.engine import DFastLLMEngine
from dfastllm.engine.dfastllm_engine import (
    EngineState,
    EngineError,
    GenerationError,
    TimeoutError as EngineTimeoutError,
    QueueFullError,
)
from dfastllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    ErrorResponse,
    HealthResponse,
    VersionResponse,
)
from dfastllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from dfastllm.entrypoints.openai.serving_chat import OpenAIServingChat
from dfastllm.metrics import setup_metrics, metrics_endpoint, record_request

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vdiff.server")


class RequestState(BaseModel):
    """Request state for tracking."""
    request_id: str
    start_time: float
    client_ip: str


class RateLimiter:
    """Simple in-memory rate limiter with sliding window."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            if client_id not in self._requests:
                self._requests[client_id] = []
            
            # Remove expired entries
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]
            
            if len(self._requests[client_id]) >= self.max_requests:
                return False
            
            self._requests[client_id].append(now)
            return True
    
    async def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client (thread-safe)."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            if client_id not in self._requests:
                return self.max_requests
            
            valid_requests = [t for t in self._requests[client_id] if t > cutoff]
            return max(0, self.max_requests - len(valid_requests))


class ServerState:
    """Global server state management."""
    
    def __init__(self):
        self.engine: Optional[DFastLLMEngine] = None
        self.completion_serving: Optional[OpenAIServingCompletion] = None
        self.chat_serving: Optional[OpenAIServingChat] = None
        self.config: Optional[DFastLLMConfig] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.start_time: float = time.time()
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._request_count: int = 0
        self._active_requests: int = 0
        self._lock = asyncio.Lock()
    
    async def increment_requests(self) -> None:
        """Thread-safe increment of request counters."""
        async with self._lock:
            self._request_count += 1
            self._active_requests += 1
    
    async def decrement_active_requests(self) -> None:
        """Thread-safe decrement of active requests."""
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
    
    @property
    def request_count(self) -> int:
        return self._request_count
    
    @property
    def active_requests(self) -> int:
        return self._active_requests
    
    @property
    def is_ready(self) -> bool:
        return self.engine is not None and self.engine.is_ready
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


# Global server state
server_state = ServerState()
security = HTTPBearer(auto_error=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Response: {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration_ms:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error: {request.method} {request.url.path} | "
                f"Error: {str(e)} | "
                f"Duration: {duration_ms:.2f}ms"
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """Verify API key if configured."""
    if server_state.config is None or not server_state.config.api_key:
        return True
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != server_state.config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return True


async def check_rate_limit(request: Request) -> bool:
    """Check rate limit for request."""
    if server_state.rate_limiter is None:
        return True
    
    client_id = request.client.host if request.client else "unknown"
    
    if not await server_state.rate_limiter.is_allowed(client_id):
        remaining = await server_state.rate_limiter.get_remaining(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Remaining: {remaining}",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Remaining": str(remaining),
            },
        )
    
    return True


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "server_error",
    request_id: Optional[str] = None,
) -> JSONResponse:
    """Create a standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
                "request_id": request_id,
            }
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup and shutdown."""
    logger.info("=" * 60)
    logger.info("Starting vdiff API Server")
    logger.info("=" * 60)
    
    if server_state.config is None:
        raise RuntimeError("Configuration not set")
    
    config = server_state.config
    
    # Initialize rate limiter
    server_state.rate_limiter = RateLimiter(
        max_requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window,
    )
    
    # Initialize engine
    logger.info(f"Loading model: {config.model}")
    logger.info(f"Device: auto-detect | Dtype: {config.dtype}")
    logger.info(f"Diffusion steps: {config.diffusion_steps} | APD: {config.enable_apd}")
    
    try:
        server_state.engine = DFastLLMEngine(
            config=config,
            max_queue_size=config.max_queue_size,
            max_concurrent=config.max_concurrent_requests,
            default_timeout=config.request_timeout,
        )
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise
    
    # Wait for engine to be ready
    max_wait = 300  # 5 minutes
    start_time = time.time()
    while not server_state.engine.is_ready:
        if time.time() - start_time > max_wait:
            raise RuntimeError("Engine failed to initialize within timeout")
        await asyncio.sleep(0.5)
    
    # Initialize serving components
    model_name = config.served_model_name or config.model
    served_model_names = [model_name]
    
    server_state.completion_serving = OpenAIServingCompletion(
        engine=server_state.engine,
        model_name=model_name,
        served_model_names=served_model_names,
    )
    
    server_state.chat_serving = OpenAIServingChat(
        engine=server_state.engine,
        model_name=model_name,
        served_model_names=served_model_names,
    )
    
    # Setup metrics
    setup_metrics(model_name)
    
    logger.info("=" * 60)
    logger.info(f"vdiff API Server ready")
    logger.info(f"Listening on http://{config.host}:{config.port}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Health check: http://{config.host}:{config.port}/health")
    logger.info(f"API docs: http://{config.host}:{config.port}/docs")
    logger.info("=" * 60)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(graceful_shutdown()),
        )
    
    yield
    
    # Shutdown
    await graceful_shutdown()


async def graceful_shutdown() -> None:
    """Perform graceful shutdown."""
    if server_state.shutdown_event.is_set():
        return
    
    server_state.shutdown_event.set()
    
    logger.info("=" * 60)
    logger.info("Initiating graceful shutdown...")
    logger.info("=" * 60)
    
    # Wait for active requests (max 30 seconds)
    shutdown_start = time.time()
    while server_state.active_requests > 0:
        if time.time() - shutdown_start > 30:
            logger.warning(f"Shutdown timeout: {server_state.active_requests} requests still active")
            break
        logger.info(f"Waiting for {server_state.active_requests} active requests...")
        await asyncio.sleep(1)
    
    # Shutdown engine
    if server_state.engine:
        await server_state.engine.shutdown()
    
    logger.info("vdiff API Server shutdown complete")


def create_app(app_config: Optional[DFastLLMConfig] = None) -> FastAPI:
    """Create the FastAPI application.
    
    Args:
        app_config: Optional configuration. If not provided, uses global config.
    
    Returns:
        Configured FastAPI application.
    """
    if app_config is not None:
        server_state.config = app_config
    
    config = server_state.config
    
    app = FastAPI(
        title="vdiff API Server",
        description=(
            "Production-ready vLLM-compatible API server for Diffusion LLMs. "
            "Supports LLaDA, Dream, and other diffusion language models with "
            "OpenAI-compatible endpoints."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add middleware (order matters - last added runs first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    
    # Add CORS middleware
    if config:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_credentials=config.allow_credentials,
            allow_methods=config.allowed_methods,
            allow_headers=config.allowed_headers,
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Register routes
    register_routes(app)
    
    # Exception handlers
    @app.exception_handler(EngineError)
    async def engine_error_handler(request: Request, exc: EngineError):
        request_id = getattr(request.state, "request_id", None)
        return create_error_response(
            status_code=500,
            message=str(exc),
            error_type="engine_error",
            request_id=request_id,
        )
    
    @app.exception_handler(GenerationError)
    async def generation_error_handler(request: Request, exc: GenerationError):
        request_id = getattr(request.state, "request_id", None)
        return create_error_response(
            status_code=500,
            message=str(exc),
            error_type="generation_error",
            request_id=request_id,
        )
    
    @app.exception_handler(EngineTimeoutError)
    async def timeout_error_handler(request: Request, exc: EngineTimeoutError):
        request_id = getattr(request.state, "request_id", None)
        return create_error_response(
            status_code=504,
            message=str(exc),
            error_type="timeout_error",
            request_id=request_id,
        )
    
    @app.exception_handler(QueueFullError)
    async def queue_full_error_handler(request: Request, exc: QueueFullError):
        request_id = getattr(request.state, "request_id", None)
        return create_error_response(
            status_code=503,
            message=str(exc),
            error_type="queue_full",
            request_id=request_id,
        )
    
    @app.exception_handler(ValueError)
    async def validation_error_handler(request: Request, exc: ValueError):
        request_id = getattr(request.state, "request_id", None)
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="validation_error",
            request_id=request_id,
        )
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    @app.get("/")
    async def root():
        """Root endpoint with server info."""
        return {
            "name": "vdiff API Server",
            "version": __version__,
            "status": "ready" if server_state.is_ready else "loading",
            "docs": "/docs",
        }
    
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Detailed health check endpoint."""
        if server_state.engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        health_status = server_state.engine.get_health()
        
        if health_status.status == "unhealthy":
            raise HTTPException(status_code=503, detail=health_status.to_dict())
        
        return health_status.to_dict()
    
    @app.get("/health/live")
    async def liveness():
        """Kubernetes liveness probe."""
        return {"status": "alive"}
    
    @app.get("/health/ready")
    async def readiness():
        """Kubernetes readiness probe."""
        if not server_state.is_ready:
            raise HTTPException(status_code=503, detail="Not ready")
        return {"status": "ready"}
    
    @app.get("/version")
    async def version() -> VersionResponse:
        """Version endpoint."""
        return VersionResponse(
            version=__version__,
            vllm_compat_version=VLLM_COMPAT_VERSION,
            model_type="diffusion-llm",
        )
    
    @app.get("/v1/models")
    async def list_models(
        _: bool = Depends(verify_api_key),
    ) -> ModelList:
        """List available models matching OpenAI/vLLM format."""
        if server_state.completion_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        return server_state.completion_serving.show_available_models()
    
    @app.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest,
        raw_request: Request,
        _auth: bool = Depends(verify_api_key),
        _rate: bool = Depends(check_rate_limit),
    ):
        """Create completion matching OpenAI/vLLM format."""
        if server_state.completion_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        request_id = getattr(raw_request.state, "request_id", str(uuid.uuid4()))
        await server_state.increment_requests()
        
        try:
            result = await server_state.completion_serving.create_completion(
                request, request_id=request_id
            )
            
            if request.stream:
                # Wrap streaming response to decrement on completion
                async def stream_with_cleanup():
                    try:
                        async for chunk in result:
                            yield chunk
                    finally:
                        await server_state.decrement_active_requests()
                
                return StreamingResponse(
                    stream_with_cleanup(),
                    media_type="text/event-stream",
                    headers={"X-Request-ID": request_id},
                )
            
            if isinstance(result, CompletionResponse) and result.usage:
                record_request(
                    success=True,
                    prompt_tokens=result.usage.prompt_tokens,
                    generated_tokens=result.usage.completion_tokens or 0,
                    parallel_tokens=result.parallel_tokens_decoded or 0,
                )
            
            await server_state.decrement_active_requests()
            return result
            
        except Exception as e:
            logger.error(f"Completion error [{request_id}]: {e}")
            record_request(success=False)
            await server_state.decrement_active_requests()
            raise
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest,
        raw_request: Request,
        _auth: bool = Depends(verify_api_key),
        _rate: bool = Depends(check_rate_limit),
    ):
        """Create chat completion matching OpenAI/vLLM format."""
        if server_state.chat_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        request_id = getattr(raw_request.state, "request_id", str(uuid.uuid4()))
        await server_state.increment_requests()
        
        try:
            result = await server_state.chat_serving.create_chat_completion(
                request, request_id=request_id
            )
            
            if request.stream:
                # Wrap streaming response to decrement on completion
                async def stream_with_cleanup():
                    try:
                        async for chunk in result:
                            yield chunk
                    finally:
                        await server_state.decrement_active_requests()
                
                return StreamingResponse(
                    stream_with_cleanup(),
                    media_type="text/event-stream",
                    headers={"X-Request-ID": request_id},
                )
            
            if isinstance(result, ChatCompletionResponse) and result.usage:
                record_request(
                    success=True,
                    prompt_tokens=result.usage.prompt_tokens,
                    generated_tokens=result.usage.completion_tokens or 0,
                    parallel_tokens=result.parallel_tokens_decoded or 0,
                )
            
            await server_state.decrement_active_requests()
            return result
            
        except Exception as e:
            logger.error(f"Chat completion error [{request_id}]: {e}")
            record_request(success=False)
            await server_state.decrement_active_requests()
            raise
    
    @app.get("/metrics")
    async def get_metrics() -> Response:
        """Prometheus metrics endpoint."""
        return metrics_endpoint()
    
    @app.get("/v1/engine/stats")
    async def get_engine_stats(
        _: bool = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """Get engine statistics (vdiff extension)."""
        if server_state.engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        
        stats = server_state.engine.get_stats()
        stats["server"] = {
            "uptime_seconds": server_state.uptime_seconds,
            "total_requests": server_state.request_count,
            "active_requests": server_state.active_requests,
        }
        return stats
    
    @app.get("/v1/engine/health")
    async def get_engine_health(
        _: bool = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """Get detailed engine health (vdiff extension)."""
        if server_state.engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return server_state.engine.get_health().to_dict()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments matching vLLM style."""
    parser = argparse.ArgumentParser(
        description="vdiff API Server - Production-ready serving for Diffusion LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Name or path of the model to serve",
    )
    model_group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of the tokenizer (defaults to model)",
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use",
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    
    # Server arguments
    server_group = parser.add_argument_group("Server Options")
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    server_group.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    server_group.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level",
    )
    server_group.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name to use in API responses",
    )
    server_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (or set VDIFF_API_KEY env var)",
    )
    
    # Resource arguments
    resource_group = parser.add_argument_group("Resource Options")
    resource_group.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of tensor parallel replicas",
    )
    resource_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)",
    )
    resource_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration",
    )
    resource_group.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens per iteration",
    )
    
    # Diffusion arguments
    diffusion_group = parser.add_argument_group("Diffusion Options")
    diffusion_group.add_argument(
        "--diffusion-steps",
        type=int,
        default=64,
        help="Number of diffusion steps for generation",
    )
    diffusion_group.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for semi-autoregressive generation",
    )
    
    # APD arguments
    apd_group = parser.add_argument_group("APD (Adaptive Parallel Decoding) Options")
    apd_group.add_argument(
        "--enable-apd",
        action="store_true",
        default=True,
        help="Enable APD for faster inference",
    )
    apd_group.add_argument(
        "--disable-apd",
        action="store_true",
        default=False,
        help="Disable APD and use standard diffusion generation",
    )
    apd_group.add_argument(
        "--apd-max-parallel",
        type=int,
        default=8,
        help="Maximum tokens to decode in parallel per step",
    )
    apd_group.add_argument(
        "--apd-threshold",
        type=float,
        default=0.3,
        help="Acceptance threshold for parallel tokens (0-1)",
    )
    
    # Production arguments
    prod_group = parser.add_argument_group("Production Options")
    prod_group.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=4,
        help="Maximum concurrent generation requests",
    )
    prod_group.add_argument(
        "--max-queue-size",
        type=int,
        default=256,
        help="Maximum pending requests in queue",
    )
    prod_group.add_argument(
        "--request-timeout",
        type=float,
        default=300,
        help="Request timeout in seconds",
    )
    prod_group.add_argument(
        "--rate-limit-requests",
        type=int,
        default=100,
        help="Maximum requests per window (0 to disable)",
    )
    prod_group.add_argument(
        "--rate-limit-window",
        type=int,
        default=60,
        help="Rate limit window in seconds",
    )
    prod_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers",
    )
    
    # Optimization arguments
    opt_group = parser.add_argument_group("Performance Optimization Options")
    opt_group.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile for faster inference (PyTorch 2.0+)",
    )
    opt_group.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile",
    )
    opt_group.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile optimization mode",
    )
    opt_group.add_argument(
        "--flash-attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 if available",
    )
    opt_group.add_argument(
        "--no-flash-attention",
        action="store_true",
        default=False,
        help="Disable Flash Attention",
    )
    opt_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["8bit", "4bit"],
        help="Enable quantization (requires bitsandbytes)",
    )
    
    return parser.parse_args()


def run_server(app_config: DFastLLMConfig) -> None:
    """Run the server with the given configuration.
    
    Args:
        app_config: Server configuration.
    """
    server_state.config = app_config
    
    app = create_app(app_config)
    
    uvicorn_config = uvicorn.Config(
        app,
        host=app_config.host,
        port=app_config.port,
        log_level=app_config.uvicorn_log_level,
        access_log=True,
        workers=app_config.workers,
    )
    
    server = uvicorn.Server(uvicorn_config)
    server.run()


def main() -> None:
    """Main entry point for the API server."""
    args = parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get("VDIFF_API_KEY")
    
    # Determine feature flags (--no-* overrides --*)
    enable_apd = args.enable_apd and not args.disable_apd
    compile_model = args.compile and not args.no_compile
    use_flash_attention = args.flash_attention and not args.no_flash_attention
    use_8bit = args.quantization == "8bit"
    use_4bit = args.quantization == "4bit"
    
    # Create configuration
    server_config = DFastLLMConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        revision=args.revision,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        host=args.host,
        port=args.port,
        uvicorn_log_level=args.uvicorn_log_level,
        served_model_name=args.served_model_name,
        api_key=api_key,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        diffusion_steps=args.diffusion_steps,
        block_size=args.block_size,
        enable_apd=enable_apd,
        apd_max_parallel=args.apd_max_parallel,
        apd_threshold=args.apd_threshold,
        max_concurrent_requests=args.max_concurrent_requests,
        max_queue_size=args.max_queue_size,
        request_timeout=args.request_timeout,
        rate_limit_requests=args.rate_limit_requests,
        rate_limit_window=args.rate_limit_window,
        workers=args.workers,
        compile_model=compile_model,
        compile_mode=args.compile_mode,
        use_flash_attention=use_flash_attention,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
    )
    
    logger.info("=" * 60)
    logger.info("vdiff API Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {server_config.model}")
    logger.info(f"Diffusion Steps: {server_config.diffusion_steps}")
    logger.info(f"APD Enabled: {server_config.enable_apd}")
    logger.info(f"Max Concurrent: {server_config.max_concurrent_requests}")
    logger.info(f"Rate Limit: {server_config.rate_limit_requests}/min")
    logger.info(f"torch.compile: {compile_model} (mode={args.compile_mode})")
    logger.info(f"Flash Attention: {use_flash_attention}")
    logger.info(f"Quantization: {args.quantization or 'none'}")
    logger.info("=" * 60)
    
    run_server(server_config)


if __name__ == "__main__":
    main()
