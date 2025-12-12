"""OpenAI-compatible API server for vdiff.

Main FastAPI application matching vLLM's API server interface.
"""

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vdiff.config import VDiffConfig
from vdiff.version import __version__, VLLM_COMPAT_VERSION
from vdiff.engine import VDiffEngine
from vdiff.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    ErrorResponse,
    HealthResponse,
    VersionResponse,
)
from vdiff.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vdiff.entrypoints.openai.serving_chat import OpenAIServingChat
from vdiff.metrics import setup_metrics, metrics_endpoint, record_request

logger = logging.getLogger(__name__)

# Global state
engine: Optional[VDiffEngine] = None
completion_serving: Optional[OpenAIServingCompletion] = None
chat_serving: Optional[OpenAIServingChat] = None
config: Optional[VDiffConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup and shutdown."""
    global engine, completion_serving, chat_serving
    
    logger.info("Starting vdiff API server...")
    
    if config is None:
        raise RuntimeError("Configuration not set")
    
    # Initialize engine
    logger.info(f"Loading model: {config.model}")
    engine = VDiffEngine(config)
    
    # Wait for engine to be ready
    while not engine.is_ready:
        await asyncio.sleep(0.1)
    
    # Initialize serving components
    model_name = config.served_model_name or config.model
    served_model_names = [model_name]
    
    completion_serving = OpenAIServingCompletion(
        engine=engine,
        model_name=model_name,
        served_model_names=served_model_names,
    )
    
    chat_serving = OpenAIServingChat(
        engine=engine,
        model_name=model_name,
        served_model_names=served_model_names,
    )
    
    # Setup metrics
    setup_metrics(model_name)
    
    logger.info(f"vdiff API server ready on {config.host}:{config.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down vdiff API server...")
    if engine:
        engine.shutdown()
    logger.info("vdiff API server shutdown complete")


def create_app(app_config: Optional[VDiffConfig] = None) -> FastAPI:
    """Create the FastAPI application.
    
    Args:
        app_config: Optional configuration. If not provided, uses global config.
    
    Returns:
        Configured FastAPI application.
    """
    global config
    
    if app_config is not None:
        config = app_config
    
    app = FastAPI(
        title="vdiff API Server",
        description="vLLM-compatible API server for Diffusion LLMs",
        version=__version__,
        lifespan=lifespan,
    )
    
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
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    @app.get("/health")
    async def health() -> HealthResponse:
        """Health check endpoint matching vLLM."""
        if engine is None or not engine.is_ready:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return HealthResponse(status="healthy")
    
    @app.get("/version")
    async def version() -> VersionResponse:
        """Version endpoint."""
        return VersionResponse(
            version=__version__,
            vllm_compat_version=VLLM_COMPAT_VERSION,
            model_type="diffusion-llm",
        )
    
    @app.get("/v1/models")
    async def list_models() -> ModelList:
        """List available models matching OpenAI/vLLM format."""
        if completion_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        return completion_serving.show_available_models()
    
    @app.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest,
        raw_request: Request,
    ):
        """Create completion matching OpenAI/vLLM format."""
        if completion_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        try:
            result = await completion_serving.create_completion(request)
            
            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                )
            
            # Record metrics
            if isinstance(result, CompletionResponse) and result.usage:
                record_request(
                    success=True,
                    prompt_tokens=result.usage.prompt_tokens,
                    generated_tokens=result.usage.completion_tokens or 0,
                    parallel_tokens=result.parallel_tokens_decoded or 0,
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Completion error: {e}")
            record_request(success=False)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    message=str(e),
                    type="server_error",
                    code=500,
                ).model_dump(),
            )
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        """Create chat completion matching OpenAI/vLLM format."""
        if chat_serving is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        try:
            result = await chat_serving.create_chat_completion(request)
            
            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                )
            
            # Record metrics
            if isinstance(result, ChatCompletionResponse) and result.usage:
                record_request(
                    success=True,
                    prompt_tokens=result.usage.prompt_tokens,
                    generated_tokens=result.usage.completion_tokens or 0,
                    parallel_tokens=result.parallel_tokens_decoded or 0,
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            record_request(success=False)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    message=str(e),
                    type="server_error",
                    code=500,
                ).model_dump(),
            )
    
    @app.get("/metrics")
    async def get_metrics() -> Response:
        """Prometheus metrics endpoint."""
        return metrics_endpoint()
    
    @app.get("/v1/engine/stats")
    async def get_engine_stats():
        """Get engine statistics (vdiff extension)."""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return engine.get_stats()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments matching vLLM style."""
    parser = argparse.ArgumentParser(
        description="vdiff API Server - vLLM-compatible serving for Diffusion LLMs"
    )
    
    # Model arguments (matching vLLM)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the model to serve",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of the tokenizer (defaults to model)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    
    # Server arguments (matching vLLM)
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name to use in API responses",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication",
    )
    
    # Parallel arguments (matching vLLM)
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Number of tensor parallel replicas",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens per iteration",
    )
    
    # Diffusion model arguments
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=64,
        help="Number of diffusion steps for generation",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for semi-autoregressive generation",
    )
    
    # APD (Adaptive Parallel Decoding) arguments
    parser.add_argument(
        "--enable-apd",
        action="store_true",
        default=True,
        help="Enable APD (Adaptive Parallel Decoding) for faster inference",
    )
    parser.add_argument(
        "--disable-apd",
        action="store_true",
        default=False,
        help="Disable APD and use standard diffusion generation",
    )
    parser.add_argument(
        "--apd-max-parallel",
        type=int,
        default=8,
        help="APD: Maximum tokens to decode in parallel per step",
    )
    parser.add_argument(
        "--apd-threshold",
        type=float,
        default=0.3,
        help="APD: Acceptance threshold for parallel tokens (0-1)",
    )
    
    return parser.parse_args()


def run_server(app_config: VDiffConfig) -> None:
    """Run the server with the given configuration.
    
    Args:
        app_config: Server configuration.
    """
    global config
    config = app_config
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.uvicorn_log_level,
    )


def main() -> None:
    """Main entry point for the API server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Parse arguments
    args = parse_args()
    
    # Determine if APD is enabled (--disable-apd overrides --enable-apd)
    enable_apd = args.enable_apd and not args.disable_apd
    
    # Create configuration
    server_config = VDiffConfig(
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
        api_key=args.api_key,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        diffusion_steps=args.diffusion_steps,
        block_size=args.block_size,
        enable_apd=enable_apd,
        apd_max_parallel=args.apd_max_parallel,
        apd_threshold=args.apd_threshold,
    )
    
    logger.info(f"Starting vdiff server with model: {server_config.model}")
    logger.info(f"Diffusion: steps={server_config.diffusion_steps}, APD={server_config.enable_apd}")
    
    # Run server
    run_server(server_config)


if __name__ == "__main__":
    main()
