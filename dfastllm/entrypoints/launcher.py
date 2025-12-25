"""Server launcher for vdiff.

Provides utilities for launching the vdiff server programmatically.
"""

import logging
import multiprocessing
from typing import Optional

from dfastllm.config import DFastLLMConfig

logger = logging.getLogger(__name__)


def launch_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tokenizer: Optional[str] = None,
    revision: Optional[str] = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enable_kv_cache: bool = True,
    enable_parallel_decoding: bool = True,
    confidence_threshold: float = 0.8,
    block_size: int = 4,
    background: bool = False,
    **kwargs,
) -> Optional[multiprocessing.Process]:
    """Launch the vdiff server.
    
    Args:
        model: Name or path of the model to serve.
        host: Host to bind to.
        port: Port to bind to.
        tokenizer: Tokenizer name/path (defaults to model).
        revision: Model revision.
        dtype: Data type for weights.
        trust_remote_code: Trust remote code.
        max_model_len: Maximum context length.
        tensor_parallel_size: TP size.
        gpu_memory_utilization: GPU memory fraction.
        enable_kv_cache: Enable KV cache.
        enable_parallel_decoding: Enable parallel decoding.
        confidence_threshold: Confidence threshold for parallel decoding.
        block_size: KV cache block size.
        background: Run in background process.
        **kwargs: Additional configuration options.
    
    Returns:
        Process if running in background, None otherwise.
    """
    config = DFastLLMConfig(
        model=model,
        tokenizer=tokenizer,
        revision=revision,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_kv_cache=enable_kv_cache,
        enable_parallel_decoding=enable_parallel_decoding,
        confidence_threshold=confidence_threshold,
        block_size=block_size,
        **kwargs,
    )
    
    if background:
        process = multiprocessing.Process(
            target=_run_server_process,
            args=(config,),
        )
        process.start()
        logger.info(f"vdiff server started in background (PID: {process.pid})")
        return process
    else:
        _run_server_process(config)
        return None


def _run_server_process(config: DFastLLMConfig) -> None:
    """Run the server in the current process."""
    from dfastllm.entrypoints.openai.api_server import run_server
    run_server(config)


def wait_for_server(
    host: str = "localhost",
    port: int = 8000,
    timeout: float = 300.0,
    check_interval: float = 1.0,
) -> bool:
    """Wait for the server to be ready.
    
    Args:
        host: Server host.
        port: Server port.
        timeout: Maximum time to wait in seconds.
        check_interval: Interval between health checks.
    
    Returns:
        True if server is ready, False if timeout.
    """
    import time
    import httpx
    
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Server ready at {host}:{port}")
                return True
        except Exception:
            pass
        
        time.sleep(check_interval)
    
    logger.error(f"Server at {host}:{port} not ready after {timeout}s")
    return False


def stop_server(process: multiprocessing.Process, timeout: float = 10.0) -> None:
    """Stop a background server process.
    
    Args:
        process: Server process to stop.
        timeout: Timeout for graceful shutdown.
    """
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
        
        if process.is_alive():
            logger.warning("Server did not terminate gracefully, killing")
            process.kill()
            process.join(timeout=5.0)
    
    logger.info("Server stopped")
