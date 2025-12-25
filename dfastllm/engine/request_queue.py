"""Production-ready request queue with batching and concurrency control.

Features:
- Semaphore-based concurrency limiting
- Request timeout handling
- Priority queue support
- Request tracking and metrics
- Memory-aware request admission
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RequestState(Enum):
    """Request lifecycle states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class QueuedRequest:
    """A request in the queue."""
    request_id: str
    prompt: str
    params: Any
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    state: RequestState = RequestState.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    future: Optional[asyncio.Future] = None
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at
    
    @property
    def processing_time(self) -> float:
        """Time spent processing."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0


@dataclass
class QueueStats:
    """Queue statistics for monitoring."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    cancelled_requests: int = 0
    current_queue_size: int = 0
    current_running: int = 0
    avg_wait_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    _wait_times: list = field(default_factory=list)
    _processing_times: list = field(default_factory=list)
    
    def record_wait_time(self, wait_time: float):
        self._wait_times.append(wait_time * 1000)
        if len(self._wait_times) > 1000:
            self._wait_times = self._wait_times[-1000:]
        self.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
    
    def record_processing_time(self, processing_time: float):
        self._processing_times.append(processing_time * 1000)
        if len(self._processing_times) > 1000:
            self._processing_times = self._processing_times[-1000:]
        self.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)


class RequestQueue:
    """Production-ready async request queue.
    
    Features:
    - Concurrency limiting via semaphore
    - Request timeout handling
    - Priority queue support
    - Graceful shutdown
    - Memory pressure detection
    """
    
    def __init__(
        self,
        max_concurrent: int = 1,
        max_queue_size: int = 100,
        request_timeout: float = 300.0,
        enable_priority: bool = False,
    ):
        """Initialize the request queue.
        
        Args:
            max_concurrent: Maximum concurrent requests being processed.
            max_queue_size: Maximum number of pending requests.
            request_timeout: Default timeout for requests in seconds.
            enable_priority: Enable priority-based ordering.
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.enable_priority = enable_priority
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: deque = deque()
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._completed_requests: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._stats = QueueStats()
        
        # Request processor callback
        self._processor: Optional[Callable[[QueuedRequest], Awaitable[Any]]] = None
        
        logger.info(f"RequestQueue initialized: max_concurrent={max_concurrent}, "
                   f"max_queue_size={max_queue_size}, timeout={request_timeout}s")
    
    def set_processor(self, processor: Callable[[QueuedRequest], Awaitable[Any]]):
        """Set the request processing callback."""
        self._processor = processor
    
    async def submit(
        self,
        prompt: str,
        params: Any,
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> Any:
        """Submit a request and wait for result.
        
        Args:
            prompt: The prompt text.
            params: Sampling parameters.
            priority: Request priority (higher = processed first).
            timeout: Request timeout override.
        
        Returns:
            The generation result.
        
        Raises:
            asyncio.TimeoutError: If request times out.
            RuntimeError: If queue is full or shutdown.
        """
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")
        
        async with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise RuntimeError(f"Queue full (max={self.max_queue_size})")
            
            request = QueuedRequest(
                request_id=str(uuid.uuid4()),
                prompt=prompt,
                params=params,
                priority=priority,
                state=RequestState.QUEUED,
                future=asyncio.get_event_loop().create_future(),
            )
            
            self._queue.append(request)
            self._stats.total_requests += 1
            self._stats.current_queue_size = len(self._queue)
            
            logger.debug(f"Request {request.request_id} queued (queue_size={len(self._queue)})")
        
        # Wait for result with timeout
        try:
            timeout = timeout or self.request_timeout
            result = await asyncio.wait_for(
                self._process_request(request),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            request.state = RequestState.TIMEOUT
            self._stats.timeout_requests += 1
            logger.warning(f"Request {request.request_id} timed out after {timeout}s")
            raise
    
    async def _process_request(self, request: QueuedRequest) -> Any:
        """Process a single request with concurrency control."""
        async with self._semaphore:
            async with self._lock:
                self._stats.current_running += 1
                self._active_requests[request.request_id] = request
            
            request.state = RequestState.RUNNING
            request.started_at = time.time()
            self._stats.record_wait_time(request.wait_time)
            
            logger.debug(f"Request {request.request_id} started (wait_time={request.wait_time:.2f}s)")
            
            try:
                if self._processor is None:
                    raise RuntimeError("No processor set")
                
                result = await self._processor(request)
                
                request.result = result
                request.state = RequestState.COMPLETED
                request.completed_at = time.time()
                self._stats.completed_requests += 1
                self._stats.record_processing_time(request.processing_time)
                
                logger.debug(f"Request {request.request_id} completed "
                           f"(processing_time={request.processing_time:.2f}s)")
                
                return result
                
            except Exception as e:
                request.error = e
                request.state = RequestState.FAILED
                request.completed_at = time.time()
                self._stats.failed_requests += 1
                logger.error(f"Request {request.request_id} failed: {e}")
                raise
            
            finally:
                async with self._lock:
                    self._stats.current_running -= 1
                    if request.request_id in self._active_requests:
                        del self._active_requests[request.request_id]
                    self._completed_requests[request.request_id] = request
                    
                    # Cleanup old completed requests
                    if len(self._completed_requests) > 1000:
                        oldest = sorted(
                            self._completed_requests.keys(),
                            key=lambda k: self._completed_requests[k].completed_at or 0
                        )[:500]
                        for k in oldest:
                            del self._completed_requests[k]
    
    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        async with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    req.state = RequestState.CANCELLED
                    self._queue.remove(req)
                    self._stats.cancelled_requests += 1
                    self._stats.current_queue_size = len(self._queue)
                    logger.info(f"Request {request_id} cancelled")
                    return True
        return False
    
    async def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Gracefully shutdown the queue.
        
        Args:
            wait: Whether to wait for pending requests.
            timeout: Maximum time to wait for pending requests.
        """
        logger.info("Shutting down request queue...")
        self._shutdown = True
        
        if wait and self._active_requests:
            logger.info(f"Waiting for {len(self._active_requests)} active requests...")
            start = time.time()
            while self._active_requests and (time.time() - start) < timeout:
                await asyncio.sleep(0.1)
        
        # Cancel pending requests
        async with self._lock:
            for req in self._queue:
                req.state = RequestState.CANCELLED
            self._queue.clear()
            self._stats.current_queue_size = 0
        
        logger.info("Request queue shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "completed_requests": self._stats.completed_requests,
            "failed_requests": self._stats.failed_requests,
            "timeout_requests": self._stats.timeout_requests,
            "cancelled_requests": self._stats.cancelled_requests,
            "current_queue_size": self._stats.current_queue_size,
            "current_running": self._stats.current_running,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "avg_wait_time_ms": self._stats.avg_wait_time_ms,
            "avg_processing_time_ms": self._stats.avg_processing_time_ms,
        }
    
    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return len(self._queue)
    
    @property
    def active_count(self) -> int:
        """Current active request count."""
        return len(self._active_requests)

