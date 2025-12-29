"""Unit tests for Continuous Batching implementation.

Tests the request batching, scheduling, and batch generation
for improved throughput in diffusion LLM inference.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

from dfastllm.engine.continuous_batching import (
    RequestBatcher,
    BatcherConfig,
    BatchedRequest,
    BatchResult,
    BatcherStats,
    RequestPriority,
    BatchedDiffusionGenerator,
    ContinuousBatchingScheduler,
    PrefixCache,
    create_continuous_batching_engine,
)


class TestBatcherConfig:
    """Tests for BatcherConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatcherConfig()
        
        assert config.max_batch_size == 8
        assert config.max_wait_time_ms == 50.0
        assert config.max_tokens_per_batch == 4096
        assert config.pad_token_id == 0
        assert config.enable_priority_queue is True
        assert config.dynamic_batch_size is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatcherConfig(
            max_batch_size=16,
            max_wait_time_ms=100.0,
            enable_priority_queue=False,
        )
        
        assert config.max_batch_size == 16
        assert config.max_wait_time_ms == 100.0
        assert config.enable_priority_queue is False


class TestBatchedRequest:
    """Tests for BatchedRequest dataclass."""
    
    def test_request_creation(self):
        """Test creating a batched request."""
        request = BatchedRequest(
            request_id="test-123",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=64,
            temperature=0.7,
        )
        
        assert request.request_id == "test-123"
        assert request.prompt_tokens == [1, 2, 3]
        assert request.max_new_tokens == 64
        assert request.temperature == 0.7
        assert request.priority == RequestPriority.NORMAL
    
    def test_request_priority_comparison(self):
        """Test request priority ordering."""
        high_priority = BatchedRequest(
            request_id="high",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.HIGH,
        )
        
        low_priority = BatchedRequest(
            request_id="low",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.LOW,
        )
        
        assert high_priority < low_priority
    
    def test_request_arrival_time_comparison(self):
        """Test requests with same priority ordered by arrival time."""
        import time
        
        earlier = BatchedRequest(
            request_id="earlier",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.NORMAL,
        )
        earlier.arrival_time = 1000.0
        
        later = BatchedRequest(
            request_id="later",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.NORMAL,
        )
        later.arrival_time = 2000.0
        
        assert earlier < later


class TestBatchResult:
    """Tests for BatchResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a batch result."""
        result = BatchResult(
            request_id="test-123",
            output_tokens=[1, 2, 3, 4, 5],
            generated_text="Hello world",
            finish_reason="stop",
            prompt_tokens=3,
            generated_tokens=5,
            latency_ms=100.5,
        )
        
        assert result.request_id == "test-123"
        assert result.generated_text == "Hello world"
        assert result.finish_reason == "stop"
        assert result.latency_ms == 100.5


class TestRequestPriority:
    """Tests for RequestPriority enum."""
    
    def test_priority_values(self):
        """Test priority enum values."""
        assert RequestPriority.LOW.value == 0
        assert RequestPriority.NORMAL.value == 1
        assert RequestPriority.HIGH.value == 2
        assert RequestPriority.CRITICAL.value == 3
    
    def test_priority_ordering(self):
        """Test priority value ordering."""
        priorities = [
            RequestPriority.CRITICAL,
            RequestPriority.HIGH,
            RequestPriority.NORMAL,
            RequestPriority.LOW,
        ]
        
        for i in range(len(priorities) - 1):
            assert priorities[i].value > priorities[i + 1].value


class TestBatcherStats:
    """Tests for BatcherStats dataclass."""
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = BatcherStats()
        
        assert stats.total_batches == 0
        assert stats.total_requests == 0
        assert stats.avg_batch_size == 0.0
        assert stats.avg_wait_time_ms == 0.0
    
    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = BatcherStats(
            total_batches=10,
            total_requests=50,
            avg_batch_size=5.0,
            avg_wait_time_ms=25.0,
            max_batch_size_seen=8,
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_batches"] == 10
        assert stats_dict["total_requests"] == 50
        assert stats_dict["avg_batch_size"] == 5.0
        assert stats_dict["max_batch_size_seen"] == 8


class TestRequestBatcher:
    """Tests for the RequestBatcher class."""
    
    @pytest.fixture
    def batcher(self):
        """Create a request batcher with test config."""
        config = BatcherConfig(
            max_batch_size=4,
            max_wait_time_ms=10.0,
        )
        return RequestBatcher(config)
    
    @pytest.mark.asyncio
    async def test_add_request(self, batcher):
        """Test adding a request to the batcher."""
        request = BatchedRequest(
            request_id="test-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        
        await batcher.add_request(request)
        
        assert batcher.get_queue_size() == 1
    
    @pytest.mark.asyncio
    async def test_batch_fills_up(self, batcher):
        """Test batch is ready when filled up."""
        for i in range(4):
            request = BatchedRequest(
                request_id=f"test-{i}",
                prompt_tokens=[1, 2, 3],
                max_new_tokens=10,
            )
            await batcher.add_request(request)
        
        batch = await batcher.get_batch()
        
        assert len(batch) == 4
        assert batcher.get_queue_size() == 0
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self, batcher):
        """Test batch is returned after timeout even if not full."""
        request = BatchedRequest(
            request_id="test-1",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        await batcher.add_request(request)
        
        batch = await batcher.get_batch()
        
        assert len(batch) == 1
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self):
        """Test priority queue orders requests correctly."""
        config = BatcherConfig(
            max_batch_size=4,
            max_wait_time_ms=10.0,
            enable_priority_queue=True,
        )
        batcher = RequestBatcher(config)
        
        low = BatchedRequest(
            request_id="low",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.LOW,
        )
        high = BatchedRequest(
            request_id="high",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.HIGH,
        )
        normal = BatchedRequest(
            request_id="normal",
            prompt_tokens=[1],
            max_new_tokens=10,
            priority=RequestPriority.NORMAL,
        )
        
        await batcher.add_request(low)
        await batcher.add_request(normal)
        await batcher.add_request(high)
        
        batch = await batcher.get_batch()
        
        assert batch[0].request_id == "high"
        assert batch[1].request_id == "normal"
        assert batch[2].request_id == "low"
    
    def test_get_stats(self, batcher):
        """Test getting batcher statistics."""
        stats = batcher.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_batches" in stats
        assert "total_requests" in stats


class TestPrefixCache:
    """Tests for the PrefixCache class."""
    
    def test_cache_initialization(self):
        """Test prefix cache initialization."""
        cache = PrefixCache(
            max_cache_size=100,
            min_prefix_length=16,
            max_prefix_length=512,
        )
        
        assert cache.max_cache_size == 100
        assert cache.min_prefix_length == 16
    
    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cache = PrefixCache(min_prefix_length=4)
        
        tokens = list(range(20))
        kv_cache = {"layer_0": "mock_kv"}
        
        cache.put(tokens, kv_cache)
        
        result = cache.get(tokens)
        assert result == kv_cache
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = PrefixCache(min_prefix_length=4)
        
        tokens = list(range(20))
        
        result = cache.get(tokens)
        assert result is None
    
    def test_cache_short_sequence_ignored(self):
        """Test short sequences are not cached."""
        cache = PrefixCache(min_prefix_length=16)
        
        short_tokens = list(range(10))
        kv_cache = {"layer_0": "mock_kv"}
        
        cache.put(short_tokens, kv_cache)
        
        result = cache.get(short_tokens)
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = PrefixCache(max_cache_size=2, min_prefix_length=4)
        
        tokens1 = list(range(20))
        tokens2 = list(range(20, 40))
        tokens3 = list(range(40, 60))
        
        cache.put(tokens1, "kv1")
        cache.put(tokens2, "kv2")
        
        cache.get(tokens1)
        
        cache.put(tokens3, "kv3")
        
        assert cache.get(tokens1) == "kv1"
        assert cache.get(tokens3) == "kv3"
    
    def test_hit_rate_tracking(self):
        """Test cache hit rate tracking."""
        cache = PrefixCache(min_prefix_length=4)
        
        tokens = list(range(20))
        cache.put(tokens, "kv")
        
        cache.get(tokens)
        cache.get(tokens)
        cache.get(list(range(30, 50)))
        
        hit_rate = cache.get_hit_rate()
        assert hit_rate == 2 / 3
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = PrefixCache(min_prefix_length=4)
        
        tokens = list(range(20))
        cache.put(tokens, "kv")
        cache.get(tokens)
        cache.get(list(range(30, 50)))
        
        stats = cache.get_stats()
        
        assert stats["cache_size"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = PrefixCache(min_prefix_length=4)
        
        tokens = list(range(20))
        cache.put(tokens, "kv")
        
        cache.clear()
        
        assert cache.get(tokens) is None
        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["hit_count"] == 0


class TestBatchedDiffusionGenerator:
    """Tests for BatchedDiffusionGenerator."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.device = "cpu"
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.mask_token_id = 126336
        return tokenizer
    
    def test_generator_initialization(self, mock_model, mock_tokenizer):
        """Test generator initialization."""
        generator = BatchedDiffusionGenerator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            mask_id=126336,
            device="cpu",
        )
        
        assert generator.model == mock_model
        assert generator.mask_id == 126336


class TestContinuousBatchingScheduler:
    """Tests for ContinuousBatchingScheduler."""
    
    @pytest.fixture
    def mock_generator(self):
        """Create a mock generator."""
        generator = Mock()
        generator.device = "cpu"
        generator.generate_batch = Mock(return_value=[])
        return generator
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="generated text")
        return tokenizer
    
    def test_scheduler_initialization(self, mock_generator, mock_tokenizer):
        """Test scheduler initialization."""
        scheduler = ContinuousBatchingScheduler(
            generator=mock_generator,
            tokenizer=mock_tokenizer,
        )
        
        assert scheduler.generator == mock_generator
        assert scheduler._running is False
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, mock_generator, mock_tokenizer):
        """Test starting and stopping scheduler."""
        scheduler = ContinuousBatchingScheduler(
            generator=mock_generator,
            tokenizer=mock_tokenizer,
        )
        
        await scheduler.start()
        assert scheduler._running is True
        
        await scheduler.stop()
        assert scheduler._running is False
    
    def test_get_stats(self, mock_generator, mock_tokenizer):
        """Test getting scheduler statistics."""
        scheduler = ContinuousBatchingScheduler(
            generator=mock_generator,
            tokenizer=mock_tokenizer,
        )
        
        stats = scheduler.get_stats()
        
        assert isinstance(stats, dict)
        assert "running" in stats


class TestCreateContinuousBatchingEngine:
    """Tests for the factory function."""
    
    def test_create_engine(self):
        """Test creating continuous batching engine."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        config = BatcherConfig(max_batch_size=4)
        
        engine = create_continuous_batching_engine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
            device="cpu",
        )
        
        assert isinstance(engine, ContinuousBatchingScheduler)
        assert engine.config.max_batch_size == 4
