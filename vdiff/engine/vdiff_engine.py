"""Core vdiff Engine for diffusion language model inference.

A standalone serving engine for Diffusion LLMs (LLaDA, Dream, etc.)
providing vLLM-compatible API for RHOAI, KServe, and llm-d deployments.

Supports:
- Standard diffusion generation (masked diffusion)
- APD (Adaptive Parallel Decoding) for improved throughput
"""

from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
import asyncio
import logging
import time
import uuid

from vdiff.config import VDiffConfig, ModelConfig
from vdiff.engine.sampling_params import SamplingParams
from vdiff.engine.outputs import CompletionOutput, RequestOutput, RequestMetrics
from vdiff.engine.tokenizer import TokenizerWrapper
from vdiff.engine.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
    diffusion_generate,
    is_diffusion_model,
)
from vdiff.engine.apd import APDDecoder, APDConfig, apd_generate

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, running in mock mode")


@dataclass
class EngineStats:
    """Runtime statistics for the engine."""
    requests_processed: int = 0
    tokens_generated: int = 0
    total_latency_ms: float = 0.0
    avg_tokens_per_step: float = 0.0


class VDiffEngine:
    """Core inference engine for diffusion language models.
    
    Provides vLLM-compatible interface for serving diffusion LLMs
    on RHOAI, KServe, llm-d, and other platforms.
    
    Supported models:
    - LLaDA (GSAI-ML/LLaDA-8B-Instruct, GSAI-ML/LLaDA-8B-Base)
    - Dream
    - Any HuggingFace diffusion LLM
    
    Features:
    - Standard masked diffusion generation
    - APD (Adaptive Parallel Decoding) for faster inference
    - vLLM-compatible API
    """
    
    def __init__(self, config: VDiffConfig):
        """Initialize the vdiff engine.
        
        Args:
            config: Engine configuration.
        """
        self.config = config
        self._model = None
        self._tokenizer: Optional[TokenizerWrapper] = None
        self._model_config: Optional[ModelConfig] = None
        self._is_ready = False
        self._stats = EngineStats()
        
        # Diffusion components
        self._diffusion_sampler: Optional[DiffusionSampler] = None
        self._apd_decoder: Optional[APDDecoder] = None
        self._is_diffusion_model = False
        self._mask_id = 126336  # Default LLaDA mask ID
        
        # Device setup
        self._device = self._get_device()
        
        # Load model and tokenizer
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model}")
        
        try:
            # Load tokenizer
            self._tokenizer = TokenizerWrapper(
                tokenizer_name=self.config.tokenizer or self.config.model,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Load model configuration
            self._model_config = ModelConfig.from_pretrained(self.config.model)
            
            if TORCH_AVAILABLE:
                self._load_torch_model()
            else:
                logger.warning("PyTorch not available, using mock model")
                self._model = MockModel()
            
            # Check if this is a diffusion model
            self._is_diffusion_model = is_diffusion_model(self.config.model)
            
            if self._is_diffusion_model:
                self._setup_diffusion_components()
            
            self._is_ready = True
            logger.info(f"Model loaded successfully on {self._device}")
            logger.info(f"Diffusion model: {self._is_diffusion_model}")
            logger.info(f"APD enabled: {self.config.enable_apd}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_diffusion_components(self) -> None:
        """Setup diffusion sampler and APD decoder."""
        # Get mask token ID
        if hasattr(self._tokenizer._tokenizer, 'mask_token_id'):
            mask_id = self._tokenizer._tokenizer.mask_token_id
            if mask_id is not None:
                self._mask_id = mask_id
        
        # Initialize diffusion sampler
        sampler_config = DiffusionSamplerConfig(
            steps=self.config.diffusion_steps,
            block_length=self.config.block_size,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=self._mask_id,
        )
        self._diffusion_sampler = DiffusionSampler(
            model=self._model,
            tokenizer=self._tokenizer._tokenizer,
            config=sampler_config,
        )
        
        # Initialize APD decoder if enabled
        if self.config.enable_apd:
            apd_config = APDConfig(
                enabled=True,
                max_parallel_tokens=self.config.apd_max_parallel,
                acceptance_threshold=self.config.apd_threshold,
                temperature=0.0,
            )
            self._apd_decoder = APDDecoder(config=apd_config)
            logger.info(f"APD decoder initialized (max_parallel={self.config.apd_max_parallel})")
        
        logger.info(f"Diffusion sampler initialized (mask_id={self._mask_id})")
    
    def _load_torch_model(self) -> None:
        """Load the PyTorch model."""
        from transformers import AutoModelForCausalLM
        
        # Determine dtype
        if self.config.dtype == "auto":
            dtype = torch.float16 if self._device == "cuda" else torch.float32
        elif self.config.dtype == "float16":
            dtype = torch.float16
        elif self.config.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        logger.info(f"Loading model with dtype: {dtype}")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        if self._device != "cuda":
            self._model = self._model.to(self._device)
        
        self._model.eval()
    
    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready for inference."""
        return self._is_ready
    
    @property
    def tokenizer(self) -> TokenizerWrapper:
        """Get the tokenizer wrapper."""
        return self._tokenizer
    
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> RequestOutput:
        """Generate text completion synchronously.
        
        Args:
            prompt: Input prompt text.
            sampling_params: Sampling parameters for generation.
            request_id: Unique request identifier.
        
        Returns:
            RequestOutput with generated text.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        metrics = RequestMetrics(arrival_time=time.time())
        
        # Tokenize input
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt" if TORCH_AVAILABLE else None)
        
        if TORCH_AVAILABLE:
            input_ids = input_ids.to(self._device)
        
        metrics.prompt_tokens = len(input_ids[0]) if TORCH_AVAILABLE else len(input_ids)
        
        try:
            start_time = time.time()
            
            if self._is_diffusion_model:
                # Use APD if enabled, otherwise standard diffusion
                if self.config.enable_apd and self._apd_decoder:
                    output_ids = self._apd_generate(input_ids, sampling_params)
                else:
                    output_ids = self._diffusion_generate(input_ids, sampling_params)
            else:
                output_ids = self._standard_generate(input_ids, sampling_params)
            
            metrics.first_token_time = time.time()
            
            # Decode output
            generated_text = self._tokenizer.decode(
                output_ids,
                skip_special_tokens=sampling_params.skip_special_tokens,
            )
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            metrics.finished_time = time.time()
            metrics.generated_tokens = len(output_ids[0]) - metrics.prompt_tokens if TORCH_AVAILABLE else 0
            
            # Create output
            completion_output = CompletionOutput(
                index=0,
                text=generated_text,
                token_ids=output_ids[0].tolist() if TORCH_AVAILABLE else output_ids,
                finish_reason="stop" if self._check_stop_condition(output_ids, sampling_params) else "length",
            )
            
            output = RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=input_ids[0].tolist() if TORCH_AVAILABLE else input_ids,
                outputs=[completion_output],
                finished=True,
                metrics=metrics,
            )
            
            # Update stats
            self._stats.requests_processed += 1
            self._stats.tokens_generated += metrics.generated_tokens
            self._stats.total_latency_ms += (time.time() - start_time) * 1000
            
            return output
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> RequestOutput:
        """Generate text completion asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, sampling_params, request_id
        )
    
    async def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Generate text completion with streaming."""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        output = await self.generate_async(prompt, sampling_params, request_id)
        yield output
    
    def _standard_generate(
        self,
        input_ids: "torch.Tensor",
        sampling_params: SamplingParams,
    ) -> "torch.Tensor":
        """Standard autoregressive generation."""
        if not TORCH_AVAILABLE:
            return input_ids + [0] * sampling_params.max_tokens
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": sampling_params.max_tokens,
                "temperature": max(sampling_params.temperature, 1e-7),
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k if sampling_params.top_k > 0 else None,
                "do_sample": sampling_params.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            output_ids = self._model.generate(input_ids, **gen_kwargs)
        
        return output_ids
    
    def _diffusion_generate(
        self,
        input_ids: "torch.Tensor",
        sampling_params: SamplingParams,
    ) -> "torch.Tensor":
        """Diffusion-based generation using masked diffusion algorithm."""
        if not TORCH_AVAILABLE:
            return self._standard_generate(input_ids, sampling_params)
        
        gen_length = sampling_params.max_tokens
        steps = min(self.config.diffusion_steps, gen_length)
        
        # Adjust block_length
        block_length = min(32, gen_length)
        if gen_length % block_length != 0:
            block_length = gen_length
        
        # Ensure steps divisible by num_blocks
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            steps = max(num_blocks, (steps // num_blocks) * num_blocks)
        
        with torch.no_grad():
            output_ids = diffusion_generate(
                model=self._model,
                prompt=input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=sampling_params.temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=self._mask_id,
            )
        
        return output_ids
    
    def _apd_generate(
        self,
        input_ids: "torch.Tensor",
        sampling_params: SamplingParams,
    ) -> "torch.Tensor":
        """APD (Adaptive Parallel Decoding) generation."""
        if not TORCH_AVAILABLE or self._apd_decoder is None:
            return self._diffusion_generate(input_ids, sampling_params)
        
        gen_length = sampling_params.max_tokens
        steps = min(self.config.diffusion_steps, gen_length)
        
        with torch.no_grad():
            output_ids = self._apd_decoder.generate(
                model=self._model,
                prompt=input_ids,
                gen_length=gen_length,
                steps=steps,
                mask_id=self._mask_id,
                temperature=sampling_params.temperature,
            )
        
        # Update stats from APD decoder
        apd_stats = self._apd_decoder.get_stats()
        self._stats.avg_tokens_per_step = apd_stats.get("avg_tokens_per_step", 0.0)
        
        return output_ids
    
    def _check_stop_condition(
        self,
        output_ids: "torch.Tensor",
        sampling_params: SamplingParams,
    ) -> bool:
        """Check if generation should stop."""
        if not TORCH_AVAILABLE:
            return True
        
        if self._tokenizer.eos_token_id is not None:
            if output_ids[0, -1].item() == self._tokenizer.eos_token_id:
                return True
        
        if sampling_params.stop_token_ids:
            if output_ids[0, -1].item() in sampling_params.stop_token_ids:
                return True
        
        return False
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "model": self.config.model,
            "tokenizer": self.config.tokenizer,
            "max_model_len": self.config.max_model_len,
            "dtype": self.config.dtype,
            "is_diffusion_model": self._is_diffusion_model,
            "apd_enabled": self.config.enable_apd,
            "diffusion_steps": self.config.diffusion_steps,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine runtime statistics."""
        stats = {
            "requests_processed": self._stats.requests_processed,
            "tokens_generated": self._stats.tokens_generated,
            "avg_latency_ms": self._stats.total_latency_ms / max(1, self._stats.requests_processed),
            "is_ready": self._is_ready,
            "device": self._device,
            "is_diffusion_model": self._is_diffusion_model,
        }
        
        if self._apd_decoder:
            stats["apd"] = self._apd_decoder.get_stats()
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        logger.info("Shutting down vdiff engine")
        
        self._is_ready = False
        
        if TORCH_AVAILABLE and self._model is not None:
            del self._model
            self._model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("vdiff engine shutdown complete")


class MockModel:
    """Mock model for testing without PyTorch."""
    
    def generate(self, input_ids: List[int], **kwargs) -> List[int]:
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        return input_ids + [0] * max_new_tokens
    
    def __call__(self, *args, **kwargs):
        return type("Output", (), {"logits": None})()


class AsyncVDiffEngine:
    """Async wrapper for VDiffEngine with request queue."""
    
    def __init__(self, config: VDiffConfig):
        self.config = config
        self._engine: Optional[VDiffEngine] = None
        self._is_running = False
    
    async def start(self) -> None:
        """Start the async engine."""
        self._engine = VDiffEngine(self.config)
        self._is_running = True
        logger.info("Async vdiff engine started")
    
    async def stop(self) -> None:
        """Stop the async engine."""
        self._is_running = False
        if self._engine:
            self._engine.shutdown()
        logger.info("Async vdiff engine stopped")
    
    @property
    def is_ready(self) -> bool:
        return self._engine is not None and self._engine.is_ready
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> RequestOutput:
        if not self._engine:
            raise RuntimeError("Engine not started")
        return await self._engine.generate_async(prompt, sampling_params, request_id)
    
    async def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[RequestOutput]:
        if not self._engine:
            raise RuntimeError("Engine not started")
        async for output in self._engine.generate_stream(prompt, sampling_params, request_id):
            yield output
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._engine:
            return {"is_ready": False}
        return self._engine.get_stats()
