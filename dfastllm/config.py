"""Configuration classes for dfastllm.

Production-ready configuration with diffusion-specific options.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os


@dataclass
class DFastLLMConfig:
    """Main configuration for the dfastllm engine.
    
    Comprehensive configuration with:
    - Diffusion-specific options (steps, block size)
    - APD (Adaptive Parallel Decoding) options
    - Production options (rate limiting, timeouts, queue management)
    
    Attributes:
        model: Name or path of the model to serve
        tokenizer: Name or path of the tokenizer (defaults to model)
        revision: Model revision to use
        max_model_len: Maximum model context length
        dtype: Data type for model weights (auto, float16, bfloat16, float32)
        trust_remote_code: Trust remote code from HuggingFace
        host: Host to bind the server to
        port: Port to bind the server to
        diffusion_steps: Number of diffusion steps for generation
        block_size: Block size for semi-autoregressive generation
        enable_apd: Enable APD for faster inference
        apd_max_parallel: Maximum tokens to decode in parallel
        apd_threshold: Acceptance threshold for parallel tokens
    """
    
    # Model configuration (matches vLLM)
    model: str = ""
    tokenizer: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    max_model_len: int = 4096
    dtype: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    
    # Server configuration (matches vLLM)
    host: str = "0.0.0.0"
    port: int = 8000
    uvicorn_log_level: str = "info"
    allow_credentials: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    served_model_name: Optional[str] = None
    
    # Parallel configuration (matches vLLM)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Resource configuration (matches vLLM)
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    
    # Diffusion model specific
    diffusion_steps: int = 64
    block_size: int = 32
    noise_schedule: str = "linear"
    
    # APD (Adaptive Parallel Decoding)
    enable_apd: bool = True
    apd_max_parallel: int = 8
    apd_threshold: float = 0.3
    apd_use_ar_verification: bool = False
    
    # Production settings
    max_concurrent_requests: int = 4
    max_queue_size: int = 256
    request_timeout: float = 300.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    workers: int = 1
    
    # Performance optimizations
    compile_model: bool = True  # Use torch.compile (PyTorch 2.0+)
    compile_mode: str = "reduce-overhead"  # torch.compile mode: reduce-overhead, max-autotune
    use_flash_attention: bool = True  # Use Flash Attention 2 if available
    use_8bit: bool = False  # 8-bit quantization (bitsandbytes)
    use_4bit: bool = False  # 4-bit quantization (bitsandbytes)
    
    # Diffusion-specific optimizations
    use_mixed_precision: bool = True  # Use BF16/FP16 for forward passes
    use_adaptive_steps: bool = True  # Dynamically reduce steps based on confidence
    confidence_threshold: float = 0.95  # Threshold for adaptive early stopping
    enable_early_stopping: bool = True  # Stop when all tokens are unmasked
    use_attention_cache: bool = False  # Cache attention maps (experimental)
    attention_cache_interval: int = 4  # Recompute attention every N steps
    
    # Dynamic quantization (native PyTorch, no bitsandbytes)
    use_dynamic_quantization: bool = False  # Apply INT8 dynamic quantization
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.tokenizer is None:
            self.tokenizer = self.model
        
        if self.served_model_name is None:
            self.served_model_name = self.model
        
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        
        if self.apd_max_parallel < 1:
            raise ValueError(f"apd_max_parallel must be >= 1, got {self.apd_max_parallel}")
        
        if not 0.0 <= self.apd_threshold <= 1.0:
            raise ValueError(f"apd_threshold must be in [0, 1], got {self.apd_threshold}")
        
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}"
            )
        
        if self.max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {self.max_queue_size}")
        
        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0, got {self.request_timeout}")
    
    @classmethod
    def from_env(cls) -> "DFastLLMConfig":
        """Create configuration from environment variables.
        
        Environment variables use VDIFF_ prefix for consistency.
        Falls back to common environment variables for compatibility.
        
        Returns:
            DFastLLMConfig instance populated from environment.
        """
        return cls(
            model=os.getenv("VDIFF_MODEL", os.getenv("MODEL_NAME", "")),
            tokenizer=os.getenv("VDIFF_TOKENIZER"),
            revision=os.getenv("VDIFF_REVISION"),
            max_model_len=int(os.getenv("VDIFF_MAX_MODEL_LEN", "4096")),
            dtype=os.getenv("VDIFF_DTYPE", "auto"),
            trust_remote_code=os.getenv("VDIFF_TRUST_REMOTE_CODE", "").lower() == "true",
            host=os.getenv("VDIFF_HOST", "0.0.0.0"),
            port=int(os.getenv("VDIFF_PORT", os.getenv("PORT", "8000"))),
            api_key=os.getenv("VDIFF_API_KEY"),
            tensor_parallel_size=int(os.getenv("VDIFF_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VDIFF_GPU_MEMORY_UTILIZATION", "0.9")),
            max_num_seqs=int(os.getenv("VDIFF_MAX_NUM_SEQS", "256")),
            max_num_batched_tokens=int(os.getenv("VDIFF_MAX_NUM_BATCHED_TOKENS", "4096")),
            diffusion_steps=int(os.getenv("VDIFF_DIFFUSION_STEPS", "64")),
            block_size=int(os.getenv("VDIFF_BLOCK_SIZE", "32")),
            enable_apd=os.getenv("VDIFF_ENABLE_APD", "true").lower() == "true",
            apd_max_parallel=int(os.getenv("VDIFF_APD_MAX_PARALLEL", "8")),
            apd_threshold=float(os.getenv("VDIFF_APD_THRESHOLD", "0.3")),
            max_concurrent_requests=int(os.getenv("VDIFF_MAX_CONCURRENT", "4")),
            max_queue_size=int(os.getenv("VDIFF_MAX_QUEUE_SIZE", "256")),
            request_timeout=float(os.getenv("VDIFF_REQUEST_TIMEOUT", "300")),
            rate_limit_requests=int(os.getenv("VDIFF_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("VDIFF_RATE_LIMIT_WINDOW", "60")),
            workers=int(os.getenv("VDIFF_WORKERS", "1")),
            compile_model=os.getenv("VDIFF_COMPILE", "true").lower() == "true",
            compile_mode=os.getenv("VDIFF_COMPILE_MODE", "reduce-overhead"),
            use_flash_attention=os.getenv("VDIFF_FLASH_ATTENTION", "true").lower() == "true",
            use_8bit=os.getenv("VDIFF_USE_8BIT", "false").lower() == "true",
            use_4bit=os.getenv("VDIFF_USE_4BIT", "false").lower() == "true",
            use_mixed_precision=os.getenv("VDIFF_MIXED_PRECISION", "true").lower() == "true",
            use_adaptive_steps=os.getenv("VDIFF_ADAPTIVE_STEPS", "true").lower() == "true",
            confidence_threshold=float(os.getenv("VDIFF_CONFIDENCE_THRESHOLD", "0.95")),
            enable_early_stopping=os.getenv("VDIFF_EARLY_STOPPING", "true").lower() == "true",
            use_attention_cache=os.getenv("VDIFF_ATTENTION_CACHE", "false").lower() == "true",
            attention_cache_interval=int(os.getenv("VDIFF_ATTENTION_CACHE_INTERVAL", "4")),
            use_dynamic_quantization=os.getenv("VDIFF_DYNAMIC_QUANT", "false").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "revision": self.revision,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "host": self.host,
            "port": self.port,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "diffusion_steps": self.diffusion_steps,
            "block_size": self.block_size,
            "noise_schedule": self.noise_schedule,
            "enable_apd": self.enable_apd,
            "apd_max_parallel": self.apd_max_parallel,
            "apd_threshold": self.apd_threshold,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queue_size": self.max_queue_size,
            "request_timeout": self.request_timeout,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        if not self.model:
            errors.append("model is required")
        
        if self.diffusion_steps < 1:
            errors.append("diffusion_steps must be >= 1")
        
        if self.block_size < 1:
            errors.append("block_size must be >= 1")
        
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            errors.append(f"dtype must be one of: auto, float16, bfloat16, float32")
        
        return errors


@dataclass
class ModelConfig:
    """Model-specific configuration loaded from the model.
    
    Attributes:
        model_type: Type of model (e.g., llama, gpt2, diffusion-llm)
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        vocab_size: Vocabulary size
        max_position_embeddings: Maximum sequence length
    """
    
    model_type: str = "diffusion-llm"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load configuration from a pretrained model.
        
        Args:
            model_path: Path or name of the pretrained model.
            
        Returns:
            ModelConfig instance with model parameters.
        """
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            return cls(
                model_type=getattr(hf_config, "model_type", "diffusion-llm"),
                hidden_size=getattr(hf_config, "hidden_size", 4096),
                num_attention_heads=getattr(hf_config, "num_attention_heads", 32),
                num_hidden_layers=getattr(hf_config, "num_hidden_layers", 32),
                vocab_size=getattr(hf_config, "vocab_size", 32000),
                max_position_embeddings=getattr(hf_config, "max_position_embeddings", 4096),
            )
        except Exception:
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation of model config.
        """
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
