"""Configuration classes for vdiff.

Matches vLLM configuration patterns while adding diffusion-specific options.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class VDiffConfig:
    """Main configuration for the vdiff engine.
    
    Attributes match vLLM's configuration with additional diffusion-specific options.
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
    
    @classmethod
    def from_env(cls) -> "VDiffConfig":
        """Create configuration from environment variables."""
        return cls(
            model=os.getenv("VDIFF_MODEL", os.getenv("MODEL_NAME", "")),
            tokenizer=os.getenv("VDIFF_TOKENIZER"),
            revision=os.getenv("VDIFF_REVISION"),
            max_model_len=int(os.getenv("VDIFF_MAX_MODEL_LEN", "4096")),
            dtype=os.getenv("VDIFF_DTYPE", "auto"),
            trust_remote_code=os.getenv("VDIFF_TRUST_REMOTE_CODE", "").lower() == "true",
            host=os.getenv("VDIFF_HOST", "0.0.0.0"),
            port=int(os.getenv("VDIFF_PORT", os.getenv("PORT", "8000"))),
            tensor_parallel_size=int(os.getenv("VDIFF_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VDIFF_GPU_MEMORY_UTILIZATION", "0.9")),
            max_num_seqs=int(os.getenv("VDIFF_MAX_NUM_SEQS", "256")),
            max_num_batched_tokens=int(os.getenv("VDIFF_MAX_NUM_BATCHED_TOKENS", "4096")),
            diffusion_steps=int(os.getenv("VDIFF_DIFFUSION_STEPS", "64")),
            block_size=int(os.getenv("VDIFF_BLOCK_SIZE", "32")),
            enable_apd=os.getenv("VDIFF_ENABLE_APD", "true").lower() == "true",
            apd_max_parallel=int(os.getenv("VDIFF_APD_MAX_PARALLEL", "8")),
            apd_threshold=float(os.getenv("VDIFF_APD_THRESHOLD", "0.3")),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
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
        }


@dataclass
class ModelConfig:
    """Model-specific configuration loaded from the model."""
    
    model_type: str = "diffusion-llm"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load configuration from a pretrained model."""
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
