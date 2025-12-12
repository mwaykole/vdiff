"""OpenAI API Protocol definitions for vdiff.

Pydantic models matching vLLM's OpenAI-compatible API exactly.
"""

from typing import Optional, List, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
import time


# ============================================================================
# Error Response
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response matching OpenAI API format."""
    
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[int] = None


# ============================================================================
# Model Information
# ============================================================================

class ModelPermission(BaseModel):
    """Model permission information."""
    
    id: str = Field(default_factory=lambda: f"modelperm-{int(time.time())}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """Model information matching OpenAI API format."""
    
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vdiff"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    """List of models matching OpenAI API format."""
    
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ============================================================================
# Usage Information
# ============================================================================

class UsageInfo(BaseModel):
    """Token usage information matching OpenAI API format."""
    
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


# ============================================================================
# Completion API
# ============================================================================

class CompletionRequest(BaseModel):
    """Completion request matching OpenAI API format."""
    
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    
    # vdiff specific extensions
    parallel_decoding: Optional[bool] = True
    confidence_threshold: Optional[float] = None


class CompletionLogProbs(BaseModel):
    """Log probabilities for completion tokens."""
    
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None


class CompletionResponseChoice(BaseModel):
    """Choice in completion response matching OpenAI API format."""
    
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Completion response matching OpenAI API format."""
    
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    
    # vdiff specific extensions
    parallel_tokens_decoded: Optional[int] = None


class CompletionStreamResponse(BaseModel):
    """Streaming completion response matching OpenAI API format."""
    
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]


# ============================================================================
# Chat Completion API
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message matching OpenAI API format."""
    
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request matching OpenAI API format."""
    
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    
    # Function calling (for compatibility)
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    # vdiff specific extensions
    parallel_decoding: Optional[bool] = True
    confidence_threshold: Optional[float] = None


class ChatCompletionResponseMessage(BaseModel):
    """Response message in chat completion."""
    
    role: str = "assistant"
    content: Optional[str] = None
    function_call: Optional[Dict[str, str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionLogProbs(BaseModel):
    """Log probabilities for chat completion."""
    
    content: Optional[List[Dict[str, Any]]] = None


class ChatCompletionResponseChoice(BaseModel):
    """Choice in chat completion response matching OpenAI API format."""
    
    index: int
    message: ChatCompletionResponseMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response matching OpenAI API format."""
    
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None
    
    # vdiff specific extensions
    parallel_tokens_decoded: Optional[int] = None


class ChatCompletionStreamResponseDelta(BaseModel):
    """Delta in streaming chat completion response."""
    
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Choice in streaming chat completion response."""
    
    index: int
    delta: ChatCompletionStreamResponseDelta
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response matching OpenAI API format."""
    
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]
    system_fingerprint: Optional[str] = None


# ============================================================================
# Health and Version
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"


class VersionResponse(BaseModel):
    """Version information response."""
    
    version: str
    vllm_compat_version: str
    model_type: str = "diffusion-llm"


# ============================================================================
# Embeddings API (for compatibility)
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Embedding request matching OpenAI API format."""
    
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """Embedding data in response."""
    
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response matching OpenAI API format."""
    
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo
