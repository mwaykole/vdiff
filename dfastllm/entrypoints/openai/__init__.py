"""OpenAI-compatible API entrypoints for dfastllm."""

from dfastllm.entrypoints.openai.api_server import create_app, run_server
from dfastllm.entrypoints.openai.serving_chat import OpenAIServingChat
from dfastllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from dfastllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    ErrorResponse,
)

__all__ = [
    "create_app",
    "run_server",
    "OpenAIServingChat",
    "OpenAIServingCompletion",
    "CompletionRequest",
    "CompletionResponse",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ModelList",
    "ErrorResponse",
]
