"""OpenAI-compatible API entrypoints for vdiff."""

from vdiff.entrypoints.openai.api_server import create_app, run_server
from vdiff.entrypoints.openai.serving_chat import OpenAIServingChat
from vdiff.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vdiff.entrypoints.openai.protocol import (
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
