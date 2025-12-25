"""OpenAI-compatible chat completion serving for vdiff.

Handles /v1/chat/completions endpoint matching vLLM's interface.
"""

from typing import Optional, List, AsyncIterator, Union
import time
import uuid
import logging

from dfastllm.engine import DFastLLMEngine, SamplingParams
from dfastllm.entrypoints.openai.protocol import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    UsageInfo,
    ErrorResponse,
)

logger = logging.getLogger(__name__)


class OpenAIServingChat:
    """OpenAI-compatible chat completion serving class.
    
    Matches vLLM's OpenAIServingChat interface.
    """
    
    def __init__(
        self,
        engine: DFastLLMEngine,
        model_name: str,
        served_model_names: Optional[list] = None,
        chat_template: Optional[str] = None,
    ):
        """Initialize chat serving.
        
        Args:
            engine: The vdiff inference engine.
            model_name: Primary model name.
            served_model_names: List of model names to serve.
            chat_template: Optional custom chat template.
        """
        self.engine = engine
        self.model_name = model_name
        self.served_model_names = served_model_names or [model_name]
        self.chat_template = chat_template
        
        self._request_counter = 0
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string.
        
        Uses the tokenizer's chat template if available, otherwise
        falls back to a simple format.
        
        Args:
            messages: List of chat messages.
        
        Returns:
            Formatted prompt string.
        """
        # Convert Pydantic models to dicts
        message_dicts = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in messages
        ]
        
        # Try to use tokenizer's chat template
        try:
            if self.engine.tokenizer:
                formatted = self.engine.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
        except Exception as e:
            logger.debug(f"Chat template not available: {e}")
        
        # Fallback to simple formatting
        formatted = ""
        for msg in messages:
            role = msg.role
            content = msg.content or ""
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
            else:
                formatted += f"{role.capitalize()}: {content}\n\n"
        
        formatted += "Assistant:"
        return formatted
    
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[str]]:
        """Create a chat completion.
        
        Args:
            request: The chat completion request.
            request_id: Optional request ID for tracking.
        
        Returns:
            ChatCompletionResponse or async iterator for streaming.
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self._request_counter += 1
        
        # Validate messages
        if not request.messages:
            raise ValueError("messages cannot be empty")
        
        # Validate model
        if request.model not in self.served_model_names:
            logger.warning(
                f"Model {request.model} not in served models, using {self.model_name}"
            )
        
        # Format messages into prompt
        prompt = self._format_messages(request.messages)
        
        # Build sampling params
        sampling_params = SamplingParams(
            n=request.n or 1,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            stop=request.stop,
            max_tokens=request.max_tokens or 1024,
            seed=request.seed,
            parallel_decoding=request.parallel_decoding if request.parallel_decoding is not None else True,
            confidence_threshold=request.confidence_threshold,
        )
        
        if request.stream:
            return self._stream_chat_completion(request_id, prompt, sampling_params, request)
        else:
            return await self._generate_chat_completion(request_id, prompt, sampling_params, request)
    
    async def _generate_chat_completion(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate non-streaming chat completion.
        
        Args:
            request_id: Unique request identifier.
            prompt: Formatted prompt string.
            sampling_params: Sampling parameters.
            request: Original request.
        
        Returns:
            ChatCompletionResponse with generated message.
        """
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_parallel_tokens = 0
        
        # Generate for each n
        for n_idx in range(sampling_params.n):
            try:
                # Generate completion
                output = await self.engine.generate_async(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}-{n_idx}",
                )
                
                # Extract generated text
                if output.outputs:
                    generated_output = output.outputs[0]
                    
                    # Create response message
                    message = ChatCompletionResponseMessage(
                        role="assistant",
                        content=generated_output.text.strip(),
                    )
                    
                    choice = ChatCompletionResponseChoice(
                        index=n_idx,
                        message=message,
                        finish_reason=generated_output.finish_reason,
                    )
                    choices.append(choice)
                    
                    # Update token counts
                    if output.metrics:
                        total_prompt_tokens += output.metrics.prompt_tokens
                        total_completion_tokens += output.metrics.generated_tokens
                        total_parallel_tokens += output.metrics.parallel_tokens_decoded
                
            except Exception as e:
                logger.error(f"Chat completion generation failed: {e}")
                raise
        
        # Build response
        response = ChatCompletionResponse(
            id=request_id,
            model=request.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
            parallel_tokens_decoded=total_parallel_tokens if total_parallel_tokens > 0 else None,
        )
        
        return response
    
    async def _stream_chat_completion(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Generate streaming chat completion.
        
        Args:
            request_id: Unique request identifier.
            prompt: Formatted prompt string.
            sampling_params: Sampling parameters.
            request: Original request.
        
        Yields:
            Server-sent event formatted strings.
        """
        # Send initial chunk with role
        initial_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(
                        role="assistant",
                        content="",
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Stream content
        previous_text = ""
        async for output in self.engine.generate_stream(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if output.outputs:
                generated_output = output.outputs[0]
                current_text = generated_output.text.strip()
                
                # Get new content (delta)
                new_content = current_text[len(previous_text):]
                previous_text = current_text
                
                if new_content or output.finished:
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=ChatCompletionStreamResponseDelta(
                                    content=new_content if new_content else None,
                                ),
                                finish_reason=generated_output.finish_reason if output.finished else None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Send done message
        yield "data: [DONE]\n\n"
    
    def create_error_response(
        self,
        message: str,
        err_type: str = "server_error",
        code: int = 500,
    ) -> ErrorResponse:
        """Create an error response.
        
        Args:
            message: Error message.
            err_type: Error type.
            code: HTTP status code.
        
        Returns:
            ErrorResponse object.
        """
        return ErrorResponse(
            message=message,
            type=err_type,
            code=code,
        )
