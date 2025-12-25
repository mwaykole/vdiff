"""OpenAI-compatible completion serving for vdiff.

Handles /v1/completions endpoint matching vLLM's interface.
"""

from typing import Optional, AsyncIterator, Union
import time
import uuid
import logging

from dfastllm.engine import DFastLLMEngine, SamplingParams
from dfastllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    UsageInfo,
    ModelList,
    ModelCard,
    ModelPermission,
    ErrorResponse,
)

logger = logging.getLogger(__name__)


class OpenAIServingCompletion:
    """OpenAI-compatible completion serving class.
    
    Matches vLLM's OpenAIServingCompletion interface.
    """
    
    def __init__(
        self,
        engine: DFastLLMEngine,
        model_name: str,
        served_model_names: Optional[list] = None,
    ):
        """Initialize completion serving.
        
        Args:
            engine: The vdiff inference engine.
            model_name: Primary model name.
            served_model_names: List of model names to serve.
        """
        self.engine = engine
        self.model_name = model_name
        self.served_model_names = served_model_names or [model_name]
        
        self._request_counter = 0
    
    async def create_completion(
        self,
        request: CompletionRequest,
        request_id: Optional[str] = None,
    ) -> Union[CompletionResponse, AsyncIterator[str]]:
        """Create a completion.
        
        Args:
            request: The completion request.
            request_id: Optional request ID for tracking.
        
        Returns:
            CompletionResponse or async iterator for streaming.
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        self._request_counter += 1
        
        # Validate model
        if request.model not in self.served_model_names:
            logger.warning(
                f"Model {request.model} not in served models, using {self.model_name}"
            )
        
        # Handle prompt (single or batch)
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        # Validate prompts
        if not prompts or all(not p for p in prompts):
            raise ValueError("prompt cannot be empty")
        
        # Build sampling params
        sampling_params = SamplingParams(
            n=request.n or 1,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            stop=request.stop,
            max_tokens=request.max_tokens or 16,
            logprobs=request.logprobs,
            seed=request.seed,
            parallel_decoding=request.parallel_decoding if request.parallel_decoding is not None else True,
            confidence_threshold=request.confidence_threshold,
        )
        
        if request.stream:
            return self._stream_completion(request_id, prompts, sampling_params, request)
        else:
            return await self._generate_completion(request_id, prompts, sampling_params, request)
    
    async def _generate_completion(
        self,
        request_id: str,
        prompts: list,
        sampling_params: SamplingParams,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Generate non-streaming completion.
        
        Args:
            request_id: Unique request identifier.
            prompts: List of prompts to complete.
            sampling_params: Sampling parameters.
            request: Original request.
        
        Returns:
            CompletionResponse with generated text.
        """
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_parallel_tokens = 0
        
        for prompt_idx, prompt in enumerate(prompts):
            # Generate for each n
            for n_idx in range(sampling_params.n):
                try:
                    # Generate completion
                    output = await self.engine.generate_async(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        request_id=f"{request_id}-{prompt_idx}-{n_idx}",
                    )
                    
                    # Extract generated text
                    if output.outputs:
                        generated_output = output.outputs[0]
                        text = generated_output.text
                        
                        # Handle echo
                        if request.echo:
                            text = prompt + text
                        
                        # Handle suffix
                        if request.suffix:
                            text = text + request.suffix
                        
                        choice = CompletionResponseChoice(
                            index=len(choices),
                            text=text,
                            finish_reason=generated_output.finish_reason,
                        )
                        choices.append(choice)
                        
                        # Update token counts
                        if output.metrics:
                            total_prompt_tokens += output.metrics.prompt_tokens
                            total_completion_tokens += output.metrics.generated_tokens
                            total_parallel_tokens += output.metrics.parallel_tokens_decoded
                    
                except Exception as e:
                    logger.error(f"Generation failed for prompt {prompt_idx}: {e}")
                    raise
        
        # Build response
        response = CompletionResponse(
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
    
    async def _stream_completion(
        self,
        request_id: str,
        prompts: list,
        sampling_params: SamplingParams,
        request: CompletionRequest,
    ) -> AsyncIterator[str]:
        """Generate streaming completion.
        
        Args:
            request_id: Unique request identifier.
            prompts: List of prompts to complete.
            sampling_params: Sampling parameters.
            request: Original request.
        
        Yields:
            Server-sent event formatted strings.
        """
        for prompt_idx, prompt in enumerate(prompts):
            async for output in self.engine.generate_stream(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"{request_id}-{prompt_idx}",
            ):
                if output.outputs:
                    generated_output = output.outputs[0]
                    
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        model=request.model,
                        choices=[
                            CompletionResponseChoice(
                                index=prompt_idx,
                                text=generated_output.text,
                                finish_reason=generated_output.finish_reason if output.finished else None,
                            )
                        ],
                    )
                    
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Send done message
        yield "data: [DONE]\n\n"
    
    def show_available_models(self) -> ModelList:
        """Return list of available models.
        
        Returns:
            ModelList with served models.
        """
        models = []
        
        for name in self.served_model_names:
            permission = ModelPermission()
            model = ModelCard(
                id=name,
                owned_by="vdiff",
                permission=[permission],
            )
            models.append(model)
        
        return ModelList(data=models)
    
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
