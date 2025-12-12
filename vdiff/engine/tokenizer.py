"""Tokenizer wrapper for vdiff.

Provides a consistent interface for tokenization across different models.
"""

from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizers.
    
    Provides a consistent interface matching vLLM's tokenizer handling.
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer_mode: str = "auto",
    ):
        """Initialize the tokenizer.
        
        Args:
            tokenizer_name: Name or path of the tokenizer.
            revision: Specific model version to use.
            trust_remote_code: Whether to trust remote code in tokenizer.
            tokenizer_mode: Mode for tokenizer ("auto", "slow", "fast").
        """
        self.tokenizer_name = tokenizer_name
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.tokenizer_mode = tokenizer_mode
        
        self._tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                use_fast=(self.tokenizer_mode != "slow"),
            )
            
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                else:
                    self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
            logger.info(f"Loaded tokenizer: {self.tokenizer_name}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_name}: {e}")
            raise
    
    @property
    def tokenizer(self):
        """Get the underlying tokenizer."""
        return self._tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._tokenizer.vocab_size
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the end-of-sequence token ID."""
        return self._tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get the beginning-of-sequence token ID."""
        return self._tokenizer.bos_token_id
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the padding token ID."""
        return self._tokenizer.pad_token_id
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """Get the mask token ID (important for diffusion models)."""
        if hasattr(self._tokenizer, "mask_token_id"):
            return self._tokenizer.mask_token_id
        return None
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], "torch.Tensor"]:
        """Encode text to token IDs.
        
        Args:
            text: Text or list of texts to encode.
            add_special_tokens: Whether to add special tokens.
            max_length: Maximum length for truncation/padding.
            truncation: Whether to truncate sequences.
            padding: Whether to pad sequences.
            return_tensors: If set, return tensors of specified type ("pt" for PyTorch).
        
        Returns:
            Token IDs as list or tensor.
        """
        kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
        }
        
        if max_length is not None:
            kwargs["max_length"] = max_length
        
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
        
        result = self._tokenizer(text, **kwargs)
        
        if return_tensors is not None:
            return result["input_ids"]
        
        if isinstance(text, str):
            return result["input_ids"]
        return result["input_ids"]
    
    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
        
        Returns:
            Decoded text.
        """
        # Handle tensor input
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        
        # Handle nested lists (batch dimension)
        if isinstance(token_ids, list) and len(token_ids) > 0:
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode multiple sequences of token IDs.
        
        Args:
            token_ids_list: List of token ID sequences.
            skip_special_tokens: Whether to skip special tokens.
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
        
        Returns:
            List of decoded texts.
        """
        return self._tokenizer.batch_decode(
            token_ids_list,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Apply chat template to messages.
        
        Args:
            messages: List of message dictionaries with "role" and "content".
            tokenize: Whether to return token IDs instead of text.
            add_generation_prompt: Whether to add the generation prompt.
        
        Returns:
            Formatted chat string or token IDs.
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Fallback for tokenizers without chat template
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    formatted += f"System: {content}\n"
                elif role == "user":
                    formatted += f"User: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
            
            if add_generation_prompt:
                formatted += "Assistant:"
            
            if tokenize:
                return self.encode(formatted)
            return formatted
    
    def get_added_vocab(self) -> dict:
        """Get the added vocabulary."""
        return self._tokenizer.get_added_vocab()
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to their IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(
        self, 
        ids: Union[int, List[int]], 
        skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """Convert IDs to their tokens."""
        return self._tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)
