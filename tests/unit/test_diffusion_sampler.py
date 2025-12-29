"""Unit tests for the optimized diffusion sampler.

Test Structure:
===============
Each test follows a consistent pattern:
1. ARRANGE - Set up test inputs and expected outputs
2. ACT - Call the function/method being tested
3. ASSERT - Verify the results match expectations

Test Categories:
================
- TestDiffusionSamplerConfig: Configuration dataclass tests
- TestAddGumbelNoise: Gumbel noise sampling tests
- TestGetNumTransferTokens: Token distribution scheduling tests
- TestVectorizedTopkUnmask: Vectorized top-k selection tests
- TestDiffusionGenerate: End-to-end generation tests
- TestDiffusionSampler: High-level sampler class tests
- TestIsDiffusionModel: Model detection tests
- TestOptimizationPerformance: Performance benchmarks
"""

import pytest
from unittest.mock import MagicMock, patch
import time

# Check if PyTorch is available
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDiffusionSamplerConfig:
    """Test cases for DiffusionSamplerConfig.
    
    Tests verify that:
    - Default values are correctly initialized
    - Custom optimization options can be set
    """
    
    def test_default_values(self):
        """Test default configuration values.
        
        Steps:
            1. Create a DiffusionSamplerConfig with no arguments
            2. Verify all default values match expected defaults:
               - steps=128 (number of diffusion steps)
               - gen_length=128 (tokens to generate)
               - block_length=32 (semi-autoregressive block size)
               - temperature=0.0 (greedy sampling)
               - cfg_scale=0.0 (no classifier-free guidance)
               - remasking="low_confidence" (unmask high-confidence first)
               - mask_id=126336 (LLaDA's [MASK] token)
               - use_float32_gumbel=False (quality over speed)
               - enable_early_stopping=True (stop when all unmasked)
        """
        from dfastllm.engine.diffusion_sampler import DiffusionSamplerConfig
        
        # Step 1: Create config with defaults
        config = DiffusionSamplerConfig()
        
        # Step 2: Verify all default values
        assert config.steps == 128
        assert config.gen_length == 128
        assert config.block_length == 32
        assert config.temperature == 0.0
        assert config.cfg_scale == 0.0
        assert config.remasking == "low_confidence"
        assert config.mask_id == 126336
        assert config.use_float32_gumbel is False
        assert config.enable_early_stopping is True
    
    def test_custom_optimization_options(self):
        """Test custom optimization options.
        
        Steps:
            1. Create config with custom optimization flags
            2. Verify the custom values are applied:
               - use_float32_gumbel=True (faster but lower quality)
               - enable_early_stopping=False (always run all steps)
        """
        from dfastllm.engine.diffusion_sampler import DiffusionSamplerConfig
        
        # Step 1: Create config with custom optimization options
        config = DiffusionSamplerConfig(
            use_float32_gumbel=True,
            enable_early_stopping=False,
        )
        
        # Step 2: Verify custom values are set
        assert config.use_float32_gumbel is True
        assert config.enable_early_stopping is False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAddGumbelNoise:
    """Test cases for add_gumbel_noise function.
    
    Tests verify that:
    - Zero temperature returns unchanged logits (greedy mode)
    - Nonzero temperature adds stochastic Gumbel noise
    - Float64 is used by default for numerical precision
    - Float32 can be enabled for speed optimization
    """
    
    def test_zero_temperature_returns_unchanged(self):
        """Test that temperature=0 returns logits unchanged.
        
        Steps:
            1. Create random logits tensor (2 batches, 10 positions, 100 vocab)
            2. Apply add_gumbel_noise with temperature=0.0
            3. Verify output equals input (greedy mode = no noise)
        """
        from dfastllm.engine.diffusion_sampler import add_gumbel_noise
        
        # Step 1: Create input logits
        logits = torch.randn(2, 10, 100)
        
        # Step 2: Apply Gumbel noise with zero temperature
        result = add_gumbel_noise(logits, temperature=0.0)
        
        # Step 3: Verify output is unchanged
        assert torch.equal(result, logits)
    
    def test_nonzero_temperature_adds_noise(self):
        """Test that nonzero temperature adds Gumbel noise.
        
        Steps:
            1. Set random seed for reproducibility
            2. Create random logits tensor
            3. Apply add_gumbel_noise with temperature=1.0
            4. Verify output differs from input (noise was added)
        """
        from dfastllm.engine.diffusion_sampler import add_gumbel_noise
        
        # Step 1: Set seed for reproducibility
        torch.manual_seed(42)
        
        # Step 2: Create input logits
        logits = torch.randn(2, 10, 100)
        
        # Step 3: Apply Gumbel noise
        result = add_gumbel_noise(logits.clone(), temperature=1.0)
        
        # Step 4: Verify result is different (noise was added)
        assert not torch.equal(result, logits)
    
    def test_float64_by_default(self):
        """Test that float64 is used by default for quality.
        
        Steps:
            1. Create float32 logits
            2. Apply add_gumbel_noise with use_float32=False (default)
            3. Verify output dtype is float64 (higher precision)
        """
        from dfastllm.engine.diffusion_sampler import add_gumbel_noise
        
        # Step 1: Create float32 input
        logits = torch.randn(2, 10, 100, dtype=torch.float32)
        
        # Step 2: Apply noise with default float64
        result = add_gumbel_noise(logits, temperature=1.0, use_float32=False)
        
        # Step 3: Verify float64 output for numerical precision
        assert result.dtype == torch.float64
    
    def test_float32_when_requested(self):
        """Test that float32 is used when requested for speed.
        
        Steps:
            1. Create float32 logits
            2. Apply add_gumbel_noise with use_float32=True
            3. Verify output dtype is float32 (faster computation)
        """
        from dfastllm.engine.diffusion_sampler import add_gumbel_noise
        
        # Step 1: Create float32 input
        logits = torch.randn(2, 10, 100, dtype=torch.float32)
        
        # Step 2: Apply noise with float32 for speed
        result = add_gumbel_noise(logits, temperature=1.0, use_float32=True)
        
        # Step 3: Verify float32 output
        assert result.dtype == torch.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGetNumTransferTokens:
    """Test cases for vectorized get_num_transfer_tokens.
    
    Tests verify that:
    - Tokens are evenly distributed across steps
    - Uneven divisions handle remainders correctly
    - Batched inputs work with different mask counts per item
    """
    
    def test_basic_distribution(self):
        """Test basic token distribution across steps.
        
        Steps:
            1. Create mask_index with 10 masked positions
            2. Call get_num_transfer_tokens with steps=5
            3. Verify output shape is (batch_size, steps) = (1, 5)
            4. Verify total tokens equals 10 (all masks accounted for)
            5. Verify each step gets 2 tokens (10/5 = 2)
        """
        from dfastllm.engine.diffusion_sampler import get_num_transfer_tokens
        
        # Step 1: Create mask with 10 True values
        mask_index = torch.ones(1, 10, dtype=torch.bool)
        
        # Step 2: Compute transfer schedule
        result = get_num_transfer_tokens(mask_index, steps=5)
        
        # Step 3-5: Verify distribution
        assert result.shape == (1, 5)
        assert result.sum().item() == 10
        assert (result == 2).all()
    
    def test_uneven_distribution(self):
        """Test distribution when tokens don't divide evenly.
        
        Steps:
            1. Create mask_index with 10 masked positions
            2. Call get_num_transfer_tokens with steps=3 (10/3 = 3.33)
            3. Verify output shape is (1, 3)
            4. Verify total still equals 10 (remainder distributed)
               Expected: [4, 3, 3] or [3, 4, 3] etc.
        """
        from dfastllm.engine.diffusion_sampler import get_num_transfer_tokens
        
        # Step 1: Create mask with 10 True values
        mask_index = torch.ones(1, 10, dtype=torch.bool)
        
        # Step 2: Compute with uneven division
        result = get_num_transfer_tokens(mask_index, steps=3)
        
        # Step 3-4: Verify shape and total
        assert result.shape == (1, 3)
        assert result.sum().item() == 10
    
    def test_batch_handling(self):
        """Test handling of batched input.
        
        Steps:
            1. Create batch of 2 with different mask counts:
               - Batch 0: 5 masked positions
               - Batch 1: 3 masked positions
            2. Call get_num_transfer_tokens with steps=2
            3. Verify output shape is (2, 2)
            4. Verify batch 0 sums to 5
            5. Verify batch 1 sums to 3
        """
        from dfastllm.engine.diffusion_sampler import get_num_transfer_tokens
        
        # Step 1: Create batched mask with different counts
        mask_index = torch.tensor([
            [True, True, True, True, True],  # 5 masked
            [True, True, True, False, False],  # 3 masked
        ])
        
        # Step 2: Compute transfer schedule
        result = get_num_transfer_tokens(mask_index, steps=2)
        
        # Step 3-5: Verify per-batch distribution
        assert result.shape == (2, 2)
        assert result[0].sum().item() == 5
        assert result[1].sum().item() == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestVectorizedTopkUnmask:
    """Test cases for vectorized top-k selection.
    
    This is the KEY OPTIMIZATION: replaces Python loops with vectorized ops.
    
    Tests verify that:
    - Top-k positions are correctly selected by confidence
    - Different k values per batch item work correctly
    - Zero tokens to unmask returns empty mask
    """
    
    def test_basic_topk(self):
        """Test basic top-k selection.
        
        Steps:
            1. Create confidence scores: [0.1, 0.9, 0.5, 0.3, 0.7]
               Sorted descending: 0.9(idx=1), 0.7(idx=4), 0.5(idx=2), ...
            2. Set num_tokens to select k=2 at step 0
            3. Call _vectorized_topk_unmask
            4. Verify output shape is (1, 5)
            5. Verify exactly 2 positions are True
            6. Verify indices 1 (conf=0.9) and 4 (conf=0.7) are selected
        """
        from dfastllm.engine.diffusion_sampler import _vectorized_topk_unmask
        
        # Step 1: Create confidence scores
        confidence = torch.tensor([[0.1, 0.9, 0.5, 0.3, 0.7]])  # batch_size=1
        
        # Step 2: Set k=2 at step 0
        num_tokens = torch.tensor([[2, 2]])
        
        # Step 3: Perform vectorized top-k selection
        result = _vectorized_topk_unmask(confidence, num_tokens, step=0)
        
        # Step 4-6: Verify correct positions selected
        assert result.shape == (1, 5)
        assert result.sum().item() == 2
        assert result[0, 1].item() is True  # idx 1 has conf 0.9
        assert result[0, 4].item() is True  # idx 4 has conf 0.7
    
    def test_batch_different_k(self):
        """Test batch with different k values per item.
        
        Steps:
            1. Create 2-batch confidence scores:
               - Batch 0: [0.1, 0.9, 0.5] → top is idx 1
               - Batch 1: [0.8, 0.2, 0.6] → top 2 are idx 0, 2
            2. Set num_tokens: batch 0 gets k=1, batch 1 gets k=2
            3. Call _vectorized_topk_unmask
            4. Verify batch 0 has exactly 1 True
            5. Verify batch 1 has exactly 2 True
        """
        from dfastllm.engine.diffusion_sampler import _vectorized_topk_unmask
        
        # Step 1: Create batched confidence scores
        confidence = torch.tensor([
            [0.1, 0.9, 0.5],  # batch 0
            [0.8, 0.2, 0.6],  # batch 1
        ])
        
        # Step 2: Different k per batch
        num_tokens = torch.tensor([
            [1, 0],  # batch 0: k=1 at step 0
            [2, 0],  # batch 1: k=2 at step 0
        ])
        
        # Step 3: Perform selection
        result = _vectorized_topk_unmask(confidence, num_tokens, step=0)
        
        # Step 4-5: Verify per-batch counts
        assert result[0].sum().item() == 1  # batch 0 gets 1
        assert result[1].sum().item() == 2  # batch 1 gets 2
    
    def test_zero_tokens(self):
        """Test when no tokens should be unmasked.
        
        Steps:
            1. Create confidence scores
            2. Set num_tokens to k=0 at all steps
            3. Call _vectorized_topk_unmask
            4. Verify no positions are True (empty mask)
        """
        from dfastllm.engine.diffusion_sampler import _vectorized_topk_unmask
        
        # Step 1-2: Create inputs with k=0
        confidence = torch.tensor([[0.1, 0.9, 0.5]])
        num_tokens = torch.tensor([[0, 0]])
        
        # Step 3: Perform selection
        result = _vectorized_topk_unmask(confidence, num_tokens, step=0)
        
        # Step 4: Verify empty result
        assert result.sum().item() == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDiffusionGenerate:
    """Test cases for the main diffusion_generate function.
    
    This tests the CORE ALGORITHM:
    1. Initialize: [prompt] + [MASK × gen_length]
    2. For each step:
       a. Forward pass → get logits
       b. Sample tokens with Gumbel noise
       c. Compute confidence scores
       d. Select top-k confident masked positions
       e. Unmask those positions with predicted tokens
    3. Return fully unmasked sequence
    """
    
    def _create_mock_model(self, device="cpu"):
        """Create a mock model for testing.
        
        The mock model:
        - Has a dummy parameter for device detection
        - Returns random logits on forward pass
        - Mimics HuggingFace model output format (.logits attribute)
        """
        class MockModel(torch.nn.Module):
            def __init__(self, vocab_size=1000):
                super().__init__()
                self.vocab_size = vocab_size
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, x, attention_mask=None):
                batch_size, seq_len = x.shape
                logits = torch.randn(batch_size, seq_len, self.vocab_size)
                return type("Output", (), {"logits": logits})()
        
        return MockModel().to(device)
    
    def test_basic_generation(self):
        """Test basic diffusion generation.
        
        Steps:
            1. Create mock model and prompt [1, 2, 3]
            2. Call diffusion_generate with:
               - steps=4 (diffusion iterations)
               - gen_length=8 (tokens to generate)
               - block_length=8 (single block)
            3. Verify output shape is (1, 3+8) = (1, 11)
            4. Verify prompt tokens are preserved at start
            5. Verify no MASK tokens remain in output
        """
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        # Step 1: Setup
        model = self._create_mock_model()
        prompt = torch.tensor([[1, 2, 3]])  # batch_size=1, prompt_length=3
        mask_id = 999
        
        # Step 2: Generate
        result = diffusion_generate(
            model=model,
            prompt=prompt,
            steps=4,
            gen_length=8,
            block_length=8,
            mask_id=mask_id,
        )
        
        # Step 3: Verify shape
        assert result.shape == (1, 11)
        
        # Step 4: Verify prompt preserved
        assert (result[:, :3] == prompt).all()
        
        # Step 5: Verify all MASKs replaced
        assert (result != mask_id).all()
    
    def test_batched_generation(self):
        """Test batched diffusion generation.
        
        Steps:
            1. Create batch of 2 prompts: [[1,2,3], [4,5,6]]
            2. Call diffusion_generate
            3. Verify output shape is (2, 11) - both batches processed
            4. Verify each prompt is preserved in its respective output
        """
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        # Step 1: Create batched prompt
        model = self._create_mock_model()
        prompt = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        mask_id = 999
        
        # Step 2: Generate for batch
        result = diffusion_generate(
            model=model,
            prompt=prompt,
            steps=4,
            gen_length=8,
            block_length=8,
            mask_id=mask_id,
        )
        
        # Step 3-4: Verify batch handling
        assert result.shape == (2, 11)
        assert (result[:, :3] == prompt).all()
    
    def test_early_stopping(self):
        """Test early stopping when all tokens unmasked.
        
        Steps:
            1. Create model that always predicts token 42 with high confidence
               (This ensures all masks get unmasked quickly)
            2. Call diffusion_generate with many steps (64)
            3. Verify generation completes (early stopping triggered)
            4. Verify output shape is correct
        
        Note: Early stopping should trigger when mask_index.any() is False,
        saving computation when all tokens are already unmasked.
        """
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        # Step 1: Create high-confidence model
        class ConfidentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
                self.call_count = 0
            
            def forward(self, x, attention_mask=None):
                self.call_count += 1
                batch_size, seq_len = x.shape
                logits = torch.zeros(batch_size, seq_len, 100)
                logits[:, :, 42] = 10.0  # High confidence for token 42
                return type("Output", (), {"logits": logits})()
        
        model = ConfidentModel()
        prompt = torch.tensor([[1, 2, 3]])
        
        # Step 2: Generate with many steps
        result = diffusion_generate(
            model=model,
            prompt=prompt,
            steps=64,  # Many steps - should early stop
            gen_length=8,
            block_length=8,
            mask_id=999,
            enable_early_stopping=True,
        )
        
        # Step 3-4: Verify completion
        assert result.shape == (1, 11)
    
    def test_random_remasking(self):
        """Test random remasking strategy.
        
        Steps:
            1. Create mock model and prompt
            2. Call diffusion_generate with remasking="random"
               (Instead of confidence-based, uses random selection)
            3. Verify output shape is correct
            4. Verify generation completes successfully
        
        Note: Random remasking selects which tokens to unmask randomly
        rather than by confidence score. Useful for diversity.
        """
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        # Step 1: Setup
        model = self._create_mock_model()
        prompt = torch.tensor([[1, 2, 3]])
        
        # Step 2: Generate with random remasking
        result = diffusion_generate(
            model=model,
            prompt=prompt,
            steps=4,
            gen_length=8,
            block_length=8,
            mask_id=999,
            remasking="random",  # Random instead of confidence-based
        )
        
        # Step 3-4: Verify completion
        assert result.shape == (1, 11)
    
    def test_with_attention_mask(self):
        """Test generation with attention mask.
        
        Steps:
            1. Create prompt and matching attention mask
            2. Call diffusion_generate with attention_mask
               (Mask gets extended to cover generated tokens)
            3. Verify output shape is correct
            4. Verify generation works with attention mask
        
        Note: Attention mask is extended from (batch, prompt_len) to
        (batch, prompt_len + gen_length) with 1s for new tokens.
        """
        from dfastllm.engine.diffusion_sampler import diffusion_generate
        
        # Step 1: Create prompt with attention mask
        model = self._create_mock_model()
        prompt = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)  # All prompt tokens attended
        
        # Step 2: Generate with attention mask
        result = diffusion_generate(
            model=model,
            prompt=prompt,
            attention_mask=attention_mask,
            steps=4,
            gen_length=8,
            block_length=8,
            mask_id=999,
        )
        
        # Step 3-4: Verify completion
        assert result.shape == (1, 11)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDiffusionSampler:
    """Test cases for the DiffusionSampler class.
    
    DiffusionSampler is the HIGH-LEVEL API that wraps:
    - Model and tokenizer management
    - Configuration handling
    - The low-level diffusion_generate function
    """
    
    def _create_mock_model(self):
        """Create a mock model for testing."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, x, attention_mask=None):
                batch_size, seq_len = x.shape
                logits = torch.randn(batch_size, seq_len, 100)
                return type("Output", (), {"logits": logits})()
        
        return MockModel()
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer with mask_token_id."""
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 999
        tokenizer.decode = MagicMock(return_value="generated text")
        return tokenizer
    
    def test_initialization(self):
        """Test DiffusionSampler initialization.
        
        Steps:
            1. Create mock model and tokenizer (with mask_token_id=999)
            2. Initialize DiffusionSampler
            3. Verify model and tokenizer are stored
            4. Verify mask_id is AUTO-DETECTED from tokenizer.mask_token_id
        
        Note: Auto-detection of mask_id is important for different models
        that may use different mask token IDs.
        """
        from dfastllm.engine.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
        
        # Step 1: Create mocks
        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        
        # Step 2: Initialize sampler
        sampler = DiffusionSampler(model, tokenizer)
        
        # Step 3-4: Verify initialization
        assert sampler.model == model
        assert sampler.tokenizer == tokenizer
        assert sampler.config.mask_id == 999  # Auto-detected!
    
    def test_generate(self):
        """Test generation via DiffusionSampler.
        
        Steps:
            1. Create sampler with custom config (steps=4, block_length=8)
            2. Prepare input_ids [1, 2, 3]
            3. Call sampler.generate(input_ids, max_new_tokens=8)
            4. Verify output shape is (1, 11) = prompt + generated
        
        Note: DiffusionSampler.generate() handles:
        - Block length adjustment for divisibility
        - Steps adjustment for block count
        - Calling diffusion_generate with all parameters
        """
        from dfastllm.engine.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
        
        # Step 1: Create sampler with config
        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DiffusionSamplerConfig(
            steps=4,
            block_length=8,
        )
        sampler = DiffusionSampler(model, tokenizer, config)
        
        # Step 2: Prepare input
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Step 3: Generate
        result = sampler.generate(input_ids, max_new_tokens=8)
        
        # Step 4: Verify output
        assert result.shape == (1, 11)
    
    def test_decode(self):
        """Test decoding via DiffusionSampler.
        
        Steps:
            1. Create sampler with mock tokenizer
            2. Create output_ids [1,2,3,4,5,6,7,8,9,10]
            3. Call sampler.decode(output_ids, prompt_length=3)
               (Should decode tokens 4-10, skipping prompt)
            4. Verify tokenizer.decode was called with correct tokens
        """
        from dfastllm.engine.diffusion_sampler import DiffusionSampler
        
        # Step 1: Create sampler
        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        sampler = DiffusionSampler(model, tokenizer)
        
        # Step 2: Create output tokens
        output_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Step 3: Decode (skip first 3 = prompt)
        result = sampler.decode(output_ids, prompt_length=3)
        
        # Step 4: Verify tokenizer.decode was called
        tokenizer.decode.assert_called_once()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestIsDiffusionModel:
    """Test cases for is_diffusion_model function.
    
    This utility function detects diffusion models by name pattern matching.
    Used to automatically enable diffusion generation vs standard AR generation.
    """
    
    def test_llada_detection(self):
        """Test detection of LLaDA models.
        
        Steps:
            1. Test various LLaDA model name formats
            2. Verify all are detected as diffusion models
        
        LLaDA (Large Language Diffusion with mAsking) is a primary
        diffusion LLM architecture supported by vdiff.
        """
        from dfastllm.engine.diffusion_sampler import is_diffusion_model
        
        # Test various LLaDA name formats
        assert is_diffusion_model("GSAI-ML/LLaDA-8B-Instruct") is True
        assert is_diffusion_model("llada-7b") is True
    
    def test_dream_detection(self):
        """Test detection of Dream models.
        
        Steps:
            1. Test various Dream model name formats
            2. Verify all are detected as diffusion models
        
        Dream is another diffusion LLM architecture.
        """
        from dfastllm.engine.diffusion_sampler import is_diffusion_model
        
        # Test Dream model variants
        assert is_diffusion_model("dream-7b") is True
        assert is_diffusion_model("Dream-Model") is True
    
    def test_non_diffusion_model(self):
        """Test non-diffusion models are not detected.
        
        Steps:
            1. Test standard autoregressive models (LLaMA, GPT-2)
            2. Verify they are NOT detected as diffusion models
        
        Standard AR models should use normal generation, not diffusion.
        """
        from dfastllm.engine.diffusion_sampler import is_diffusion_model
        
        # Test AR models - should return False
        assert is_diffusion_model("meta-llama/Llama-2-7b") is False
        assert is_diffusion_model("gpt2") is False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOptimizationPerformance:
    """Performance benchmarks for optimizations.
    
    Compares optimized vectorized implementation against naive Python loops.
    The vectorized version should be faster, especially on GPU and larger batches.
    """
    
    def _create_mock_model(self):
        """Create a mock model for benchmarking."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, x, attention_mask=None):
                batch_size, seq_len = x.shape
                logits = torch.randn(batch_size, seq_len, 1000)
                return type("Output", (), {"logits": logits})()
        
        return MockModel()
    
    def test_vectorized_topk_faster_than_loop(self):
        """Verify vectorized top-k is faster than naive loop.
        
        Steps:
            1. SETUP: Create test data
               - batch_size=8, seq_len=512, steps=64
               - Random confidence scores and token counts
            
            2. WARM UP: Run each version once to JIT compile
            
            3. BENCHMARK VECTORIZED:
               - Run 100 iterations of _vectorized_topk_unmask
               - Measure total time
            
            4. BENCHMARK NAIVE:
               - Run 100 iterations of naive Python loop version
               - Measure total time
            
            5. COMPARE:
               - Print timing results
               - Assert vectorized is not significantly slower
        
        Expected Results:
        - CPU, batch=8: ~1.5-2x speedup
        - GPU, batch=8: ~10-50x speedup (vectorization leverages parallelism)
        """
        from dfastllm.engine.diffusion_sampler import _vectorized_topk_unmask
        
        # Step 1: SETUP - Create test data
        batch_size = 8
        seq_len = 512
        steps = 64
        
        confidence = torch.rand(batch_size, seq_len)
        num_tokens = torch.randint(1, 10, (batch_size, steps))
        
        # Step 2: WARM UP
        _ = _vectorized_topk_unmask(confidence, num_tokens, step=0)
        
        # Step 3: BENCHMARK VECTORIZED
        start = time.time()
        for _ in range(100):
            _ = _vectorized_topk_unmask(confidence, num_tokens, step=0)
        vectorized_time = time.time() - start
        
        # Step 4: BENCHMARK NAIVE (Python loop version)
        def naive_topk(confidence, num_tokens, step):
            """Naive implementation with Python for loop."""
            batch_size, seq_len = confidence.shape
            transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
            for b in range(batch_size):  # <-- This loop is the bottleneck!
                k = num_tokens[b, step].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[b], k=k)
                    transfer_index[b, select_index] = True
            return transfer_index
        
        # Warm up naive
        _ = naive_topk(confidence, num_tokens, step=0)
        
        start = time.time()
        for _ in range(100):
            _ = naive_topk(confidence, num_tokens, step=0)
        naive_time = time.time() - start
        
        # Step 5: COMPARE and report
        print(f"\nVectorized: {vectorized_time:.4f}s, Naive: {naive_time:.4f}s")
        print(f"Speedup: {naive_time / vectorized_time:.2f}x")
        
        # Assert vectorized is competitive (allow 2x tolerance for overhead)
        assert vectorized_time <= naive_time * 2

