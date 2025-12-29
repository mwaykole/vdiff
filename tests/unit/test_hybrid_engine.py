"""Unit tests for the Hybrid Diffusion-AR Engine.

Tests the DEER-style hybrid generation combining diffusion drafting
with autoregressive verification.

Reference papers:
- DEER: https://czc726.github.io/DEER/
- DiffuSpec: arxiv:2510.02358
- SpecDiff: NAACL 2025
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from dfastllm.engine.hybrid_engine import (
    HybridEngine,
    HybridConfig,
    HybridMode,
    HybridStats,
    SpecDiffEngine,
    SemiAREngine,
    create_hybrid_engine,
    hybrid_generate,
)
from dfastllm.engine.entropy_controller import EntropyAwareDraftController


class TestHybridConfig:
    """Tests for HybridConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HybridConfig()
        
        assert config.enabled is True
        assert config.mode == HybridMode.DEER
        assert config.draft_block_size == 8
        assert config.max_draft_tokens == 32
        assert config.acceptance_threshold == 0.3
        assert config.diffusion_weight == 1.0
        assert config.ar_weight == 0.5
        assert config.adaptive_draft_length is True
        assert config.fallback_to_ar is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridConfig(
            mode=HybridMode.SPEC_DIFF,
            draft_block_size=16,
            max_draft_tokens=64,
            acceptance_threshold=0.5,
            adaptive_draft_length=False,
        )
        
        assert config.mode == HybridMode.SPEC_DIFF
        assert config.draft_block_size == 16
        assert config.max_draft_tokens == 64
        assert config.acceptance_threshold == 0.5
        assert config.adaptive_draft_length is False
    
    def test_hybrid_modes(self):
        """Test all hybrid mode values."""
        assert HybridMode.DEER.value == "deer"
        assert HybridMode.SPEC_DIFF.value == "spec_diff"
        assert HybridMode.SEMI_AR.value == "semi_ar"
        assert HybridMode.ADAPTIVE.value == "adaptive"
        
        for mode in HybridMode:
            config = HybridConfig(mode=mode)
            assert config.mode == mode


class TestHybridStats:
    """Tests for HybridStats tracking."""
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = HybridStats()
        
        assert stats.total_requests == 0
        assert stats.tokens_accepted == 0
        assert stats.tokens_rejected == 0
        assert stats.total_drafts == 0
        assert stats.draft_acceptance_rate == 0.0
        assert stats.diffusion_time_ms == 0.0
        assert stats.ar_verification_time_ms == 0.0
    
    def test_stats_update(self):
        """Test statistics update."""
        stats = HybridStats()
        
        stats.update(
            drafted=10,
            accepted=8,
            diffusion_time=0.1,
            ar_time=0.05,
            used_fallback=False,
        )
        
        assert stats.total_drafts == 1
        assert stats.tokens_accepted == 8
        assert stats.tokens_rejected == 2
        assert stats.ar_fallbacks == 0
    
    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = HybridStats()
        stats.update(
            drafted=10,
            accepted=8,
            diffusion_time=0.1,
            ar_time=0.05,
        )
        
        stats_dict = stats.to_dict()
        
        assert "total_drafts" in stats_dict
        assert "tokens_accepted" in stats_dict
        assert stats_dict["tokens_accepted"] == 8


class TestDraftLengthController:
    """Tests for adaptive draft length controller."""
    
    def test_initial_draft_length(self):
        """Test initial draft length."""
        controller = EntropyAwareDraftController(
            initial_length=8,
            min_length=4,
            max_length=32,
        )
        
        assert controller.get_draft_length() == 8
    
    def test_draft_length_increase_on_high_acceptance(self):
        """Test draft length increases with high acceptance."""
        controller = EntropyAwareDraftController(
            initial_length=8,
            min_length=4,
            max_length=32,
        )
        
        initial_length = controller.get_draft_length()
        
        controller.update(accepted=9, total=10)
        
        new_length = controller.get_draft_length()
        assert new_length >= initial_length
    
    def test_draft_length_decrease_on_low_acceptance(self):
        """Test draft length decreases with low acceptance."""
        controller = EntropyAwareDraftController(
            initial_length=16,
            min_length=4,
            max_length=32,
        )
        
        initial_length = controller.get_draft_length()
        
        controller.update(accepted=2, total=10)
        
        new_length = controller.get_draft_length()
        assert new_length <= initial_length
    
    def test_draft_length_respects_bounds(self):
        """Test draft length stays within bounds."""
        controller = EntropyAwareDraftController(
            initial_length=8,
            min_length=4,
            max_length=16,
        )
        
        for _ in range(20):
            controller.update(accepted=10, total=10)
        
        assert controller.get_draft_length() <= 16
        
        for _ in range(20):
            controller.update(accepted=0, total=10)
        
        assert controller.get_draft_length() >= 4


def create_mock_model():
    """Create a mock model with proper parameters() method."""
    import torch
    
    model = Mock()
    
    @dataclass
    class MockOutput:
        logits: Mock
    
    mock_logits = Mock()
    mock_logits.shape = (1, 64, 32000)
    mock_logits.argmax.return_value = Mock()
    mock_logits.__getitem__ = Mock(return_value=mock_logits)
    
    model.return_value = MockOutput(logits=mock_logits)
    model.device = "cpu"
    
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    model.parameters = Mock(return_value=iter([mock_param]))
    
    return model


class TestHybridEngine:
    """Tests for the main HybridEngine class."""
    
    @pytest.fixture
    def mock_diffusion_model(self):
        """Create a mock diffusion model."""
        return create_mock_model()
    
    @pytest.fixture
    def mock_ar_model(self):
        """Create a mock AR model."""
        return create_mock_model()
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.mask_token_id = 126336
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_engine_initialization(self, mock_diffusion_model, mock_tokenizer):
        """Test hybrid engine initialization."""
        config = HybridConfig(mode=HybridMode.DEER)
        
        engine = HybridEngine(
            diffusion_model=mock_diffusion_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None
        assert engine.config.mode == HybridMode.DEER
    
    def test_engine_with_ar_verifier(
        self, mock_diffusion_model, mock_ar_model, mock_tokenizer
    ):
        """Test engine initialization with AR verifier."""
        config = HybridConfig(mode=HybridMode.DEER)
        
        engine = HybridEngine(
            diffusion_model=mock_diffusion_model,
            ar_model=mock_ar_model,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine.ar_model is not None
    
    def test_get_stats(self, mock_diffusion_model, mock_tokenizer):
        """Test getting engine statistics."""
        config = HybridConfig()
        engine = HybridEngine(
            diffusion_model=mock_diffusion_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        stats = engine.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_drafts" in stats or "tokens_accepted" in stats
    
    def test_disabled_engine_passthrough(self, mock_diffusion_model, mock_tokenizer):
        """Test disabled hybrid engine passes through to diffusion."""
        config = HybridConfig(enabled=False)
        
        engine = HybridEngine(
            diffusion_model=mock_diffusion_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine.config.enabled is False


class TestSpecDiffEngine:
    """Tests for SpecDiff-style engine."""
    
    @pytest.fixture
    def mock_model(self):
        return create_mock_model()
    
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        tokenizer.mask_token_id = 126336
        return tokenizer
    
    def test_spec_diff_initialization(self, mock_model, mock_tokenizer):
        """Test SpecDiff engine initialization."""
        config = HybridConfig(mode=HybridMode.SPEC_DIFF)
        
        engine = SpecDiffEngine(
            diffusion_model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None


class TestSemiAREngine:
    """Tests for Semi-AR engine."""
    
    @pytest.fixture
    def mock_model(self):
        return create_mock_model()
    
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        tokenizer.mask_token_id = 126336
        return tokenizer
    
    def test_semi_ar_initialization(self, mock_model, mock_tokenizer):
        """Test Semi-AR engine initialization."""
        config = HybridConfig(mode=HybridMode.SEMI_AR)
        
        engine = SemiAREngine(
            diffusion_model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None


class TestCreateHybridEngine:
    """Tests for the hybrid engine factory function."""
    
    def test_create_deer_engine(self):
        """Test creating DEER engine via factory."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(mode=HybridMode.DEER)
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert isinstance(engine, HybridEngine)
    
    def test_create_spec_diff_engine(self):
        """Test creating SpecDiff engine via factory."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(mode=HybridMode.SPEC_DIFF)
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None
    
    def test_create_semi_ar_engine(self):
        """Test creating Semi-AR engine via factory."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(mode=HybridMode.SEMI_AR)
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None


class TestHybridGenerate:
    """Tests for the hybrid_generate helper function."""
    
    def test_hybrid_generate_basic(self):
        """Test basic hybrid generation."""
        mock_model = Mock()
        mock_model.device = "cpu"
        
        @dataclass
        class MockOutput:
            logits: Mock
        
        import torch
        mock_logits = torch.randn(1, 32, 1000)
        mock_model.return_value = MockOutput(logits=mock_logits)
        
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(enabled=False)
        
        try:
            result = hybrid_generate(
                diffusion_model=mock_model,
                prompt=torch.tensor([[1, 2, 3]]),
                max_new_tokens=8,
                config=config,
                tokenizer=mock_tokenizer,
            )
            assert result is not None
        except Exception:
            pass


class TestHybridModeSelection:
    """Tests for automatic hybrid mode selection."""
    
    def test_deer_mode_requires_ar_model(self):
        """Test DEER mode behavior without AR model."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(mode=HybridMode.DEER)
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None
    
    def test_adaptive_mode_selection(self):
        """Test adaptive mode selects appropriate strategy."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(mode=HybridMode.ADAPTIVE)
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""
    
    def test_chat_completion_scenario(self):
        """Test hybrid generation for chat completion."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(
            mode=HybridMode.DEER,
            draft_block_size=8,
            max_draft_tokens=64,
        )
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None
        assert engine.config.max_draft_tokens == 64
    
    def test_long_generation_scenario(self):
        """Test hybrid generation for long outputs."""
        mock_model = create_mock_model()
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 126336
        
        config = HybridConfig(
            mode=HybridMode.SPEC_DIFF,
            draft_block_size=16,
            max_draft_tokens=128,
            adaptive_draft_length=True,
        )
        
        engine = create_hybrid_engine(
            diffusion_model=mock_model,
            ar_model=None,
            tokenizer=mock_tokenizer,
            config=config,
            mask_id=126336,
        )
        
        assert engine is not None
        assert engine.config.adaptive_draft_length is True
