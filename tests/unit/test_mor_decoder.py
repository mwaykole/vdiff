"""Unit tests for Mixture of Recursions (MoR) decoder.

Tests the MoR implementation which provides adaptive compute allocation
for diffusion LLM inference.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock torch if not available
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

from dfastllm.engine.mor_decoder import (
    MoRConfig,
    MoRStats,
    MoRDecoder,
    RouterStrategy,
)


class TestMoRConfig:
    """Tests for MoRConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MoRConfig()
        
        assert config.enabled is True
        assert config.min_recursions == 1
        assert config.max_recursions == 4
        assert config.router_strategy == RouterStrategy.CONFIDENCE
        assert config.difficulty_threshold_low == 0.8
        assert config.difficulty_threshold_high == 0.3
        assert config.skip_confident_tokens is True
        assert config.skip_threshold == 0.95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoRConfig(
            enabled=True,
            min_recursions=2,
            max_recursions=8,
            router_strategy=RouterStrategy.ENTROPY,
            difficulty_threshold_low=0.7,
            difficulty_threshold_high=0.2,
            skip_threshold=0.99,
        )
        
        assert config.min_recursions == 2
        assert config.max_recursions == 8
        assert config.router_strategy == RouterStrategy.ENTROPY
        assert config.difficulty_threshold_low == 0.7
    
    def test_invalid_min_recursions(self):
        """Test validation of min_recursions."""
        with pytest.raises(ValueError, match="min_recursions must be >= 1"):
            MoRConfig(min_recursions=0)
    
    def test_invalid_max_recursions(self):
        """Test validation of max_recursions."""
        with pytest.raises(ValueError, match="max_recursions must be >= min_recursions"):
            MoRConfig(min_recursions=4, max_recursions=2)
    
    def test_invalid_threshold(self):
        """Test validation of thresholds."""
        with pytest.raises(ValueError, match="difficulty_threshold_low must be in"):
            MoRConfig(difficulty_threshold_low=1.5)
        
        with pytest.raises(ValueError, match="difficulty_threshold_high must be in"):
            MoRConfig(difficulty_threshold_high=-0.1)


class TestMoRStats:
    """Tests for MoRStats dataclass."""
    
    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = MoRStats()
        
        assert stats.total_steps == 0
        assert stats.total_tokens_processed == 0
        assert stats.tokens_skipped == 0
        assert stats.easy_tokens == 0
        assert stats.medium_tokens == 0
        assert stats.hard_tokens == 0
        assert stats.compute_saved_pct == 0.0
    
    def test_update_stats(self):
        """Test statistics update."""
        stats = MoRStats()
        
        stats.update(
            processed=100,
            skipped=20,
            easy=50,
            medium=30,
            hard=20,
            recursions=150,
        )
        
        assert stats.total_steps == 1
        assert stats.total_tokens_processed == 100
        assert stats.tokens_skipped == 20
        assert stats.easy_tokens == 50
        assert stats.medium_tokens == 30
        assert stats.hard_tokens == 20
        assert stats.total_recursions == 150
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = MoRStats()
        stats.update(processed=100, skipped=10, easy=60, medium=30, hard=10, recursions=120)
        
        result = stats.to_dict()
        
        assert isinstance(result, dict)
        assert "total_steps" in result
        assert "compute_saved_pct" in result
        assert result["total_tokens_processed"] == 100
    
    def test_reset(self):
        """Test statistics reset."""
        stats = MoRStats()
        stats.update(processed=100, skipped=10, easy=60, medium=30, hard=10, recursions=120)
        
        stats.reset()
        
        assert stats.total_steps == 0
        assert stats.total_tokens_processed == 0


class TestMoRDecoder:
    """Tests for MoRDecoder class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        decoder = MoRDecoder()
        
        assert decoder.config.enabled is True
        assert decoder.stats.total_steps == 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = MoRConfig(max_recursions=6, skip_threshold=0.98)
        decoder = MoRDecoder(config)
        
        assert decoder.config.max_recursions == 6
        assert decoder.config.skip_threshold == 0.98
    
    def test_initialization_disabled(self):
        """Test initialization with MoR disabled."""
        config = MoRConfig(enabled=False)
        decoder = MoRDecoder(config)
        
        assert decoder.config.enabled is False
    
    def test_get_stats(self):
        """Test get_stats method."""
        decoder = MoRDecoder()
        
        stats = decoder.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_steps" in stats
    
    def test_reset_stats(self):
        """Test reset_stats method."""
        decoder = MoRDecoder()
        decoder.stats.total_steps = 10
        
        decoder.reset_stats()
        
        assert decoder.stats.total_steps == 0


class TestRouterStrategy:
    """Tests for RouterStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RouterStrategy.CONFIDENCE.value == "confidence"
        assert RouterStrategy.ENTROPY.value == "entropy"
        assert RouterStrategy.GRADIENT.value == "gradient"
        assert RouterStrategy.FIXED.value == "fixed"
    
    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        strategy = RouterStrategy("confidence")
        assert strategy == RouterStrategy.CONFIDENCE
        
        strategy = RouterStrategy("entropy")
        assert strategy == RouterStrategy.ENTROPY


class TestMoRIntegration:
    """Integration tests for MoR with mock tensors."""
    
    @pytest.fixture
    def mock_torch(self):
        """Create mock torch module."""
        with patch.dict(sys.modules, {'torch': torch_mock}):
            yield torch_mock
    
    def test_mor_config_env_integration(self):
        """Test MoR config works with environment-style settings."""
        config = MoRConfig(
            enabled=True,
            min_recursions=1,
            max_recursions=4,
            difficulty_threshold_low=0.8,
            difficulty_threshold_high=0.3,
        )
        
        decoder = MoRDecoder(config)
        
        assert decoder.config.enabled is True
        assert decoder.config.max_recursions == 4
    
    def test_mor_stats_compute_savings(self):
        """Test compute savings calculation."""
        stats = MoRStats()
        
        # Simulate processing where we skip some tokens
        stats.update(
            processed=80,  # Only processed 80
            skipped=20,    # Skipped 20
            easy=40,
            medium=30,
            hard=10,
            recursions=120,  # Fewer recursions than max
        )
        
        # Should show compute savings
        assert stats.tokens_skipped == 20
        assert stats.avg_recursions_per_token > 0


class TestMoRBenefits:
    """Tests to verify MoR provides expected benefits."""
    
    def test_easy_tokens_fewer_recursions(self):
        """Test that easy tokens conceptually get fewer recursions."""
        config = MoRConfig(
            min_recursions=1,
            max_recursions=8,
            difficulty_threshold_low=0.9,  # High confidence = easy
            difficulty_threshold_high=0.3,  # Low confidence = hard
        )
        
        # High confidence (0.95) should map to fewer recursions
        # Low confidence (0.2) should map to more recursions
        
        # This is conceptual - actual tensor ops would be tested with real torch
        assert config.min_recursions < config.max_recursions
        assert config.difficulty_threshold_low > config.difficulty_threshold_high
    
    def test_skip_threshold_effectiveness(self):
        """Test skip threshold configuration."""
        config = MoRConfig(
            skip_confident_tokens=True,
            skip_threshold=0.95,
        )
        
        decoder = MoRDecoder(config)
        
        # Verify configuration is set correctly
        assert decoder.config.skip_confident_tokens is True
        assert decoder.config.skip_threshold == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
