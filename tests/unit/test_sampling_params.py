"""Unit tests for SamplingParams."""

import pytest
from vdiff.engine.sampling_params import SamplingParams


class TestSamplingParams:
    """Test cases for SamplingParams."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = SamplingParams()
        
        assert params.n == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.max_tokens == 16
        assert params.parallel_decoding is True
    
    def test_custom_values(self):
        """Test custom parameter values."""
        params = SamplingParams(
            n=3,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
        )
        
        assert params.n == 3
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 100
        assert params.stop == ["END"]
    
    def test_stop_normalization(self):
        """Test that stop strings are normalized to list."""
        # Single string
        params = SamplingParams(stop="END")
        assert params.stop == ["END"]
        
        # List of strings
        params = SamplingParams(stop=["END", "STOP"])
        assert params.stop == ["END", "STOP"]
        
        # None
        params = SamplingParams(stop=None)
        assert params.stop == []
    
    def test_best_of_default(self):
        """Test best_of defaults to n."""
        params = SamplingParams(n=5)
        assert params.best_of == 5
    
    def test_validation_n(self):
        """Test validation for n parameter."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            SamplingParams(n=0)
    
    def test_validation_best_of(self):
        """Test validation for best_of parameter."""
        with pytest.raises(ValueError, match="best_of must be >= n"):
            SamplingParams(n=5, best_of=3)
    
    def test_validation_presence_penalty(self):
        """Test validation for presence_penalty."""
        with pytest.raises(ValueError, match="presence_penalty must be in"):
            SamplingParams(presence_penalty=3.0)
    
    def test_validation_frequency_penalty(self):
        """Test validation for frequency_penalty."""
        with pytest.raises(ValueError, match="frequency_penalty must be in"):
            SamplingParams(frequency_penalty=-3.0)
    
    def test_validation_temperature(self):
        """Test validation for temperature."""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            SamplingParams(temperature=-0.1)
    
    def test_validation_top_p(self):
        """Test validation for top_p."""
        with pytest.raises(ValueError, match="top_p must be in"):
            SamplingParams(top_p=0.0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            SamplingParams(top_p=1.5)
    
    def test_validation_top_k(self):
        """Test validation for top_k."""
        with pytest.raises(ValueError, match="top_k must be -1 or positive"):
            SamplingParams(top_k=0)
    
    def test_validation_max_tokens(self):
        """Test validation for max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            SamplingParams(max_tokens=0)
    
    def test_validation_confidence_threshold(self):
        """Test validation for confidence_threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            SamplingParams(confidence_threshold=1.5)
    
    def test_from_openai_params(self):
        """Test creation from OpenAI parameters."""
        params = SamplingParams.from_openai_params(
            n=2,
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
            stop=["END"],
        )
        
        assert params.n == 2
        assert params.temperature == 0.8
        assert params.top_p == 0.95
        assert params.max_tokens == 50
        assert params.stop == ["END"]
    
    def test_clone(self):
        """Test cloning sampling parameters."""
        original = SamplingParams(
            n=3,
            temperature=0.5,
            max_tokens=200,
        )
        
        cloned = original.clone()
        
        assert cloned.n == original.n
        assert cloned.temperature == original.temperature
        assert cloned.max_tokens == original.max_tokens
        assert cloned is not original
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = SamplingParams(
            n=2,
            temperature=0.7,
            max_tokens=100,
        )
        
        d = params.to_dict()
        
        assert d["n"] == 2
        assert d["temperature"] == 0.7
        assert d["max_tokens"] == 100
        assert "parallel_decoding" in d
    
    def test_vdiff_specific_params(self):
        """Test vdiff-specific parameters."""
        params = SamplingParams(
            parallel_decoding=True,
            confidence_threshold=0.85,
        )
        
        assert params.parallel_decoding is True
        assert params.confidence_threshold == 0.85
