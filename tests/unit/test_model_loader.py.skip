"""Unit tests for model loader."""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock

from dfastllm.engine.model_loader import (
    is_local_path,
    needs_trust_remote_code,
    fix_config_if_needed,
    get_mask_token_id,
    MODEL_TYPE_MAPPINGS,
    CUSTOM_ARCHITECTURE_MODELS,
)


class TestIsLocalPath:
    """Tests for is_local_path function."""
    
    def test_absolute_path(self):
        assert is_local_path("/models/llama") is True
        assert is_local_path("/home/user/model") is True
    
    def test_hub_path(self):
        # These don't exist as files, so should return False
        assert is_local_path("meta-llama/Llama-2-7b") is False
        assert is_local_path("GSAI-ML/LLaDA-8B-Instruct") is False
    
    def test_existing_path(self, tmp_path):
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        assert is_local_path(str(model_dir)) is True


class TestNeedsTrustRemoteCode:
    """Tests for needs_trust_remote_code function."""
    
    def test_llada_models(self):
        assert needs_trust_remote_code("GSAI-ML/LLaDA-8B-Instruct") is True
        assert needs_trust_remote_code("/models/llada-8b") is True
    
    def test_dream_models(self):
        assert needs_trust_remote_code("dream-org/Dream-v1") is True
    
    def test_mdlm_models(self):
        assert needs_trust_remote_code("mdlm/MDLM-Base") is True
    
    def test_standard_models(self):
        assert needs_trust_remote_code("meta-llama/Llama-2-7b") is False
        assert needs_trust_remote_code("mistralai/Mistral-7B") is False


class TestFixConfigIfNeeded:
    """Tests for fix_config_if_needed function."""
    
    def test_no_config_file(self, tmp_path):
        model_dir = tmp_path / "no_config"
        model_dir.mkdir()
        assert fix_config_if_needed(str(model_dir)) is False
    
    def test_config_with_model_type(self, tmp_path):
        model_dir = tmp_path / "with_type"
        model_dir.mkdir()
        config = {"model_type": "llama", "hidden_size": 4096}
        (model_dir / "config.json").write_text(json.dumps(config))
        
        assert fix_config_if_needed(str(model_dir)) is False
        
        # Verify config unchanged
        loaded = json.loads((model_dir / "config.json").read_text())
        assert loaded["model_type"] == "llama"
    
    def test_config_missing_model_type_llada(self, tmp_path):
        model_dir = tmp_path / "llada-model"
        model_dir.mkdir()
        config = {"hidden_size": 4096, "vocab_size": 32000}
        (model_dir / "config.json").write_text(json.dumps(config))
        
        assert fix_config_if_needed(str(model_dir)) is True
        
        # Verify model_type added
        loaded = json.loads((model_dir / "config.json").read_text())
        assert loaded["model_type"] == "llama"


class TestGetMaskTokenId:
    """Tests for get_mask_token_id function."""
    
    def test_llada_model(self):
        assert get_mask_token_id("GSAI-ML/LLaDA-8B-Instruct") == 126336
        assert get_mask_token_id("/models/llada-8b") == 126336
    
    def test_bert_model(self):
        assert get_mask_token_id("bert-base-uncased") == 103
    
    def test_mdlm_model(self):
        assert get_mask_token_id("mdlm/MDLM-Base") == 103
    
    def test_gpt2_model(self):
        assert get_mask_token_id("gpt2") == 50256
    
    def test_dream_model(self):
        assert get_mask_token_id("dream-org/Dream-v1") == 50256
    
    def test_with_tokenizer(self):
        mock_tokenizer = Mock()
        mock_tokenizer.mask_token_id = 999
        
        assert get_mask_token_id("unknown-model", mock_tokenizer) == 999
    
    def test_default_fallback(self):
        assert get_mask_token_id("unknown-model-xyz") == 126336


class TestModelTypeMappings:
    """Tests for model type mappings."""
    
    def test_known_mappings(self):
        assert "llada" in MODEL_TYPE_MAPPINGS
        assert MODEL_TYPE_MAPPINGS["llada"] == "llama"
        
        assert "dream" in MODEL_TYPE_MAPPINGS
        assert "mdlm" in MODEL_TYPE_MAPPINGS


class TestCustomArchitectureModels:
    """Tests for custom architecture model list."""
    
    def test_contains_llada(self):
        assert "llada" in CUSTOM_ARCHITECTURE_MODELS
    
    def test_contains_dream(self):
        assert "dream" in CUSTOM_ARCHITECTURE_MODELS
    
    def test_contains_mdlm(self):
        assert "mdlm" in CUSTOM_ARCHITECTURE_MODELS

