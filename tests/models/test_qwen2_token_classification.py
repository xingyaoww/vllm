# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen2ForTokenClassification model."""

import pytest

from vllm.model_executor.models.registry import ModelRegistry


@pytest.mark.parametrize("model_arch", ["Qwen2ForTokenClassification"])
def test_model_registration(model_arch):
    """Test that the model is properly registered."""
    assert model_arch in ModelRegistry.get_supported_archs()


@pytest.mark.parametrize("model_arch", ["Qwen2ForTokenClassification"])
def test_model_loading(model_arch):
    """Test that the model can be loaded."""
    model_cls, _ = ModelRegistry.resolve_model_cls([model_arch])
    assert model_cls.__name__ == "Qwen2ForTokenClassification"


@pytest.mark.parametrize("model_arch", ["Qwen2ForTokenClassification"])
def test_model_inheritance(model_arch):
    """Test that the model inherits from Qwen2RewardBaseModel."""
    model_cls, _ = ModelRegistry.resolve_model_cls([model_arch])
    from vllm.model_executor.models.qwen2_rm import Qwen2RewardBaseModel
    assert issubclass(model_cls, Qwen2RewardBaseModel)


@pytest.mark.parametrize("model_arch", ["Qwen2ForTokenClassification"])
def test_model_interfaces(model_arch):
    """Test that the model implements the correct interfaces."""
    model_cls, _ = ModelRegistry.resolve_model_cls([model_arch])
    from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP, SupportsV0Only
    
    # Check that the model inherits from the correct interfaces
    assert issubclass(model_cls, SupportsLoRA)
    assert issubclass(model_cls, SupportsPP)
    assert issubclass(model_cls, SupportsV0Only)
    
    # Check that the model is a pooling model (for token-level predictions)
    from vllm.model_executor.models.interfaces_base import is_pooling_model
    assert is_pooling_model(model_cls)