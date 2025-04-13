# SPDX-License-Identifier: Apache-2.0

"""Inference-only Qwen2ForTokenClassification model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsLoRA, SupportsPP, SupportsV0Only
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class Qwen2ForTokenClassification(nn.Module, SupportsLoRA, SupportsPP, SupportsV0Only):
    """Qwen2 model with a token classification head on top.
    
    This model is designed for token-level classification tasks such as NER,
    POS tagging, or token-level quality assessment.
    """
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        # Initialize the base model
        self.model = Qwen2Model(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "model"))

        # Initialize the token classification head
        # Disable quantization for the classification head to avoid dimension issues
        if quant_config is not None:
            logger.info(
                "Quantization is disabled for the token classification head to avoid "
                "dimension compatibility issues. The rest of the model remains quantized."
            )
        
        # Create the classification head (simple linear layer)
        self.classifier = RowParallelLinear(
            config.hidden_size,
            config.num_labels,
            quant_config=None,  # Force full precision for classification head
            bias=True,
            prefix=maybe_prefix(prefix, "classifier")
        )

        # Initialize the pooler for token-level predictions
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.ALL,  # Use ALL to get representations for all tokens
            normalize=False,
            softmax=False
        )
        
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get hidden states from the base model
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        
        # Apply the classification head to get token-level logits
        logits, _ = self.classifier(hidden_states)
        
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self,
                                  ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)