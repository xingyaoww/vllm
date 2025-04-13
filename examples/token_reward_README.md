# Token-Level Reward Models with vLLM

This document explains how to use vLLM for token-level reward modeling tasks, such as token-level quality assessment, token-level alignment, or any other task that requires token-level predictions.

## Overview

Token-level reward models predict a score or label for each token in the input sequence. vLLM now supports token-level reward models through an enhanced reward model adapter that:

1. Processes the entire input sequence
2. Applies a classification head to each token's representation
3. Returns token-level logits for each token in the sequence

## Usage

### Loading a Token-Level Reward Model

To use a token-level reward model with vLLM, specify the `task="token_reward"` parameter when initializing the LLM:

```python
from vllm import LLM

llm = LLM(
    model="path/to/your/model",
    task="token_reward",
    model_config={"num_labels": 2},  # Specify the number of labels for your task
)
```

### Getting Token-Level Predictions

Use the `encode` method to get token-level predictions:

```python
# Get token-level rewards
outputs = llm.encode("The quick brown fox jumps over the lazy dog.")

# Access the token-level logits
token_logits = outputs[0].data

# Convert logits to probabilities if needed
import torch
token_probs = torch.nn.functional.softmax(torch.tensor(token_logits), dim=-1)
```

### Example

See the `token_reward_example.py` script for a complete example of how to use vLLM for token-level reward modeling.

## Running the Example

```bash
python token_reward_example.py --model path/to/your/model --num-labels 2
```

## Supported Models

The token-level reward adapter works with any model that can be loaded by vLLM. For example:

- BertForTokenClassification
- RobertaForTokenClassification
- And many more!

### Native vLLM Implementations

vLLM now includes native implementations for the following token classification models:

- **Qwen2ForTokenClassification**: A native vLLM implementation that provides optimal performance for token classification tasks with Qwen2 models.

See the `qwen2_token_classification_example.py` script for a demonstration of using the native Qwen2ForTokenClassification implementation.

## Implementation Details

The token-level reward adapter:

1. Uses the `ALL` pooling type to get representations for all tokens
2. Applies a linear classification head to each token representation
3. Returns the logits for each token

This implementation extends the standard reward model adapter in vLLM to support token-level predictions, making it suitable for fine-grained token-level assessment tasks.

## Quantization Note

When using token-level reward models with quantized models (e.g., FP8, INT8), the classification head will automatically use full precision (FP16/FP32) even if the rest of the model is quantized. This is because quantization libraries like CUTLASS require tensor dimensions to be multiples of 16, which is often not the case for the classification head's output dimension (num_labels).