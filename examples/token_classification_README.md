# Token Classification with vLLM

This document explains how to use vLLM for token classification tasks, such as named entity recognition (NER), part-of-speech tagging, or any other task that requires token-level predictions.

## Overview

Token classification models predict a label for each token in the input sequence. vLLM now supports token classification models through a dedicated adapter that:

1. Processes the entire input sequence
2. Applies a classification head to each token's representation
3. Returns token-level logits for each token in the sequence

## Usage

### Loading a Token Classification Model

To use a token classification model with vLLM, specify the `task="token_classification"` parameter when initializing the LLM:

```python
from vllm import LLM

llm = LLM(
    model="path/to/your/model",
    task="token_classification",
    model_config={"num_labels": 2},  # Specify the number of labels for your task
)
```

### Getting Token-Level Predictions

Use the `encode` method to get token-level predictions:

```python
# Get token-level classifications
outputs = llm.encode("The quick brown fox jumps over the lazy dog.")

# Access the token-level logits
token_logits = outputs[0].data

# Convert logits to probabilities if needed
import torch
token_probs = torch.nn.functional.softmax(torch.tensor(token_logits), dim=-1)
```

### Example

See the `token_classification_example.py` script for a complete example of how to use vLLM for token classification.

## Running the Example

```bash
python token_classification_example.py --model path/to/your/model --num-labels 2
```

## Supported Models

The token classification adapter works with any model that can be loaded by vLLM. For example:

- Qwen2ForTokenClassification
- BertForTokenClassification
- RobertaForTokenClassification
- And many more!

## Implementation Details

The token classification adapter:

1. Uses the `ALL` pooling type to get representations for all tokens
2. Applies a dropout layer followed by a linear classification head
3. Returns the logits for each token

This implementation is similar to how token classification models work in Hugging Face Transformers.