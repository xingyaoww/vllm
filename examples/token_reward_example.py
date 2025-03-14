#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Example script for using vLLM with token-level reward models.
This example shows how to use a model like Qwen2ForTokenClassification with vLLM
by adapting it as a token-level reward model.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from vllm import LLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script for token-level reward models with vLLM")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--tokenizer", type=str, help="Tokenizer name or path")
    parser.add_argument(
        "--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument(
        "--input-text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of labels for token-level rewards")
    return parser.parse_args()


def main(args):
    # Initialize the tokenizer
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize the LLM for token-level rewards
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        task="token_reward",
        model_config={"num_labels": args.num_labels},
    )
    
    # Tokenize the input text
    inputs = tokenizer(args.input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Get token-level rewards
    outputs = llm.encode(args.input_text)
    
    # Process and print the results
    print(f"Input text: {args.input_text}")
    print("\nToken-level rewards:")
    
    # Get the token-level logits
    token_logits = outputs[0].data
    
    # Convert logits to probabilities
    token_probs = torch.nn.functional.softmax(
        torch.tensor(token_logits), dim=-1)
    
    # Print token-by-token results
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    for i, (token, probs) in enumerate(zip(tokens, token_probs)):
        label_probs = {f"Label {j}": float(p) for j, p in enumerate(probs)}
        predicted_label = probs.argmax().item()
        
        print(
            f"Token {i}: '{token}' - "
            f"Predicted: Label {predicted_label} - "
            f"Probabilities: {json.dumps(label_probs)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)