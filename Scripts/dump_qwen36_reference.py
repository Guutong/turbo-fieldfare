#!/usr/bin/env python3
"""Dump a Qwen3.6-35B-A3B reference fixture for TurboFieldfare parity tests.

Loads mlx-community/Qwen3.6-35B-A3B-4bit, runs a fixed prompt through all 40
layers, and writes one safetensors file containing:
  - input_token_ids: the tokenized prompt
  - embedding_output: output of the embedding layer (before any layers)
  - hidden_in.<layer>: hidden state at the INPUT of each layer (0..39)
  - hidden_out.<layer>: hidden state at the OUTPUT of each layer (0..39)
  - router_logits.0, router_logits.3: MoE gate logits for layers 0 and 3
  - expert_ids.0, expert_ids.3: selected expert indices for layers 0 and 3

Fixed prompt: "The capital of France is"
Greedy decoding (temperature 0), deterministic.
"""

import argparse
import json
import numpy as np
import safetensors
import safetensors.numpy
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


def mlx_to_np(tree):
    """Recursively convert an MLX pytree to numpy for safetensors."""
    if isinstance(tree, mx.array):
        # Explicitly copy to host memory, then convert to numpy
        return np.array(np.frombuffer(tree.tobytes(), dtype=tree.dtype), dtype=tree.dtype).reshape(tree.shape)
    if isinstance(tree, (list, tuple)):
        return type(tree)(mlx_to_np(x) for x in tree)
    if isinstance(tree, dict):
        return {k: mlx_to_np(v) for k, v in tree.items()}
    return tree


class HookedQwen3Next(nn.Module):
    """Qwen3Next model with hooks to capture per-layer activations."""

    def __init__(self, model):
        super().__init__()
        self._base_model = model
        self._hooks = {}
        self._captured = {}

    def __call__(self, inputs, cache=None):
        return self._base_model(inputs, cache=cache)

    def register_hook(self, layer_idx, capture_in=True, capture_out=True, capture_router=False):
        key = layer_idx
        if key not in self._hooks:
            self._hooks[key] = {"in": None, "out": None, "router": False}
        self._hooks[key]["in"] = capture_in
        self._hooks[key]["out"] = capture_out
        self._hooks[key]["router"] = capture_router

    def capture(self, prompt_tokens, max_new_tokens=1):
        """Run forward pass and capture all registered hooks."""
        import mlx.core as mx

        model = self._base_model
        layers = model.model.layers

        # Embedding
        hidden_states = model.model.embed_tokens(prompt_tokens)

        # We need to hook each decoder layer. Since we can't easily inject
        # hooks into the existing module, we'll wrap each layer.
        captured = {
            "embedding_output": mlx_to_np(hidden_states),
            "hidden_in": {},
            "hidden_out": {},
            "router_logits": {},
            "expert_ids": {},
        }

        cache = model.make_cache()

        fa_mask = None
        ssm_mask = None
        # We need to recreate the mask logic from Qwen3NextModel
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(hidden_states, cache[model.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[model.ssm_idx])

        for layer_idx, (layer, c) in enumerate(zip(layers, cache)):
            # Capture input
            if layer_idx in self._hooks and self._hooks[layer_idx]["in"]:
                captured["hidden_in"][str(layer_idx)] = mlx_to_np(hidden_states)

            # Wrap the layer's __call__ to capture outputs and router info
            original_call = layer.__call__

            def make_wrapped_call(orig, idx):
                def wrapped_call(x, mask=None, cache=None):
                    result = orig(x, mask=mask, cache=cache)
                    return result
                return wrapped_call

            # For MoE layers, we need to capture router logits and expert ids
            # We'll do this by calling the layer directly and inspecting
            if layer_idx in self._hooks and self._hooks[layer_idx]["router"]:
                # Manually run the layer with instrumentation
                if layer.is_linear:
                    r = layer.linear_attn(layer.input_layernorm(hidden_states), ssm_mask, cache)
                else:
                    r = layer.self_attn(layer.input_layernorm(hidden_states), fa_mask, cache)
                h = hidden_states + r

                # Capture MLP/MoE output
                mlp_input = layer.post_attention_layernorm(h)

                # Check if this layer has MoE
                has_moe = (layer_idx not in self._base_model.args.mlp_only_layers) and \
                          (self._base_model.args.num_experts > 0 and
                           (layer_idx + 1) % self._base_model.args.decoder_sparse_step == 0)

                if has_moe:
                    # Capture router logits and expert ids
                    gates = layer.mlp.gate(h)
                    gates = mx.softmax(gates, axis=-1, precise=True)
                    k = layer.mlp.top_k
                    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    if layer.mlp.norm_topk_prob:
                        scores = scores / scores.sum(axis=-1, keepdims=True)

                    captured["router_logits"][str(layer_idx)] = mlx_to_np(gates)
                    captured["expert_ids"][str(layer_idx)] = mlx_to_np(inds)

                    y = layer.mlp.switch_mlp(h, inds)
                    y = (y * scores[..., None]).sum(axis=-2)
                    shared_y = layer.mlp.shared_expert(h)
                    shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h)) * shared_y
                    mlp_out = y + shared_y
                else:
                    mlp_out = layer.mlp(mlp_input)

                hidden_states = h + mlp_out
            else:
                hidden_states = layer(hidden_states, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)

            # Capture output
            if layer_idx in self._hooks and self._hooks[layer_idx]["out"]:
                captured["hidden_out"][str(layer_idx)] = mlx_to_np(hidden_states)

        # Final norm
        hidden_states = model.model.norm(hidden_states)

        # LM head
        if not model.args.tie_word_embeddings:
            logits = model.lm_head(hidden_states)
        else:
            logits = model.model.embed_tokens.as_linear(hidden_states)

        return logits, captured


def main():
    parser = argparse.ArgumentParser(description="Dump Qwen3.6 reference fixture")
    parser.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit",
                        help="Model ID or path")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Fixed prompt for the dump")
    parser.add_argument("--output", default="qwen36_fixture.safetensors",
                        help="Output safetensors file")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (optional)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load config
    import os
    model_path = args.model
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load model and tokenizer via mlx_lm. This handles config parsing,
    # quantization, and weight loading correctly (including renamed/dropped
    # config keys and quantized weight shapes) -- do not reimplement it.
    from mlx_lm import load
    model, tokenizer = load(model_path)

    print(f"Config layers: {config.get('num_hidden_layers')}, hidden: {config.get('hidden_size')}, vocab: {config.get('vocab_size')}")

    # Create hooked model
    hooked = HookedQwen3Next(model)

    # Register hooks for all layers (capture in/out) and layers 0, 3 (capture router)
    for i in range(40):
        hooked.register_hook(i, capture_in=True, capture_out=True, capture_router=(i in (0, 3)))

    # Tokenize prompt
    prompt_tokens = mx.array(tokenizer.encode(args.prompt))
    print(f"Prompt tokens: {prompt_tokens.shape[0]} tokens")
    print(f"Tokens: {tokenizer.decode(prompt_tokens.tolist())}")

    # Run forward pass
    print("Running forward pass...")
    logits, captured = hooked.capture(prompt_tokens, max_new_tokens=1)

    # Add input token ids
    captured["input_token_ids"] = mlx_to_np(prompt_tokens)

    # Add config metadata
    captured["_config"] = config

    # Save
    print(f"Saving fixture to {args.output}...")
    safetensors.numpy.save_file(captured, args.output)

    # Print summary
    print(f"\nFixture contents:")
    print(f"  input_token_ids: {captured['input_token_ids'].shape}")
    print(f"  embedding_output: {captured['embedding_output'].shape}")
    for i in range(40):
        print(f"  hidden_in.{i}: {captured['hidden_in'][str(i)].shape}")
        print(f"  hidden_out.{i}: {captured['hidden_out'][str(i)].shape}")
    for layer_id in (0, 3):
        if str(layer_id) in captured["router_logits"]:
            print(f"  router_logits.{layer_id}: {captured['router_logits'][str(layer_id)].shape}")
            print(f"  expert_ids.{layer_id}: {captured['expert_ids'][str(layer_id)].shape}")

    print(f"\nDone. Fixture saved to {args.output}")


if __name__ == "__main__":
    main()
