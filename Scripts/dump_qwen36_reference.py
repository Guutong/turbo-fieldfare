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
        # numpy has no native bfloat16; upcast to float32 before conversion.
        if tree.dtype == mx.bfloat16:
            tree = tree.astype(mx.float32)
        return np.array(tree)
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
        """Run forward pass and capture all registered hooks.

        The checkpoint's mlx_lm architecture is qwen3_5 / qwen3_5_moe, not
        qwen3_next: the top-level Model wraps a `language_model` (TextModel),
        whose `.model` (Qwen3_5TextModel) holds the actual decoder layers,
        embed_tokens, norm, and the fa_idx/ssm_idx mask indices. Every layer
        has an MoE mlp in this checkpoint (num_experts > 0 unconditionally),
        so there is no dense-only mlp_only_layers distinction to make.
        """
        model = self._base_model
        lm = model.language_model
        text_model = lm.model
        layers = text_model.layers

        # Batch dim: GatedDeltaNet and the mask helpers expect (B, S, ...).
        tokens = prompt_tokens[None]

        # Embedding
        hidden_states = text_model.embed_tokens(tokens)

        captured = {
            "embedding_output": mlx_to_np(hidden_states[0]),
            "hidden_in": {},
            "hidden_out": {},
            "router_logits": {},
            "expert_ids": {},
        }

        cache = model.make_cache()

        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(hidden_states, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

        for layer_idx, (layer, c) in enumerate(zip(layers, cache)):
            # Capture input
            if layer_idx in self._hooks and self._hooks[layer_idx]["in"]:
                captured["hidden_in"][str(layer_idx)] = mlx_to_np(hidden_states[0])

            if layer_idx in self._hooks and self._hooks[layer_idx]["router"]:
                # Manually run the layer so we can inspect the router.
                if layer.is_linear:
                    r = layer.linear_attn(layer.input_layernorm(hidden_states), ssm_mask, c)
                else:
                    r = layer.self_attn(layer.input_layernorm(hidden_states), fa_mask, c)
                h = hidden_states + r
                mlp_input = layer.post_attention_layernorm(h)

                gates = layer.mlp.gate(mlp_input)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = layer.mlp.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if layer.mlp.norm_topk_prob:
                    scores = scores / scores.sum(axis=-1, keepdims=True)

                captured["router_logits"][str(layer_idx)] = mlx_to_np(gates[0])
                captured["expert_ids"][str(layer_idx)] = mlx_to_np(inds[0])

                y = layer.mlp.switch_mlp(mlp_input, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                shared_y = layer.mlp.shared_expert(mlp_input)
                shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(mlp_input)) * shared_y
                mlp_out = y + shared_y

                hidden_states = h + mlp_out
            else:
                hidden_states = layer(hidden_states, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)

            # Capture output
            if layer_idx in self._hooks and self._hooks[layer_idx]["out"]:
                captured["hidden_out"][str(layer_idx)] = mlx_to_np(hidden_states[0])

        # Final norm
        hidden_states = text_model.norm(hidden_states)

        # LM head
        if not lm.args.tie_word_embeddings:
            logits = lm.lm_head(hidden_states)
        else:
            logits = text_model.embed_tokens.as_linear(hidden_states)

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

    # Text config lives under "text_config" for this checkpoint's (qwen3_5_moe)
    # multimodal-shaped config.json rather than flat at the top level.
    text_config = config.get("text_config", config)
    print(f"Config layers: {text_config.get('num_hidden_layers')}, hidden: {text_config.get('hidden_size')}, vocab: {text_config.get('vocab_size')}")

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

    # safetensors.save_file needs a flat {name: ndarray} map; captured's
    # per-layer dicts (hidden_in, hidden_out, router_logits, expert_ids)
    # need flattening to "hidden_in.<layer>" style keys, matching the
    # module docstring's documented fixture layout.
    flat = {
        "input_token_ids": captured["input_token_ids"],
        "embedding_output": captured["embedding_output"],
    }
    for i in range(40):
        flat[f"hidden_in.{i}"] = captured["hidden_in"][str(i)]
        flat[f"hidden_out.{i}"] = captured["hidden_out"][str(i)]
    for layer_id in (0, 3):
        if str(layer_id) in captured["router_logits"]:
            flat[f"router_logits.{layer_id}"] = captured["router_logits"][str(layer_id)]
            flat[f"expert_ids.{layer_id}"] = captured["expert_ids"][str(layer_id)]

    # Save (config goes in safetensors metadata, not as a tensor value)
    print(f"Saving fixture to {args.output}...")
    safetensors.numpy.save_file(flat, args.output, metadata={"config": json.dumps(config)})

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
