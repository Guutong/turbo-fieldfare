#!/usr/bin/env python3
"""Dump layer-0 MLP/MoE intermediate tensors (post-DeltaNet) for stage-by-
stage comparison, companion to dump_qwen36_deltanet_stages.py. See that
script's docstring and PHASE-LOG.md P3-3 for context: the DeltaNet block
itself was verified to match Swift to <1.1% relL2 per stage; this narrows
down whether the remaining ~28% divergence is in the router/shared-expert/
routed-MoE combine.
"""
import argparse
import numpy as np
import safetensors.numpy
import mlx.core as mx
import mlx.nn as nn


def mlx_to_np(x):
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    return np.array(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Users/guutong/models/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--fixture", default="Tests/Fixtures/qwen36_fixture.safetensors")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", default="qwen36_layer0_mlp_stages.safetensors")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    lm = model.language_model
    text_model = lm.model
    layer0 = text_model.layers[0]

    prompt_tokens = mx.array(tokenizer.encode(args.prompt))
    tokens = prompt_tokens[None]
    hidden_in = text_model.embed_tokens(tokens)
    S = hidden_in.shape[1]

    r = layer0.linear_attn(layer0.input_layernorm(hidden_in), None, None)
    h = hidden_in + r
    mlp_input = layer0.post_attention_layernorm(h)

    mlp = layer0.mlp
    gates = mlp.gate(mlp_input)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = mlp.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mlp.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)

    y = mlp.switch_mlp(mlp_input, inds)
    routed_y_per_expert_sum = (y * scores[..., None]).sum(axis=-2)
    shared_y_raw = mlp.shared_expert(mlp_input)
    gate_scalar = mx.sigmoid(mlp.shared_expert_gate(mlp_input))
    shared_y = gate_scalar * shared_y_raw
    mlp_out = routed_y_per_expert_sum + shared_y
    h_out = h + mlp_out

    stages = {
        "mlp_input": mlx_to_np(mlp_input[0]),
        "gates": mlx_to_np(gates[0]),
        "expert_inds": mlx_to_np(inds[0]).astype(np.int64),
        "scores": mlx_to_np(scores[0]),
        "shared_y_raw": mlx_to_np(shared_y_raw[0]),
        "gate_scalar": mlx_to_np(gate_scalar[0]),
        "shared_y": mlx_to_np(shared_y[0]),
        "routed_y": mlx_to_np(routed_y_per_expert_sum[0]),
        "mlp_out": mlx_to_np(mlp_out[0]),
        "h_out": mlx_to_np(h_out[0]),
        "h": mlx_to_np(h[0]),
    }
    for name, arr in stages.items():
        if arr.dtype == np.int64:
            print(f"  {name}: {arr.shape} {arr.dtype}")
        else:
            print(f"  {name}: {arr.shape} {arr.dtype} rms={np.sqrt(np.mean(arr.astype(np.float64)**2)):.6g}")

    fixture = safetensors.numpy.load_file(args.fixture)
    fho = fixture["hidden_out.0"]
    diff = np.abs(mlx_to_np(h_out[0]) - fho)
    print("h_out vs fixture hidden_out.0 max abs diff:", diff.max())

    safetensors.numpy.save_file(stages, args.output)
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
