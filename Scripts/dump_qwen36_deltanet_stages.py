#!/usr/bin/env python3
"""Dump intermediate DeltaNet-block tensors for layer 0, stage by stage.

Companion to dump_qwen36_reference.py: that script only captures
hidden_in.0/hidden_out.0 (the layer boundary). This script captures MLX's
own intermediate tensors *inside* layer 0's GatedDeltaNet block, by
re-running the exact body of `Qwen3NextGatedDeltaNet.__call__`
(mlx_lm/models/qwen3_next.py) inline so every stage can be captured. Used
to bisect the P3-3 numeric divergence (see PHASE-LOG.md 2026-08-08 entry)
against the plain-Swift-fp32 layer-0 isolation test.

Fixed prompt: "The capital of France is" (same as the main fixture, so the
input tokens/hidden states line up token-for-token with
Tests/Fixtures/qwen36_fixture.safetensors's hidden_in.0).

Captured stages (each `[S, ...]`, batch dim squeezed):
  norm_in            - input_layernorm(hidden_in.0)
  qkvz_raw           - in_proj_qkvz(norm_in), pre-split
  ba_raw             - in_proj_ba(norm_in), pre-split
  q_preconv, k_preconv, v_preconv  - post fix_query_key_value_ordering, pre-conv
  z                  - the z (output-gate) split, never touched by conv
  b_raw, a_raw       - the b/a split from ba_raw, pre-sigmoid/pre-decay
  conv_out           - silu(conv1d(...)), pre-split back to q/k/v
  q_postconv, k_postconv, v_postconv - conv_out split back to heads
  q_normed, k_normed - after the inv_scale**2 / inv_scale rms_norm (pre head-expansion)
  beta               - sigmoid(b)
  g                  - decay gate, exp(-exp(A_log)*softplus(a+dt_bias))
  y_recurrence       - gated_delta_update output, pre output-gate
  y_gated            - self.norm(y_recurrence, z)  (RMSNormGated + swiglu)
  delta_out          - out_proj(y_gated), pre residual-add
  hidden_out_check   - hidden_in.0 + delta_out (should equal fixture hidden_out.0)
"""

import argparse
import json
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
    parser.add_argument("--output", default="qwen36_layer0_deltanet_stages.safetensors")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    lm = model.language_model
    text_model = lm.model
    layer0 = text_model.layers[0]
    assert layer0.is_linear, "layer 0 must be a DeltaNet (linear_attn) layer"
    dn = layer0.linear_attn

    prompt_tokens = mx.array(tokenizer.encode(args.prompt))
    print(f"Prompt tokens: {prompt_tokens.tolist()}")
    tokens = prompt_tokens[None]  # [1, S]

    # Recompute embedding + hidden_in.0 exactly like dump_qwen36_reference.py,
    # so this script is self-contained and doesn't depend on the fixture for
    # the input (but we'll cross-check against the fixture below).
    hidden_in = text_model.embed_tokens(tokens)  # [1, S, D]
    S = hidden_in.shape[1]

    # --- Cross-check against the existing fixture's hidden_in.0 ---
    fixture = safetensors.numpy.load_file(args.fixture)
    fixture_hidden_in0 = fixture["hidden_in.0"]
    diff = np.abs(mlx_to_np(hidden_in[0]) - fixture_hidden_in0).max()
    print(f"hidden_in.0 cross-check max-abs diff vs fixture: {diff}")

    stages = {}

    def cap(name, x):
        stages[name] = mlx_to_np(x[0] if x.ndim == 3 else x)

    # ---- Inline Qwen3NextGatedDeltaNet.__call__ body ----
    norm_in = layer0.input_layernorm(hidden_in)
    cap("norm_in", norm_in)

    # This checkpoint's arch is mlx_lm.models.qwen3_5.GatedDeltaNet (NOT
    # qwen3_next.py's packed-qkvz/ba variant): separate in_proj_qkv /
    # in_proj_z / in_proj_b / in_proj_a projections. Confirmed by inspecting
    # type(dn).__module__ == 'mlx_lm.models.qwen3_5' at runtime.
    qkv_raw = dn.in_proj_qkv(norm_in)
    z_raw = dn.in_proj_z(norm_in)
    b_raw = dn.in_proj_b(norm_in)
    a_raw = dn.in_proj_a(norm_in)
    cap("qkv_raw", qkv_raw)
    cap("z_raw", z_raw)
    cap("b_raw", b_raw)
    cap("a_raw", a_raw)

    z = z_raw.reshape(1, S, dn.num_v_heads, dn.head_v_dim)
    cap("z", z)

    B = 1
    conv_state = mx.zeros((B, dn.conv_kernel_size - 1, dn.conv_dim), dtype=hidden_in.dtype)
    conv_input = mx.concatenate([conv_state, qkv_raw], axis=1)
    conv_out = nn.silu(dn.conv1d(conv_input))
    cap("conv_out", conv_out)

    q2, k2, v2 = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [dn.key_dim, 2 * dn.key_dim], -1),
            [dn.num_k_heads, dn.num_k_heads, dn.num_v_heads],
            [dn.head_k_dim, dn.head_k_dim, dn.head_v_dim],
        )
    ]
    cap("q_postconv", q2)
    cap("k_postconv", k2)
    cap("v_postconv", v2)

    inv_scale = k2.shape[-1] ** -0.5
    q_normed = (inv_scale**2) * mx.fast.rms_norm(q2, None, 1e-6)
    k_normed = inv_scale * mx.fast.rms_norm(k2, None, 1e-6)
    cap("q_normed", q_normed)
    cap("k_normed", k_normed)

    from mlx_lm.models.gated_delta import gated_delta_update, compute_g

    beta = mx.sigmoid(b_raw)
    g = compute_g(dn.A_log, a_raw, dn.dt_bias)
    cap("beta", beta)
    cap("g", g)

    y_recurrence, _state = gated_delta_update(
        q_normed, k_normed, v2, a_raw, b_raw, dn.A_log, dn.dt_bias, None, None, use_kernel=False
    )
    cap("y_recurrence", y_recurrence)

    y_gated = dn.norm(y_recurrence, z)
    cap("y_gated", y_gated)

    delta_out = dn.out_proj(y_gated.reshape(B, S, -1))
    cap("delta_out", delta_out)

    hidden_out_check = hidden_in + delta_out
    cap("hidden_out_check", hidden_out_check)

    fixture_hidden_out0 = fixture["hidden_out.0"]
    diff_out = np.abs(mlx_to_np(hidden_out_check[0]) - fixture_hidden_out0).max()
    print(f"hidden_out_check vs fixture hidden_out.0 max-abs diff: {diff_out}")
    # NOTE: hidden_out.0 in the main fixture is captured AFTER the full
    # decoder layer (attn + MLP/MoE residual), so it will legitimately
    # differ from hidden_out_check (attn-only residual) -- this diff is
    # expected to be nonzero and is not itself evidence of a bug. It's
    # printed for visibility only.

    for name, arr in stages.items():
        print(f"  {name}: {arr.shape} {arr.dtype} rms={np.sqrt(np.mean(arr.astype(np.float64)**2)):.6g}")

    safetensors.numpy.save_file(stages, args.output)
    print(f"Saved {len(stages)} stage tensors to {args.output}")


if __name__ == "__main__":
    main()
