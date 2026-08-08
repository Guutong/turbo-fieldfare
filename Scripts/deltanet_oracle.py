#!/usr/bin/env python3
"""Independent oracle for the Gated DeltaNet recurrence (P3-2).

Plain-Python, naive loops, no numpy/mlx — a transcription of the equations
verified against ml-explore/mlx-lm main on 2026-08-08 (see PHASE-LOG and
docs/adr/0002). Its printed constants are pasted into
Tests/TurboFieldfare/Core/Runtime/DeltaNet/DeltaNetRuleTests.swift; re-run
this script to re-verify them.

Sources transcribed:
  - mlx_lm/models/qwen3_5.py        GatedDeltaNet.__call__
        q = (inv_scale**2) * rms_norm(q, None, 1e-6)      # inv_scale = Dk**-0.5
        k = inv_scale * rms_norm(k, None, 1e-6)
        out = self.norm(out, z)                            # silu(z) * rms_norm(y)
        head expansion: mx.repeat(q, Hv // Hk, -2)          (repeat_interleave)
  - mlx_lm/models/gated_delta.py    gated_delta_ops / _gated_delta_step_ops
        beta = sigmoid(b)
        g = exp(-exp(A_log) * softplus(a + dt_bias))       (fp32)
        state = state * g
        kv_mem = (state * k).sum(-1)
        delta = (v - kv_mem) * beta
        state = state + outer(k, delta)
        y = (state * q).sum(-1)                            (write-then-read)
  - mlx_lm/models/qwen3_next.py     Qwen3NextRMSNormGated / _precise_swiglu
        out = silu(z, fp32) * rms_norm(y, weight, eps)     (fp32 math)

Inference semantics: no causal mask exists (the ssm mask is a padding mask
and None for the fixture dump) — prefill IS the sequential loop, so this
oracle runs one timestep at a time.
"""

import math

EPS = 1e-6

# --- Case: Hk=1, Hv=2 (repeat_factor 2), Dk=Dv=2, T=3. Rational inputs. ---
Hk, Hv, Dk, Dv, T = 1, 2, 2, 2, 3

A_log = [0.0, math.log(2.0)]       # exp(A_log) = [1, 2]
dt_bias = [0.0, 0.5]
norm_weight = [1.0, 0.5]

tokens = [
    dict(q=[[1.0, 2.0]], k=[[1.0, 0.0]], v=[[1.0, 0.0], [0.0, 1.0]],
         a=[0.0, 0.0], b=[0.0, 0.0], z=[[1.0, -1.0], [0.5, 2.0]]),
    dict(q=[[0.5, -1.0]], k=[[1.0, 1.0]], v=[[2.0, 0.0], [0.0, 0.5]],
         a=[0.25, -0.5], b=[1.0, -1.0], z=[[0.0, 0.0], [1.0, 1.0]]),
    dict(q=[[2.0, 1.0]], k=[[0.0, 1.0]], v=[[1.0, 1.0], [-1.0, 0.0]],
         a=[1.0, 2.0], b=[0.5, 0.0], z=[[-2.0, 1.0], [0.25, -0.25]]),
]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softplus(x):  # mlx nn.softplus = logaddexp(x, 0)
    return math.log1p(math.exp(x)) if x <= 20 else x + math.log1p(math.exp(-x))


def rms_norm(x, weight=None, eps=EPS):
    mean_sq = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(mean_sq + eps)
    out = [v * inv for v in x]
    if weight is not None:
        out = [o * w for o, w in zip(out, weight)]
    return out


def swift_floats(values):
    return "[" + ", ".join(f"{v!r}" for v in values) + "]"


def main():
    inv_scale = Dk ** -0.5
    state = [[[0.0] * Dk for _ in range(Dv)] for _ in range(Hv)]
    print(f"// Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv} T={T}; inv_scale={inv_scale!r}")
    print(f"// A_log {swift_floats(A_log)}  dt_bias {swift_floats(dt_bias)}")
    print(f"// norm_weight {swift_floats(norm_weight)}")

    for t, tok in enumerate(tokens):
        # Gating (fp32 in the source).
        beta = [sigmoid(x) for x in tok["b"]]
        g = [math.exp(-math.exp(al) * softplus(a + dt))
             for al, a, dt in zip(A_log, tok["a"], dt_bias)]

        # Fixed-scale RMSNorm on key heads, BEFORE head expansion.
        q = [[inv_scale ** 2 * v for v in rms_norm(head)] for head in tok["q"]]
        k = [[inv_scale * v for v in rms_norm(head)] for head in tok["k"]]
        # repeat_interleave: value head h <- key head h // (Hv // Hk).
        rf = Hv // Hk
        q_exp = [q[h // rf] for h in range(Hv)]
        k_exp = [k[h // rf] for h in range(Hv)]

        y = [[0.0] * Dv for _ in range(Hv)]
        for h in range(Hv):
            for dv in range(Dv):
                kv_mem = 0.0
                for i in range(Dk):
                    state[h][dv][i] *= g[h]                    # decay
                    kv_mem += state[h][dv][i] * k_exp[h][i]    # read
                delta = (tok["v"][h][dv] - kv_mem) * beta[h]
                acc = 0.0
                for i in range(Dk):
                    state[h][dv][i] += k_exp[h][i] * delta     # write
                    acc += state[h][dv][i] * q_exp[h][i]       # then read
                y[h][dv] = acc

        # Output gate: silu(z) * rms_norm(y, weight), fp32.
        gated = []
        for h in range(Hv):
            gated.extend(silu_g * v for silu_g, v in zip(
                [z / (1.0 + math.exp(-z)) for z in tok["z"][h]],
                rms_norm(y[h], norm_weight)))

        print(f"// t{t}: beta {swift_floats(beta)}")
        print(f"//      g    {swift_floats(g)}")
        print(f"//      q_norm[0] {swift_floats(q[0])}")
        print(f"//      k_norm[0] {swift_floats(k[0])}")
        print(f"//      y    {swift_floats([v for row in y for v in row])}")
        print(f"//      out  {swift_floats(gated)}")

    flat_state = [v for h in range(Hv) for row in state[h] for v in row]
    print(f"// final state {swift_floats(flat_state)}")


if __name__ == "__main__":
    main()
