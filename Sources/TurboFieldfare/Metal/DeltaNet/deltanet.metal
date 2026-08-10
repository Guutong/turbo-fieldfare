#include <metal_stdlib>
using namespace metal;

// ============================================================================
// deltanet — P4-1: Gated DeltaNet decode step on the GPU, with the conv and
// recurrent state resident in Metal buffers.
//
// This is a direct port of the plain-Swift-fp32 oracle chain
// (`DeltaNetCPUBlock` + `DeltaNetConv` / `DeltaNetQKNorm` /
// `DeltaNetHeadExpansion` / `DeltaNetGate` / `DeltaNetRecurrence` /
// `DeltaNetOutputGate`), which ADR-0001 deliberately kept obviously-correct
// so Phase 4 would have an exact diff target.
//
// Numeric doctrine: everything is fp32, and every reduction is performed
// SEQUENTIALLY IN THE SAME ORDER AS THE SWIFT REFERENCE by a single thread.
// No tree/SIMD reductions, no atomics. That costs some redundant arithmetic
// (a row's sum of squares is recomputed by each thread that needs it) but at
// these dimensions the cost is negligible, and it makes the Metal path
// bit-comparable to the CPU path so any divergence is a real bug rather than
// a reassociation artifact (ADR-0002: tolerances are never widened).
//
// All kernels are `dn_`-prefixed: shader modules are concatenated into one
// runtime library, so symbol names are global.
// ============================================================================

static inline float dn_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float dn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// MSL has no `log1p`; this is the standard accurate substitute — the
// `log(u) * (y / (u - 1))` correction cancels the rounding of `1 + y` and is
// exact-in-the-limit for tiny y, matching libm's log1p closely enough that
// `dn_softplus` tracks `DeltaNetGate.softplus` to fp32 ULPs.
static inline float dn_log1p(float y) {
    const float u = 1.0f + y;
    if (u == 1.0f) return y;
    return log(u) * (y / (u - 1.0f));
}

// log(1 + e^x), stable for large x — matches `DeltaNetGate.softplus`.
static inline float dn_softplus(float x) {
    return x > 20.0f ? x + dn_log1p(exp(-x)) : dn_log1p(exp(x));
}

// ---------------------------------------------------------------------------
// Residual-stream bridging. The runner's `hidden` is FP16; the DeltaNet block
// works in fp32, exactly like the CPU block (which read `Float(ptr[i])` and
// wrote back `Float16(x[i] + deltaOut[i])`).
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_load_hidden(
    device const half*  hidden [[buffer(0)]],
    device float*       x      [[buffer(1)]],
    constant uint&      d      [[buffer(2)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= d) return;
    x[tid] = float(hidden[tid]);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_store_hidden(
    device half*        hidden   [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device const float* deltaOut [[buffer(2)]],
    constant uint&      d        [[buffer(3)]],
    uint                tid      [[thread_position_in_grid]]
) {
    if (tid >= d) return;
    hidden[tid] = half(x[tid] + deltaOut[tid]);
}

// ---------------------------------------------------------------------------
// Input RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i].
// One thread per element; each recomputes the whole sum in reference order.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_rmsnorm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      d      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= d) return;
    float sumSquares = 0.0f;
    for (uint i = 0; i < d; ++i) {
        sumSquares += x[i] * x[i];
    }
    const float inv = 1.0f / sqrt(sumSquares / float(d) + eps);
    out[tid] = x[tid] * inv * weight[tid];
}

// ---------------------------------------------------------------------------
// Dense fp32 mat-vec: y[r] = sum_c w[r * cols + c] * x[c].
// One thread per output row, accumulating in the reference's loop order.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_matvec(
    device const float* w    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      rows [[buffer(3)]],
    constant uint&      cols [[buffer(4)]],
    uint                tid  [[thread_position_in_grid]]
) {
    if (tid >= rows) return;
    const uint base = tid * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        acc += w[base + c] * x[c];
    }
    y[tid] = acc;
}

// ---------------------------------------------------------------------------
// Causal depthwise conv1d (width 4) + silu, then slide the conv state.
// One thread per channel; a channel's three state rows are touched only by
// that channel's thread, so the in-place slide is race-free.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_conv_step(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       state   [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      convDim [[buffer(4)]],
    uint                tid     [[thread_position_in_grid]]
) {
    if (tid >= convDim) return;
    const uint tap = tid * 4;
    const float x = input[tid];
    const float acc =
          weight[tap + 0] * state[tid]
        + weight[tap + 1] * state[convDim + tid]
        + weight[tap + 2] * state[2 * convDim + tid]
        + weight[tap + 3] * x;
    out[tid] = dn_silu(acc);

    // Slide the window: drop the oldest row, append this token.
    state[tid] = state[convDim + tid];
    state[convDim + tid] = state[2 * convDim + tid];
    state[2 * convDim + tid] = x;
}

// ---------------------------------------------------------------------------
// Fixed-scale per-key-head RMSNorm fused with the key->value head expansion
// (`repeat_interleave`). One thread per EXPANDED element; the source head's
// sum of squares is recomputed per thread, in reference order.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_qknorm_expand(
    device const float* src           [[buffer(0)]],
    device float*       out           [[buffer(1)]],
    constant uint&      numKeyHeads   [[buffer(2)]],
    constant uint&      numValueHeads [[buffer(3)]],
    constant uint&      headDim       [[buffer(4)]],
    constant float&     scale         [[buffer(5)]],
    constant float&     eps           [[buffer(6)]],
    uint                tid           [[thread_position_in_grid]]
) {
    if (tid >= numValueHeads * headDim) return;
    const uint valueHead = tid / headDim;
    const uint lane = tid % headDim;
    const uint repeatFactor = numValueHeads / numKeyHeads;
    const uint base = (valueHead / repeatFactor) * headDim;

    float sumSquares = 0.0f;
    for (uint i = 0; i < headDim; ++i) {
        const float v = src[base + i];
        sumSquares += v * v;
    }
    const float factor = scale / sqrt(sumSquares / float(headDim) + eps);
    out[tid] = src[base + lane] * factor;
}

// ---------------------------------------------------------------------------
// beta = sigmoid(b); g = exp(-exp(A_log) * softplus(a + dt_bias)).
// One thread per value head.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_gates(
    device const float* aRaw   [[buffer(0)]],
    device const float* bRaw   [[buffer(1)]],
    device const float* aLog   [[buffer(2)]],
    device const float* dtBias [[buffer(3)]],
    device float*       beta   [[buffer(4)]],
    device float*       g      [[buffer(5)]],
    constant uint&      heads  [[buffer(6)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= heads) return;
    beta[tid] = dn_sigmoid(bRaw[tid]);
    g[tid] = exp(-exp(aLog[tid]) * dn_softplus(aRaw[tid] + dtBias[tid]));
}

// ---------------------------------------------------------------------------
// The gated delta-rule recurrence, one decode timestep, state GPU-resident.
//
//   decay state by g -> kv_mem = sum(state * k) -> delta = (v - kv_mem) * beta
//   -> state += outer(k, delta) -> y = sum(state * q)     [write-then-read]
//
// One thread per (value head, value index) pair. Each thread owns exactly one
// `headKDim`-long row of the state, so the whole step is race-free and needs
// no barrier.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_recurrence(
    device const float* q             [[buffer(0)]],
    device const float* k             [[buffer(1)]],
    device const float* v             [[buffer(2)]],
    device const float* beta          [[buffer(3)]],
    device const float* g             [[buffer(4)]],
    device float*       state         [[buffer(5)]],
    device float*       y             [[buffer(6)]],
    constant uint&      numValueHeads [[buffer(7)]],
    constant uint&      headVDim      [[buffer(8)]],
    constant uint&      headKDim      [[buffer(9)]],
    uint                tid           [[thread_position_in_grid]]
) {
    if (tid >= numValueHeads * headVDim) return;
    const uint head = tid / headVDim;
    const float decay = g[head];
    const float writeScale = beta[head];
    const uint kBase = head * headKDim;
    const uint stateBase = tid * headKDim;

    float kvMemory = 0.0f;
    for (uint i = 0; i < headKDim; ++i) {
        state[stateBase + i] *= decay;
        kvMemory += state[stateBase + i] * k[kBase + i];
    }
    const float delta = (v[tid] - kvMemory) * writeScale;
    float acc = 0.0f;
    for (uint i = 0; i < headKDim; ++i) {
        state[stateBase + i] += k[kBase + i] * delta;
        acc += state[stateBase + i] * q[kBase + i];
    }
    y[tid] = acc;
}

// ---------------------------------------------------------------------------
// Output gate: out = silu(z) * rms_norm(y, weight), per value-head row.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_output_gate(
    device const float* y          [[buffer(0)]],
    device const float* z          [[buffer(1)]],
    device const float* normWeight [[buffer(2)]],
    device float*       out        [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    constant uint&      headVDim   [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint                tid        [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const uint lane = tid % headVDim;
    const uint base = tid - lane;

    float sumSquares = 0.0f;
    for (uint i = 0; i < headVDim; ++i) {
        const float value = y[base + i];
        sumSquares += value * value;
    }
    const float invRms = 1.0f / sqrt(sumSquares / float(headVDim) + eps);
    out[tid] = dn_silu(z[tid]) * y[tid] * normWeight[lane] * invRms;
}

// ============================================================================
// BATCHED VARIANTS — process K draft tokens in one kernel launch.
//
// Layout: token t's data for dimension d is at index t * dim + d (row-major).
// Scratch buffers expand from [dim] to [batch * dim].
// Shared weights stay at fixed offsets.
// Global state (convState, recurrentState) expands from [stateCount] to
//   [batch * stateCount], token t occupying offset t * stateCount.
// ============================================================================

// ---------------------------------------------------------------------------
// Batched residual-stream bridging. One thread per element across all tokens.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_load_hidden_batched(
    device const half*  hidden [[buffer(0)]],
    device float*       x      [[buffer(1)]],
    constant uint&      batch  [[buffer(2)]],
    constant uint&      D      [[buffer(3)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= batch * D) return;
    x[tid] = float(hidden[tid]);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_store_hidden_batched(
    device half*        hidden   [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device const float* deltaOut [[buffer(2)]],
    constant uint&      batch    [[buffer(3)]],
    constant uint&      D        [[buffer(4)]],
    uint                tid      [[thread_position_in_grid]]
) {
    if (tid >= batch * D) return;
    hidden[tid] = half(x[tid] + deltaOut[tid]);
}

// ---------------------------------------------------------------------------
// Batched input RMSNorm: each token independent, one thread per element per
// token. Each thread computes its own row's sum-of-squares.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_rmsnorm_batched(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      batch  [[buffer(3)]],
    constant uint&      D      [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= batch * D) return;
    const uint token = tid / D;
    const uint elem  = tid % D;
    const uint base  = token * D;

    float sumSquares = 0.0f;
    for (uint i = 0; i < D; ++i) {
        sumSquares += x[base + i] * x[base + i];
    }
    const float inv = 1.0f / sqrt(sumSquares / float(D) + eps);
    out[base + elem] = x[base + elem] * inv * weight[elem];
}

// ---------------------------------------------------------------------------
// Batched dense mat-vec: fully parallel over both batch and output rows.
// Weight matrix shared, input/output are batched.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_matvec_batched(
    device const float* w     [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    device float*       y     [[buffer(2)]],
    constant uint&      rows  [[buffer(3)]],
    constant uint&      cols  [[buffer(4)]],
    constant uint&      batch [[buffer(5)]],
    uint                tid   [[thread_position_in_grid]]
) {
    if (tid >= batch * rows) return;
    const uint token = tid / rows;
    const uint row   = tid % rows;
    const uint srcBase = token * cols;
    const uint dstBase = token * rows;
    const uint wBase = row * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        acc += w[wBase + c] * x[srcBase + c];
    }
    y[dstBase + row] = acc;
}

// ---------------------------------------------------------------------------
// Batched causal depthwise conv1d (width 4) + silu, then slide the conv
// state per-token. Uses uint tid for the channel dimension and iterates
// tokens inside — with convDim=8192 that's 8192 threads each looping over K.
//
// State layout: [batch × 3 × convDim]. Token t occupies regions starting
// at offset t × 3 × convDim. Region r (0–2) starts at (t × 3 + r) × convDim.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_conv_step_batched(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       state   [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      batch   [[buffer(4)]],
    constant uint&      convDim [[buffer(5)]],
    uint                tid     [[thread_position_in_grid]]
) {
    if (tid >= convDim) return;

    const uint tap = tid * 4;
    for (uint t = 0; t < batch; ++t) {
        const uint sOff = t * 3 * convDim;
        const float x = input[t * convDim + tid];

        const float acc =
              weight[tap + 0] * state[sOff + tid]
            + weight[tap + 1] * state[sOff + convDim + tid]
            + weight[tap + 2] * state[sOff + 2 * convDim + tid]
            + weight[tap + 3] * x;

        out[t * convDim + tid] = dn_silu(acc);

        // Slide the window — safe because state is per-token (no race).
        state[sOff + tid]                   = state[sOff + convDim + tid];
        state[sOff + convDim + tid]         = state[sOff + 2 * convDim + tid];
        state[sOff + 2 * convDim + tid]     = x;
    }
}

// ---------------------------------------------------------------------------
// Batched QK-RMSNorm fused with key->value head expansion. Per-token
// per-element, one thread per expanded output element.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_qknorm_expand_batched(
    device const float* src           [[buffer(0)]],
    device float*       out           [[buffer(1)]],
    constant uint&      numKeyHeads   [[buffer(2)]],
    constant uint&      numValueHeads [[buffer(3)]],
    constant uint&      headDim       [[buffer(4)]],
    constant float&     scale         [[buffer(5)]],
    constant float&     eps           [[buffer(6)]],
    constant uint&      batch         [[buffer(7)]],
    uint                tid           [[thread_position_in_grid]]
) {
    if (tid >= batch * numValueHeads * headDim) return;
    const uint token  = tid / (numValueHeads * headDim);
    const uint elem   = tid % (numValueHeads * headDim);

    const uint valueHead = elem / headDim;
    const uint lane = elem % headDim;
    const uint repeatFactor = numValueHeads / numKeyHeads;
    const uint base = (valueHead / repeatFactor) * headDim;
    const uint srcBase = token * numKeyHeads * headDim;
    const uint outBase = token * numValueHeads * headDim;

    float sumSquares = 0.0f;
    for (uint i = 0; i < headDim; ++i) {
        const float v = src[srcBase + base + i];
        sumSquares += v * v;
    }
    const float factor = scale / sqrt(sumSquares / float(headDim) + eps);
    out[outBase + elem] = src[srcBase + base + lane] * factor;
}

// ---------------------------------------------------------------------------
// Batched beta/g gates. Independent per token, per head.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_gates_batched(
    device const float* aRaw   [[buffer(0)]],
    device const float* bRaw   [[buffer(1)]],
    device const float* aLog   [[buffer(2)]],
    device const float* dtBias [[buffer(3)]],
    device float*       beta   [[buffer(4)]],
    device float*       g      [[buffer(5)]],
    constant uint&      heads  [[buffer(6)]],
    constant uint&      batch  [[buffer(7)]],
    uint                tid    [[thread_position_in_grid]]
) {
    if (tid >= batch * heads) return;
    const uint token = tid / heads;
    const uint head  = tid % heads;
    const uint aIdx  = token * heads + head;
    beta[token * heads + head] = dn_sigmoid(bRaw[aIdx]);
    g[token * heads + head] = exp(-exp(aLog[head]) * dn_softplus(aRaw[aIdx] + dtBias[head]));
}

// ---------------------------------------------------------------------------
// Batched gated delta-rule recurrence. One thread per (value head, value
// index) pair per token. Each thread owns one headKDim-long slice of the
// per-token state buffer.
//
// State layout: [batch × numValueHeads × headVDim × headKDim].
// Thread tid maps to local idx = tid % (numValueHeads * headVDim),
// which spans the flat (head, vIdx) plane.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_recurrence_batched(
    device const float* q             [[buffer(0)]],
    device const float* k             [[buffer(1)]],
    device const float* v             [[buffer(2)]],
    device const float* beta          [[buffer(3)]],
    device const float* g             [[buffer(4)]],
    device float*       state         [[buffer(5)]],
    device float*       y             [[buffer(6)]],
    constant uint&      numValueHeads [[buffer(7)]],
    constant uint&      headVDim      [[buffer(8)]],
    constant uint&      headKDim      [[buffer(9)]],
    constant uint&      batch         [[buffer(10)]],
    uint                tid           [[thread_position_in_grid]]
) {
    if (tid >= batch * numValueHeads * headVDim) return;
    const uint token  = tid / (numValueHeads * headVDim);
    const uint local  = tid % (numValueHeads * headVDim);
    const uint head   = local / headVDim;
    const uint vIdx   = local % headVDim;

    const float decay      = g[token * numValueHeads + head];
    const float writeScale = beta[token * numValueHeads + head];
    const uint ekDim       = numValueHeads * headKDim;  // expanded key dim
    const uint kvBase      = token * ekDim + head * headKDim;
    const uint qBase       = kvBase;
    const uint vBase       = token * numValueHeads * headVDim + vIdx;
    const uint stateBase   = token * numValueHeads * headVDim * headKDim + local * headKDim;

    float kvMemory = 0.0f;
    for (uint i = 0; i < headKDim; ++i) {
        state[stateBase + i] *= decay;
        kvMemory += state[stateBase + i] * k[kvBase + i];
    }
    const float delta = (v[vBase] - kvMemory) * writeScale;
    float acc = 0.0f;
    for (uint i = 0; i < headKDim; ++i) {
        state[stateBase + i] += k[kvBase + i] * delta;
        acc += state[stateBase + i] * q[qBase + i];
    }
    y[token * numValueHeads * headVDim + vIdx] = acc;
}

// ---------------------------------------------------------------------------
// Batched output gate: out = silu(z) * rms_norm(y, weight), per token,
// per value-head row.
// ---------------------------------------------------------------------------

[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_output_gate_batched(
    device const float* y          [[buffer(0)]],
    device const float* z          [[buffer(1)]],
    device const float* normWeight [[buffer(2)]],
    device float*       out        [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    constant uint&      headVDim   [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    constant uint&      batch      [[buffer(7)]],
    uint                tid        [[thread_position_in_grid]]
) {
    if (tid >= batch * count) return;
    const uint token = tid / count;
    const uint local = tid % count;
    const uint lane  = local % headVDim;
    const uint base  = token * count + (local - lane);

    float sumSquares = 0.0f;
    for (uint i = 0; i < headVDim; ++i) {
        const float value = y[base + i];
        sumSquares += value * value;
    }
    const float invRms = 1.0f / sqrt(sumSquares / float(headVDim) + eps);
    out[tid] = dn_silu(z[tid]) * y[base + lane] * normWeight[lane] * invRms;
}
