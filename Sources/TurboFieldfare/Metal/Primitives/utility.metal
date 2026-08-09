#include <metal_stdlib>
using namespace metal;

// Activation helpers — shared library so both INT4 and INT8 shared-expert
// paths use the same implementation without compiling private modules.
// gelu_pytorch_tanh and silu are defined in moe.metal (loaded before utility
// in the concatenated library).

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    constant bool&     use_silu [[buffer(4)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half( (use_silu ? silu(g) : gelu_pytorch_tanh(g)) * u );
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(silu(g) * u);
}

// Elementwise FP16 add: a += b. Used for raw residuals in pre-norm topologies.
[[kernel, max_total_threads_per_threadgroup(256)]]
void add_fp16(
    device half*       a     [[buffer(0)]],
    device const half* b     [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    a[tid] += b[tid];
}
