#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_subbyte — MLX `affine` 5-bit and 6-bit dequant.
//
// Packing (per row of length N, `bits` in {5,6}):
//   MLX packs unsigned `bits`-wide codes into a byte-addressed, LSB-first,
//   little-endian continuous bitstream: element i's code occupies bit range
//   [i*bits, i*bits+bits) of the row's packed bytes, with no padding between
//   elements — see the doc comment on
//   Sources/TurboFieldfare/Infrastructure/ModelIO/QuantizationSubByte.swift
//   for the full derivation. VERIFIED byte-for-byte against real MLX output
//   (mlx.core.quantize(..., bits=5|6), group_size 64 and 128) — see that
//   file's doc comment for how, and QuantizationSubByteTests for the baked-in
//   MLX fixture. This kernel's extract_subbyte_code implements the same
//   convention as that file's packBits/unpackBits, exercised end-to-end by
//   DequantInt5GEMVTests/DequantInt6GEMVTests against the CPU reference.
//   Because group sizes used here (64/128) are multiples of 8, group and row
//   boundaries always land on whole-byte boundaries, so no group-edge
//   special-casing is needed; a 5- or 6-bit code spans at most 2 bytes.
//
//   scales  : N/groupSize BF16, one per group.
//   biases  : N/groupSize BF16, one per group.
//   value   : w[i] = float(code[i]) * scale[i/groupSize] + bias[i/groupSize].
//
// Unlike dequant_int4/dequant_int8 (which hardcode group size 64), groupSize
// is a runtime kernel parameter here: oQ4e's 5/6-bit layer-0 projections use
// group_size=64, but the routed-expert 4-bit path (unaffected by this file)
// uses group_size=128, and nothing here should assume a fixed group size.
// ============================================================================

// Extracts the BITS-wide unsigned code starting at element index `elem_idx`
// within a group, where `group_base` points at byte 0 of that group's packed
// data. A code spans at most 2 bytes for BITS <= 8, so a 16-bit combine of
// at most two byte loads is always sufficient. The second byte is only read
// when the code actually straddles into it, so this never reads past the
// group's own byte range.
template <uint BITS>
inline uint extract_subbyte_code(device const uint8_t* group_base, uint elem_idx) {
    const uint bit_offset  = elem_idx * BITS;
    const uint byte_idx    = bit_offset >> 3;
    const uint bit_in_byte = bit_offset & 7u;
    const uint b0 = uint(group_base[byte_idx]);
    const uint b1 = (bit_in_byte + BITS > 8u) ? uint(group_base[byte_idx + 1]) : 0u;
    const uint combined = b0 | (b1 << 8);
    constexpr uint mask = (1u << BITS) - 1u;
    return (combined >> bit_in_byte) & mask;
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group (32 threads) per output row;
// each lane strides across a group's elements by 32 so this works for any
// groupSize that's a multiple of 32 (64 and 128, the two sizes this codebase
// uses). Affine factoring: sum_k (q_k*s + b) * x_k = s*sum_k(q_k*x_k) +
// b*sum_k(x_k), so scale/bias cost one FMA per group instead of per element.
template <uint BITS>
static inline void dequant_subbyte_gemv_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  groupSize,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups    = N / groupSize;
    const uint row_bytes   = (N * BITS) / 8u;
    const uint group_bytes = (groupSize * BITS) / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        device const uint8_t* group_base = W_row + g * group_bytes;
        device const half*    x_group    = x + g * groupSize;
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint elem = lane; elem < groupSize; elem += 32u) {
            const uint  code = extract_subbyte_code<BITS>(group_base, elem);
            const float xv   = float(x_group[elem]);
            dot = fma(float(code), xv, dot);
            sum += xv;
        }
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void dequant_int5_gemv_simd(
    device const uint8_t* W         [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const half*    x         [[buffer(3)]],
    device half*          y         [[buffer(4)]],
    constant uint&        M         [[buffer(5)]],
    constant uint&        N         [[buffer(6)]],
    constant uint&        groupSize [[buffer(7)]],
    uint                  tg_idx    [[threadgroup_position_in_grid]],
    uint                  sg_idx    [[simdgroup_index_in_threadgroup]],
    uint                  lane      [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    dequant_subbyte_gemv_body<5>(W, scales, biases, x, y, M, N, groupSize,
                                  rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void dequant_int6_gemv_simd(
    device const uint8_t* W         [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const half*    x         [[buffer(3)]],
    device half*          y         [[buffer(4)]],
    constant uint&        M         [[buffer(5)]],
    constant uint&        N         [[buffer(6)]],
    constant uint&        groupSize [[buffer(7)]],
    uint                  tg_idx    [[threadgroup_position_in_grid]],
    uint                  sg_idx    [[simdgroup_index_in_threadgroup]],
    uint                  lane      [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    dequant_subbyte_gemv_body<6>(W, scales, biases, x, y, M, N, groupSize,
                                  rows_per_tg, tg_idx, sg_idx, lane);
}
