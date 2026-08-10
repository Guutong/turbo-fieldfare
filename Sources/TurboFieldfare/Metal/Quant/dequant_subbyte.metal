#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_subbyte — MLX `affine` dequant for bit widths that need a general
// bitstream reader. 5- and 6-bit are the reason it exists; 8-bit rides along
// because the same reader degenerates to a byte load there and this file's
// loop is group-size generic, which dequant_int8.metal's is not. See the note
// on `dequant_int8_qkv_gemv_simd` below.
//
// Packing (per row of length N, `bits` in {5,6,8}):
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
    uint                  lane,
    uint                  outputStride
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
        y[row * outputStride] = half(acc);
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
                                  rows_per_tg, tg_idx, sg_idx, lane, 1u);
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
                                  rows_per_tg, tg_idx, sg_idx, lane, 1u);
}

// ============================================================================
// Fused Q/K/V projection at sub-byte precision.
//
// Same row-partitioning as `dequant_int4_qkv_gemv_simd`: one dispatch covers
// Q's Mq rows followed by K's and V's Mkv rows each, so the three projections
// share one encoder and one pass over `x` instead of three. A simdgroup owns
// one output row; `global_row` picks which of the three weight matrices that
// row belongs to.
//
// Two deliberate differences from the int4 version:
//
//  - Shapes are runtime arguments, not function constants. The int4 path
//    specializes on Gemma's two known decode shapes; there is no equivalent
//    fixed set here yet, and guessing one would bake in shapes no measurement
//    has justified.
//  - `groupSize` is a runtime argument, because that is what
//    `dequant_subbyte_gemv_body` already takes. Laguna's 5-bit attention is
//    group 64, but nothing here needs to assume it.
//
// Templated on BITS, and instantiated at both 5 and 8 — see the note above
// `dequant_int8_qkv_gemv_simd` for why 8-bit lands here rather than on the
// int8 path. 6-bit is one instantiation away if a checkpoint ever needs it.
// ============================================================================
template <uint BITS>
static inline void subbyte_qkv_gemv_body(
    device const uint8_t* qW, device const bfloat* qScales, device const bfloat* qBiases,
    device const uint8_t* kW, device const bfloat* kScales, device const bfloat* kBiases,
    device const uint8_t* vW, device const bfloat* vScales, device const bfloat* vBiases,
    device const half*    x,
    device half*          qY,
    device half*          kY,
    device half*          vY,
    uint Mq, uint Mkv, uint N, uint groupSize,
    uint outputStride,
    uint tg_idx, uint sg_idx, uint lane
) {
    constexpr uint rows_per_tg = 8;
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = Mq + 2u * Mkv;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat*  scales;
    device const bfloat*  biases;
    device half*          y;
    uint local_row;
    uint M;
    if (global_row < Mq) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = Mq;
    } else if (global_row < Mq + Mkv) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - Mq;
        M = Mkv;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - Mq - Mkv;
        M = Mkv;
    }
    // rows_per_tg=1 with sg_idx=0 makes the body treat `local_row` as its row
    // directly — the same reuse the int4 fused kernel relies on.
    dequant_subbyte_gemv_body<BITS>(W, scales, biases, x, y, M, N, groupSize,
                                    1u, local_row, 0u, lane, outputStride);
}

kernel void dequant_int5_qkv_gemv_simd(
    device const uint8_t* qW        [[buffer(0)]],
    device const bfloat*  qScales   [[buffer(1)]],
    device const bfloat*  qBiases   [[buffer(2)]],
    device const uint8_t* kW        [[buffer(3)]],
    device const bfloat*  kScales   [[buffer(4)]],
    device const bfloat*  kBiases   [[buffer(5)]],
    device const uint8_t* vW        [[buffer(6)]],
    device const bfloat*  vScales   [[buffer(7)]],
    device const bfloat*  vBiases   [[buffer(8)]],
    device const half*    x         [[buffer(9)]],
    device half*          qY        [[buffer(10)]],
    device half*          kY        [[buffer(11)]],
    device half*          vY        [[buffer(12)]],
    constant uint&        Mq        [[buffer(13)]],
    constant uint&        Mkv       [[buffer(14)]],
    constant uint&        N         [[buffer(15)]],
    constant uint&        groupSize [[buffer(16)]],
    constant uint&        outStr    [[buffer(17)]],
    uint                  tg_idx    [[threadgroup_position_in_grid]],
    uint                  sg_idx    [[simdgroup_index_in_threadgroup]],
    uint                  lane      [[thread_index_in_simdgroup]]
) {
    const uint outStride = outStr > 0 ? outStr : 1u;
    subbyte_qkv_gemv_body<5>(qW, qScales, qBiases,
                             kW, kScales, kBiases,
                             vW, vScales, vBiases,
                             x, qY, kY, vY,
                             Mq, Mkv, N, groupSize,
                             outStride,
                             tg_idx, sg_idx, lane);
}

// The 8-bit fused QKV, and yes, it lives in the sub-byte file.
//
// At BITS=8 `extract_subbyte_code` degenerates exactly to a plain byte load:
// bit_offset is a whole number of bytes, so bit_in_byte is 0, the mask is
// 0xFF, and the straddle test `bit_in_byte + BITS > 8` is never true — the
// second byte is never even read, so this cannot over-read past a group.
// Verified exhaustively against a direct byte load before relying on it.
//
// That makes this instantiation *correct by the same construction* as the
// 5-bit one, and — the actual reason it is here — **group-size generic**.
// `dequant_int8_gemv_simd` in dequant_int8.metal hardcodes group 64 and maps
// two elements per lane (`lane * 2`), which silently produces wrong numbers at
// group 128. Laguna's 8-bit attention is group 64 and would fit that mapping,
// but its 8-bit *shared experts* are group 128, so the codebase needs a
// group-generic 8-bit path regardless; building the fused QKV on the strided-
// lane loop gets it without a second bespoke kernel.
//
// The cost is speed: a strided-lane loop over `groupSize` is slower than the
// two-per-lane load it replaces. Nothing runs this checkpoint end to end yet,
// so correctness wins for now — the same trade the group-128 int4 work made.
// If profiling later says this is hot, add a group-64 fast path beside it
// rather than reshaping this one.
kernel void dequant_int8_qkv_gemv_simd(
    device const uint8_t* qW        [[buffer(0)]],
    device const bfloat*  qScales   [[buffer(1)]],
    device const bfloat*  qBiases   [[buffer(2)]],
    device const uint8_t* kW        [[buffer(3)]],
    device const bfloat*  kScales   [[buffer(4)]],
    device const bfloat*  kBiases   [[buffer(5)]],
    device const uint8_t* vW        [[buffer(6)]],
    device const bfloat*  vScales   [[buffer(7)]],
    device const bfloat*  vBiases   [[buffer(8)]],
    device const half*    x         [[buffer(9)]],
    device half*          qY        [[buffer(10)]],
    device half*          kY        [[buffer(11)]],
    device half*          vY        [[buffer(12)]],
    constant uint&        Mq        [[buffer(13)]],
    constant uint&        Mkv       [[buffer(14)]],
    constant uint&        N         [[buffer(15)]],
    constant uint&        groupSize [[buffer(16)]],
    constant uint&        outStr    [[buffer(17)]],
    uint                  tg_idx    [[threadgroup_position_in_grid]],
    uint                  sg_idx    [[simdgroup_index_in_threadgroup]],
    uint                  lane      [[thread_index_in_simdgroup]]
) {
    const uint outStride = outStr > 0 ? outStr : 1u;
    subbyte_qkv_gemv_body<8>(qW, qScales, qBiases,
                             kW, kScales, kBiases,
                             vW, vScales, vBiases,
                             x, qY, kY, vY,
                             Mq, Mkv, N, groupSize,
                             outStride,
                             tg_idx, sg_idx, lane);
}

kernel void dequant_int8_gemv_generic(
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
    dequant_subbyte_gemv_body<8>(W, scales, biases, x, y, M, N, groupSize,
                                  rows_per_tg, tg_idx, sg_idx, lane, 1u);
}
