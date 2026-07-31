import Foundation

/// MLX `affine` 5-bit and 6-bit dequant support, for the mixed-precision
/// `mlx-community/Laguna-S-2.1-oQ4e` conversion (poolside/Laguna-S-2.1).
/// Gemma's 4-bit/8-bit paths live in `Quantization.swift` and are untouched
/// by this file — this is purely additive.
///
/// Unlike `Quantization.groupSize` (a hardcoded global used by the existing
/// 4-bit/8-bit paths), every function here takes `groupSize` as an explicit
/// parameter: oQ4e mixes group_size=64 (layer-0 5/6-bit projections,
/// lm_head/embed at 8-bit) with group_size=128 (routed experts at 4-bit).
///
/// MARK: - Bit packing convention
///
/// MLX packs sub-byte quantized weights into a byte-addressed, LSB-first,
/// little-endian continuous bitstream: element `i`'s unsigned `bits`-wide
/// code occupies bit range `[i*bits, i*bits+bits)` of the row's packed byte
/// array, with no padding between elements. Concretely, if you laid the
/// packed bytes out 4 at a time as little-endian `uint32` words, element `i`
/// would sit at bit offset `i*bits` counting from the LSB of word 0 onward —
/// this is the same layout whether you address it by byte or by 32-bit word,
/// because a little-endian `uint32`'s 4 bytes preserve LSB-first bit order
/// across the byte boundary.
///
/// For 4-bit and 8-bit this happens to divide evenly within a byte (2 codes
/// per byte, 1 code per byte) so no code ever straddles a byte boundary —
/// that's what `Quantization.swift`'s nibble/byte packing implements. For
/// 5-bit and 6-bit it does NOT divide evenly: a 5-bit code straddles a byte
/// boundary unless its bit offset is a multiple of 8 (only 1-in-8 codes), and
/// likewise for 6-bit (1-in-4 codes at best). `packBits`/`unpackBits` below
/// implement the general continuous-bitstream packing so callers never need
/// to special-case the straddling codes.
///
/// Group boundaries always land on whole-byte boundaries for the group sizes
/// this codebase uses (32/64/128, all multiples of 8), because
/// `groupSize * bits` is then always a multiple of 8 for bits in {5, 6}
/// (`groupSize` a multiple of 8 handles 5-bit since gcd(5,8)=1; a multiple of
/// 4 suffices for 6-bit since gcd(3,4)=1, and 8 | groupSize implies 4 |
/// groupSize). So no cross-group special-casing is needed either.
///
/// VERIFICATION STATUS: VERIFIED against real MLX output. The default
/// `python3` in this environment is 3.14 and lacks MLX
/// (`ModuleNotFoundError`), but Homebrew's `python3.11` has `mlx==0.31.2`
/// installed, and `mlx.core.quantize(w, group_size=64|128, bits=5|6)` was
/// used to check this layout byte-for-byte:
///
///   1. Quantize a random row with real MLX, giving packed `wq` + BF16
///      `scales`/`biases`.
///   2. Recover MLX's own integer codes in a packing-agnostic way: affine
///      dequant is exact given the code, so
///      `code = round((dequantize(wq) - bias) / scale)` recovers the
///      integer codes without assuming anything about `wq`'s bit layout.
///   3. Re-pack those codes with an independent Python reimplementation of
///      "byte-addressed, LSB-first, little-endian continuous bitstream"
///      (a different code path from this file's `packBits`, not a call into
///      it) and diff against `wq`'s raw bytes.
///
/// All of (bits=5, group=64), (bits=6, group=64), (bits=5, group=128),
/// (bits=6, group=128), at multiple row lengths, matched byte-for-byte.
/// `QuantizationSubByteTests.packBits_fiveBit_matchesRealMLXGroundTruth` /
/// `_sixBit_...` bake one such fixture (real MLX codes + real MLX packed
/// bytes) into the Swift suite as a permanent regression check against this
/// file's `packBits`/`unpackBits`.
public enum QuantizationSubByte {

    // MARK: - Generic bit packing

    /// Packs `values` (each in `[0, 2^bits)`) into a byte-addressed,
    /// LSB-first, little-endian continuous bitstream — see the type-level
    /// doc comment for the exact convention this implements.
    static func packBits(_ values: [Int], bits: Int) -> [UInt8] {
        precondition(bits > 0 && bits <= 8, "packBits only supports 1...8 bits, got \(bits)")
        let n = values.count
        let totalBits = n * bits
        precondition(totalBits % 8 == 0,
                     "packed row of \(n) x \(bits)-bit values is not byte-aligned")
        var out = [UInt8](repeating: 0, count: totalBits / 8)
        let maxVal = (1 << bits) - 1
        for i in 0..<n {
            let v = values[i]
            precondition(v >= 0 && v <= maxVal,
                         "value \(v) out of range for \(bits)-bit pack (max \(maxVal))")
            var remaining = bits
            var vv = UInt32(v)
            var pos = i * bits
            while remaining > 0 {
                let byteIdx = pos / 8
                let bitInByte = pos % 8
                let bitsAvail = 8 - bitInByte
                let take = min(remaining, bitsAvail)
                let mask: UInt32 = (1 << take) - 1
                let chunk = vv & mask
                out[byteIdx] |= UInt8(chunk << bitInByte)
                vv >>= take
                pos += take
                remaining -= take
            }
        }
        return out
    }

    /// Inverse of `packBits`: recovers `count` unsigned `bits`-wide values
    /// from the continuous bitstream in `packed`.
    static func unpackBits(_ packed: [UInt8], count n: Int, bits: Int) -> [Int] {
        precondition(bits > 0 && bits <= 8, "unpackBits only supports 1...8 bits, got \(bits)")
        let totalBits = n * bits
        precondition(totalBits % 8 == 0,
                     "packed row of \(n) x \(bits)-bit values is not byte-aligned")
        precondition(packed.count == totalBits / 8,
                     "packed.count \(packed.count) != expected \(totalBits / 8)")
        var out = [Int](repeating: 0, count: n)
        for i in 0..<n {
            var remaining = bits
            var pos = i * bits
            var shift = 0
            var value: UInt32 = 0
            while remaining > 0 {
                let byteIdx = pos / 8
                let bitInByte = pos % 8
                let bitsAvail = 8 - bitInByte
                let take = min(remaining, bitsAvail)
                let mask: UInt32 = (1 << take) - 1
                let chunk = (UInt32(packed[byteIdx]) >> bitInByte) & mask
                value |= chunk << shift
                shift += take
                pos += take
                remaining -= take
            }
            out[i] = Int(value)
        }
        return out
    }

    // MARK: - Shared affine quantize/dequantize (group min/max, BF16 scale+bias)

    private static func quantizeAffineCodes(
        _ row: [Float], groupSize: Int, bits: Int, maxQ: Int
    ) -> (codes: [Int], scales: [UInt16], biases: [UInt16]) {
        precondition(groupSize > 0 && row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of groupSize \(groupSize)")
        precondition((groupSize * bits) % 8 == 0,
                     "groupSize \(groupSize) does not byte-align \(bits)-bit packing")

        let nGroups = row.count / groupSize
        var codes = [Int](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / Float(maxQ)
                biasF  = wmin
            }
            let sBits = Quantization.bf16Bits(scaleF)
            let bBits = Quantization.bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = Quantization.bf16ToFloat(sBits)
            let bias  = Quantization.bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(maxQ, q))
                codes[g * groupSize + k] = q
            }
        }
        return (codes, scales, biases)
    }

    private static func dequantizeAffineCodes(
        codes: [Int], scales: [UInt16], biases: [UInt16], groupSize: Int
    ) -> [Float] {
        let n = codes.count
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = Quantization.bf16ToFloat(scales[g])
            let bias  = Quantization.bf16ToFloat(biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(codes[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT5 affine

    /// MLX `affine` 5-bit row. Packed unsigned 5-bit codes (continuous
    /// bitstream, see type doc), BF16 scale + bias per group of `groupSize`.
    public struct Int5AffineRow {
        public let packed: [UInt8]   // ceil(N * 5 / 8) bytes, always exact (N a multiple of groupSize)
        public let scales: [UInt16]  // N / groupSize BF16 bits
        public let biases: [UInt16]  // N / groupSize BF16 bits
        public let groupSize: Int

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16], groupSize: Int) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
            self.groupSize = groupSize
        }
    }

    /// Affine 5-bit quantize: `q ∈ [0..31]`, `w ≈ q * scale + bias`.
    /// Test-fixture / repacker helper — mirrors `Quantization.quantizeInt4Affine`
    /// but with sub-byte bit packing and a caller-supplied group size.
    public static func quantizeInt5Affine(_ row: [Float], groupSize: Int) -> Int5AffineRow {
        let (codes, scales, biases) = quantizeAffineCodes(row, groupSize: groupSize, bits: 5, maxQ: 31)
        let packed = packBits(codes, bits: 5)
        return Int5AffineRow(packed: packed, scales: scales, biases: biases, groupSize: groupSize)
    }

    public static func dequantizeInt5Affine(_ r: Int5AffineRow, n: Int) -> [Float] {
        precondition(n % r.groupSize == 0, "n \(n) is not a multiple of groupSize \(r.groupSize)")
        precondition(r.packed.count == (n * 5) / 8,
                     "packed.count \(r.packed.count) != expected \((n * 5) / 8)")
        let codes = unpackBits(r.packed, count: n, bits: 5)
        return dequantizeAffineCodes(codes: codes, scales: r.scales, biases: r.biases, groupSize: r.groupSize)
    }

    // MARK: - INT6 affine

    /// MLX `affine` 6-bit row. Packed unsigned 6-bit codes (continuous
    /// bitstream, see type doc), BF16 scale + bias per group of `groupSize`.
    public struct Int6AffineRow {
        public let packed: [UInt8]   // N * 6 / 8 bytes, always exact
        public let scales: [UInt16]  // N / groupSize BF16 bits
        public let biases: [UInt16]  // N / groupSize BF16 bits
        public let groupSize: Int

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16], groupSize: Int) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
            self.groupSize = groupSize
        }
    }

    /// Affine 6-bit quantize: `q ∈ [0..63]`, `w ≈ q * scale + bias`.
    public static func quantizeInt6Affine(_ row: [Float], groupSize: Int) -> Int6AffineRow {
        let (codes, scales, biases) = quantizeAffineCodes(row, groupSize: groupSize, bits: 6, maxQ: 63)
        let packed = packBits(codes, bits: 6)
        return Int6AffineRow(packed: packed, scales: scales, biases: biases, groupSize: groupSize)
    }

    public static func dequantizeInt6Affine(_ r: Int6AffineRow, n: Int) -> [Float] {
        precondition(n % r.groupSize == 0, "n \(n) is not a multiple of groupSize \(r.groupSize)")
        precondition(r.packed.count == (n * 6) / 8,
                     "packed.count \(r.packed.count) != expected \((n * 6) / 8)")
        let codes = unpackBits(r.packed, count: n, bits: 6)
        return dequantizeAffineCodes(codes: codes, scales: r.scales, biases: r.biases, groupSize: r.groupSize)
    }
}
