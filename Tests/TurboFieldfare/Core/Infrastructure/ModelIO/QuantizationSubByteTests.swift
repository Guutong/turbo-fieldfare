import Testing
import Foundation
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QuantizationSubByteTests {

    // MARK: - Bit-packing layout

    /// Independent reimplementation of the packing convention, built by
    /// materializing every bit into a flat array first (rather than
    /// streaming through a bit-writer like the production `packBits`), then
    /// coalescing into bytes. A different code path for the same spec, so a
    /// match against `QuantizationSubByte.packBits` is a real cross-check of
    /// the "byte-addressed, LSB-first, little-endian continuous bitstream"
    /// convention rather than a tautology.
    private static func referencePack(_ values: [Int], bits: Int) -> [UInt8] {
        let totalBits = values.count * bits
        precondition(totalBits % 8 == 0)
        var bitArray = [UInt8](repeating: 0, count: totalBits)
        for (i, v) in values.enumerated() {
            for p in 0..<bits {
                bitArray[i * bits + p] = UInt8((v >> p) & 1)
            }
        }
        var bytes = [UInt8](repeating: 0, count: totalBits / 8)
        for bitIdx in 0..<totalBits where bitArray[bitIdx] == 1 {
            bytes[bitIdx / 8] |= (1 << (bitIdx % 8))
        }
        return bytes
    }

    @Test("packBits matches an independently-derived bit layout", arguments: [5, 6])
    func packBitsMatchesIndependentReference(bits: Int) {
        // groupSize=64 -> byte-aligned for both 5-bit (40 bytes) and 6-bit (48 bytes).
        let n = 64
        var rng = SeedTree(0x5601).key("subbyte-pack-bits\(bits)")
        let maxVal = (1 << bits) - 1
        let values = (0..<n).map { _ in Int(rng.uniform(0, Float(maxVal + 1))) }.map { min($0, maxVal) }

        let produced = QuantizationSubByte.packBits(values, bits: bits)
        let expected = Self.referencePack(values, bits: bits)
        #expect(produced == expected, "bits=\(bits) packed bytes diverge from independent reference")
    }

    /// Hand-computed spot check for 5-bit packing: values 1..8 (5-bit each).
    /// byte0 = low 5 bits of v0=1 (00001) plus low 3 bits of v1=2 (010) in
    /// bits 5..7 => 0b010_00001 = 65. Verifies the "no padding, LSB-first,
    /// straddles into next byte" convention against arithmetic worked out by
    /// hand, independent of any Swift implementation.
    @Test func packBits_fiveBit_handComputedFirstByte() {
        let values = Array(1...8)
        let packed = QuantizationSubByte.packBits(values, bits: 5)
        #expect(packed[0] == 65, "byte0 = \(packed[0]), expected 65 (0b01000001)")
    }

    /// Hand-computed spot check for 6-bit packing: values 1,2,3,4 (6-bit
    /// each; 4 values so the row is byte-aligned: 4*6=24 bits = 3 bytes).
    /// byte0 = low 6 bits of v0=1 (000001) plus low 2 bits of v1=2 (10) in
    /// bits 6..7 => 0b10_000001 = 0x81 = 129.
    @Test func packBits_sixBit_handComputedFirstByte() {
        let values = [1, 2, 3, 4]
        let packed = QuantizationSubByte.packBits(values, bits: 6)
        #expect(packed[0] == 129, "byte0 = \(packed[0]), expected 129 (0b10000001)")
    }

    // MARK: - Real MLX ground truth
    //
    // Fixtures generated with `mlx.core.quantize(w, group_size=64, bits=5|6)`
    // on Homebrew's python3.11 (mlx 0.31.2 — importable there even though the
    // default `python3` on this machine is 3.14 and lacks it). `codes` were
    // recovered from MLX's own output in a packing-agnostic way: MLX's affine
    // dequant is exact given the code, so
    // `code = round((dequant(wq) - bias) / scale)` recovers the integer
    // codes without assuming anything about `wq`'s bit layout. `expectedBytes`
    // is `np.array(wq).view(np.uint8).tobytes()` — MLX's actual packed
    // buffer. A match here means `packBits` reproduces MLX's real packed
    // output byte-for-byte, not just an internally-consistent guess.
    @Test func packBits_fiveBit_matchesRealMLXGroundTruth() {
        let codes = [25, 14, 28, 22, 3, 31, 24, 25, 4, 14, 12, 30, 21, 26, 14, 7, 18, 2, 27, 20,
                     24, 11, 31, 29, 25, 6, 15, 1, 5, 22, 24, 31, 10, 12, 15, 6, 4, 15, 7, 22,
                     14, 27, 23, 10, 27, 26, 12, 9, 22, 4, 6, 0, 25, 21, 23, 25, 15, 18, 4, 3,
                     21, 15, 18, 25]
        let expectedBytes: [UInt8] = [217, 113, 59, 62, 206, 196, 49, 95, 181, 59, 82, 108, 138, 215, 239, 217,
                                       188, 80, 44, 254, 138, 61, 67, 222, 177, 110, 95, 181, 53, 75, 150, 24,
                                       144, 235, 205, 79, 146, 81, 159, 204]
        let packed = QuantizationSubByte.packBits(codes, bits: 5)
        #expect(packed == expectedBytes, "5-bit packBits diverges from real MLX quantize() output")
        let unpacked = QuantizationSubByte.unpackBits(expectedBytes, count: 64, bits: 5)
        #expect(unpacked == codes, "5-bit unpackBits diverges from real MLX quantize() codes")
    }

    @Test func packBits_sixBit_matchesRealMLXGroundTruth() {
        let codes = [50, 28, 55, 45, 6, 63, 49, 51, 8, 29, 24, 60, 41, 53, 28, 14, 36, 4, 53, 41,
                     49, 23, 63, 58, 50, 12, 30, 2, 10, 44, 48, 62, 21, 24, 30, 12, 8, 30, 14, 43,
                     28, 54, 45, 20, 54, 52, 25, 18, 44, 9, 13, 0, 51, 43, 45, 50, 29, 36, 9, 7,
                     43, 30, 36, 49]
        let expectedBytes: [UInt8] = [50, 119, 183, 198, 31, 207, 72, 135, 241, 105, 205, 57, 36, 81, 167, 241,
                                       245, 235, 50, 227, 9, 10, 11, 251, 21, 230, 49, 136, 231, 172, 156, 221,
                                       82, 54, 157, 73, 108, 210, 0, 243, 218, 202, 29, 153, 28, 171, 71, 198]
        let packed = QuantizationSubByte.packBits(codes, bits: 6)
        #expect(packed == expectedBytes, "6-bit packBits diverges from real MLX quantize() output")
        let unpacked = QuantizationSubByte.unpackBits(expectedBytes, count: 64, bits: 6)
        #expect(unpacked == codes, "6-bit unpackBits diverges from real MLX quantize() codes")
    }

    @Test("pack/unpack round-trips exactly", arguments: [5, 6])
    func packUnpackRoundtrip(bits: Int) {
        let n = 128
        var rng = SeedTree(0x5602).key("subbyte-roundtrip-bits\(bits)")
        let maxVal = (1 << bits) - 1
        let values = (0..<n).map { _ in min(Int(rng.uniform(0, Float(maxVal + 1))), maxVal) }

        let packed = QuantizationSubByte.packBits(values, bits: bits)
        let recovered = QuantizationSubByte.unpackBits(packed, count: n, bits: bits)
        #expect(recovered == values, "bits=\(bits) unpack(pack(x)) != x")
    }

    // MARK: - Affine quantize/dequantize round-trip

    @Test("INT5 affine round-trip stays within derived bound", arguments: [64, 128])
    func quantizeDequantizeInt5Affine_roundtrip(groupSize: Int) {
        var rng = SeedTree(0x5603).key("int5-roundtrip-g\(groupSize)")
        let n = groupSize * 3
        let row = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let q = QuantizationSubByte.quantizeInt5Affine(row, groupSize: groupSize)
        #expect(q.packed.count == (n * 5) / 8)
        #expect(q.scales.count == n / groupSize)
        #expect(q.biases.count == n / groupSize)

        let r = QuantizationSubByte.dequantizeInt5Affine(q, n: n)
        let groups = n / groupSize
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(q.scales[g])
            // 5-bit affine: rounding <= scale/2, plus BF16 scale rounding
            // (rel <= 2^-7) applied against the group's dynamic range.
            let bound = scale + 1e-4
            for k in 0..<groupSize {
                let i = g * groupSize + k
                #expect(abs(r[i] - row[i]) <= bound,
                        "group=\(g) k=\(k) diff=\(abs(r[i] - row[i])) bound=\(bound)")
            }
        }
    }

    @Test("INT6 affine round-trip stays within derived bound", arguments: [64, 128])
    func quantizeDequantizeInt6Affine_roundtrip(groupSize: Int) {
        var rng = SeedTree(0x5604).key("int6-roundtrip-g\(groupSize)")
        let n = groupSize * 3
        let row = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let q = QuantizationSubByte.quantizeInt6Affine(row, groupSize: groupSize)
        #expect(q.packed.count == (n * 6) / 8)
        #expect(q.scales.count == n / groupSize)
        #expect(q.biases.count == n / groupSize)

        let r = QuantizationSubByte.dequantizeInt6Affine(q, n: n)
        let groups = n / groupSize
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(q.scales[g])
            let bound = scale + 1e-4
            for k in 0..<groupSize {
                let i = g * groupSize + k
                #expect(abs(r[i] - row[i]) <= bound,
                        "group=\(g) k=\(k) diff=\(abs(r[i] - row[i])) bound=\(bound)")
            }
        }
    }

    /// 5-bit should resolve finer than 4-bit on the same input range: fewer
    /// quantization levels wasted, tighter round-trip error.
    @Test func int5Affine_tighterThanInt4OnSameRange() {
        var rng = SeedTree(0x5605).key("int5-vs-int4")
        let row = (0..<64).map { _ in rng.uniform(-1.0, 1.0) }

        let q4 = Quantization.quantizeInt4Affine(row)
        let r4 = Quantization.dequantizeInt4Affine(q4, n: 64)
        let err4 = RelError.maxAbsDiff(r4, row)

        let q5 = QuantizationSubByte.quantizeInt5Affine(row, groupSize: 64)
        let r5 = QuantizationSubByte.dequantizeInt5Affine(q5, n: 64)
        let err5 = RelError.maxAbsDiff(r5, row)

        #expect(err5 < err4, "5-bit err=\(err5) should be < 4-bit err=\(err4)")
    }

    /// 6-bit should resolve finer than 5-bit on the same input range.
    @Test func int6Affine_tighterThanInt5OnSameRange() {
        var rng = SeedTree(0x5606).key("int6-vs-int5")
        let row = (0..<64).map { _ in rng.uniform(-1.0, 1.0) }

        let q5 = QuantizationSubByte.quantizeInt5Affine(row, groupSize: 64)
        let r5 = QuantizationSubByte.dequantizeInt5Affine(q5, n: 64)
        let err5 = RelError.maxAbsDiff(r5, row)

        let q6 = QuantizationSubByte.quantizeInt6Affine(row, groupSize: 64)
        let r6 = QuantizationSubByte.dequantizeInt6Affine(q6, n: 64)
        let err6 = RelError.maxAbsDiff(r6, row)

        #expect(err6 < err5, "6-bit err=\(err6) should be < 5-bit err=\(err5)")
    }

    /// Constant-group sanity: a flat group reproduces exactly (mirrors the
    /// existing INT4 test in QuantizationAffineTests.swift).
    @Test func quantizeInt5Affine_constantGroupRoundtripsExact() {
        let row = [Float](repeating: 0.42, count: 64)
        let q = QuantizationSubByte.quantizeInt5Affine(row, groupSize: 64)
        let r = QuantizationSubByte.dequantizeInt5Affine(q, n: 64)
        let bf16Rounded = Quantization.bf16ToFloat(Quantization.bf16Bits(0.42))
        for i in 0..<64 {
            #expect(abs(r[i] - bf16Rounded) < 1e-6, "i=\(i) got=\(r[i]) ref=\(bf16Rounded)")
        }
    }

    @Test func quantizeInt6Affine_constantGroupRoundtripsExact() {
        let row = [Float](repeating: -0.17, count: 64)
        let q = QuantizationSubByte.quantizeInt6Affine(row, groupSize: 64)
        let r = QuantizationSubByte.dequantizeInt6Affine(q, n: 64)
        let bf16Rounded = Quantization.bf16ToFloat(Quantization.bf16Bits(-0.17))
        for i in 0..<64 {
            #expect(abs(r[i] - bf16Rounded) < 1e-6, "i=\(i) got=\(r[i]) ref=\(bf16Rounded)")
        }
    }

    @Test func quantizeInt5Affine_positiveOnlyUsesFullCodebook() {
        let row = (0..<64).map { Float($0) / 63.0 + 1.0 } // 1.0 .. 2.0
        let q = QuantizationSubByte.quantizeInt5Affine(row, groupSize: 64)
        let codes = QuantizationSubByte.unpackBits(q.packed, count: 64, bits: 5)
        #expect(codes.contains(0),  "affine 5-bit on positive-only range should hit q=0")
        #expect(codes.contains(31), "affine 5-bit on positive-only range should hit q=31")
    }

    @Test func quantizeInt6Affine_positiveOnlyUsesFullCodebook() {
        let row = (0..<64).map { Float($0) / 63.0 + 1.0 } // 1.0 .. 2.0
        let q = QuantizationSubByte.quantizeInt6Affine(row, groupSize: 64)
        let codes = QuantizationSubByte.unpackBits(q.packed, count: 64, bits: 6)
        #expect(codes.contains(0),  "affine 6-bit on positive-only range should hit q=0")
        #expect(codes.contains(63), "affine 6-bit on positive-only range should hit q=63")
    }
}
