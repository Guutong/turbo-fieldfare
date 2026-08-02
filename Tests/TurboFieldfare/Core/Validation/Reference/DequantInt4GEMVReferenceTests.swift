import Testing
import Foundation
import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates `DequantInt4GemvRef.apply` against a scalar reference that
/// dequantizes inline (per-element nibble unpack + scalar accumulate).
/// vDSP path and scalar path must agree to `Tolerance.identity`.
///
/// Also pins the affine quantization round-trip bound: per-group worst-case
/// rounding is `scale/2` (codebook step) plus the BF16 rounding of the scale
/// and bias.
@Suite struct DequantInt4GEMVReferenceTests {

    /// Scalar reference: dequantize inline, scalar accumulate. No vDSP.
    private static func scalarRef(
        weightRows: [Quantization.Int4AffineRow],
        x: [Float],
        n: Int,
        groupSize: Int = Quantization.groupSize
    ) -> [Float] {
        let m = weightRows.count
        var y = [Float](repeating: 0, count: m)
        for row in 0..<m {
            let w = Quantization.dequantizeInt4Affine(weightRows[row], n: n, groupSize: groupSize)
            var sum: Float = 0
            for i in 0..<n { sum += w[i] * x[i] }
            y[row] = sum
        }
        return y
    }

    @Test("vDSP ref matches scalar ref", arguments: [
        (4,   128, UInt64(0xF1)),
        (8,   256, UInt64(0xF2)),
        (16,  512, UInt64(0xF3)),
        (32, 2816, UInt64(0xF4)),
    ])
    func vDSPMatchesScalar(m: Int, n: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("int4-gemv-ref-m\(m)-n\(n)")
        var rows: [Quantization.Int4AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            rows.append(Quantization.quantizeInt4Affine(raw))
        }
        let x = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let vdsp = DequantInt4GemvRef.apply(weightRows: rows, x: x, n: n)
        let scalar = Self.scalarRef(weightRows: rows, x: x, n: n)

        let rel = RelError.compute(actual: vdsp, reference: scalar)
        #expect(rel < Tolerance.identity, "M=\(m) N=\(n) rel=\(rel)")
    }

    /// Same vDSP-vs-scalar cross-check at groupSize 128 (oQ4e's routed
    /// experts), so the CPU reference itself is proven correct for group-128
    /// before the GPU kernel is compared against it in
    /// `DequantInt4GEMVTests.gemv_group128_*`.
    @Test("vDSP ref matches scalar ref at groupSize 128", arguments: [
        (4,   128, UInt64(0xF6)),
        (8,   256, UInt64(0xF7)),
        (16,  512, UInt64(0xF8)),
        (32, 2816, UInt64(0xF9)),
    ])
    func vDSPMatchesScalarGroup128(m: Int, n: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("int4-gemv-ref-g128-m\(m)-n\(n)")
        var rows: [Quantization.Int4AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            rows.append(Quantization.quantizeInt4Affine(raw, groupSize: 128))
        }
        let x = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let vdsp = DequantInt4GemvRef.apply(weightRows: rows, x: x, n: n, groupSize: 128)
        let scalar = Self.scalarRef(weightRows: rows, x: x, n: n, groupSize: 128)

        let rel = RelError.compute(actual: vdsp, reference: scalar)
        #expect(rel < Tolerance.identity, "M=\(m) N=\(n) rel=\(rel)")
    }

    /// INT4 affine round-trip: error bounded by one codebook step `scale`.
    /// Rounding into `q ∈ [0..15]` contributes scale/2 and BF16 rounding of
    /// scale/bias contributes at most ulp(BF16(scale)) * 15 — both absorbed
    /// by `scale + small`.
    @Test("INT4 affine round-trip stays within derived bound", arguments: [128, 256, 2816])
    func quantizationDerivedBound(n: Int) {
        var rng = SeedTree(0xF5).key("int4-affine-roundtrip-n\(n)")
        let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let q = Quantization.quantizeInt4Affine(raw)
        let recovered = Quantization.dequantizeInt4Affine(q, n: n)

        let groups = n / Quantization.groupSize
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(q.scales[g])
            let bound = scale + 1e-4
            for k in 0..<Quantization.groupSize {
                let i = g * Quantization.groupSize + k
                #expect(abs(recovered[i] - raw[i]) <= bound,
                        "group=\(g) k=\(k) diff=\(abs(recovered[i] - raw[i])) bound=\(bound)")
            }
        }
    }

    /// Same round-trip bound at groupSize 128: the bound derivation is
    /// per-group and doesn't depend on how many elements are in a group, but
    /// this pins that `quantizeInt4Affine`/`dequantizeInt4Affine` actually
    /// honor a non-default `groupSize` argument end to end (wrong group
    /// slicing would still average min/max over *some* window and could
    /// still look numerically plausible).
    @Test("INT4 affine round-trip stays within derived bound at groupSize 128",
          arguments: [128, 256, 2816])
    func quantizationDerivedBoundGroup128(n: Int) {
        var rng = SeedTree(0xF5).key("int4-affine-roundtrip-g128-n\(n)")
        let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let q = Quantization.quantizeInt4Affine(raw, groupSize: 128)
        let recovered = Quantization.dequantizeInt4Affine(q, n: n, groupSize: 128)

        let groups = n / 128
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(q.scales[g])
            let bound = scale + 1e-4
            for k in 0..<128 {
                let i = g * 128 + k
                #expect(abs(recovered[i] - raw[i]) <= bound,
                        "group=\(g) k=\(k) diff=\(abs(recovered[i] - raw[i])) bound=\(bound)")
            }
        }
    }
}
