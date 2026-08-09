import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Covers `dequant_int5_qkv_gemv_simd`, the fused Q/K/V projection at 5-bit.
///
/// The risk this kernel carries is not the arithmetic — `dequant_subbyte_gemv_body`
/// is already covered by `DequantInt5GEMVTests` — it is the **row partitioning**.
/// One dispatch spans Q's rows then K's then V's, and an off-by-one in the
/// boundaries, or a swapped weight/output pairing, still produces plausible
/// numbers of the right magnitude. So every projection here gets independently
/// drawn weights, and each output is compared against its own reference: any
/// crossing of the streams shows up as a large error rather than a subtle one.
@Suite struct FusedQKVGEMVGenericTests {

    /// Quantized rows plus their flattened GPU-side buffers, for whichever
    /// width is under test. `dequantized` is captured up front so the CPU
    /// reference never has to re-derive it per bit width.
    private struct Projection {
        var dequantized: [[Float]]
        var packed: [UInt8]
        var scales: [UInt16]
        var biases: [UInt16]
    }

    private static func makeProjection(
        m: Int, n: Int, groupSize: Int, bits: FusedQKVGEMVGeneric.Bits,
        rng: inout SplitMix64
    ) -> Projection {
        let groupsPerRow = n / groupSize
        var dequantized: [[Float]] = []
        var perRowPacked: [[UInt8]] = []
        var perRowScales: [[UInt16]] = []
        var perRowBiases: [[UInt16]] = []
        dequantized.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            switch bits {
            case .five:
                let r = QuantizationSubByte.quantizeInt5Affine(raw, groupSize: groupSize)
                dequantized.append(QuantizationSubByte.dequantizeInt5Affine(r, n: n))
                perRowPacked.append(r.packed)
                perRowScales.append(r.scales)
                perRowBiases.append(r.biases)
            case .eight:
                let r = QuantizationSubByte.quantizeInt8Affine(raw, groupSize: groupSize)
                dequantized.append(QuantizationSubByte.dequantizeInt8Affine(r, n: n))
                perRowPacked.append(r.packed)
                perRowScales.append(r.scales)
                perRowBiases.append(r.biases)
            }
        }
        let rowBytes = (n * bits.rawValue) / 8
        var packed = [UInt8](repeating: 0, count: m * rowBytes)
        var scales = [UInt16](repeating: 0, count: m * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: m * groupsPerRow)
        for row in 0..<m {
            for i in 0..<rowBytes { packed[row * rowBytes + i] = perRowPacked[row][i] }
            for g in 0..<groupsPerRow {
                scales[row * groupsPerRow + g] = perRowScales[row][g]
                biases[row * groupsPerRow + g] = perRowBiases[row][g]
            }
        }
        return Projection(dequantized: dequantized, packed: packed,
                          scales: scales, biases: biases)
    }

    private static func scalarRef(_ p: Projection, x: [Float], n: Int) -> [Float] {
        p.dequantized.map { w in
            var sum: Float = 0
            for i in 0..<n { sum += w[i] * x[i] }
            return sum
        }
    }

    /// - Returns: relative error of Q, K and V against their own references.
    private static func runAndCompare(
        qRows: Int, kvRows: Int, n: Int, groupSize: Int,
        bits: FusedQKVGEMVGeneric.Bits = .five, seed: UInt64
    ) throws -> (q: Float, k: Float, v: Float) {
        precondition(n % groupSize == 0)
        var rng = SeedTree(seed)
            .key("qkv\(bits.rawValue)-q\(qRows)-kv\(kvRows)-n\(n)-g\(groupSize)")

        let qp = makeProjection(m: qRows, n: n, groupSize: groupSize, bits: bits, rng: &rng)
        let kp = makeProjection(m: kvRows, n: n, groupSize: groupSize, bits: bits, rng: &rng)
        let vp = makeProjection(m: kvRows, n: n, groupSize: groupSize, bits: bits, rng: &rng)

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try FusedQKVGEMVGeneric(context: ctx, bits: bits)

        func buf(_ bytes: [UInt8]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: bytes, length: bytes.count,
                                  options: .storageModeShared)
        }
        func buf(_ halves: [UInt16]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: halves,
                                  length: halves.count * MemoryLayout<UInt16>.size,
                                  options: .storageModeShared)
        }
        guard let qW = buf(qp.packed), let qS = buf(qp.scales), let qB = buf(qp.biases),
              let kW = buf(kp.packed), let kS = buf(kp.scales), let kB = buf(kp.biases),
              let vW = buf(vp.packed), let vS = buf(vp.scales), let vB = buf(vp.biases),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let qY = Fp16Buffer.make(ctx.device, count: qRows),
              let kY = Fp16Buffer.make(ctx.device, count: kvRows),
              let vY = Fp16Buffer.make(ctx.device, count: kvRows) else {
            Issue.record("Failed to allocate buffers")
            return (.infinity, .infinity, .infinity)
        }
        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer")
            return (.infinity, .infinity, .infinity)
        }
        kernel.encode(commandBuffer: cmd,
                      qWeights: qW, qScales: qS, qBiases: qB,
                      kWeights: kW, kScales: kS, kBiases: kB,
                      vWeights: vW, vScales: vS, vBiases: vB,
                      x: xBuf, qOut: qY, kOut: kY, vOut: vY,
                      qRows: UInt32(qRows), kvRows: UInt32(kvRows),
                      n: UInt32(n), groupSize: UInt32(groupSize))
        cmd.commit()
        cmd.waitUntilCompleted()

        return (RelError.compute(actual: Fp16Buffer.read(qY, count: qRows),
                                 reference: scalarRef(qp, x: xRef, n: n)),
                RelError.compute(actual: Fp16Buffer.read(kY, count: kvRows),
                                 reference: scalarRef(kp, x: xRef, n: n)),
                RelError.compute(actual: Fp16Buffer.read(vY, count: kvRows),
                                 reference: scalarRef(vp, x: xRef, n: n)))
    }

    /// Laguna's 5-bit attention shape: hidden 3072, 48 query heads and 8 KV
    /// heads at head_dim 128, quantized at group 64.
    @Test func lagunaFullAttentionShape() throws {
        let rel = try Self.runAndCompare(qRows: 48 * 128, kvRows: 8 * 128,
                                         n: 3072, groupSize: 64, seed: 0x5A1)
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    /// The sliding-attention layers carry 72 query heads, so the Q/KV row split
    /// is not the same on every layer — which is exactly what the partitioning
    /// has to get right at runtime rather than by specialization.
    @Test func lagunaSlidingAttentionShape() throws {
        let rel = try Self.runAndCompare(qRows: 72 * 128, kvRows: 8 * 128,
                                         n: 3072, groupSize: 64, seed: 0x5A2)
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    /// Row counts that are not multiples of the 8-rows-per-threadgroup tiling,
    /// so the last threadgroup is partial and the `global_row >= total_rows`
    /// bound is actually exercised.
    @Test(arguments: [(qRows: 12, kvRows: 5), (qRows: 33, kvRows: 3), (qRows: 7, kvRows: 1)])
    func raggedRowCounts(shape: (qRows: Int, kvRows: Int)) throws {
        let rel = try Self.runAndCompare(qRows: shape.qRows, kvRows: shape.kvRows,
                                         n: 128, groupSize: 64,
                                         seed: UInt64(shape.qRows * 977 + shape.kvRows))
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    /// Group size is a runtime argument here, unlike the int4 fused kernel.
    @Test func groupSize128() throws {
        let rel = try Self.runAndCompare(qRows: 64, kvRows: 16,
                                         n: 1024, groupSize: 128, seed: 0x5A3)
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    // MARK: - 8-bit

    /// Laguna runs 8-bit attention on 28 of its 48 layers, so this shape
    /// carries more of the model than the 5-bit one does.
    @Test func lagunaEightBitAttentionShape() throws {
        let rel = try Self.runAndCompare(qRows: 48 * 128, kvRows: 8 * 128,
                                         n: 3072, groupSize: 64,
                                         bits: .eight, seed: 0x8A1)
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    /// The reason 8-bit rides on the sub-byte body rather than
    /// `dequant_int8_gemv_simd`: that kernel maps two elements per lane and is
    /// hardcoded to group 64, so it cannot do this at all. Laguna's 8-bit
    /// shared experts are group 128.
    @Test func eightBitAtGroup128() throws {
        let rel = try Self.runAndCompare(qRows: 96, kvRows: 32,
                                         n: 1024, groupSize: 128,
                                         bits: .eight, seed: 0x8A2)
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    @Test(arguments: [(qRows: 12, kvRows: 5), (qRows: 33, kvRows: 3)])
    func eightBitRaggedRowCounts(shape: (qRows: Int, kvRows: Int)) throws {
        let rel = try Self.runAndCompare(qRows: shape.qRows, kvRows: shape.kvRows,
                                         n: 128, groupSize: 64, bits: .eight,
                                         seed: UInt64(shape.qRows * 811 + shape.kvRows))
        #expect(rel.q < Tolerance.fp16Reduction, "q rel=\(rel.q)")
        #expect(rel.k < Tolerance.fp16Reduction, "k rel=\(rel.k)")
        #expect(rel.v < Tolerance.fp16Reduction, "v rel=\(rel.v)")
    }

    /// The group-generic 8-bit reference must agree byte-for-byte with the
    /// existing fixed-group-64 one, or the new kernel is being validated
    /// against a different format than the rest of the codebase writes.
    /// At 8 bits the bitstream packer degenerates to one byte per code, so
    /// this should hold exactly — no tolerance.
    @Test func int8ReferencesAgreeAtGroup64() {
        var rng = SeedTree(0x8A3).key("int8-ref-equivalence")
        for _ in 0..<8 {
            let raw = (0..<256).map { _ in rng.uniform(-2.0, 2.0) }
            let fixed = Quantization.quantizeInt8Affine(raw)
            let generic = QuantizationSubByte.quantizeInt8Affine(raw, groupSize: 64)
            #expect(fixed.packed == generic.packed)
            #expect(fixed.scales == generic.scales)
            #expect(fixed.biases == generic.biases)
            #expect(Quantization.dequantizeInt8Affine(fixed, n: 256)
                    == QuantizationSubByte.dequantizeInt8Affine(generic, n: 256))
        }
    }

    /// Q and the two KV projections must stay distinguishable: this fails loudly
    /// if the kernel ever routes a row to the wrong weight matrix or output,
    /// which the per-projection references above are designed to catch.
    @Test func projectionsAreIndependentlyDrawn() throws {
        var rng = SeedTree(0x5A4).key("independence")
        let a = Self.makeProjection(m: 8, n: 64, groupSize: 64, bits: .five, rng: &rng)
        let b = Self.makeProjection(m: 8, n: 64, groupSize: 64, bits: .five, rng: &rng)
        #expect(a.packed != b.packed, "fixtures must differ or the test proves nothing")
    }
}
