import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// CPU reference implementation for attention gating.
enum AttentionGatingRef {
    /// Computes CPU reference for per-head attention gating:
    /// `attnOut[t, h, d] *= softplus(gOut[t, h])` where `softplus(g) = (g > 20.0) ? g : log1p(exp(g))`.
    static func applyPerHead(
        attnOut: [Float],
        gOut: [Float],
        tokens: Int,
        numHeads: Int,
        headDim: Int
    ) -> [Float] {
        var result = attnOut
        for t in 0..<tokens {
            for h in 0..<numHeads {
                let g = gOut[t * numHeads + h]
                let softplusG = (g > 20.0) ? g : log1p(exp(g))
                for d in 0..<headDim {
                    let idx = (t * numHeads + h) * headDim + d
                    result[idx] = attnOut[idx] * softplusG
                }
            }
        }
        return result
    }
}

/// Unit tests comparing the Metal `apply_attention_gating_per_head` kernel against CPU reference.
@Suite struct AttentionGatingTests {

    private static func runAndComparePerHead(
        tokens: Int,
        numHeads: Int,
        headDim: Int,
        seed: UInt64
    ) throws {
        let attnCount = tokens * numHeads * headDim
        let gCount = tokens * numHeads

        var rng = SeedTree(seed).key("attn-gating-per-head-\(tokens)-\(numHeads)-\(headDim)")
        let attnFp32 = (0..<attnCount).map { _ in rng.uniform(-1.0, 1.0) }
        let gFp32 = (0..<gCount).map { _ in rng.uniform(-5.0, 25.0) }

        let attnFp16 = attnFp32.map { Float16($0) }
        let gFp16 = gFp32.map { Float16($0) }

        let attnRefInput = attnFp16.map { Float($0) }
        let gRefInput = gFp16.map { Float($0) }

        let refOutput = AttentionGatingRef.applyPerHead(
            attnOut: attnRefInput,
            gOut: gRefInput,
            tokens: tokens,
            numHeads: numHeads,
            headDim: headDim
        )

        let ctx = try MetalContext()
        let kernel = try AttentionGatingKernel(context: ctx)

        guard let attnBuf = Fp16Buffer.make(ctx.device, halves: attnFp16),
              let gBuf = Fp16Buffer.make(ctx.device, halves: gFp16) else {
            Issue.record("alloc failed")
            return
        }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodePerHead(
            commandBuffer: cb,
            attnOut: attnBuf,
            gOut: gBuf,
            tokens: UInt32(tokens),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim)
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(attnBuf, count: attnCount)
        let relErr = RelError.compute(actual: actual, reference: refOutput)
        let maxAbs = RelError.maxAbsDiff(actual, refOutput)
        #expect(relErr < Tolerance.fp16Reduction, "relErr=\(relErr) maxAbs=\(maxAbs)")
    }

    @Test func perHeadGating_singleTokenDecode_matchesCpuReference() throws {
        try Self.runAndComparePerHead(tokens: 1, numHeads: 16, headDim: 256, seed: 0x41)
    }

    @Test func perHeadGating_multiTokenPrefill_matchesCpuReference() throws {
        try Self.runAndComparePerHead(tokens: 8, numHeads: 8, headDim: 128, seed: 0x42)
    }

    @Test func perHeadGating_largeGateValues_branchesCorrectly() throws {
        let tokens = 2, numHeads = 4, headDim = 64
        let attnCount = tokens * numHeads * headDim

        let attnFp16 = (0..<attnCount).map { Float16(Float($0 % 10 - 5) * 0.1) }
        let gFp16: [Float16] = [21.0, 50.0, 100.0, -10.0, 0.0, 5.0, 19.9, 20.1]

        let refOutput = AttentionGatingRef.applyPerHead(
            attnOut: attnFp16.map { Float($0) },
            gOut: gFp16.map { Float($0) },
            tokens: tokens,
            numHeads: numHeads,
            headDim: headDim
        )

        let ctx = try MetalContext()
        let kernel = try AttentionGatingKernel(context: ctx)

        guard let attnBuf = Fp16Buffer.make(ctx.device, halves: attnFp16),
              let gBuf = Fp16Buffer.make(ctx.device, halves: gFp16) else {
            Issue.record("alloc failed")
            return
        }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodePerHead(
            commandBuffer: cb,
            attnOut: attnBuf,
            gOut: gBuf,
            tokens: UInt32(tokens),
            numHeads: UInt32(numHeads),
            headDim: UInt32(headDim)
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(attnBuf, count: attnCount)
        let relErr = RelError.compute(actual: actual, reference: refOutput)
        #expect(relErr < Tolerance.fp16Reduction, "large gate values relErr=\(relErr)")
    }
}
