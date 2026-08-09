import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct YaRoPTests {
    private static let testScaling = RopeScaling(
        factor: 4.0,
        originalMaxPositionEmbeddings: 4096,
        betaFast: 1.0,
        betaSlow: 32.0,
        attentionFactor: 1.0
    )

    @Test func defaultNeoxMatchesYarnCPUReference() throws {
        let tokens = 1
        let heads = 16
        let headDim = 256
        let position = 8192
        let theta: Float = 10_000
        let count = tokens * heads * headDim
        let input = Self.randomInputs(count: count, seed: 0x9A, label: "yarop-neox-default")
        let context = try MetalContext()
        let kernel = try RoPE(context: context)
        let buffer = try #require(Fp16Buffer.make(context.device, values: input))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeDefaultNeox(
            commandBuffer: commandBuffer,
            data: buffer,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(heads),
            numTokens: UInt32(tokens),
            theta: theta,
            scaling: Self.testScaling
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = Fp16Buffer.read(buffer, count: count)
        let reference = RopeRef.applyNeox(
            input: input,
            numTokens: tokens,
            numHeads: heads,
            headDim: headDim,
            rotatedPairs: headDim / 2,
            position: position,
            theta: theta,
            scaling: Self.testScaling
        )
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction, "YaRoP default Neox rel=\(error)")
    }

    @Test func proportionalNeoxMatchesYarnCPUReference() throws {
        let tokens = 1
        let heads = 16
        let headDim = 512
        let rotatedPairs = 64
        let position = 16384
        let theta: Float = 1_000_000
        let count = tokens * heads * headDim
        let input = Self.randomInputs(count: count, seed: 0x9B, label: "yarop-neox-proportional")
        let context = try MetalContext()
        let kernel = try RoPE(context: context)
        let buffer = try #require(Fp16Buffer.make(context.device, values: input))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeProportionalNeox(
            commandBuffer: commandBuffer,
            data: buffer,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(heads),
            rotatedPairs: UInt32(rotatedPairs),
            numTokens: UInt32(tokens),
            theta: theta,
            scaling: Self.testScaling
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = Fp16Buffer.read(buffer, count: count)
        let reference = RopeRef.applyNeox(
            input: input,
            numTokens: tokens,
            numHeads: heads,
            headDim: headDim,
            rotatedPairs: rotatedPairs,
            position: position,
            theta: theta,
            scaling: Self.testScaling
        )
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction, "YaRoP proportional Neox rel=\(error)")
    }

    @Test func prefillRoPEMatchesYarnCPUReference() throws {
        let tokens = 4
        let heads = 8
        let headDim = 256
        let rotatedPairs = 64
        let startPosition = 4096
        let theta: Float = 1_000_000
        let count = tokens * heads * headDim
        let input = Self.randomInputs(count: count, seed: 0x9C, label: "yarop-prefill")
        let context = try MetalContext()
        let kernel = try PrefillRoPE(context: context)
        let buffer = try #require(Fp16Buffer.make(context.device, values: input))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeProportionalNeox(
            commandBuffer: commandBuffer,
            data: buffer,
            startPosition: UInt32(startPosition),
            queryCount: UInt32(tokens),
            headDim: UInt32(headDim),
            numHeads: UInt32(heads),
            rotatedPairs: UInt32(rotatedPairs),
            tokenStrideElements: UInt32(heads * headDim),
            theta: theta,
            scaling: Self.testScaling
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = Fp16Buffer.read(buffer, count: count)
        var reference = [Float]()
        for t in 0..<tokens {
            let slice = Array(input[(t * heads * headDim)...((t + 1) * heads * headDim - 1)])
            let refSlice = RopeRef.applyNeox(
                input: slice,
                numTokens: 1,
                numHeads: heads,
                headDim: headDim,
                rotatedPairs: rotatedPairs,
                position: startPosition + t,
                theta: theta,
                scaling: Self.testScaling
            )
            reference.append(contentsOf: refSlice)
        }
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction, "YaRoP prefill RoPE rel=\(error)")
    }

    private static func randomInputs(count: Int, seed: UInt64, label: String) -> [Float] {
        var random = SeedTree(seed).key(label)
        return (0..<count).map { _ in random.uniform(-1, 1) }
    }
}
