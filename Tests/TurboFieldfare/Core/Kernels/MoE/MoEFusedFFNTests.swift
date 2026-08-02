import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test func productionRoutedPipelineAndHitSplitMatchReference() throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<Self.topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<Self.topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let lowSlots = context.device.makeBuffer(
                bytes: [UInt32](0...3),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let highSlots = context.device.makeBuffer(
                bytes: [UInt32](4...7),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in [([UInt32](0...3), lowSlots),
                                     ([UInt32](4...7), highSlots)] {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    // MARK: - K != 8 coverage (Laguna-S-2.1 uses top-10; kMoEMaxTopK caps at 16)

    /// Confirms `moe_phase2_down_reduce`'s routed output matches a CPU reference
    /// for K != 8, exercising both the selection-loop shift-down fix (indirectly,
    /// via correct expert routing upstream) and the `topK * 32` threadgroup width
    /// dispatched by `encodeRoutedPersistentPhase2Reduce`. A width past the
    /// pipeline's `maxTotalThreadsPerThreadgroup` would either fail the command
    /// buffer or read garbage into `partial[]`; either shows up as a mismatch here.
    @Test("routed phase1+phase2 pipeline matches CPU reference for K != 8",
          arguments: [2, 10, 16])
    func routedPipelineMatchesReferenceForVariousK(_ topK: Int) throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe-k\(topK)")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let acts = Fp16Buffer.make(context.device, count: topK * Self.intermediate),
              let output = Fp16Buffer.make(context.device, count: Self.dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(topK)),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: acts,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: acts,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: output,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = Fp16Buffer.read(output, count: Self.dimension)
        #expect(RelError.compute(actual: actual, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    // MARK: - Group size 128 coverage (Laguna-S-2.1's routed experts use group 128)

    /// The `groupSize` parameter on the three routed encoders defaults to
    /// `Quantization.groupSize` (64) precisely so existing callers (including
    /// this file's K != 8 test above) keep compiling and keep dispatching the
    /// group-64 block-vectorized pipelines unchanged. This proves that promise:
    /// omitting `groupSize` and passing `groupSize: 64` explicitly must produce
    /// bit-identical output, because both must select the exact same pipeline
    /// objects and buffer layout.
    @Test func defaultGroupSizeMatchesExplicitGroupSize64() throws {
        var rng = SeedTree(0x2D3).key("group64-default-vs-explicit")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<Self.topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let defaultActs = Fp16Buffer.make(context.device, count: Self.topK * Self.intermediate),
              let explicitActs = Fp16Buffer.make(context.device, count: Self.topK * Self.intermediate),
              let defaultOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let explicitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        // Default groupSize (omitted).
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: defaultActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: defaultActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: defaultOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))

        // Explicit groupSize: 64.
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: explicitActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK),
            groupSize: UInt32(Quantization.groupSize))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: explicitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: explicitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK),
            groupSize: UInt32(Quantization.groupSize))

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let defaultResult = Fp16Buffer.read(defaultOutput, count: Self.dimension)
        let explicitResult = Fp16Buffer.read(explicitOutput, count: Self.dimension)
        #expect(defaultResult == explicitResult)
    }

    /// Confirms the generic group-128 pipelines
    /// (`moe_phase1_gate_up_act_u16load_generic`,
    /// `moe_phase1_gate_up_act_subset_u16load_generic`,
    /// `moe_phase2_down_reduce_generic`) match the CPU reference at group size
    /// 128, for both a group count that is a multiple of 4 (512 = 4*128 — the
    /// same shape the group-64 fast path block-vectorizes four groups at a
    /// time) and one that is not (384 = 3*128). The generic path has no block
    /// structure at all (strided-lane loop, see moe.metal), so this is really
    /// checking the strided loop's group-boundary handling rather than any
    /// block/remainder split — but the two shapes are exactly what the task
    /// asked to distinguish, and matching a shape the fast path treats
    /// specially is a reasonable sanity check in its own right.
    ///
    /// Also runs the phase-1 pipeline both as one `...U16Load` call over all
    /// experts and split across two `...SubsetU16Load` calls (low/high half),
    /// the same hit/miss split the runtime uses, to exercise the generic
    /// subset kernel too and prove the split doesn't change the answer.
    @Test("routed phase1+phase2 pipeline matches CPU reference at group size 128",
          arguments: [3, 4])
    func routedPipelineMatchesReferenceForGroupSize128(_ groupCount: Int) throws {
        let groupSize = 128
        let dimension = groupCount * groupSize
        let intermediate = groupCount * groupSize
        let topK = 4
        var rng = SeedTree(0x2D3).key("routed-moe-group128-\(groupCount)")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: intermediate, columns: dimension))
            ups.append(matrix(rows: intermediate, columns: dimension))
            downs.append(matrix(rows: dimension, columns: intermediate))
        }
        let x = (0..<dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0, groupSize: groupSize) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0, groupSize: groupSize) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0, groupSize: groupSize) }
            },
            indices: Array(0..<topK),
            routingWeights: routingWeights,
            d: dimension,
            f: intermediate,
            groupSize: groupSize)

        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0], groupSize: groupSize)
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        let half = topK / 2
        let lowSlots = Array(0..<UInt32(half))
        let highSlots = Array(UInt32(half)..<UInt32(topK))
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(context.device, count: topK * intermediate),
              let splitActs = Fp16Buffer.make(context.device, count: topK * intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: dimension),
              let lowSlotsBuffer = context.device.makeBuffer(
                bytes: lowSlots,
                length: lowSlots.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let highSlotsBuffer = context.device.makeBuffer(
                bytes: highSlots,
                length: highSlots.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            groupSize: UInt32(groupSize))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            groupSize: UInt32(groupSize))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in [(lowSlots, lowSlotsBuffer), (highSlots, highSlotsBuffer)] {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(dimension),
                f: UInt32(intermediate),
                topK: UInt32(topK),
                groupSize: UInt32(groupSize))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            groupSize: UInt32(groupSize))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: dimension)
        let split = Fp16Buffer.read(splitOutput, count: dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    /// `encodeRoutedPersistentPhase2Reduce` dispatches `topK * 32` threads per
    /// threadgroup (one simdgroup per routed expert). At the kMoEMaxTopK cap of
    /// 16 that's 512 threads; if the compiled pipeline's device-reported limit
    /// ever dropped below that, the dispatch would silently violate the Metal
    /// contract. Checked against the exact, unhinted pipeline MoE.swift builds
    /// (`context.pipeline("moe_phase2_down_reduce")`, no maxTotalThreadsPerThreadgroup
    /// override), not a hypothetical one.
    @Test func phase2ThreadgroupWidthFitsDeviceLimitAtMaxTopK() throws {
        let context = try MetalContext()
        let pipeline = try context.pipeline("moe_phase2_down_reduce")
        #expect(pipeline.maxTotalThreadsPerThreadgroup >= MoE.maxStreamedExperts * 32)
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]],
                                 groupSize: Int = Quantization.groupSize) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0, groupSize: groupSize) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
