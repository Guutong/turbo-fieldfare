import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Temporary scratch verification for the silu activation path added by the
/// "generalize arch for MoE" work. Covers every activation site:
///   - gelu_mul_fp16 (utility.metal) via the buffer(4) flag — direct kernel
///     dispatch plus the SharedExpertInt4 pipeline;
///   - shared_int8_gate_up_act_simd (dequant_int8.metal) via FC_USE_SILU;
///   - moe_phase1_gate_up_act_u16load / subset (moe.metal) via FC_USE_SILU;
///   - prefill_grouped_routed_moe_batched_phase1 (prefill.metal) PSO build.
/// Delete this file after the caller has reviewed the results.
@Suite struct SiluActivationScratchTests {

    // MARK: - shared helpers

    /// silu(x) = x * sigmoid(x) with the same -20 clamp as the shader.
    private static func silu(_ x: Float) -> Float {
        let s = Float(1.0 / (1.0 + exp(-Double(max(x, -20.0)))))
        return s * x
    }

    /// act = half( silu(g) * u ) — matches `half( (use_silu ? silu(g) : gelu(g)) * u )`.
    private static func actSilu(_ g: Float, _ u: Float) -> Float {
        Float(Float16(silu(g) * u))
    }

    // MARK: - direct gelu_mul_fp16 dispatch (buffer(4) flag, clamp behavior)

    @Test func geluMulFp16SiluClampMatchesReference() throws {
        let context = try MetalContext()
        let pso = try context.pipeline("gelu_mul_fp16")
        let gates: [Float] = [-50.0, -21.0, -20.0, -10.0, -5.0, -2.0, 0.0, 0.5, 3.0, 50.0]
        let ups = [Float](repeating: 1.0, count: gates.count)
        let reference: [Float] = zip(gates, ups).map { g, u in
            let g16 = Float(Float16(g))
            return Float(Float16(Self.silu(g16) * u))
        }
        guard let gateBuf = Fp16Buffer.make(context.device, values: gates),
              let upBuf = Fp16Buffer.make(context.device, values: ups),
              let outBuf = Fp16Buffer.make(context.device, count: gates.count),
              let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("alloc failed"); return
        }
        var count = UInt32(gates.count)
        var useSilu = true
        enc.setComputePipelineState(pso)
        enc.setBuffer(gateBuf, offset: 0, index: 0)
        enc.setBuffer(upBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&useSilu, length: MemoryLayout<Bool>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: gates.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let actual = Fp16Buffer.read(outBuf, count: gates.count)
        for (i, (a, r)) in zip(actual, reference).enumerated() {
            // The -50/-21 inputs live in the subnormal FP16 range; require the
            // clamp to bound them near zero instead of producing inf/NaN.
            if abs(r) < 1e-6 {
                #expect(a.isFinite, "gate[\(i)] produced non-finite output")
                #expect(abs(a - r) < 1e-5, "gate[\(i)] abs=\(a) expected≈\(r)")
            } else {
                #expect(abs(a - r) <= 1e-3 * abs(r),
                        "gate[\(i)] actual=\(a) reference=\(r)")
            }
        }
    }

    // MARK: - SharedExpertInt4 (gelu_mul_fp16 via wrapper)

    @Test func int4SiluMatchesReference() throws {
        var rng = SeedTree(0x604).key("silu-scratch-int4")
        let d = 128, f = 64
        let x = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<d).map { _ in (0..<f).map { _ in rng.uniform(-0.4, 0.4) } }
        let gatePack = Self.packInt4(gate)
        let upPack = Self.packInt4(up)
        let downPack = Self.packInt4(down)
        let x16 = x.map { Float(Float16($0)) }
        let gateOut = DequantInt4GemvRef.apply(weightRows: gatePack.rows, x: x16, n: d)
        let upOut = DequantInt4GemvRef.apply(weightRows: upPack.rows, x: x16, n: d)
        let act = zip(gateOut, upOut).map(Self.actSilu)
        let reference = DequantInt4GemvRef.apply(weightRows: downPack.rows, x: act, n: f)

        let context = try MetalContext()
        let runtime = try SharedExpertInt4(context: context, useSilu: true)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let yBuffer = try #require(Fp16Buffer.make(context.device, count: d))
        let gateScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let upScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let actScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try runtime.encode(commandBuffer: commandBuffer,
                           x: xBuffer,
                           gate: Self.projection(context, gatePack, rows: f, cols: d),
                           up: Self.projection(context, upPack, rows: f, cols: d),
                           down: Self.projection(context, downPack, rows: d, cols: f),
                           y: yBuffer,
                           scratchGate: gateScratch,
                           scratchUp: upScratch,
                           scratchAct: actScratch)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let actual = Fp16Buffer.read(yBuffer, count: d)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.quantInt4 * 4, "int4 silu rel=\(error)")
    }

    // MARK: - SharedExpertInt8 (FC_USE_SILU constant)

    @Test func int8SiluMatchesReference() throws {
        var rng = SeedTree(0x601).key("silu-scratch-int8")
        let d = 128, f = 64
        let xFp32 = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<d).map { _ in (0..<f).map { _ in rng.uniform(-0.4, 0.4) } }
        let xFp16 = xFp32.map { Float(Float16($0)) }
        let gatePack = Self.packInt8(gate)
        let upPack = Self.packInt8(up)
        let downPack = Self.packInt8(down)
        let gateOut = DequantInt8GemvRef.apply(weightRows: gatePack.rows, x: xFp16, n: d)
        let upOut = DequantInt8GemvRef.apply(weightRows: upPack.rows, x: xFp16, n: d)
        let act = zip(gateOut, upOut).map(Self.actSilu)
        let yRef = DequantInt8GemvRef.apply(weightRows: downPack.rows, x: act, n: f)

        let ctx = try MetalContext()
        let wrapper = try SharedExpertInt8(context: ctx, useSilu: true)
        guard let xBuf = Fp16Buffer.make(ctx.device, values: xFp32),
              let yBuf = Fp16Buffer.make(ctx.device, count: d),
              let sa = Fp16Buffer.make(ctx.device, count: f) else {
            Issue.record("alloc failed"); return
        }
        let gateProj = Self.projectionInt8(ctx, gatePack, rows: f, cols: d)
        let upProj = Self.projectionInt8(ctx, upPack, rows: f, cols: d)
        let downProj = Self.projectionInt8(ctx, downPack, rows: d, cols: f)
        let cb = ctx.queue.makeCommandBuffer()!
        try wrapper.encode(commandBuffer: cb,
                           x: xBuf, gate: gateProj, up: upProj, down: downProj,
                           y: yBuf, scratchAct: sa)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let rel = RelError.compute(actual: actual, reference: yRef)
        #expect(rel < Tolerance.quantInt8 * 4, "int8 silu rel=\(rel)")
    }

    // MARK: - routed decode MoE (FC_USE_SILU on moe.metal phase 1)

    @Test func decodeRoutedSiluMatchesReference() throws {
        let d = 128, f = 64, topK = 8
        var rng = SeedTree(0x2D3).key("silu-scratch-routed")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: f, columns: d))
            ups.append(matrix(rows: f, columns: d))
            downs.append(matrix(rows: d, columns: f))
        }
        let x = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }

        // silu reference: y = residual + Σ w_e · down( silu(gate_e(x)) * up_e(x) )
        var yRef = residual
        for e in 0..<topK {
            let gateRows = gates[e].map { Quantization.quantizeInt4Affine($0) }
            let upRows = ups[e].map { Quantization.quantizeInt4Affine($0) }
            let downRows = downs[e].map { Quantization.quantizeInt4Affine($0) }
            let gateOut = DequantInt4GemvRef.apply(weightRows: gateRows, x: x, n: d)
            let upOut = DequantInt4GemvRef.apply(weightRows: upRows, x: x, n: d)
            let act = zip(gateOut, upOut).map(Self.actSilu)
            let out = DequantInt4GemvRef.apply(weightRows: downRows, x: act, n: f)
            let w = routingWeights[e]
            yRef = zip(yRef, out).map { $0 + w * $1 }
        }

        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }
        let context = try MetalContext()
        let kernel = try MoE(context: context, useSilu: true)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(context.device, count: topK * f),
              let splitActs = Fp16Buffer.make(context.device, count: topK * f),
              let fullOutput = Fp16Buffer.make(context.device, count: d),
              let splitOutput = Fp16Buffer.make(context.device, count: d),
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
            d: UInt32(d),
            f: UInt32(f),
            topK: UInt32(topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(d),
            f: UInt32(f),
            topK: UInt32(topK))
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
                d: UInt32(d),
                f: UInt32(f),
                topK: UInt32(topK))
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
            d: UInt32(d),
            f: UInt32(f),
            topK: UInt32(topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: d)
        let split = Fp16Buffer.read(splitOutput, count: d)
        #expect(full == split)
        let fullError = RelError.compute(actual: full, reference: yRef)
        #expect(fullError < Tolerance.fp16ChainedReduction,
                "decode routed silu rel=\(fullError)")
    }

    // MARK: - prefill PSO smoke (FC_USE_SILU on prefill.metal)

    @Test func prefillGroupedSiluPSOBuilds() throws {
        let context = try MetalContext()
        // Constructing the wrapper builds the phase-1 PSO with function
        // constant 87 (FC_USE_SILU) plus the argument encoder from the
        // specialized function; a bad constant index would throw here.
        _ = try PrefillGroupedRoutedMoE(context: context, useSilu: true)
    }

    // MARK: - packers / blobs (mirror the existing suite helpers)

    private static func packInt4(_ values: [[Float]]) ->
        (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let rows = values.map(Quantization.quantizeInt4Affine)
        return (rows,
                rows.flatMap(\.packed),
                rows.flatMap(\.scales),
                rows.flatMap(\.biases))
    }

    private static func packInt8(_ rows: [[Float]]) ->
        (rows: [Quantization.Int8AffineRow],
         packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let m = rows.count
        let n = rows[0].count
        let gpr = n / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: m * n)
        var scales = [UInt16](repeating: 0, count: m * gpr)
        var biases = [UInt16](repeating: 0, count: m * gpr)
        var rowsOut: [Quantization.Int8AffineRow] = []
        rowsOut.reserveCapacity(m)
        for r in 0..<m {
            let q = Quantization.quantizeInt8Affine(rows[r])
            for i in 0..<n { packed[r * n + i] = q.packed[i] }
            for g in 0..<gpr {
                scales[r * gpr + g] = q.scales[g]
                biases[r * gpr + g] = q.biases[g]
            }
            rowsOut.append(q)
        }
        return (rowsOut, packed, scales, biases)
    }

    private static func projection(
        _ context: MetalContext,
        _ packed: (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: context.device.makeBuffer(bytes: packed.packed,
                                                length: packed.packed.count,
                                                options: .storageModeShared)!,
            scales: context.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * 2,
                                               options: .storageModeShared)!,
            biases: context.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * 2,
                                               options: .storageModeShared)!,
            rows: UInt32(rows), cols: UInt32(cols))
    }

    private static func projectionInt8(
        _ context: MetalContext,
        _ packed: (rows: [Quantization.Int8AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertInt8Proj {
        SharedExpertInt8Proj(
            weights: context.device.makeBuffer(bytes: packed.packed,
                                                length: packed.packed.count,
                                                options: .storageModeShared)!,
            scales: context.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * 2,
                                               options: .storageModeShared)!,
            biases: context.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * 2,
                                               options: .storageModeShared)!,
            rows: UInt32(rows), cols: UInt32(cols))
    }

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
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
