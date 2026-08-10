import Foundation
import Metal

/// P4-1: the DeltaNet decode step on the GPU, with the conv and recurrent
/// state resident in Metal buffers.
///
/// Direct port of `DeltaNetCPUBlock` — same chain, same order, same fp32
/// arithmetic (input RMSNorm -> qkv/z/a/b projections -> causal conv1d ->
/// QK-RMSNorm -> head expansion -> beta/g gating -> gated-delta recurrence ->
/// gated RMSNorm/swiglu output -> out_proj), plus the FP16 residual bridging
/// the CPU block did inline in `RealForwardRunner`.
///
/// ADR-0001 kept the Phase 3 CPU block obviously-correct precisely so this
/// port would have an exact diff target; `DeltaNetMetalParityTests` is that
/// diff. The kernels therefore keep the reference's sequential reduction
/// order rather than using SIMD reductions — see `deltanet.metal`.
final class DeltaNetMetalBlock {

    // MARK: - Per-layer weights, dequantized to fp32 GPU buffers once.

    /// One DeltaNet layer's weights as fp32 Metal buffers. Dequantized on
    /// first use and reused for the life of the runner, exactly like
    /// `DeltaNetCPUBlock.LayerWeights` — same values, same layout, just
    /// device-resident instead of `[Float]`.
    final class LayerWeights {
        let inputNorm: MTLBuffer   // [D]
        let qkv: MTLBuffer         // [convDim x D]
        let z: MTLBuffer           // [valueDim x D]
        let a: MTLBuffer           // [numValueHeads x D]
        let b: MTLBuffer           // [numValueHeads x D]
        let out: MTLBuffer         // [D x valueDim]
        let conv: MTLBuffer        // [convDim * 4]
        let deltaNorm: MTLBuffer   // [headVDim]
        let aLog: MTLBuffer        // [numValueHeads]
        let dtBias: MTLBuffer      // [numValueHeads]

        init(device: MTLDevice, model: Model, layer L: Int, D: Int,
             dims: DeltaNetDimensions) throws {
            let convDim = dims.convDim
            let keyDim = dims.numKeyHeads * dims.headKDim
            let valueDim = dims.numValueHeads * dims.headVDim
            precondition(convDim == 2 * keyDim + valueDim)

            // Reuses the CPU block's proven dequant helpers, then hands the
            // values to the device and drops the host array.
            func upload(_ values: [Float], _ label: String) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(
                    length: max(values.count, 1) * MemoryLayout<Float>.size,
                    options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                buffer.label = "deltanet.\(label).L\(L)"
                let dst = buffer.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<values.count { dst[i] = values[i] }
                return buffer
            }

            typealias CPU = DeltaNetCPUBlock.LayerWeights
            inputNorm = try upload(CPU.bf16Vector(try model.inputNorm(layer: L), count: D),
                                   "input_norm")
            qkv = try upload(CPU.dequantResident(try model.deltaQKVProj(layer: L),
                                                 rows: convDim, cols: D, bits: 4), "qkv")
            z = try upload(CPU.dequantResident(try model.deltaZProj(layer: L),
                                               rows: valueDim, cols: D, bits: 4), "z")
            a = try upload(CPU.dequantResident(try model.deltaAProj(layer: L),
                                               rows: dims.numValueHeads, cols: D, bits: 4), "a")
            b = try upload(CPU.dequantResident(try model.deltaBProj(layer: L),
                                               rows: dims.numValueHeads, cols: D, bits: 4), "b")
            out = try upload(CPU.dequantResident(try model.deltaOutProj(layer: L),
                                                 rows: D, cols: valueDim, bits: 4), "out")
            conv = try upload(CPU.bf16Vector(try model.deltaConv1d(layer: L),
                                             count: convDim * 4), "conv")
            deltaNorm = try upload(CPU.bf16Vector(try model.deltaNorm(layer: L),
                                                  count: dims.headVDim), "norm")
            aLog = try upload(CPU.bf16Vector(try model.deltaALog(layer: L),
                                             count: dims.numValueHeads), "a_log")
            dtBias = try upload(CPU.bf16Vector(try model.deltaDtBias(layer: L),
                                               count: dims.numValueHeads), "dt_bias")
        }
    }

    /// Lazily dequantizes and caches per-layer GPU weights on first use.
    final class WeightsCache: @unchecked Sendable {
        private let device: MTLDevice
        private let model: Model
        private let D: Int
        private let dims: DeltaNetDimensions
        private var cache: [Int: LayerWeights] = [:]

        init(device: MTLDevice, model: Model, hiddenSize: Int, dims: DeltaNetDimensions) {
            self.device = device
            self.model = model
            self.D = hiddenSize
            self.dims = dims
        }

        func weights(layer L: Int) throws -> LayerWeights {
            if let w = cache[L] { return w }
            let w = try LayerWeights(device: device, model: model, layer: L, D: D, dims: dims)
            cache[L] = w
            return w
        }
    }

    // MARK: - GPU-resident state.

    /// P4-1: the per-session DeltaNet state, resident in Metal buffers rather
    /// than the `[Float]` arrays of `DeltaNetStateStore`. Same shapes, same
    /// fp32 contents, same "indexed by GLOBAL layer index" convention; full-
    /// attention layers get no buffer at all.
    final class GPUStateStore: @unchecked Sendable {
        let numLayers: Int
        let dims: DeltaNetDimensions
        let isLinearLayer: [Bool]
        private(set) var convState: [MTLBuffer?]
        private(set) var recurrentState: [MTLBuffer?]

        init(device: MTLDevice, numLayers: Int, dims: DeltaNetDimensions,
             isLinearLayer: [Bool]) throws {
            precondition(isLinearLayer.count == numLayers)
            self.numLayers = numLayers
            self.dims = dims
            self.isLinearLayer = isLinearLayer

            func state(_ count: Int, _ label: String) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(
                    length: count * MemoryLayout<Float>.size,
                    options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                buffer.label = label
                memset(buffer.contents(), 0, buffer.length)
                return buffer
            }
            var conv: [MTLBuffer?] = []
            var recurrent: [MTLBuffer?] = []
            for L in 0..<numLayers {
                guard isLinearLayer[L] else {
                    conv.append(nil)
                    recurrent.append(nil)
                    continue
                }
                conv.append(try state(dims.convStateCount, "deltanet.conv_state.L\(L)"))
                recurrent.append(try state(dims.recurrentStateCount,
                                           "deltanet.recurrent_state.L\(L)"))
            }
            self.convState = conv
            self.recurrentState = recurrent
        }

        /// Zero every DeltaNet state, so a reused runner never leaks one
        /// conversation's state into the next.
        func reset() {
            for L in 0..<numLayers {
                if let c = convState[L] { memset(c.contents(), 0, c.length) }
                if let r = recurrentState[L] { memset(r.contents(), 0, r.length) }
            }
        }
    }

    // MARK: - Pipelines and scratch.

    private let ctx: MetalContext
    private let dims: DeltaNetDimensions
    private let D: Int

    private let psoLoadHidden: MTLComputePipelineState
    private let psoStoreHidden: MTLComputePipelineState
    private let psoRMSNorm: MTLComputePipelineState
    private let psoMatVec: MTLComputePipelineState
    private let psoConvStep: MTLComputePipelineState
    private let psoQKNormExpand: MTLComputePipelineState
    private let psoGates: MTLComputePipelineState
    private let psoRecurrence: MTLComputePipelineState
    private let psoOutputGate: MTLComputePipelineState

    // MARK: - Batched pipelines.

    private let psoLoadHiddenB: MTLComputePipelineState
    private let psoStoreHiddenB: MTLComputePipelineState
    private let psoRMSNormB: MTLComputePipelineState
    private let psoMatVecB: MTLComputePipelineState
    private let psoConvStepB: MTLComputePipelineState
    private let psoQKNormExpandB: MTLComputePipelineState
    private let psoGatesB: MTLComputePipelineState
    private let psoRecurrenceB: MTLComputePipelineState
    private let psoOutputGateB: MTLComputePipelineState

    /// Conservative max draft count for scratch-buffer sizing. Actual tk at
    /// encoding time must never exceed this. 128 covers typical Mac hardware.
    static let defaultMaxTK = 128

    // Per-token scratch, allocated once and reused across layers. About
    // 220 KB total at Qwen3.6 shape. Batched paths require [maxTK × dim].
    private let maxTK: Int
    private let xBuf: MTLBuffer          // [maxTK × D]
    private let normedBuf: MTLBuffer     // [maxTK × D]
    private let qkvBuf: MTLBuffer        // [maxTK × convDim]
    private let convOutBuf: MTLBuffer    // [maxTK × convDim]
    private let zBuf: MTLBuffer          // [maxTK × valueDim]
    private let aRawBuf: MTLBuffer       // [maxTK × numValueHeads]
    private let bRawBuf: MTLBuffer       // [maxTK × numValueHeads]
    private let qExpBuf: MTLBuffer       // [maxTK × expandedKeyDim]
    private let kExpBuf: MTLBuffer       // [maxTK × expandedKeyDim]
    private let betaBuf: MTLBuffer       // [maxTK × numValueHeads]
    private let gBuf: MTLBuffer          // [maxTK × numValueHeads]
    private let yBuf: MTLBuffer          // [maxTK × valueDim]
    private let gatedBuf: MTLBuffer      // [maxTK × valueDim]
    private let deltaOutBuf: MTLBuffer   // [maxTK × D]

    init(context: MetalContext, hiddenSize: Int, dims: DeltaNetDimensions,
         maxTK: Int = DeltaNetMetalBlock.defaultMaxTK) throws {
        self.ctx = context
        self.dims = dims
        self.D = hiddenSize
        self.maxTK = maxTK

        psoLoadHidden = try context.pipeline("dn_load_hidden")
        psoStoreHidden = try context.pipeline("dn_store_hidden")
        psoRMSNorm = try context.pipeline("dn_rmsnorm")
        psoMatVec = try context.pipeline("dn_matvec")
        psoConvStep = try context.pipeline("dn_conv_step")
        psoQKNormExpand = try context.pipeline("dn_qknorm_expand")
        psoGates = try context.pipeline("dn_gates")
        psoRecurrence = try context.pipeline("dn_recurrence")
        psoOutputGate = try context.pipeline("dn_output_gate")

        // Batched pipelines.
        psoLoadHiddenB  = try context.pipeline("dn_load_hidden_batched")
        psoStoreHiddenB = try context.pipeline("dn_store_hidden_batched")
        psoRMSNormB     = try context.pipeline("dn_rmsnorm_batched")
        psoMatVecB      = try context.pipeline("dn_matvec_batched")
        psoConvStepB    = try context.pipeline("dn_conv_step_batched")
        psoQKNormExpandB = try context.pipeline("dn_qknorm_expand_batched")
        psoGatesB       = try context.pipeline("dn_gates_batched")
        psoRecurrenceB  = try context.pipeline("dn_recurrence_batched")
        psoOutputGateB  = try context.pipeline("dn_output_gate_batched")

        let valueDim = dims.numValueHeads * dims.headVDim
        let expandedKeyDim = dims.numValueHeads * dims.headKDim
        func buf(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(
                length: max(count, 1) * MemoryLayout<Float>.size,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = "deltanet.\(label)"
            return b
        }
        xBuf = try buf(maxTK * hiddenSize, "x")
        normedBuf = try buf(maxTK * hiddenSize, "normed")
        qkvBuf = try buf(maxTK * dims.convDim, "qkv")
        convOutBuf = try buf(maxTK * dims.convDim, "conv_out")
        zBuf = try buf(maxTK * valueDim, "z")
        aRawBuf = try buf(maxTK * dims.numValueHeads, "a_raw")
        bRawBuf = try buf(maxTK * dims.numValueHeads, "b_raw")
        qExpBuf = try buf(maxTK * expandedKeyDim, "q_expanded")
        kExpBuf = try buf(maxTK * expandedKeyDim, "k_expanded")
        betaBuf = try buf(maxTK * dims.numValueHeads, "beta")
        gBuf = try buf(maxTK * dims.numValueHeads, "g")
        yBuf = try buf(maxTK * valueDim, "y")
        gatedBuf = try buf(maxTK * valueDim, "gated")
        deltaOutBuf = try buf(maxTK * hiddenSize, "gated_out")
    }

    // MARK: - Encoding.

    private func dispatch(_ enc: MTLComputeCommandEncoder,
                          _ pso: MTLComputePipelineState,
                          threads: Int) {
        enc.setComputePipelineState(pso)
        let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func matVec(_ enc: MTLComputeCommandEncoder,
                        w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
                        rows: Int, cols: Int) {
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var r = UInt32(rows), c = UInt32(cols)
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(enc, psoMatVec, threads: rows)
    }

    /// Encodes one token's DeltaNet layer into `cb`: reads the FP16 residual
    /// stream `hidden`, and writes back `hidden = hidden + deltaOut` — the
    /// exact contract the CPU call site had in `RealForwardRunner`.
    ///
    /// All kernels are encoded into a single compute encoder; Metal's
    /// serial-dispatch ordering within an encoder gives the read-after-write
    /// dependency each stage needs, and no stage races within itself (each
    /// thread owns its own state row).
    func encode(commandBuffer cb: MTLCommandBuffer,
                hidden: MTLBuffer,
                weights: LayerWeights,
                convState: MTLBuffer,
                recurrentState: MTLBuffer,
                eps: Float = 1e-6) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "deltanet"
        let keyDim = dims.numKeyHeads * dims.headKDim
        let valueDim = dims.numValueHeads * dims.headVDim
        let expandedKeyDim = dims.numValueHeads * dims.headKDim
        let floatSize = MemoryLayout<Float>.size
        var epsValue = eps

        // hidden (FP16) -> x (fp32).
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        var d = UInt32(D)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(enc, psoLoadHidden, threads: D)

        // Input RMSNorm.
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(weights.inputNorm, offset: 0, index: 1)
        enc.setBuffer(normedBuf, offset: 0, index: 2)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsValue, length: MemoryLayout<Float>.size, index: 4)
        dispatch(enc, psoRMSNorm, threads: D)

        // Projections.
        matVec(enc, w: weights.qkv, x: normedBuf, y: qkvBuf, rows: dims.convDim, cols: D)
        matVec(enc, w: weights.z, x: normedBuf, y: zBuf, rows: valueDim, cols: D)
        matVec(enc, w: weights.a, x: normedBuf, y: aRawBuf, rows: dims.numValueHeads, cols: D)
        matVec(enc, w: weights.b, x: normedBuf, y: bRawBuf, rows: dims.numValueHeads, cols: D)

        // Causal conv1d + silu, conv state resident in `convState`.
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(weights.conv, offset: 0, index: 1)
        enc.setBuffer(convState, offset: 0, index: 2)
        enc.setBuffer(convOutBuf, offset: 0, index: 3)
        var convDim = UInt32(dims.convDim)
        enc.setBytes(&convDim, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(enc, psoConvStep, threads: dims.convDim)

        // QK-RMSNorm fused with the key -> value head expansion. q occupies
        // the first keyDim lanes of the conv output, k the next keyDim.
        var numKeyHeads = UInt32(dims.numKeyHeads)
        var numValueHeads = UInt32(dims.numValueHeads)
        var headKDim = UInt32(dims.headKDim)
        func qkNorm(sourceOffset: Int, out: MTLBuffer, scale: Float) {
            enc.setBuffer(convOutBuf, offset: sourceOffset, index: 0)
            enc.setBuffer(out, offset: 0, index: 1)
            enc.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&headKDim, length: MemoryLayout<UInt32>.size, index: 4)
            var s = scale
            enc.setBytes(&s, length: MemoryLayout<Float>.size, index: 5)
            var e = DeltaNetQKNorm.eps
            enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 6)
            dispatch(enc, psoQKNormExpand, threads: expandedKeyDim)
        }
        qkNorm(sourceOffset: 0, out: qExpBuf,
               scale: DeltaNetQKNorm.qScale(headKDim: dims.headKDim))
        qkNorm(sourceOffset: keyDim * floatSize, out: kExpBuf,
               scale: DeltaNetQKNorm.kScale(headKDim: dims.headKDim))

        // beta / g gates.
        enc.setBuffer(aRawBuf, offset: 0, index: 0)
        enc.setBuffer(bRawBuf, offset: 0, index: 1)
        enc.setBuffer(weights.aLog, offset: 0, index: 2)
        enc.setBuffer(weights.dtBias, offset: 0, index: 3)
        enc.setBuffer(betaBuf, offset: 0, index: 4)
        enc.setBuffer(gBuf, offset: 0, index: 5)
        enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 6)
        dispatch(enc, psoGates, threads: dims.numValueHeads)

        // Gated delta-rule recurrence; v is the tail of the conv output.
        enc.setBuffer(qExpBuf, offset: 0, index: 0)
        enc.setBuffer(kExpBuf, offset: 0, index: 1)
        enc.setBuffer(convOutBuf, offset: 2 * keyDim * floatSize, index: 2)
        enc.setBuffer(betaBuf, offset: 0, index: 3)
        enc.setBuffer(gBuf, offset: 0, index: 4)
        enc.setBuffer(recurrentState, offset: 0, index: 5)
        enc.setBuffer(yBuf, offset: 0, index: 6)
        var headVDim = UInt32(dims.headVDim)
        enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&headVDim, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&headKDim, length: MemoryLayout<UInt32>.size, index: 9)
        dispatch(enc, psoRecurrence, threads: valueDim)

        // Gated RMSNorm / swiglu output gate.
        enc.setBuffer(yBuf, offset: 0, index: 0)
        enc.setBuffer(zBuf, offset: 0, index: 1)
        enc.setBuffer(weights.deltaNorm, offset: 0, index: 2)
        enc.setBuffer(gatedBuf, offset: 0, index: 3)
        var count = UInt32(valueDim)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&headVDim, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&epsValue, length: MemoryLayout<Float>.size, index: 6)
        dispatch(enc, psoOutputGate, threads: valueDim)

        // out_proj, then fold back into the FP16 residual stream.
        matVec(enc, w: weights.out, x: gatedBuf, y: deltaOutBuf, rows: D, cols: valueDim)

        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(deltaOutBuf, offset: 0, index: 2)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(enc, psoStoreHidden, threads: D)

        enc.endEncoding()
    }

    // MARK: - Batched encoding.

    /// Encodes `tk` tokens' DeltaNet layer in one pass: hidden[tokens 0..tk-1].
    ///
    /// Reads from and writes back into `hidden` at offsets `[t*D .. (t+1)*D]`
    /// for each token t. All scratch buffers must therefore have capacity
    /// >= tk × D. State buffers expand to per-token layout as well.
    func encodeBatched(commandBuffer cb: MTLCommandBuffer,
                       hidden: MTLBuffer,
                       weights: LayerWeights,
                       convState: MTLBuffer,
                       recurrentState: MTLBuffer,
                       tk: Int,
                       eps: Float = 1e-6) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "deltanet.batched"
        let keyDim = dims.numKeyHeads * dims.headKDim
        let valueDim = dims.numValueHeads * dims.headVDim
        let expandedKeyDim = dims.numValueHeads * dims.headKDim
        let floatSize = MemoryLayout<Float>.size
        var k32 = UInt32(tk), d32 = UInt32(D)
        var epsValue = eps

        // hidden (FP16) -> x (fp32).
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&d32, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(enc, psoLoadHiddenB, threads: tk * D)

        // Input RMSNorm.
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(weights.inputNorm, offset: 0, index: 1)
        enc.setBuffer(normedBuf, offset: 0, index: 2)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&d32, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsValue, length: MemoryLayout<Float>.size, index: 5)
        dispatch(enc, psoRMSNormB, threads: tk * D)

        // Projections — batched mat-vec on all tk tokens at once.
        matVec(enc, w: weights.qkv, x: normedBuf, y: qkvBuf,
               rows: dims.convDim * tk, cols: D)
        matVec(enc, w: weights.z, x: normedBuf, y: zBuf,
               rows: valueDim * tk, cols: D)
        matVec(enc, w: weights.a, x: normedBuf, y: aRawBuf,
               rows: dims.numValueHeads * tk, cols: D)
        matVec(enc, w: weights.b, x: normedBuf, y: bRawBuf,
               rows: dims.numValueHeads * tk, cols: D)

        // Causal conv1d + silu, per-token state in convState.
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(weights.conv, offset: 0, index: 1)
        enc.setBuffer(convState, offset: 0, index: 2)
        enc.setBuffer(convOutBuf, offset: 0, index: 3)
        var convDim = UInt32(dims.convDim)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&convDim, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch(enc, psoConvStepB, threads: dims.convDim)

        // QK-RMSNorm fused with head expansion. q occupies first keyDim lanes,
        // k occupies next keyDim. Both outputs go to separate per-token buffers.
        var numKeyHeads = UInt32(dims.numKeyHeads)
        var numValueHeads = UInt32(dims.numValueHeads)
        var headKDim = UInt32(dims.headKDim)
        func qkNorm(sourceBuf: MTLBuffer, sourceOffset: Int,
                    outBuf: MTLBuffer, scale: Float) {
            enc.setBuffer(sourceBuf, offset: sourceOffset, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&headKDim, length: MemoryLayout<UInt32>.size, index: 4)
            var s = scale
            enc.setBytes(&s, length: MemoryLayout<Float>.size, index: 5)
            var e = DeltaNetQKNorm.eps
            enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 6)
            dispatch(enc, psoQKNormExpandB, threads: expandedKeyDim * tk)
        }
        qkNorm(sourceBuf: convOutBuf, sourceOffset: 0,
               outBuf: qExpBuf,
               scale: DeltaNetQKNorm.qScale(headKDim: dims.headKDim))
        qkNorm(sourceBuf: convOutBuf,
               sourceOffset: keyDim * MemoryLayout<Float>.size * tk,
               outBuf: kExpBuf,
               scale: DeltaNetQKNorm.kScale(headKDim: dims.headKDim))

        // beta / g gates.
        enc.setBuffer(aRawBuf, offset: 0, index: 0)
        enc.setBuffer(bRawBuf, offset: 0, index: 1)
        enc.setBuffer(weights.aLog, offset: 0, index: 2)
        enc.setBuffer(weights.dtBias, offset: 0, index: 3)
        enc.setBuffer(betaBuf, offset: 0, index: 4)
        enc.setBuffer(gBuf, offset: 0, index: 5)
        enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 7)
        dispatch(enc, psoGatesB, threads: dims.numValueHeads * tk)

        // Gated delta-rule recurrence; v is the tail of the conv output.
        enc.setBuffer(qExpBuf, offset: 0, index: 0)
        enc.setBuffer(kExpBuf, offset: 0, index: 1)
        enc.setBuffer(convOutBuf, offset: 2 * keyDim * floatSize * tk, index: 2)
        enc.setBuffer(betaBuf, offset: 0, index: 3)
        enc.setBuffer(gBuf, offset: 0, index: 4)
        enc.setBuffer(recurrentState, offset: 0, index: 5)
        enc.setBuffer(yBuf, offset: 0, index: 6)
        var headVDim = UInt32(dims.headVDim)
        enc.setBytes(&numValueHeads, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&headVDim, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&headKDim, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 10)
        dispatch(enc, psoRecurrenceB, threads: valueDim * tk)

        // Gated RMSNorm / swiglu output gate.
        enc.setBuffer(yBuf, offset: 0, index: 0)
        enc.setBuffer(zBuf, offset: 0, index: 1)
        enc.setBuffer(weights.deltaNorm, offset: 0, index: 2)
        enc.setBuffer(gatedBuf, offset: 0, index: 3)
        var count = UInt32(valueDim)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&headVDim, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&epsValue, length: MemoryLayout<Float>.size, index: 6)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 7)
        dispatch(enc, psoOutputGateB, threads: valueDim * tk)

        // out_proj, then fold back into FP16 residual stream.
        matVec(enc, w: weights.out, x: gatedBuf, y: deltaOutBuf,
               rows: D * tk, cols: valueDim)

        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(deltaOutBuf, offset: 0, index: 2)
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&d32, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(enc, psoStoreHiddenB, threads: tk * D)

        enc.endEncoding()
    }

    /// The last encoded token's `deltaOut` (the D-dim residual contribution),
    /// for tests that diff this path against `DeltaNetCPUBlock.forward`.
    /// Valid only after the encoding command buffer has completed.
    var lastDeltaOut: [Float] {
        let ptr = deltaOutBuf.contents().assumingMemoryBound(to: Float.self)
        return (0..<D).map { ptr[$0] }
    }
}
