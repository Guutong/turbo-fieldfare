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
        // P7-7: default path keeps the five big projections int4-resident
        // (`qkvTV`/`zTV`/`aTV`/`bTV`/`outTV` — TensorViews into the model's
        // existing mmap'd resident buffer, no copy, no dequant). Set
        // `TFF_DELTANET_FP32=1` to fall back to the legacy fp32-dequantized
        // path (`qkv`/`z`/`a`/`b`/`out`), kept for A/B measurement and
        // rollback. Task 3 wires `encode()` to consume the TensorViews via
        // `DequantInt4GEMV`; until then the fp32 fields are the only ones
        // `encode()` reads, so they stay force-unwrapped there.
        static let useFP32 = ProcessInfo.processInfo.environment["TFF_DELTANET_FP32"] == "1"

        let inputNorm: MTLBuffer   // [D]
        let qkv: MTLBuffer?        // [convDim x D], fp32 legacy path only
        let z: MTLBuffer?          // [valueDim x D], fp32 legacy path only
        let a: MTLBuffer?          // [numValueHeads x D], fp32 legacy path only
        let b: MTLBuffer?          // [numValueHeads x D], fp32 legacy path only
        let out: MTLBuffer?        // [D x valueDim], fp32 legacy path only
        let qkvTV: TensorView      // int4-resident, [convDim x D]
        let zTV: TensorView        // int4-resident, [valueDim x D]
        let aTV: TensorView        // int4-resident, [numValueHeads x D]
        let bTV: TensorView        // int4-resident, [numValueHeads x D]
        let outTV: TensorView      // int4-resident, [D x valueDim]
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

            qkvTV = try model.deltaQKVProj(layer: L)
            zTV = try model.deltaZProj(layer: L)
            aTV = try model.deltaAProj(layer: L)
            bTV = try model.deltaBProj(layer: L)
            outTV = try model.deltaOutProj(layer: L)

            if Self.useFP32 {
                qkv = try upload(CPU.dequantResident(qkvTV, rows: convDim, cols: D, bits: 4), "qkv")
                z = try upload(CPU.dequantResident(zTV, rows: valueDim, cols: D, bits: 4), "z")
                a = try upload(CPU.dequantResident(aTV, rows: dims.numValueHeads, cols: D, bits: 4), "a")
                b = try upload(CPU.dequantResident(bTV, rows: dims.numValueHeads, cols: D, bits: 4), "b")
                out = try upload(CPU.dequantResident(outTV, rows: D, cols: valueDim, bits: 4), "out")
            } else {
                qkv = nil
                z = nil
                a = nil
                b = nil
                out = nil
            }

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

    // P7-7: int4 GEMV for the five big projections (int4-resident path),
    // plus casts bridging it to the fp32 chain around it (see deltanet.metal).
    private let int4GEMV: DequantInt4GEMV
    private let psoCastToHalf: MTLComputePipelineState
    private let psoCastToFloat: MTLComputePipelineState

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

    // P7-7: half-precision scratch for the int4 GEMV boundary. Only the
    // GEMV's direct in/out need a half copy; everything downstream reads the
    // existing fp32 buffers unchanged (populated by a cast kernel).
    private let normedBufF16: MTLBuffer    // [D], GEMV input, shared by qkv/z/a/b
    private let qkvBufF16: MTLBuffer       // [convDim]
    private let zBufF16: MTLBuffer         // [valueDim]
    private let aRawBufF16: MTLBuffer      // [numValueHeads]
    private let bRawBufF16: MTLBuffer      // [numValueHeads]
    private let gatedBufF16: MTLBuffer     // [valueDim], GEMV input for out_proj
    private let deltaOutBufF16: MTLBuffer  // [D]

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

        int4GEMV = try DequantInt4GEMV(context: context)
        psoCastToHalf = try context.pipeline("dn_cast_f32_to_f16")
        psoCastToFloat = try context.pipeline("dn_cast_f16_to_f32")

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

        func bufH(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(
                length: max(count, 1) * MemoryLayout<Float16>.size,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = "deltanet.\(label).f16"
            return b
        }
        // Single-token sized (not maxTK ×) — the int4 GEMV boundary is only
        // used by the non-batched `encode()` path; `encodeBatched` keeps the
        // fp32 dn_matvec_batched path regardless of `useFP32Weights`.
        normedBufF16 = try bufH(hiddenSize, "normed")
        qkvBufF16 = try bufH(dims.convDim, "qkv")
        zBufF16 = try bufH(valueDim, "z")
        aRawBufF16 = try bufH(dims.numValueHeads, "a_raw")
        bRawBufF16 = try bufH(dims.numValueHeads, "b_raw")
        gatedBufF16 = try bufH(valueDim, "gated")
        deltaOutBufF16 = try bufH(hiddenSize, "gated_out")
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

    private func cast(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState,
                      src: MTLBuffer, srcOffset: Int = 0,
                      dst: MTLBuffer, dstOffset: Int = 0, count: Int) {
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(dst, offset: dstOffset, index: 1)
        var c = UInt32(count)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(enc, pso, threads: count)
    }

    /// int4-resident matVec: casts `xF32` (fp32) to a half scratch buffer,
    /// runs the int4 SIMD GEMV kernel, casts the half result back into
    /// `yF32` — so callers downstream of `y` see the same fp32 buffer they
    /// always did. `w` is int4-packed with BF16 scales/biases, addressed by
    /// `TensorView` offsets into the model's resident weight arena.
    /// `xF32Offset`/`yF32Offset` are byte offsets into the (possibly
    /// batched, `[tk x dim]`) fp32 scratch; `xF16`/`yF16` are always the
    /// single-token half scratch (reused per token in the batched loop).
    private func matVecInt4(_ enc: MTLComputeCommandEncoder,
                            w: TensorView,
                            xF32: MTLBuffer, xF32Offset: Int = 0, xF16: MTLBuffer,
                            yF16: MTLBuffer, yF32: MTLBuffer, yF32Offset: Int = 0,
                            rows: Int, cols: Int) {
        cast(enc, psoCastToHalf, src: xF32, srcOffset: xF32Offset, dst: xF16, count: cols)
        int4GEMV.encode(encoder: enc,
                        weights: w.buffer, weightsOffset: Int(w.offset),
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        x: xF16, y: yF16,
                        m: UInt32(rows), n: UInt32(cols))
        cast(enc, psoCastToFloat, src: yF16, dst: yF32, dstOffset: yF32Offset, count: rows)
    }

    /// Batched int4-resident matVec: DequantInt4GEMV has no batch dimension,
    /// so this loops the single-token path over `tk` tokens, each iteration
    /// reusing the same half scratch (serial dispatch order within one
    /// encoder makes this race-free — see `encode()`'s doc comment).
    private func matVecInt4Batched(_ enc: MTLComputeCommandEncoder,
                                   w: TensorView, xF32: MTLBuffer, xF16: MTLBuffer,
                                   yF16: MTLBuffer, yF32: MTLBuffer,
                                   rows: Int, cols: Int, tk: Int) {
        let floatSize = MemoryLayout<Float>.size
        for t in 0..<tk {
            matVecInt4(enc, w: w,
                      xF32: xF32, xF32Offset: t * cols * floatSize, xF16: xF16,
                      yF16: yF16, yF32: yF32, yF32Offset: t * rows * floatSize,
                      rows: rows, cols: cols)
        }
    }

    private func matVecBatched(_ enc: MTLComputeCommandEncoder,
                               w: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
                               rows: Int, cols: Int, batch: Int) {
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var r = UInt32(rows), c = UInt32(cols), b = UInt32(batch)
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch(enc, psoMatVecB, threads: rows * batch)
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
                hiddenOffset: Int = 0,
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

        // hidden (FP16) -> x (fp32). hiddenOffset selects this token's slice
        // when `hidden` is a shared multi-token buffer (e.g. DraftVerifier's
        // K-token scratch), 0 for a single-token buffer.
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
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

        // Projections. int4-resident path (default) uses the SIMD GEMV
        // kernel through a half scratch boundary; fp32 path (TFF_DELTANET_
        // FP32=1) keeps the original one-thread-per-row dn_matvec.
        if DeltaNetMetalBlock.LayerWeights.useFP32 {
            matVec(enc, w: weights.qkv!, x: normedBuf, y: qkvBuf, rows: dims.convDim, cols: D)
            matVec(enc, w: weights.z!, x: normedBuf, y: zBuf, rows: valueDim, cols: D)
            matVec(enc, w: weights.a!, x: normedBuf, y: aRawBuf, rows: dims.numValueHeads, cols: D)
            matVec(enc, w: weights.b!, x: normedBuf, y: bRawBuf, rows: dims.numValueHeads, cols: D)
        } else {
            matVecInt4(enc, w: weights.qkvTV, xF32: normedBuf, xF16: normedBufF16,
                      yF16: qkvBufF16, yF32: qkvBuf, rows: dims.convDim, cols: D)
            matVecInt4(enc, w: weights.zTV, xF32: normedBuf, xF16: normedBufF16,
                      yF16: zBufF16, yF32: zBuf, rows: valueDim, cols: D)
            matVecInt4(enc, w: weights.aTV, xF32: normedBuf, xF16: normedBufF16,
                      yF16: aRawBufF16, yF32: aRawBuf, rows: dims.numValueHeads, cols: D)
            matVecInt4(enc, w: weights.bTV, xF32: normedBuf, xF16: normedBufF16,
                      yF16: bRawBufF16, yF32: bRawBuf, rows: dims.numValueHeads, cols: D)
        }

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
        if DeltaNetMetalBlock.LayerWeights.useFP32 {
            matVec(enc, w: weights.out!, x: gatedBuf, y: deltaOutBuf, rows: D, cols: valueDim)
        } else {
            matVecInt4(enc, w: weights.outTV, xF32: gatedBuf, xF16: gatedBufF16,
                      yF16: deltaOutBufF16, yF32: deltaOutBuf, rows: D, cols: valueDim)
        }

        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
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
        // P7-7 Task 6: batched int4 path loops the single-token GEMV per
        // token (see matVecInt4Batched) — DequantInt4GEMV has no native
        // batch dimension. maxTK caps tk, so the loop is bounded.
        precondition(tk <= maxTK, "encodeBatched: tk (\(tk)) exceeds maxTK (\(maxTK))")
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

        // Projections — batched mat-vec on all tk tokens at once (fp32 path)
        // or looped single-token int4 GEMV (int4 path — see matVecInt4Batched).
        if DeltaNetMetalBlock.LayerWeights.useFP32 {
            matVecBatched(enc, w: weights.qkv!, x: normedBuf, y: qkvBuf,
                          rows: dims.convDim, cols: D, batch: tk)
            matVecBatched(enc, w: weights.z!, x: normedBuf, y: zBuf,
                          rows: valueDim, cols: D, batch: tk)
            matVecBatched(enc, w: weights.a!, x: normedBuf, y: aRawBuf,
                          rows: dims.numValueHeads, cols: D, batch: tk)
            matVecBatched(enc, w: weights.b!, x: normedBuf, y: bRawBuf,
                          rows: dims.numValueHeads, cols: D, batch: tk)
        } else {
            matVecInt4Batched(enc, w: weights.qkvTV, xF32: normedBuf, xF16: normedBufF16,
                              yF16: qkvBufF16, yF32: qkvBuf, rows: dims.convDim, cols: D, tk: tk)
            matVecInt4Batched(enc, w: weights.zTV, xF32: normedBuf, xF16: normedBufF16,
                              yF16: zBufF16, yF32: zBuf, rows: valueDim, cols: D, tk: tk)
            matVecInt4Batched(enc, w: weights.aTV, xF32: normedBuf, xF16: normedBufF16,
                              yF16: aRawBufF16, yF32: aRawBuf, rows: dims.numValueHeads, cols: D, tk: tk)
            matVecInt4Batched(enc, w: weights.bTV, xF32: normedBuf, xF16: normedBufF16,
                              yF16: bRawBufF16, yF32: bRawBuf, rows: dims.numValueHeads, cols: D, tk: tk)
        }

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
        var convDimU32 = UInt32(dims.convDim)
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
            enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 7)
            enc.setBytes(&convDimU32, length: MemoryLayout<UInt32>.size, index: 8)
            dispatch(enc, psoQKNormExpandB, threads: expandedKeyDim * tk)
        }
        // convOutBuf is token-major [q(keyDim), k(keyDim), v(valueDim)] with
        // stride convDim; q starts at sub-offset 0, k at sub-offset keyDim —
        // a per-call constant, not multiplied by the batch count.
        qkNorm(sourceBuf: convOutBuf, sourceOffset: 0,
               outBuf: qExpBuf,
               scale: DeltaNetQKNorm.qScale(headKDim: dims.headKDim))
        qkNorm(sourceBuf: convOutBuf,
               sourceOffset: keyDim * MemoryLayout<Float>.size,
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
        // convOutBuf is token-major [q,k,v] with stride convDim, so v's
        // per-call start offset is a constant (skip q+k of token 0), and the
        // kernel needs convDim as v's row stride between tokens.
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
        enc.setBytes(&k32, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&convDimU32, length: MemoryLayout<UInt32>.size, index: 11)
        dispatch(enc, psoRecurrenceB, threads: valueDim)

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
        if DeltaNetMetalBlock.LayerWeights.useFP32 {
            matVecBatched(enc, w: weights.out!, x: gatedBuf, y: deltaOutBuf,
                          rows: D, cols: valueDim, batch: tk)
        } else {
            matVecInt4Batched(enc, w: weights.outTV, xF32: gatedBuf, xF16: gatedBufF16,
                              yF16: deltaOutBufF16, yF32: deltaOutBuf, rows: D, cols: valueDim, tk: tk)
        }

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

    /// The `deltaOut` for one token of a completed `encodeBatched` call, for
    /// tests that diff this path against K sequential `DeltaNetCPUBlock.forward`
    /// calls. Valid only after the encoding command buffer has completed.
    func batchedDeltaOut(token: Int) -> [Float] {
        let ptr = deltaOutBuf.contents().assumingMemoryBound(to: Float.self)
        let base = token * D
        return (0..<D).map { ptr[base + $0] }
    }
}
