import Foundation
import Metal

/// P3-4: wires the plain-Swift-fp32 DeltaNet block (proven correct in
/// isolation by P3-3's `Layer0IsolationNumericTests`) into the real
/// `RealForwardRunner` decode loop.
///
/// This deliberately mirrors, line for line, the DeltaNet chain in
/// `Layer0IsolationNumericTests.swift` (input RMSNorm -> qkv/z/a/b
/// projections -> causal conv1d -> QK-norm -> head expansion -> beta/g gates
/// -> gated-delta recurrence -> gated RMSNorm/swiglu output -> out_proj). It
/// does NOT include the residual add or the post-attention norm/MoE — those
/// stay in `RealForwardRunner`, matching how full-attention layers work.
enum DeltaNetCPUBlock {

    // MARK: - Dequantized per-layer weights, cached for the life of the runner.

    /// One DeltaNet layer's weights, fully dequantized to Float32 once and
    /// reused across every token in the session. ADR-0001: obviously-correct
    /// over fast — re-dequantizing per token would be needlessly slow AND
    /// would not change the numeric story, since the weights never change
    /// mid-generation.
    final class LayerWeights {
        let inputNorm: [Float]
        let qkv: [Float]      // [convDim x D]
        let z: [Float]        // [valueDim x D]
        let a: [Float]        // [numValueHeads x D]
        let b: [Float]        // [numValueHeads x D]
        let out: [Float]      // [D x valueDim]
        let conv: [Float]     // [convDim * 4]
        let deltaNorm: [Float] // [headVDim]
        let aLog: [Float]     // [numValueHeads]
        let dtBias: [Float]   // [numValueHeads]

        init(model: Model, layer L: Int, D: Int, dims: DeltaNetDimensions) throws {
            let convDim = dims.convDim
            let keyDim = dims.numKeyHeads * dims.headKDim
            let valueDim = dims.numValueHeads * dims.headVDim
            precondition(convDim == 2 * keyDim + valueDim)

            inputNorm = Self.bf16Vector(try model.inputNorm(layer: L), count: D)
            qkv = Self.dequantResident(try model.deltaQKVProj(layer: L), rows: convDim, cols: D, bits: 4)
            z = Self.dequantResident(try model.deltaZProj(layer: L), rows: valueDim, cols: D, bits: 4)
            a = Self.dequantResident(try model.deltaAProj(layer: L), rows: dims.numValueHeads, cols: D, bits: 4)
            b = Self.dequantResident(try model.deltaBProj(layer: L), rows: dims.numValueHeads, cols: D, bits: 4)
            out = Self.dequantResident(try model.deltaOutProj(layer: L), rows: D, cols: valueDim, bits: 4)
            conv = Self.bf16Vector(try model.deltaConv1d(layer: L), count: convDim * 4)
            deltaNorm = Self.bf16Vector(try model.deltaNorm(layer: L), count: dims.headVDim)
            aLog = Self.bf16Vector(try model.deltaALog(layer: L), count: dims.numValueHeads)
            dtBias = Self.bf16Vector(try model.deltaDtBias(layer: L), count: dims.numValueHeads)
        }

        // MARK: CPU dequant helpers — identical convention to
        // Layer0IsolationNumericTests's dequantResident/bf16Vector (affine
        // int4, groupSize 64, row-major; bf16 raw-bit read).

        static func dequantResident(_ tv: TensorView, rows: Int, cols: Int, bits: Int) -> [Float] {
            let groupSize = 64
            let groupsPerRow = cols / groupSize
            let base = tv.buffer.contents()
            let scalePtr = base.advanced(by: Int(tv.scaleOffset)).assumingMemoryBound(to: UInt16.self)
            let biasPtr = base.advanced(by: Int(tv.biasOffset)).assumingMemoryBound(to: UInt16.self)
            var result = [Float](repeating: 0, count: rows * cols)
            precondition(bits == 4, "only int4 DeltaNet weights are expected on this path")
            let wPtr = base.advanced(by: Int(tv.offset)).assumingMemoryBound(to: UInt8.self)
            let bytesPerRow = cols / 2
            for r in 0..<rows {
                for g in 0..<groupsPerRow {
                    let scale = Quantization.bf16ToFloat(scalePtr[r * groupsPerRow + g])
                    let bias = Quantization.bf16ToFloat(biasPtr[r * groupsPerRow + g])
                    for k in 0..<groupSize {
                        let c = g * groupSize + k
                        let byte = wPtr[r * bytesPerRow + c / 2]
                        let nibble = (c & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                        result[r * cols + c] = Float(nibble) * scale + bias
                    }
                }
            }
            return result
        }

        static func bf16Vector(_ tv: TensorView, count: Int) -> [Float] {
            let ptr = tv.buffer.contents().advanced(by: Int(tv.offset)).assumingMemoryBound(to: UInt16.self)
            return (0..<count).map { Quantization.bf16ToFloat(ptr[$0]) }
        }
    }

    /// Lazily dequantizes and caches per-layer weights on first use.
    final class WeightsCache: @unchecked Sendable {
        private let model: Model
        private let D: Int
        private let dims: DeltaNetDimensions
        private var cache: [Int: LayerWeights] = [:]

        init(model: Model, hiddenSize: Int, dims: DeltaNetDimensions) {
            self.model = model
            self.D = hiddenSize
            self.dims = dims
        }

        func weights(layer L: Int) throws -> LayerWeights {
            if let w = cache[L] { return w }
            let w = try LayerWeights(model: model, layer: L, D: D, dims: dims)
            cache[L] = w
            return w
        }
    }

    // MARK: - Plain math (matches Layer0IsolationNumericTests exactly).

    private static func matVec(_ w: [Float], rows: Int, cols: Int, _ x: [Float]) -> [Float] {
        precondition(w.count == rows * cols)
        precondition(x.count == cols)
        var y = [Float](repeating: 0, count: rows)
        w.withUnsafeBufferPointer { wp in
            x.withUnsafeBufferPointer { xp in
                for r in 0..<rows {
                    var acc: Float = 0
                    let base = r * cols
                    for c in 0..<cols { acc += wp[base + c] * xp[c] }
                    y[r] = acc
                }
            }
        }
        return y
    }

    private static func rmsNorm(_ x: [Float], weight: [Float], eps: Float = 1e-6) -> [Float] {
        let n = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / sqrt(ss / Float(n) + eps)
        return (0..<n).map { x[$0] * inv * weight[$0] }
    }

    /// One token's DeltaNet forward, returning `deltaOut` (D-dim), the
    /// contribution to be added to the residual stream by the caller — the
    /// same split `Layer0IsolationNumericTests` uses (`h = x + deltaOut`).
    static func forward(x: [Float],
                        weights: LayerWeights,
                        dims: DeltaNetDimensions,
                        convState: inout [Float],
                        recurrentState: inout [Float]) -> [Float] {
        let keyDim = dims.numKeyHeads * dims.headKDim
        let valueDim = dims.numValueHeads * dims.headVDim
        let D = weights.out.count / valueDim

        let a = rmsNorm(x, weight: weights.inputNorm)
        let qkv = matVec(weights.qkv, rows: dims.convDim, cols: D, a)
        let z = matVec(weights.z, rows: valueDim, cols: D, a)
        let aGateRaw = matVec(weights.a, rows: dims.numValueHeads, cols: D, a)
        let bGateRaw = matVec(weights.b, rows: dims.numValueHeads, cols: D, a)

        let convOut = DeltaNetConv.step(input: qkv, weight: weights.conv,
                                        convDim: dims.convDim, convState: &convState)

        var q = Array(convOut[0..<keyDim])
        var k = Array(convOut[keyDim..<(2 * keyDim)])
        let v = Array(convOut[(2 * keyDim)..<(2 * keyDim + valueDim)])

        DeltaNetQKNorm.applyInPlace(&q, numHeads: dims.numKeyHeads, headDim: dims.headKDim,
                                    scale: DeltaNetQKNorm.qScale(headKDim: dims.headKDim))
        DeltaNetQKNorm.applyInPlace(&k, numHeads: dims.numKeyHeads, headDim: dims.headKDim,
                                    scale: DeltaNetQKNorm.kScale(headKDim: dims.headKDim))

        let qExp = DeltaNetHeadExpansion.expand(q, numKeyHeads: dims.numKeyHeads,
                                                numValueHeads: dims.numValueHeads, headDim: dims.headKDim)
        let kExp = DeltaNetHeadExpansion.expand(k, numKeyHeads: dims.numKeyHeads,
                                                numValueHeads: dims.numValueHeads, headDim: dims.headKDim)

        let beta = DeltaNetGate.beta(bGateRaw)
        let g = DeltaNetGate.decay(a: aGateRaw, aLog: weights.aLog, dtBias: weights.dtBias)

        let y = DeltaNetRecurrence.step(q: qExp, k: kExp, v: v, beta: beta, g: g,
                                        numValueHeads: dims.numValueHeads,
                                        headVDim: dims.headVDim, headKDim: dims.headKDim,
                                        state: &recurrentState)

        let gated = DeltaNetOutputGate.apply(y: y, z: z, normWeight: weights.deltaNorm,
                                             numValueHeads: dims.numValueHeads, headVDim: dims.headVDim)

        return matVec(weights.out, rows: D, cols: valueDim, gated)
    }
}
