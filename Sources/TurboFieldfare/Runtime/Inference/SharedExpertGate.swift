import Foundation
import Metal

/// Qwen3.5-MoE (`qwen3_5_moe`, the Qwen3.6 checkpoint's architecture) gates the
/// shared expert by a per-token scalar:
///
///     shared_out = sigmoid(mlp.shared_expert_gate · h_norm) * shared_expert(h_norm)
///
/// `mlp.shared_expert_gate.weight` is a single `[1 x hiddenSize]` row, quantized
/// int8-affine with group size 64 (see `config.json`'s per-tensor `quantization`
/// overrides — the same treatment as `mlp.gate`, the router).
///
/// Gemma has no such gate, so this applies to the `.qwen36` topology only.
/// ADR-0001: computed in plain Swift fp32 — it is one dot product per layer per
/// token, far below the noise floor of the 35B model's real cost.
struct SharedExpertGateWeights {
    private let model: Model
    private let hiddenSize: Int
    private var cache: [Int: [Float]] = [:]

    init(model: Model, hiddenSize: Int) {
        self.model = model
        self.hiddenSize = hiddenSize
    }

    /// Dequantized `[hiddenSize]` gate row for `layer`, cached for the session.
    mutating func weight(layer L: Int) throws -> [Float] {
        if let w = cache[L] { return w }
        let tv = try model.resident(
            name: "language_model.model.layers.\(L).mlp.shared_expert_gate.weight")
        let w = Self.dequantAffineInt8Row(tv, cols: hiddenSize)
        cache[L] = w
        return w
    }

    /// `sigmoid(weight · x)` where `x` is the FP16 post-attention-normed hidden.
    static func gateValue(weight: [Float], x: UnsafePointer<Float16>, count: Int) -> Float {
        var acc: Float = 0
        for i in 0..<count { acc += weight[i] * Float(x[i]) }
        return 1 / (1 + exp(-acc))
    }

    private static func dequantAffineInt8Row(_ tv: TensorView, cols: Int) -> [Float] {
        let groupSize = 64
        let groups = cols / groupSize
        let base = tv.buffer.contents()
        let wPtr = base.advanced(by: Int(tv.offset)).assumingMemoryBound(to: UInt8.self)
        let sPtr = base.advanced(by: Int(tv.scaleOffset)).assumingMemoryBound(to: UInt16.self)
        let bPtr = base.advanced(by: Int(tv.biasOffset)).assumingMemoryBound(to: UInt16.self)
        var out = [Float](repeating: 0, count: cols)
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(sPtr[g])
            let bias = Quantization.bf16ToFloat(bPtr[g])
            for k in 0..<groupSize {
                let c = g * groupSize + k
                out[c] = Float(wPtr[c]) * scale + bias
            }
        }
        return out
    }
}
