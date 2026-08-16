import Foundation
import Metal

/// P8-2: GPU kernels for Qwen3.6 gated attention, replacing CPU readback
/// points that forced 3-CB splits per full-attention layer.
///
/// `gated_qkv_split`: de-interleave [heads, 2*headDim] into separate Q and gate
/// `gated_attn_output_sigmoid`: attnOut *= sigmoid(gate) elementwise
final class GatedAttentionFusion {
    private let splitPSO: MTLComputePipelineState
    private let sigmoidPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.splitPSO = try context.pipeline("gated_qkv_split")
        self.sigmoidPSO = try context.pipeline("gated_attn_output_sigmoid")
    }

    /// De-interleave gated QKV output into separate Q and gate buffers.
    ///
    /// Input: qRaw [heads, 2*headDim] FP16, interleaved [q, gate] per head.
    /// Output: qOut [heads, headDim], gateOut [heads, headDim].
    func encodeSplit(
        commandBuffer: MTLCommandBuffer,
        qRaw: MTLBuffer,
        qRawOffset: Int = 0,
        qOut: MTLBuffer,
        qOutOffset: Int = 0,
        gateOut: MTLBuffer,
        gateOutOffset: Int = 0,
        numHeads: UInt32,
        headDim: UInt32
    ) {
        let total = Int(numHeads) * Int(headDim)
        guard total > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(splitPSO)
        encoder.setBuffer(qRaw, offset: qRawOffset, index: 0)
        encoder.setBuffer(qOut, offset: qOutOffset, index: 1)
        encoder.setBuffer(gateOut, offset: gateOutOffset, index: 2)
        var hVar = numHeads
        var dVar = headDim
        encoder.setBytes(&hVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)

        let tgSize = min(total, 256)
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Apply sigmoid gate to attention output: attnOut *= sigmoid(gate).
    ///
    /// Both buffers are [qDim] FP16 where qDim = numHeads * headDim.
    func encodeOutputGate(
        commandBuffer: MTLCommandBuffer,
        attnOut: MTLBuffer,
        attnOutOffset: Int = 0,
        gate: MTLBuffer,
        gateOffset: Int = 0,
        count: UInt32
    ) {
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(sigmoidPSO)
        encoder.setBuffer(attnOut, offset: attnOutOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var cVar = count
        encoder.setBytes(&cVar, length: MemoryLayout<UInt32>.size, index: 2)

        let total = Int(count)
        let tgSize = min(total, 256)
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
