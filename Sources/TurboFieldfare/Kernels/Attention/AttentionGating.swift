import Foundation
import Metal

/// Swift wrapper for attention gating Metal kernels.
final class AttentionGatingKernel {
    private let psoPerHead: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoPerHead = try context.pipeline("apply_attention_gating_per_head")
    }

    /// Encodes per-head attention gating in-place on `attnOut`.
    ///
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer.
    ///   - attnOut: Attention output buffer `[tokens, numHeads, headDim]` (FP16).
    ///   - attnOutOffset: Byte offset in `attnOut`.
    ///   - gOut: Per-head gate projection output buffer `[tokens, numHeads]` (FP16).
    ///   - gOutOffset: Byte offset in `gOut`.
    ///   - tokens: Number of tokens (`t`).
    ///   - numHeads: Number of query attention heads (`h`).
    ///   - headDim: Head dimension (`d`).
    func encodePerHead(
        commandBuffer: MTLCommandBuffer,
        attnOut: MTLBuffer,
        attnOutOffset: Int = 0,
        gOut: MTLBuffer,
        gOutOffset: Int = 0,
        tokens: UInt32,
        numHeads: UInt32,
        headDim: UInt32
    ) {
        guard tokens > 0, numHeads > 0, headDim > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(psoPerHead)
        encoder.setBuffer(attnOut, offset: attnOutOffset, index: 0)
        encoder.setBuffer(gOut, offset: gOutOffset, index: 1)
        var tVar = tokens
        var hVar = numHeads
        var dVar = headDim
        encoder.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&hVar, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)

        let gridSize = MTLSize(width: Int(headDim), height: Int(numHeads), depth: Int(tokens))
        let threadgroupSize = MTLSize(
            width: min(Int(headDim), 256),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
