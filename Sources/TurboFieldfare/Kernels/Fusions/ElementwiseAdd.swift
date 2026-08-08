import Foundation
import Metal

/// Elementwise FP16 in-place add: a[d] += b[d].
/// Used for raw residuals in pre-norm topologies (Qwen3.6).
final class ElementwiseAdd {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("add_fp16")
    }

    /// a += b, elementwise over `count` half-precision elements.
    func encode(commandBuffer cb: MTLCommandBuffer,
                a: MTLBuffer, aOffset: Int = 0,
                b: MTLBuffer, bOffset: Int = 0,
                count: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(a, offset: aOffset, index: 0)
        enc.setBuffer(b, offset: bOffset, index: 1)
        var c = count
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)
        let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }
}
