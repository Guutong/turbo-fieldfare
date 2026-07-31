import Foundation
import Metal

/// Swift wrapper for `dequant_int6_gemv_simd`.
///
///   y = W * x
///
/// where W (M rows, N cols) is MLX-affine 6-bit (sub-byte packed, see
/// `QuantizationSubByte.swift`) with BF16 scale + BF16 bias per group of
/// `groupSize`, x is FP16 [N], y is FP16 [M].
///
/// Unlike `DequantInt4GEMV`/`DequantInt8GEMV`, group size is not fixed at 64
/// — it's a runtime parameter, since the mixed-precision oQ4e conversion
/// this kernel targets pairs 6-bit layer-0 projections (group_size=64) with
/// other tensors at other group sizes.
///
/// One SIMD group (32 threads) runs per output row; a threadgroup handles
/// 8 rows (one SIMD group each), mirroring the INT8 GEMV's dispatch shape.
final class DequantInt6GEMV {
    private static let rowsPerThreadgroup = 8

    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int6_gemv_simd")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales:  MTLBuffer, scalesOffset:  Int = 0,
                biases:  MTLBuffer, biasesOffset:  Int = 0,
                x:       MTLBuffer, xOffset: Int = 0,
                y:       MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32, groupSize: UInt32) {
        precondition(n % groupSize == 0, "N must be a multiple of groupSize")
        precondition(xOffset >= 0 && yOffset >= 0, "buffer offsets must be non-negative")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales,  offset: scalesOffset,  index: 1)
        enc.setBuffer(biases,  offset: biasesOffset,  index: 2)
        enc.setBuffer(x,       offset: xOffset,       index: 3)
        enc.setBuffer(y,       offset: yOffset,       index: 4)
        var mVar = m
        var nVar = n
        var gVar = groupSize
        enc.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&gVar, length: MemoryLayout<UInt32>.size, index: 7)

        let tgSize  = MTLSize(width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1)
        let tgCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
