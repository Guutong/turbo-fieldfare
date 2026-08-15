import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]
    // Group sizes other than 64 (currently 128, for oQ4e's routed experts)
    // go through the generic strided-loop kernel instead of the hand-
    // vectorized group-64 fast path above, so `pipeline`/`specializedPipelines`
    // and their dispatch stay byte-for-byte what they were before group-128
    // support existed.
    private let genericPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines

        self.genericPipeline = try context.pipeline(
            "dequant_int4_gemv_generic",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32,
                groupSize: Int = Quantization.groupSize) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encode(encoder: encoder, weights: weights, weightsOffset: weightsOffset,
               scales: scales, scalesOffset: scalesOffset,
               biases: biases, biasesOffset: biasesOffset,
               x: x, xOffset: xOffset, y: y, yOffset: yOffset,
               m: m, n: n, groupSize: groupSize)
        encoder.endEncoding()
    }

    /// P7-7: encoder-level entry point so callers that already have an open
    /// `MTLComputeCommandEncoder` (e.g. `DeltaNetMetalBlock.encode`, which
    /// keeps every stage of one layer in a single encoder) can dispatch this
    /// kernel without paying for an extra encoder begin/end pair. Same
    /// dispatch as the command-buffer entry point above, which now just
    /// wraps this and calls `endEncoding()` itself — behavior unchanged.
    func encode(encoder: MTLComputeCommandEncoder,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32,
                groupSize: Int = Quantization.groupSize) {
        precondition(n % UInt32(groupSize) == 0,
                     "N must be a multiple of \(groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        // groupSize 64 keeps using the hand-vectorized fast path (identical
        // pipeline objects and dispatch to before group-128 existed); any
        // other groupSize (currently only 128) goes through the generic
        // strided-loop kernel, which takes groupSize as a runtime buffer.
        let useGeneric = groupSize != Quantization.groupSize
        encoder.setComputePipelineState(
            useGeneric ? genericPipeline : (specializedPipelines[Shape(m: m, n: n)] ?? pipeline))
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        if useGeneric {
            var gValue = UInt32(groupSize)
            encoder.setBytes(&gValue, length: MemoryLayout<UInt32>.size, index: 7)
        }

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
    }
}
