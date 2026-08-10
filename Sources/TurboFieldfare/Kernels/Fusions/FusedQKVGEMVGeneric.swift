import Foundation
import Metal

/// Fused Q/K/V projection at 5- or 8-bit, group-size generic.
///
/// Wraps `dequant_int5_qkv_gemv_simd` / `dequant_int8_qkv_gemv_simd`, which are
/// two instantiations of one templated Metal body. "Generic" here means what it
/// means elsewhere on this branch (`dequant_int4_gemv_generic`,
/// `moe_phase1_gate_up_act_u16load_generic`): group size is a runtime argument
/// rather than a baked-in 64.
///
/// A sibling of `FusedQKVGEMV` rather than a `bits:` parameter on it. That type
/// specializes on Gemma's two known decode shapes via function constants and is
/// on the only path that currently runs end to end; leaving it byte-for-byte
/// alone is the regression argument, the same one that shaped the group-128
/// work. When a checkpoint needs several widths in one model — Laguna does,
/// 5-bit attention on 20 layers and 8-bit on 28 — the caller picks per layer
/// from `ManifestQuantSlot.resolved(atLayer:)`, which is where that decision
/// belongs.
///
/// Unlike the int4 wrapper this takes `groupSize` at runtime and does not
/// specialize on shape: Laguna's attention is group 64 but its 8-bit shared
/// experts are group 128, and there is no measured shape set worth baking in.
final class FusedQKVGEMVGeneric {
    /// Which packed width this instance dispatches. Both are served by the same
    /// templated Metal body, so the only thing that varies is the entry point —
    /// hence one wrapper rather than two near-identical ones.
    enum Bits: Int {
        case five = 5
        case eight = 8

        var functionName: String {
            switch self {
            case .five:  return "dequant_int5_qkv_gemv_simd"
            case .eight: return "dequant_int8_qkv_gemv_simd"
            }
        }
    }

    private static let rowsPerThreadgroup = 8

    let bits: Bits
    private let pso: MTLComputePipelineState

    init(context: MetalContext, bits: Bits = .five) throws {
        self.bits = bits
        self.pso = try context.pipeline(bits.functionName,
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: 512)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                qScales: MTLBuffer, qScalesOffset: Int = 0,
                qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                kScales: MTLBuffer, kScalesOffset: Int = 0,
                kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                vScales: MTLBuffer, vScalesOffset: Int = 0,
                vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                qOut: MTLBuffer, qOutOffset: Int = 0,
                kOut: MTLBuffer, kOutOffset: Int = 0,
                vOut: MTLBuffer, vOutOffset: Int = 0,
                qRows: UInt32,
                kvRows: UInt32,
                n: UInt32,
                groupSize: UInt32,
                outputTokenStride: Int = 1) {
        precondition(n % groupSize == 0, "N must be a multiple of groupSize")
        precondition(groupSize % 32 == 0,
                     "the strided-lane loop needs a groupSize that is a multiple of 32")
        // At 5 bits a code straddles bytes, so a row's packed data only starts
        // on a byte boundary when N * bits is a multiple of 8. Every group size
        // this supports is a multiple of 32, which makes group and row starts
        // whole bytes at any width — but the offsets still have to be
        // byte-addressable, which is what this checks.
        precondition(qWeightsOffset >= 0 && kWeightsOffset >= 0 && vWeightsOffset >= 0,
                     "buffer offsets must be non-negative")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: xOffset, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        var gVar = groupSize
        var strideVar = UInt32(outputTokenStride > 0 ? outputTokenStride : 1)
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        enc.setBytes(&gVar, length: MemoryLayout<UInt32>.size, index: 16)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 17)

        let totalRows = Int(qRows) + 2 * Int(kvRows)
        let tgCount = (totalRows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        enc.dispatchThreadgroups(
            MTLSize(width: tgCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
