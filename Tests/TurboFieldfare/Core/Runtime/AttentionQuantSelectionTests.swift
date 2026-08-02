import Testing
import Foundation
@testable import TurboFieldfare

/// The forward pass now picks a QKV kernel per layer from the manifest.
/// These cover the selection itself — which layer gets which width, and what
/// happens for checkpoints that predate the field — because that is where the
/// mistakes are. The kernels themselves are covered by
/// `FusedQKVGEMVGenericTests`.
@Suite struct AttentionQuantSelectionTests {

    private static func slot(weightBits: Int,
                             groupSize: Int = 64,
                             perLayer: [(Int, Int, Int)]? = nil) throws -> ManifestQuantSlot {
        var dict: [String: Any] = [
            "weightBits": weightBits,
            "scheme": "affine",
            "scaleType": "bf16",
            "biasType": "bf16",
            "groupSize": groupSize,
        ]
        if let perLayer {
            dict["perLayer"] = perLayer.map { layer, bits, group in
                ["layer": layer, "weightBits": bits, "groupSize": group] as [String: Any]
            }
        }
        let data = try JSONSerialization.data(withJSONObject: dict)
        return try JSONDecoder().decode(ManifestQuantSlot.self, from: data)
    }

    /// Gemma: one width everywhere. Every layer must resolve to 4-bit group 64,
    /// which is what keeps it on the original `FusedQKVGEMV` dispatch.
    @Test func uniformSlotResolvesTheSameAtEveryLayer() throws {
        let s = try Self.slot(weightBits: 4)
        for layer in 0..<30 {
            #expect(s.resolved(atLayer: layer).weightBits == 4)
            #expect(s.resolved(atLayer: layer).groupSize == 64)
        }
        #expect(s.distinctConfigurations.count == 1)
    }

    /// Laguna's real attention layout: 5-bit on 20 named layers, 8-bit on the
    /// remaining 28. Both the per-layer answers and the set of pipelines a
    /// runner would have to build are checked, since the runner builds from
    /// `distinctConfigurations` but dispatches from `resolved(atLayer:)` — a
    /// disagreement between those two is exactly the bug that would trap at
    /// decode time.
    @Test func lagunaAttentionResolvesFiveAndEightBitByLayer() throws {
        let fiveBit = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                       31, 33, 34, 37, 38, 41, 42]
        let s = try Self.slot(weightBits: 8,
                              perLayer: fiveBit.map { ($0, 5, 64) })
        var counts: [Int: Int] = [:]
        for layer in 0..<48 {
            counts[s.resolved(atLayer: layer).weightBits, default: 0] += 1
        }
        #expect(counts[5] == 20)
        #expect(counts[8] == 28)
        for layer in fiveBit {
            #expect(s.resolved(atLayer: layer).weightBits == 5, "layer \(layer)")
        }
        #expect(s.resolved(atLayer: 2).weightBits == 8)

        // Every width dispatch can ask for must be one the runner built.
        let built = Set(s.distinctConfigurations.map(\.weightBits))
        let dispatched = Set((0..<48).map { s.resolved(atLayer: $0).weightBits })
        #expect(dispatched.isSubset(of: built))
        #expect(built == [5, 8])
    }

    /// A manifest with no `quant` block at all is the pre-quant-metadata case.
    /// It has to keep landing on 4-bit group 64 at every layer, or the runner
    /// would look for a wide pipeline it never built and trip the
    /// `preconditionFailure` that guards unknown widths.
    @Test func missingQuantBlockFallsBackToFourBitGroup64() {
        for layer in [0, 1, 17, 47] {
            let q = Model.attentionQuant(nil, atLayer: layer)
            #expect(q.weightBits == 4, "layer \(layer)")
            #expect(q.groupSize == Quantization.groupSize, "layer \(layer)")
        }
    }

    /// And when a quant block *is* present, the same entry point must defer to
    /// it rather than keep returning the fallback.
    @Test func presentQuantBlockOverridesTheFallback() throws {
        let s = try Self.slot(weightBits: 8, groupSize: 64, perLayer: [(3, 5, 64)])
        #expect(Model.attentionQuant(s, atLayer: 3).weightBits == 5)
        #expect(Model.attentionQuant(s, atLayer: 4).weightBits == 8)
    }

    /// Only widths with a kernel may appear. 6-bit dequant exists but no fused
    /// QKV kernel is instantiated at 6, so `Bits` must not claim to cover it —
    /// otherwise the runner would build a pipeline for a function that does not
    /// exist and fail at init with a confusing Metal error instead of the
    /// explicit one.
    @Test func onlyImplementedWidthsAreConstructible() {
        #expect(FusedQKVGEMVGeneric.Bits(rawValue: 5) == .five)
        #expect(FusedQKVGEMVGeneric.Bits(rawValue: 8) == .eight)
        #expect(FusedQKVGEMVGeneric.Bits(rawValue: 6) == nil)
        #expect(FusedQKVGEMVGeneric.Bits(rawValue: 4) == nil)
        #expect(FusedQKVGEMVGeneric.Bits.five.functionName == "dequant_int5_qkv_gemv_simd")
        #expect(FusedQKVGEMVGeneric.Bits.eight.functionName == "dequant_int8_qkv_gemv_simd")
    }
}
