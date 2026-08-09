import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct PrefillAttentionQuantTests {

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

    /// Tests that prefill attention quantization resolution maps correctly per layer
    /// for mixed 5-bit and 8-bit checkpoints as used in Laguna.
    @Test func prefillAttentionQuantizationResolvesCorrectlyByLayer() throws {
        let fiveBitLayers = [0, 1, 3, 4, 5, 6]
        let slot = try Self.slot(weightBits: 8, perLayer: fiveBitLayers.map { ($0, 5, 64) })

        for layer in 0..<12 {
            let (bits, groupSize) = Model.attentionQuant(slot, atLayer: layer)
            if fiveBitLayers.contains(layer) {
                #expect(bits == 5, "Layer \(layer) should resolve to 5-bit")
            } else {
                #expect(bits == 8, "Layer \(layer) should resolve to 8-bit")
            }
            #expect(groupSize == 64)
        }
    }

    /// Tests fallback for missing quant block in prefill.
    @Test func prefillAttentionQuantizationFallbackIsFourBitGroup64() {
        let (bits, groupSize) = Model.attentionQuant(nil, atLayer: 0)
        #expect(bits == 4)
        #expect(groupSize == 64)
    }
}
