import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ManifestReaderLagunaTests {

    @Test func lagunaManifestQuantSpecValidatesSuccessfully() throws {
        let quantDict: [String: Any] = [
            "embedding": [
                "weightBits": 8,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": 64
            ],
            "attention": [
                "weightBits": 4,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": 128
            ],
            "router": [
                "weightBits": 4,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": 64
            ],
            "sharedExpert": [
                "weightBits": 8,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": 128
            ],
            "routedExpert": [
                "weightBits": 4,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": 128
            ]
        ]

        let (dir, config) = try ManifestReaderTests.writeToyManifest(
            ["quant": quantDict],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }

        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        let quant = try #require(manifest.quant)

        #expect(quant.embedding.weightBits == 8)
        #expect(quant.embedding.groupSize == 64)

        #expect(quant.attention.weightBits == 4)
        #expect(quant.attention.groupSize == 128)

        #expect(quant.router.weightBits == 4)
        #expect(quant.router.groupSize == 64)

        #expect(quant.sharedExpert.weightBits == 8)
        #expect(quant.sharedExpert.groupSize == 128)

        #expect(quant.routedExpert.weightBits == 4)
        #expect(quant.routedExpert.groupSize == 128)
    }
}
