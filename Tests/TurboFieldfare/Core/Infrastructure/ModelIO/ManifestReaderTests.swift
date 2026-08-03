import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ManifestReaderTests {

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .gemma4Toy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    /// - Parameter perLayerOverrides: slot name -> `[(layer, bits, groupSize)]`.
    ///   Omitted entirely when empty, so the default output is byte-identical
    ///   to what this helper produced before per-layer quant existed — which is
    ///   what makes every other test in this file a backward-compat guard.
    static func quant(sharedExpertBits: Int = 4,
                      routerBits: Int = 8,
                      groupSizeOverrides: [String: Int] = [:],
                      perLayerOverrides: [String: [(Int, Int, Int)]] = [:]) -> [String: Any] {
        func slot(_ name: String, _ bits: Int) -> [String: Any] {
            var s: [String: Any] = [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": groupSizeOverrides[name] ?? Quantization.groupSize,
            ]
            if let overrides = perLayerOverrides[name] {
                s["perLayer"] = overrides.map { layer, b, g in
                    ["layer": layer, "weightBits": b, "groupSize": g] as [String: Any]
                }
            }
            return s
        }
        return [
            "embedding": slot("embedding", 4),
            "attention": slot("attention", 4),
            "router": slot("router", routerBits),
            "sharedExpert": slot("sharedExpert", sharedExpertBits),
            "routedExpert": slot("routedExpert", 4),
        ]
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    /// `routedExpert` joined this set once the generic MoE kernels
    /// (`moe_phase1_gate_up_act_u16load_generic` and friends) learned group
    /// 128 — it is Laguna's routed-expert layout, and the whole point of that
    /// kernel work.
    @Test func productionManifestAcceptsGroup128ForInt4GenericSlots() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(groupSizeOverrides: ["embedding": 128,
                                                      "attention": 128,
                                                      "routedExpert": 128])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.embedding.groupSize == 128)
        #expect(manifest.quant?.attention.groupSize == 128)
        #expect(manifest.quant?.routedExpert.groupSize == 128)
    }

    /// `router` is 8-bit (`router_gemv_gemma4_r4`). Accepting 128 would load
    /// cleanly and then produce silently wrong numbers, so the rejection is the
    /// feature. Relax a slot only in the same change that teaches its kernel group 128.
    @Test(arguments: ["router"])
    func productionManifestRejectsGroup128ForSlotsWhoseKernelIsGroup64(
        _ slotName: String
    ) throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(groupSizeOverrides: [slotName: 128])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains(slotName)
        }
    }

    // MARK: - Per-layer quantization

    /// A manifest written before per-layer quant existed has no `perLayer` key
    /// at all. It must still decode, with the slot reporting its scalar width
    /// at every layer — otherwise this format change bricks installed models.
    @Test func manifestWithoutPerLayerKeyDecodesAsUniform() throws {
        let (dir, config) = try Self.writeToyManifest(["quant": Self.quant()],
                                                      config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        let attention = try #require(manifest.quant?.attention)
        #expect(attention.perLayer == nil)
        for layer in [0, 7, 29] {
            #expect(attention.resolved(atLayer: layer).weightBits == 4)
            #expect(attention.resolved(atLayer: layer).groupSize == Quantization.groupSize)
        }
        #expect(attention.distinctConfigurations.count == 1)
    }

    /// The shape Laguna needs: one slot, two widths, selected by layer. Uses
    /// group 128 on the overridden layers so the values stay inside what the
    /// int4-generic kernels actually decode — the point is the *mechanism*,
    /// and a guard-clearing fixture keeps this test about that.
    @Test func perLayerOverrideSelectsWidthByLayer() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(perLayerOverrides: ["attention": [(3, 4, 128), (5, 4, 128)]])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        let attention = try #require(manifest.quant?.attention)
        #expect(attention.resolved(atLayer: 3).groupSize == 128)
        #expect(attention.resolved(atLayer: 5).groupSize == 128)
        // Unlisted layers fall back to the slot default, not to the override.
        #expect(attention.resolved(atLayer: 4).groupSize == Quantization.groupSize)
        #expect(attention.resolved(atLayer: 0).groupSize == Quantization.groupSize)
        #expect(attention.distinctConfigurations.count == 2)
    }

    /// Validation has to clear every configuration a slot can present, not the
    /// scalar default alone. A slot whose default is fine but whose override
    /// names an unsupported width would otherwise load cleanly and then hand
    /// the kernels bits they cannot decode.
    @Test func perLayerOverrideWithUnsupportedWidthIsRejected() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(perLayerOverrides: ["attention": [(3, 5, 64)]])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("attention") && detail.contains("5-bit")
        }
    }

    /// Same rule for group size: `router` is group-64-only, and an
    /// override must not be a side door around the slot-level gate.
    @Test func perLayerOverrideCannotBypassAGroupSizeGate() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(perLayerOverrides: ["router": [(1, 4, 128)]])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("router")
        }
    }

    /// Two entries for the same layer are two different answers to "what is the
    /// width here". `resolved(atLayer:)` would quietly take the first; rejecting
    /// is the honest response.
    @Test func duplicatePerLayerOverrideIsRejected() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(perLayerOverrides: ["attention": [(3, 4, 128), (3, 4, 64)]])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("duplicate per-layer quant override")
        }
    }

    /// Laguna's actual attention slot, as a whole: 5-bit on 20 layers, 8-bit on
    /// 28. The format can now *express* it; the kernels still cannot decode
    /// 5-bit attention, so loading must still fail — and fail naming the width,
    /// not with the old "cannot be represented" silence.
    @Test func lagunaMixedAttentionIsExpressibleButStillRejectedByTheKernelGate() throws {
        let fiveBitLayers = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(
                groupSizeOverrides: ["attention": 64],
                perLayerOverrides: ["attention": fiveBitLayers.map { ($0, 5, 64) }])],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("attention") && detail.contains("5-bit")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func missingLayerFileThrowsMissingFile() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            // intentionally do not list layer_0.bin or layer_1.bin
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 2 layers (both full), hidden 64,
    /// vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }
}
