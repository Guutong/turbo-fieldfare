import Foundation

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]

    // Optional so manifests written before these fields existed still decode.
    // Absent means "Gemma's behaviour", matching the `ArchConfig` defaults.
    public let headsPerLayer: [Int]?
    public let denseMLPLayerMask: [Int]?
    public let denseMLPIntermediateSize: Int?
    public let fullPartialRotaryFactor: Double?
    public let fullRopeScaling: ManifestRopeScaling?
    public let attentionGating: String?
    public let routedScalingFactor: Double?
}

public struct ManifestRopeScaling: Decodable, Equatable, Sendable {
    public let factor: Double
    public let originalMaxPositionEmbeddings: Int
    public let betaFast: Double
    public let betaSlow: Double
    public let attentionFactor: Double

    var asRopeScaling: RopeScaling {
        RopeScaling(factor: factor,
                    originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
                    betaFast: betaFast,
                    betaSlow: betaSlow,
                    attentionFactor: attentionFactor)
    }
}

extension ManifestArch {
    /// Build the `ArchConfig` this manifest describes. The manifest's arch
    /// block is authoritative; when no separate expected config is provided
    /// at load time, this becomes both the actual and the expected, so the
    /// arch-validation gate passes trivially for a correctly-written manifest.
    var asArchConfig: ArchConfig {
        let headsPerLayer = headsPerLayer ?? []
        return ArchConfig(
            hiddenSize: hiddenSize,
            intermediateSize: ffnIntermediate,
            moeIntermediateSize: moeIntermediateSize,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            numFullKVHeads: numFullKVHeads,
            headDim: headDim,
            fullHeadDim: fullHeadDim,
            vocabSize: vocabSize,
            slidingWindow: slidingWindow,
            finalLogitSoftcap: finalLogitSoftcap,
            ropeTheta: ropeTheta,
            fullRopeTheta: fullRopeTheta,
            partialRotaryFactor: partialRotaryFactor,
            numLayers: numLayers,
            numExperts: numExperts,
            topKExperts: topKExperts,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionKEqV: attentionKEqV,
            fullAttentionLayerMask: fullAttentionLayerMask.map { UInt8($0) },
            hiddenActivation: hiddenActivation,
            headsPerLayer: headsPerLayer,
            denseMLPLayerMask: denseMLPLayerMask?.map { UInt8($0) } ?? [],
            denseMLPIntermediateSize: denseMLPIntermediateSize ?? 0,
            fullPartialRotaryFactor: fullPartialRotaryFactor,
            fullRopeScaling: fullRopeScaling?.asRopeScaling,
            attentionGating: AttentionGating(rawValue: attentionGating ?? "none") ?? .none,
            routedScalingFactor: routedScalingFactor ?? 1.0
        )
    }
}

/// One layer's exception to a slot's `weightBits`/`groupSize`.
///
/// Only the two fields that vary are overridable. `scheme`/`scaleType`/
/// `biasType` stay on the slot because no known checkpoint varies them per
/// layer, and a field that can differ is a field every reader has to check.
public struct ManifestQuantLayerOverride: Decodable, Equatable, Sendable {
    public let layer: Int
    public let weightBits: Int
    public let groupSize: Int
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
    /// Sparse per-layer exceptions, or nil when the slot is uniform.
    ///
    /// Some checkpoints quantize per *layer*, not per role. Laguna-S-2.1
    /// quantizes attention at 5 bits on 20 layers and 8 bits on the other 28,
    /// which a single `weightBits` cannot describe at all. The shape here
    /// mirrors the upstream `config.json`: scalar defaults plus a sparse
    /// override table, so the common uniform case costs nothing.
    ///
    /// Optional on purpose — existing manifests have no such key and decode
    /// with this nil, which is what keeps already-installed `.gturbo` files
    /// readable byte-for-byte.
    public let perLayer: [ManifestQuantLayerOverride]?

    /// The `(weightBits, groupSize)` in effect at `layer`, which is the slot's
    /// own values unless an override names that layer.
    public func resolved(atLayer layer: Int) -> (weightBits: Int, groupSize: Int) {
        if let o = perLayer?.first(where: { $0.layer == layer }) {
            return (o.weightBits, o.groupSize)
        }
        return (weightBits, groupSize)
    }

    /// Every distinct `(weightBits, groupSize)` this slot can present, which is
    /// what validation has to clear — checking only the scalar would admit a
    /// manifest whose overrides name a width no kernel implements.
    public var distinctConfigurations: [(weightBits: Int, groupSize: Int)] {
        var seen: [(weightBits: Int, groupSize: Int)] = [(weightBits, groupSize)]
        for o in perLayer ?? [] where !seen.contains(where: {
            $0.weightBits == o.weightBits && $0.groupSize == o.groupSize
        }) {
            seen.append((o.weightBits, o.groupSize))
        }
        return seen
    }
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert"
    ]

    /// Required file entries (relative to `model.gturbo/`). Layer files
    /// `packed_experts/layer_<L>.bin` for L in 0..<numLayers are checked
    /// after decode against `numLayers` (with the zero-padded "layer_%02d"
    /// naming the writer produces; falling back to plain "layer_<L>" when
    /// only the unpadded form is present, for toy synthetics).
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig?,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        // When no external expected config is provided, use the manifest's own
        // arch — the manifest is authoritative about what model it contains.
        // A caller that passes a specific config (e.g. for a service that only
        // runs one model) still gets the full archMismatch gate.
        let expected = expecting ?? manifest.arch.asArchConfig
        try validate(manifest, against: expected,
                     directoryURL: directoryURL)
        return manifest
    }

    private static func metadataFileSize(_ url: URL,
                                         fileName: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "\(fileName): file size unavailable")
        }
        return number.uint64Value
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig,
                         directoryURL: URL) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        for L in 0..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant) throws {
        // Group sizes are allowed per slot, not globally, because each slot is
        // read by a different kernel and they do not all handle 128 yet.
        // `dequant_int4_gemv_generic` strides by lane and so takes any
        // multiple of 32; `routedExpert` is decoded by the group-size-generic
        // MoE kernels (`moe_phase1_gate_up_act_u16load_generic` /
        // `moe_phase1_gate_up_act_subset_u16load_generic` /
        // `moe_phase2_down_reduce_generic` in Metal/MoE/moe.metal), which now
        // handle 128 the same way. `router` and `sharedExpert` still feed the
        // group-64-only kernels (`router_gemv_gemma4_r4`,
        // `moe_int4_gemv_row_simd_dev_vec` for the shared MLP path) — widen
        // those together with their kernels, not before. Admitting 128 for a
        // slot its kernel cannot decode would turn a clean load-time rejection
        // into silently wrong numbers at inference.
        let defaultGroup: Set<Int> = [Quantization.groupSize]
        let int4GenericGroups: Set<Int> = [Quantization.groupSize, 128]
        let slots: [(String, ManifestQuantSlot, Set<Int>, Set<Int>)] = [
            ("embedding", quant.embedding, [4, 8], int4GenericGroups),
            ("attention", quant.attention, [4, 5, 8], int4GenericGroups),
            ("router", quant.router, [4, 8], int4GenericGroups),
            ("sharedExpert", quant.sharedExpert, [4, 8], int4GenericGroups),
            ("routedExpert", quant.routedExpert, [4], int4GenericGroups),
        ]
        for (name, slot, allowedBits, allowedGroupSizes) in slots {
            guard slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16" else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
            // Every configuration the slot can present has to clear the gate,
            // not just the scalar default. A per-layer override is a value the
            // kernels will actually be handed, so admitting the slot on its
            // default alone would let an override smuggle in a bit width or
            // group size nothing can decode — exactly the silent-wrong-numbers
            // trade this guard exists to prevent.
            for config in slot.distinctConfigurations {
                guard allowedBits.contains(config.weightBits),
                      allowedGroupSizes.contains(config.groupSize) else {
                    throw ModelError.indexCorrupt(
                        detail: "unsupported quantization for \(name): "
                            + "\(config.weightBits)-bit group \(config.groupSize)")
                }
            }
            // A duplicate layer entry means two different answers to "what is
            // the width at layer N", and `resolved(atLayer:)` would silently
            // take the first. Reject rather than pick.
            let layers = (slot.perLayer ?? []).map(\.layer)
            guard Set(layers).count == layers.count else {
                throw ModelError.indexCorrupt(
                    detail: "duplicate per-layer quant override for \(name)")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)

        // Fields absent from older manifests default to the `ArchConfig`
        // defaults, so a Gemma manifest written before they existed still
        // validates against `ArchConfig.gemma4_26B_A4B` unchanged.
        try check("headsPerLayer",
                  (a.headsPerLayer ?? []).description,
                  e.headsPerLayer.description)
        try check("denseMLPLayerMask",
                  (a.denseMLPLayerMask ?? []).map { UInt8($0) }.description,
                  e.denseMLPLayerMask.description)
        try check("denseMLPIntermediateSize",
                  a.denseMLPIntermediateSize ?? 0,
                  e.denseMLPIntermediateSize)
        try check("fullPartialRotaryFactor",
                  (a.fullPartialRotaryFactor.map { "\($0)" } ?? "nil"),
                  (e.fullPartialRotaryFactor.map { "\($0)" } ?? "nil"))
        try check("fullRopeScaling",
                  (a.fullRopeScaling?.asRopeScaling).map { "\($0)" } ?? "nil",
                  e.fullRopeScaling.map { "\($0)" } ?? "nil")
        try check("attentionGating",
                  a.attentionGating ?? AttentionGating.none.rawValue,
                  e.attentionGating.rawValue)
        try check("routedScalingFactor",
                  a.routedScalingFactor ?? 1.0,
                  e.routedScalingFactor)

        // Per-layer arrays, when present, must cover every layer.
        if let heads = a.headsPerLayer, !heads.isEmpty, heads.count != a.numLayers {
            throw ModelError.archMismatch(field: "headsPerLayer.count",
                                          expected: "\(a.numLayers)",
                                          actual: "\(heads.count)")
        }
        if let dense = a.denseMLPLayerMask, !dense.isEmpty, dense.count != a.numLayers {
            throw ModelError.archMismatch(field: "denseMLPLayerMask.count",
                                          expected: "\(a.numLayers)",
                                          actual: "\(dense.count)")
        }
    }
}
