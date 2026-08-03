import Foundation

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = "GTURBO"
    static let versionMajor = 1
    static let versionMinor = 0

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
        /// Per-slot group size when it differs from plan.baseGroupSize
        /// (Laguna: attention/embedding/router are g64 while base is 128).
        var slotGroupSizes: [String: Int] = [:]
        /// Per-layer overrides for slots that vary by layer (Laguna attention).
        /// Keyed by slot name: "attention", "sharedExpert", etc.
        var perLayer: [String: [(layer: Int, weightBits: Int, groupSize: Int)]]?
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) }
        ]
        // Emitted only when the model actually differs from Gemma, so a Gemma
        // repack still writes a byte-identical `arch` block and the runtime's
        // optional decoding of these keys stays exercised on both paths.
        if !arch.headsPerLayer.isEmpty {
            archDict["headsPerLayer"] = arch.headsPerLayer
        }
        if !arch.denseMLPLayerMask.isEmpty {
            archDict["denseMLPLayerMask"] = arch.denseMLPLayerMask.map { Int($0) }
            archDict["denseMLPIntermediateSize"] = arch.denseMLPIntermediateSize
        }
        if let prf = arch.fullPartialRotaryFactor {
            archDict["fullPartialRotaryFactor"] = prf
        }
        if let s = arch.fullRopeScaling {
            archDict["fullRopeScaling"] = [
                "factor": s.factor,
                "originalMaxPositionEmbeddings": s.originalMaxPositionEmbeddings,
                "betaFast": s.betaFast,
                "betaSlow": s.betaSlow,
                "attentionFactor": s.attentionFactor,
            ]
        }
        if arch.attentionGating != "none" {
            archDict["attentionGating"] = arch.attentionGating
        }
        if arch.routedScalingFactor != 1.0 {
            archDict["routedScalingFactor"] = arch.routedScalingFactor
        }
        let quantBits = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        var quantDict: [String: Any] = [:]
        for (slot, bits) in quantBits {
            // When perLayer overrides exist, derive the scalar groupSize from
            // them (all overrides should share the same groupSize when they
            // exist; if mixed, the first one defines the scalar). Otherwise
            // use any slot-specific override or the plan's base group size.
            let perLayer = bitWidths.perLayer?[slot] ?? []
            let slotGroupSize: Int
            if let first = perLayer.first {
                slotGroupSize = first.groupSize
            } else if let override = bitWidths.slotGroupSizes[slot] {
                slotGroupSize = override
            } else {
                slotGroupSize = plan.baseGroupSize
            }
            var entry: [String: Any] = [
                "weightBits": bits,
                "scheme": plan.baseMode,
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": slotGroupSize
            ]
            if !perLayer.isEmpty {
                let overrides = perLayer.filter {
                    $0.weightBits != bits || $0.groupSize != slotGroupSize
                }
                if !overrides.isEmpty {
                    entry["perLayer"] = overrides.map { o in
                        ["layer": o.layer, "weightBits": o.weightBits,
                         "groupSize": o.groupSize]
                    }
                }
            }
            quantDict[slot] = entry
        }

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        let manifest: [String: Any] = [
            "magic": GTurboJSON.magic,
            "versionMajor": GTurboJSON.versionMajor,
            "versionMinor": GTurboJSON.versionMinor,
            "flags": [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
            "bitWidthOverridesHonored": plan.bitsOverrideCount
        ]
        return try JSONSerialization.data(withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size":   slice.sizeInExpertBlob,
                        "dtype":  slice.dtype == 0 ? "U32" : "BF16",
                        "shape":  slice.logicalShape.map { Int($0) }
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size":   lp.expertStride,
                    "tensors": tensors
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file":  layerFile,
                "experts": experts
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            // Dense-MLP layers hold zero experts, so the sparse-layer count has
            // to come from the first layer that actually has any.
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        // Written compact: this file is machine-read only and scales with
        // layers x experts x sub-tensors (tens of MB on a large MoE).
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.sortedKeys, .withoutEscapingSlashes])
    }
}
