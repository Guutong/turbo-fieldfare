import Foundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    /// Three-way layer kind: 0=sliding, 1=full, 2=linear. Indexed by layer.
    let layerKindMask: [UInt8]
    let hiddenActivation: String

    // Fields below describe variation Gemma 4 does not exercise. They keep
    // Gemma's defaults when the source config omits them, so a Gemma repack
    // produces a byte-identical `arch` block to before.
    let headsPerLayer: [Int]
    let denseMLPLayerMask: [UInt8]
    let denseMLPIntermediateSize: Int
    let fullPartialRotaryFactor: Double?
    let fullRopeScaling: RopeScalingInfo?
    let attentionGating: String
    let routedScalingFactor: Double
    let normTopology: String
    let routerScoring: String

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // Gemma nests the language model under `text_config`; Qwen3.6, Laguna,
        // and other text-only checkpoints put the same keys at the root.
        let tc = (root["text_config"] as? [String: Any]) ?? root
        guard tc["hidden_size"] != nil else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "no text_config and no hidden_size at root")
        }

        func optInt(_ k: String) -> Int? {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
        }
        func optDouble(_ k: String) -> Double? {
            (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue
        }
        func i(_ k: String) throws -> Int {
            guard let n = optInt(k) else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }

        let numLayers = try i("num_hidden_layers")

        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        // Reject architectures the runtime has no kernels for, rather than
        // silently importing them as if they were plain attention.
        if let unsupported = layerTypes.first(where: {
            $0 != "full_attention" && $0 != "sliding_attention" && $0 != "linear_attention"
        }) {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "unsupported layer_type \"\(unsupported)\"; only full_attention, " +
                        "sliding_attention, and linear_attention are implemented")
        }
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        // Three-way layer kind: 0=sliding, 1=full, 2=linear (Qwen3.6 DeltaNet).
        let kindMask = layerTypes.map { s -> UInt8 in
            switch s {
            case "full_attention": return 1
            case "linear_attention": return 2
            default: return 0
            }
        }

        // Gemma nests per-layer-kind rope params under full_attention/sliding_attention
        // sub-keys; Qwen3.6's rope_parameters is flat (single rope_theta for the whole
        // model, since it has no sliding-attention layers). Fall back to the flat dict
        // itself when the nested sub-keys are absent.
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? rope
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? rope

        func ropeDouble(_ d: [String: Any], _ k: String) -> Double? {
            (d[k] as? Double) ?? (d[k] as? NSNumber)?.doubleValue
        }

        // Gemma carries one partial rotary factor; Laguna differs per layer
        // type (0.5 full, 1.0 sliding). When both agree we leave the
        // full-attention override nil so Gemma's/Qwen3.6's arch block is
        // unchanged.
        let prfFull = ropeDouble(ropeFull, "partial_rotary_factor")
            ?? optDouble("partial_rotary_factor") ?? 0.25
        let prfSWA = ropeDouble(ropeSWA, "partial_rotary_factor") ?? prfFull
        let fullPRF: Double? = (prfFull == prfSWA) ? nil : prfFull

        let fullTheta = ropeDouble(ropeFull, "rope_theta") ?? 1_000_000.0
        let swaTheta  = ropeDouble(ropeSWA, "rope_theta") ?? 10_000.0

        var ropeScaling: RopeScalingInfo?
        if (ropeFull["rope_type"] as? String) == "yarn" {
            guard let factor = ropeDouble(ropeFull, "factor"),
                  let origMax = (ropeFull["original_max_position_embeddings"] as? Int)
                      ?? (ropeFull["original_max_position_embeddings"] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "rope_type yarn without factor/original_max_position_embeddings")
            }
            ropeScaling = RopeScalingInfo(
                factor: factor,
                originalMaxPositionEmbeddings: origMax,
                betaFast: ropeDouble(ropeFull, "beta_fast") ?? 32.0,
                betaSlow: ropeDouble(ropeFull, "beta_slow") ?? 1.0,
                attentionFactor: ropeDouble(ropeFull, "attention_factor") ?? 1.0)
        }

        // "shared expert FFN" size: Gemma uses intermediate_size; Laguna names
        // it shared_expert_intermediate_size and repurposes intermediate_size
        // for its dense (non-MoE) layers; Qwen3.6 has no shared-mlp dense
        // layer, only a shared *expert* sized by
        // shared_expert_intermediate_size (moe_intermediate_size is the
        // per-routed-expert size — a different field that is coincidentally
        // equal for Qwen3.6).
        let sharedFFN: Int
        if let v = try? i("shared_expert_intermediate_size") {
            sharedFFN = v
        } else if let v = try? i("intermediate_size") {
            sharedFFN = v
        } else {
            sharedFFN = try i("moe_intermediate_size")
        }

        // `mlp_only_layers` lists dense layer indices; `mlp_layer_types` spells
        // the same thing out per layer. Accept either.
        var denseMask = [UInt8](repeating: 0, count: numLayers)
        var hasDense = false
        if let mlpTypes = tc["mlp_layer_types"] as? [String], !mlpTypes.isEmpty {
            for (idx, t) in mlpTypes.enumerated() where idx < numLayers && t == "dense" {
                denseMask[idx] = 1
                hasDense = true
            }
        } else if let onlyLayers = tc["mlp_only_layers"] as? [Int], !onlyLayers.isEmpty {
            for idx in onlyLayers where idx >= 0 && idx < numLayers {
                denseMask[idx] = 1
                hasDense = true
            }
        }
        let denseFFN = hasDense ? (optInt("intermediate_size") ?? 0) : 0

        // Per-layer query-head counts, when the model varies them.
        var headsPerLayer: [Int] = []
        if let perLayer = tc["num_attention_heads_per_layer"] as? [Int],
           perLayer.count == numLayers,
           Set(perLayer).count > 1 {
            headsPerLayer = perLayer
        }

        // The decoder-block shape is not spelled out in any config key, so it
        // is keyed off `model_type`. Unknown types get Gemma's arrangement,
        // which is what every other field here defaults to — but note that a
        // third architecture with a different block will produce wrong numbers
        // rather than an error, so add it to this table before repacking it.
        //
        // `hidden_act` is also read from this table: Laguna's config.json omits
        // the key entirely and `configuration_laguna.py` defaults it to silu,
        // so falling back to Gemma's gelu would silently use the wrong
        // activation. Qwen3.6's checkpoint reports `model_type: "qwen3_5_moe"`
        // (not the superficially-similar "qwen3_next").
        let modelType = (tc["model_type"] as? String) ?? (root["model_type"] as? String) ?? ""
        let normTopology: String
        let routerScoring: String
        let defaultActivation: String
        switch modelType {
        case "laguna", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next":
            normTopology = "preNorm"
            routerScoring = "sigmoidTopK"
            defaultActivation = "silu"
        default:
            normTopology = "sandwich"
            routerScoring = "softmaxTopK"
            defaultActivation = "gelu_pytorch_tanh"
        }

        // `gating: true` means per-element in the reference implementation.
        let gatingRaw = tc["gating"]
        let gating: String
        switch gatingRaw {
        case let s as String: gating = (s == "per-head") ? "perHead" : "perElement"
        case let b as Bool:   gating = b ? "perElement" : "none"
        default:              gating = "none"
        }

        let headDim = try i("head_dim")
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: sharedFFN,
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            // Absent means the model has no distinct full-attention KV head
            // count — 0 for Qwen3.6, which has no separate full-attention KV
            // width, rather than falling back to `numKVHeads`.
            numFullKVHeads: optInt("num_global_key_value_heads") ?? 0,
            headDim: headDim,
            fullHeadDim: optInt("global_head_dim") ?? headDim,
            vocabSize: try i("vocab_size"),
            // Absent means no sliding window at all (Qwen3.6), not a required key.
            slidingWindow: optInt("sliding_window") ?? 0,
            finalLogitSoftcap: optDouble("final_logit_softcapping") ?? 0.0,
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prfSWA,
            numLayers: numLayers,
            numExperts: try i("num_experts"),
            topKExperts: (try? i("top_k_experts")) ?? (try? i("num_experts_per_tok")) ?? 1,
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: (tc["attention_k_eq_v"] as? Bool) ?? false,
            fullAttentionLayerMask: mask,
            layerKindMask: kindMask,
            hiddenActivation: (tc["hidden_act"] as? String)
                ?? (tc["hidden_activation"] as? String) ?? defaultActivation,
            headsPerLayer: headsPerLayer,
            denseMLPLayerMask: hasDense ? denseMask : [],
            denseMLPIntermediateSize: denseFFN,
            fullPartialRotaryFactor: fullPRF,
            fullRopeScaling: ropeScaling,
            attentionGating: gating,
            routedScalingFactor: optDouble("moe_routed_scaling_factor") ?? 1.0,
            normTopology: normTopology,
            routerScoring: routerScoring)
    }
}

struct RopeScalingInfo: Sendable, Equatable {
    let factor: Double
    let originalMaxPositionEmbeddings: Int
    let betaFast: Double
    let betaSlow: Double
    let attentionFactor: Double
}
