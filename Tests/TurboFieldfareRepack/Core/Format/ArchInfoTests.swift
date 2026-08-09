import Testing
import Foundation
@testable import TurboFieldfareRepackCore

/// `ArchInfo.load` used to assume Gemma's config shape. These tests pin the
/// two things that matter after generalizing it: Gemma must parse to exactly
/// what it parsed to before, and a genuinely different architecture must
/// either parse correctly or be rejected loudly.
@Suite struct ArchInfoTests {

    private static func write(_ json: String) throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("archinfo-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    /// Trimmed from mlx-community/gemma-4-26b-a4b-it-4bit at the pinned
    /// revision 0d77464eeb233a2da68ebf9d7dc4edaac7db956d. Keys the loader
    /// reads are verbatim; the layer list is truncated to 6 for brevity.
    private static let gemmaConfig = """
    {
      "text_config": {
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "moe_intermediate_size": 704,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 262144,
        "sliding_window": 1024,
        "final_logit_softcapping": 30.0,
        "num_hidden_layers": 6,
        "num_experts": 128,
        "top_k_experts": 8,
        "tie_word_embeddings": true,
        "attention_k_eq_v": true,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
          "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
          },
          "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
        }
      }
    }
    """

    @Test func gemmaParsesToItsHistoricalValues() throws {
        let a = try ArchInfo.load(configPath: Self.write(Self.gemmaConfig))

        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numHeads == 16)
        #expect(a.numKVHeads == 8)
        #expect(a.numFullKVHeads == 2)
        #expect(a.headDim == 256)
        #expect(a.fullHeadDim == 512)
        #expect(a.vocabSize == 262144)
        #expect(a.slidingWindow == 1024)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.ropeTheta == 10_000.0)
        #expect(a.fullRopeTheta == 1_000_000.0)
        #expect(a.topKExperts == 8)
        #expect(a.tieWordEmbeddings)
        #expect(a.attentionKEqV)
        #expect(a.hiddenActivation == "gelu_pytorch_tanh")
        #expect(a.fullAttentionLayerMask == [0, 0, 0, 0, 0, 1])

        // `sliding_attention` carries no partial_rotary_factor, so it inherits
        // the full-attention value — the behaviour before generalization.
        #expect(a.partialRotaryFactor == 0.25)
    }

    /// Every field added for other architectures must collapse to Gemma's
    /// defaults, or a Gemma repack would silently change.
    @Test func gemmaLeavesAllNewFieldsAtDefaults() throws {
        let a = try ArchInfo.load(configPath: Self.write(Self.gemmaConfig))

        #expect(a.headsPerLayer.isEmpty)
        #expect(a.denseMLPLayerMask.isEmpty)
        #expect(a.denseMLPIntermediateSize == 0)
        #expect(a.fullPartialRotaryFactor == nil)
        // rope_type is "proportional", not "yarn".
        #expect(a.fullRopeScaling == nil)
        #expect(a.attentionGating == "none")
        #expect(a.routedScalingFactor == 1.0)
    }

    /// Trimmed from poolside/Laguna-S-2.1. Flat (no `text_config`), varies
    /// heads per layer, has a dense layer 0, YaRN on full-attention only, and
    /// per-head gating. Truncated to 8 layers keeping the period-4 pattern.
    private static let lagunaConfig = """
    {
      "hidden_size": 3072,
      "intermediate_size": 12288,
      "shared_expert_intermediate_size": 1024,
      "moe_intermediate_size": 1024,
      "num_attention_heads": 48,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "vocab_size": 100352,
      "sliding_window": 512,
      "num_hidden_layers": 8,
      "num_experts": 256,
      "num_experts_per_tok": 10,
      "tie_word_embeddings": false,
      "hidden_act": "silu",
      "gating": "per-head",
      "moe_routed_scaling_factor": 2.5,
      "mlp_only_layers": [0],
      "num_attention_heads_per_layer": [48, 72, 72, 72, 48, 72, 72, 72],
      "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                      "sliding_attention", "full_attention", "sliding_attention",
                      "sliding_attention", "sliding_attention"],
      "rope_parameters": {
        "full_attention": {
          "rope_theta": 500000.0,
          "rope_type": "yarn",
          "factor": 128.0,
          "original_max_position_embeddings": 8192,
          "beta_slow": 1.0,
          "beta_fast": 32.0,
          "attention_factor": 1.4852030263919618,
          "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
          "rope_type": "default",
          "rope_theta": 10000.0,
          "partial_rotary_factor": 1.0
        }
      }
    }
    """

    @Test func lagunaParsesFromAFlatConfig() throws {
        let a = try ArchInfo.load(configPath: Self.write(Self.lagunaConfig))

        #expect(a.hiddenSize == 3072)
        // `intermediate_size` is the dense width here; the shared expert is
        // the separately named key.
        #expect(a.intermediateSize == 1024)
        #expect(a.denseMLPIntermediateSize == 12288)
        #expect(a.moeIntermediateSize == 1024)
        #expect(a.vocabSize == 100352)
        #expect(a.hiddenActivation == "silu")

        // Absent keys fall back rather than throwing.
        #expect(a.numFullKVHeads == 8)
        #expect(a.fullHeadDim == 128)
        #expect(a.finalLogitSoftcap == 0.0)
        #expect(!a.attentionKEqV)

        // top-K comes from num_experts_per_tok when top_k_experts is absent.
        #expect(a.topKExperts == 10)
        #expect(a.numExperts == 256)
    }

    @Test func lagunaCarriesPerLayerVariation() throws {
        let a = try ArchInfo.load(configPath: Self.write(Self.lagunaConfig))

        #expect(a.fullAttentionLayerMask == [1, 0, 0, 0, 1, 0, 0, 0])
        #expect(a.headsPerLayer == [48, 72, 72, 72, 48, 72, 72, 72])
        #expect(a.denseMLPLayerMask == [1, 0, 0, 0, 0, 0, 0, 0])

        // Distinct per layer type, so the full-attention override is set.
        #expect(a.partialRotaryFactor == 1.0)
        #expect(a.fullPartialRotaryFactor == 0.5)
        #expect(a.ropeTheta == 10_000.0)
        #expect(a.fullRopeTheta == 500_000.0)

        #expect(a.attentionGating == "perHead")
        #expect(a.routedScalingFactor == 2.5)
    }

    @Test func lagunaParsesYarnOnFullAttentionOnly() throws {
        let a = try ArchInfo.load(configPath: Self.write(Self.lagunaConfig))
        let s = try #require(a.fullRopeScaling)

        #expect(s.factor == 128.0)
        #expect(s.originalMaxPositionEmbeddings == 8192)
        #expect(s.betaFast == 32.0)
        #expect(s.betaSlow == 1.0)
        #expect(s.attentionFactor == 1.4852030263919618)
    }

    /// Qwen3.6-35B-A3B is 36/40 linear-attention layers and the runtime has no
    /// kernels for it. Importing it must fail at config parse rather than
    /// producing a model that decodes garbage.
    @Test func linearAttentionLayersAreRejected() throws {
        let config = """
        {
          "hidden_size": 2048, "intermediate_size": 512, "moe_intermediate_size": 512,
          "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
          "vocab_size": 248320, "sliding_window": 512, "num_hidden_layers": 4,
          "num_experts": 256, "num_experts_per_tok": 8,
          "layer_types": ["linear_attention", "linear_attention",
                          "linear_attention", "full_attention"]
        }
        """
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: Self.write(config))
        }
    }

    @Test func aConfigWithNoRecognizableShapeIsRejected() throws {
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: Self.write(#"{"model_type": "mystery"}"#))
        }
    }
}
