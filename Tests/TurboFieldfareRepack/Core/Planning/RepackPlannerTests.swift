import Testing
@testable import TurboFieldfareRepackCore

/// `RepackPlanner.classify` used to understand only Gemma's tensor naming.
/// These tests pin two things: Gemma must classify exactly as it did before
/// Laguna's naming was taught to the planner (a regression guard), and
/// Laguna's naming — including its dense layer 0 — must classify correctly.
@Suite struct RepackPlannerTests {

    // MARK: - Gemma regression guard

    @Test func gemmaRoutedExpertsClassifyByRole() {
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.experts.switch_glu.gate_proj.weight",
            numLayers: 30) == .routedExpert(role: "gate", layer: 3))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.experts.switch_glu.up_proj.weight",
            numLayers: 30) == .routedExpert(role: "up", layer: 3))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.experts.switch_glu.down_proj.weight",
            numLayers: 30) == .routedExpert(role: "down", layer: 3))
    }

    @Test func gemmaNonExpertLanguageModelTensorsAreResident() {
        let names = [
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.k_proj.weight",
            "language_model.model.layers.0.self_attn.v_proj.weight",
            "language_model.model.layers.0.self_attn.o_proj.weight",
            "language_model.model.layers.0.self_attn.q_norm.weight",
            "language_model.model.layers.0.self_attn.k_norm.weight",
            "language_model.model.layers.0.router.proj.weight",
            "language_model.model.layers.0.router.scale",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.0.input_layernorm.weight",
        ]
        for n in names {
            #expect(RepackPlanner.classify(n, numLayers: 30) == .lmResident, "\(n)")
        }
    }

    @Test func gemmaMultimodalTensorsAreExcluded() {
        let names = [
            "vision_tower.encoder.layers.0.input_layernorm.weight",
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            "embed_vision.embedding_projection.weight",
            "audio_tower.encoder.layers.0.weight",
        ]
        for n in names {
            #expect(RepackPlanner.classify(n, numLayers: 30) == .excludedMultimodal, "\(n)")
        }
    }

    @Test func unrecognizedPrefixIsUnknown() {
        #expect(RepackPlanner.classify("some_other_module.weight", numLayers: 30) == .unknown)
    }

    // MARK: - Laguna naming

    @Test func lagunaRoutedExpertsClassifyByRole() {
        #expect(RepackPlanner.classify(
            "language_model.model.layers.5.mlp.switch_mlp.gate_proj.weight",
            numLayers: 48) == .routedExpert(role: "gate", layer: 5))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.5.mlp.switch_mlp.up_proj.weight",
            numLayers: 48) == .routedExpert(role: "up", layer: 5))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.5.mlp.switch_mlp.down_proj.weight",
            numLayers: 48) == .routedExpert(role: "down", layer: 5))
    }

    @Test func lagunaNonExpertTensorsAreResident() {
        let names = [
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.4.mlp.gate.proj.weight",              // router
            "language_model.model.layers.4.mlp.gate.e_score_correction_bias",  // router bias
            "language_model.model.layers.4.mlp.shared_expert.down_proj.weight",
            "language_model.model.layers.4.mlp.shared_expert.gate_proj.weight",
            "language_model.model.layers.4.mlp.shared_expert.up_proj.weight",
            "language_model.model.layers.4.self_attn.q_proj.weight",
            "language_model.model.layers.4.self_attn.k_proj.weight",
            "language_model.model.layers.4.self_attn.v_proj.weight",
            "language_model.model.layers.4.self_attn.o_proj.weight",
            "language_model.model.layers.4.self_attn.g_proj.weight",   // per-head gate
            "language_model.model.layers.4.self_attn.q_norm.weight",
            "language_model.model.layers.4.self_attn.k_norm.weight",
            "language_model.model.layers.4.input_layernorm.weight",
            "language_model.model.layers.4.post_attention_layernorm.weight",
        ]
        for n in names {
            #expect(RepackPlanner.classify(n, numLayers: 48, denseMLPLayerMask: Self.lagunaDenseMask)
                == .lmResident, "\(n)")
        }
    }

    /// Laguna's layer 0 is a plain dense MLP (`mlp_only_layers: [0]`), named
    /// like the shared-expert tensors rather than `mlp.switch_mlp.*`. It must
    /// never be routed into per-expert packing.
    @Test func lagunaLayerZeroDenseMLPIsNotRoutedExpert() {
        let names = [
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
        ]
        for n in names {
            #expect(RepackPlanner.classify(n, numLayers: 48, denseMLPLayerMask: Self.lagunaDenseMask)
                == .lmResident, "\(n)")
        }
    }

    /// Defense in depth: even if a tensor name matches a routed-expert
    /// marker on a layer the config marks dense, the mask keeps it out of
    /// the routed-expert bucket instead of silently feeding a dense weight
    /// into per-expert packing.
    @Test func denseMLPLayerMaskOverridesAMatchingMarker() {
        let hypothetical = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        #expect(RepackPlanner.classify(hypothetical, numLayers: 48,
                                       denseMLPLayerMask: Self.lagunaDenseMask) == .lmResident)
        // Without the mask (or on a sparse layer) the same name does route.
        #expect(RepackPlanner.classify(hypothetical, numLayers: 48) == .routedExpert(role: "gate", layer: 0))
    }

    /// The manifest records one bit width per quant slot, probed from a
    /// representative tensor. Laguna's dense layer-0 MLP ends in the same
    /// `.mlp.gate_proj.weight` the generic probe looks for, so an unordered
    /// probe would write layer 0's bit width into the sharedExpert slot — a
    /// manifest that loads cleanly and describes the wrong quantization.
    /// Order is the fix, so pin it.
    @Test func sharedExpertProbePrefersTheQualifiedNameOverADenseMLP() {
        let probes = RepackPlanner.sharedExpertProbeSuffixes
        let qualified = ".mlp.shared_expert.gate_proj.weight"
        let generic = ".mlp.gate_proj.weight"
        let qualifiedIndex = probes.firstIndex(of: qualified)
        let genericIndex = probes.firstIndex(of: generic)
        #expect(qualifiedIndex != nil)
        #expect(genericIndex != nil)
        if let q = qualifiedIndex, let g = genericIndex {
            #expect(q < g, "qualified shared-expert probe must be tested first")
        }
        // The trap the ordering defends against: Laguna's dense layer-0 tensor
        // really does match the generic probe.
        #expect("language_model.model.layers.0.mlp.gate_proj.weight".hasSuffix(generic))
    }

    /// Gemma spells the router `.router.proj.`, Laguna `.mlp.gate.proj.`.
    /// The dot matters — `gate_proj` is an MLP, `gate.proj` is the router.
    @Test func routerProbeCoversBothFamiliesWithoutMatchingAnMLP() {
        let probes = RepackPlanner.routerProbeSuffixes
        let gemma = "language_model.model.layers.1.router.proj.weight"
        let laguna = "language_model.model.layers.1.mlp.gate.proj.weight"
        let mlp = "language_model.model.layers.0.mlp.gate_proj.weight"
        #expect(probes.contains(where: gemma.hasSuffix))
        #expect(probes.contains(where: laguna.hasSuffix))
        #expect(!probes.contains(where: mlp.hasSuffix))
    }

    // MARK: - Slot bit-width uniformity

    @Test func uniformBitsReturnsTheSingleWidthWhenTheSlotAgrees() throws {
        let obs = (0..<48).map { (name: "layers.\($0).self_attn.q_proj.weight", bits: 4) }
        #expect(try RepackPlanner.uniformBits(slot: "attention", observations: obs) == 4)
    }

    /// No matching tensor is not an error — the caller keeps its default.
    @Test func uniformBitsReturnsNilForAnEmptySlot() throws {
        #expect(try RepackPlanner.uniformBits(slot: "attention", observations: []) == nil)
    }

    /// The real shape of Laguna's attention slot: 5-bit on 20 layers, 8-bit on
    /// the other 28, taken from `mlx-community/Laguna-S-2.1-oQ4e/config.json`.
    /// The manifest holds one width per slot, so there is no correct answer
    /// and the only safe outcome is refusal. Before this check the probe was
    /// `bits.attention = s.bits` inside a loop — last-write-wins, which would
    /// have recorded 8 and misdescribed 20 layers in a manifest that loads
    /// cleanly.
    @Test func uniformBitsRefusesLagunaMixedAttentionRatherThanPickingOne() {
        let fiveBitLayers: Set<Int> = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                                       31, 33, 34, 37, 38, 41, 42]
        let obs = (0..<48).map { layer in
            (name: "language_model.model.layers.\(layer).self_attn.q_proj.weight",
             bits: fiveBitLayers.contains(layer) ? 5 : 8)
        }
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.uniformBits(slot: "attention", observations: obs)
        }
    }

    /// The error has to name the slot and both widths, otherwise it sends the
    /// reader hunting through 48 layers for the disagreement.
    @Test func nonUniformSlotErrorNamesTheSlotAndBothWidths() {
        let obs = [(name: "layers.0.self_attn.q_proj.weight", bits: 5),
                   (name: "layers.1.self_attn.q_proj.weight", bits: 8)]
        do {
            _ = try RepackPlanner.uniformBits(slot: "attention", observations: obs)
            Issue.record("expected a refusal")
        } catch let error as RepackError {
            guard case .quantSlotNotUniform(let slot, let bits, let sample) = error else {
                Issue.record("wrong case: \(error)")
                return
            }
            #expect(slot == "attention")
            #expect(Set(bits) == [5, 8])
            // One representative per width, not a prefix of the majority.
            #expect(sample.count == 2)
            #expect(error.description.contains("attention"))
            #expect(error.description.contains("5"))
            #expect(error.description.contains("8"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    /// Gemma is uniform in every slot, so the new check must be inert for it.
    @Test func gemmaSlotsStayUniformUnderTheNewCheck() throws {
        for (slot, bits) in [("embedding", 4), ("attention", 4),
                             ("router", 8), ("sharedExpert", 8), ("routedExpert", 4)] {
            let obs = (0..<30).map { (name: "layers.\($0).\(slot)", bits: bits) }
            #expect(try RepackPlanner.uniformBits(slot: slot, observations: obs) == bits)
        }
    }

    private static let lagunaDenseMask: [UInt8] = {
        var mask = [UInt8](repeating: 0, count: 48)
        mask[0] = 1
        return mask
    }()
}
