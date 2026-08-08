import Foundation

/// Fixed per-layer dimensions of the gated DeltaNet stack.
///
/// Qwen3.6-35B-A3B: `convDim` 8192 (q 2048 + k 2048 + v 4096), 16 key heads ×
/// 128 expanded to 32 value heads × 128 by the recurrence.
struct DeltaNetDimensions: Sendable, Equatable {
    let convDim: Int
    let numKeyHeads: Int
    let numValueHeads: Int
    let headKDim: Int
    let headVDim: Int

    init(convDim: Int,
         numKeyHeads: Int,
         numValueHeads: Int,
         headKDim: Int,
         headVDim: Int) {
        self.convDim = convDim
        self.numKeyHeads = numKeyHeads
        self.numValueHeads = numValueHeads
        self.headKDim = headKDim
        self.headVDim = headVDim
    }

    /// Qwen3.6-35B-A3B dimensions, from its config.json.
    static let qwen36_35B_A3B = DeltaNetDimensions(
        convDim: 8192,
        numKeyHeads: 16,
        numValueHeads: 32,
        headKDim: 128,
        headVDim: 128)

    /// Elements of one layer's conv state: the last 3 pre-conv qkv vectors.
    var convStateCount: Int { 3 * convDim }

    /// Elements of one layer's recurrent state: `[Hv, Dv, Dk]` fp32.
    var recurrentStateCount: Int { numValueHeads * headVDim * headKDim }
}

/// All DeltaNet layer states for one generation session.
///
/// fp32 throughout: Phase 3 deliberately prefers obviously-correct over fast
/// (ADR-0001). Footprint for Qwen3.6 is 30 linear layers ×
/// (3×8192 + 32×128×128) floats ≈ 63 MB, constant in context length — this is
/// what replaces a growing KV cache for the 30 DeltaNet layers.
///
/// Indexed by GLOBAL layer index (0..<numLayers); full-attention layers hold
/// empty arrays so the decode loop can address state by layer without a rank
/// mapping.
final class DeltaNetStateStore: @unchecked Sendable {
    let numLayers: Int
    let dims: DeltaNetDimensions
    /// `true` at global indices that are DeltaNet layers.
    let isLinearLayer: [Bool]

    /// Per layer, `[3 × convDim]` fp32 rows oldest-first, empty for
    /// full-attention layers. Mutable by the decode loop and tests.
    var convState: [[Float]]
    /// Per layer, `[Hv × Dv × Dk]` fp32, empty for full-attention layers.
    var recurrentState: [[Float]]

    init(numLayers: Int, dims: DeltaNetDimensions, isLinearLayer: [Bool]) {
        precondition(isLinearLayer.count == numLayers)
        self.numLayers = numLayers
        self.dims = dims
        self.isLinearLayer = isLinearLayer
        self.convState = (0..<numLayers).map { layer in
            isLinearLayer[layer]
                ? [Float](repeating: 0, count: dims.convStateCount) : []
        }
        self.recurrentState = (0..<numLayers).map { layer in
            isLinearLayer[layer]
                ? [Float](repeating: 0, count: dims.recurrentStateCount) : []
        }
    }

    /// Zero every DeltaNet state. Called at the start of each generation so a
    /// reused runner never leaks one conversation's state into the next.
    func reset() {
        for layer in 0..<numLayers where isLinearLayer[layer] {
            for index in convState[layer].indices { convState[layer][index] = 0 }
            for index in recurrentState[layer].indices {
                recurrentState[layer][index] = 0
            }
        }
    }
}
