import Foundation
import Testing
@testable import TurboFieldfare

/// P3-1: causal conv1d (width 4) + state container, plain Swift fp32.
///
/// Hand-checked fixture — every expected number below is derived by pencil
/// from the tap equation, so a failing test points at the code, not at an
/// oracle nobody re-derived. Setup: convDim = 2, per-channel taps (tap 0 =
/// oldest input, matching cross-correlation over concat(conv_state, qkv)):
///
///     channel 0: [1, 2, 3, 4]
///     channel 1: [0, 1, 0, 1]
///
/// Input tokens (each 2 channels): x0 = [1, 1], x1 = [2, 1],
/// x2 = [3, 1], x3 = [4, 1]. Pre-silu outputs, from
/// out[t][c] = w0·x[t-3] + w1·x[t-2] + w2·x[t-1] + w3·x[t], x[<0] = 0:
///
///     t0: ch0 = 4·1 = 4                   ch1 = 1·1 = 1          → [4, 1]
///     t1: ch0 = 3·1 + 4·2 = 11            ch1 = 1·1 = 1          → [11, 1]
///     t2: ch0 = 2·1 + 3·2 + 4·3 = 20      ch1 = 1·1 + 1·1 = 2    → [20, 2]
///     t3: ch0 = 1·1 + 2·2 + 3·3 + 4·4 = 30
///                                ch1 = 1·1 + 1·1 = 2             → [30, 2]
///
/// (t3 exercises the sliding window: x0 has left the state and still
/// contributes through tap 0; after t3 the state rows are x1, x2, x3.)
/// Expected values below are silu(pre) computed at Double precision.
@Suite struct DeltaNetConvStateTests {
    private static let convDim = 2
    private static let weight: [Float] = [1, 2, 3, 4,   // channel 0 taps
                                          0, 1, 0, 1]   // channel 1 taps
    private static let inputs: [[Float]] = [[1, 1], [2, 1], [3, 1], [4, 1]]
    /// Hand-derived pre-silu activations, per token per channel.
    private static let preSilu: [[Double]] = [[4, 1], [11, 1], [20, 2], [30, 2]]

    private static func silu(_ x: Double) -> Double { x / (1 + exp(-x)) }

    @Test func convStepMatchesHandComputedSequence() {
        var state = [Float](repeating: 0, count: 3 * Self.convDim)
        for (step, input) in Self.inputs.enumerated() {
            let output = DeltaNetConv.step(input: input,
                                           weight: Self.weight,
                                           convDim: Self.convDim,
                                           convState: &state)
            #expect(output.count == Self.convDim)
            for channel in 0..<Self.convDim {
                let expected = Float(Self.silu(Self.preSilu[step][channel]))
                #expect(abs(output[channel] - expected) < 1e-6,
                        "t\(step) ch\(channel): \(output[channel]) vs \(expected)")
            }
        }
    }

    @Test func convStateHoldsLastThreeInputsOldestFirst() {
        var state = [Float](repeating: 0, count: 3 * Self.convDim)
        _ = DeltaNetConv.step(input: Self.inputs[0], weight: Self.weight,
                              convDim: Self.convDim, convState: &state)
        // After one step: rows [zeros, zeros, x0].
        #expect(state == [0, 0,  0, 0,  1, 1])

        for input in Self.inputs[1...] {
            _ = DeltaNetConv.step(input: input, weight: Self.weight,
                                  convDim: Self.convDim, convState: &state)
        }
        // After all four steps: rows [x1, x2, x3].
        #expect(state == [2, 1,  3, 1,  4, 1])
    }

    @Test func stateStoreSizesFollowDimensionsAndMask() {
        let dims = DeltaNetDimensions(convDim: 2, numKeyHeads: 1,
                                      numValueHeads: 2, headKDim: 3,
                                      headVDim: 3)
        let store = DeltaNetStateStore(numLayers: 4, dims: dims,
                                       isLinearLayer: [true, true, false, true])
        #expect(store.convState[0].count == 6)          // 3 × convDim
        #expect(store.recurrentState[0].count == 18)    // 2 × 3 × 3
        #expect(store.convState[2].isEmpty)             // full-attn layer
        #expect(store.recurrentState[2].isEmpty)
    }

    @Test func stateStoreQwen36Footprint() {
        let dims = DeltaNetDimensions.qwen36_35B_A3B
        #expect(dims.convStateCount == 24_576)          // 3 × 8192
        #expect(dims.recurrentStateCount == 524_288)    // 32 × 128 × 128
        // Sanity: ≈63 MB across the 30 linear layers (plan's state budget):
        // 30 × (24_576 + 524_288) × 4 = 65_863_680 bytes = 62.8 MiB.
        let bytes = 30 * (dims.convStateCount + dims.recurrentStateCount) * 4
        #expect(bytes == 65_863_680)
    }

    @Test func resetZeroesOnlyLinearLayers() {
        let dims = DeltaNetDimensions(convDim: 2, numKeyHeads: 1,
                                      numValueHeads: 1, headKDim: 2,
                                      headVDim: 2)
        let store = DeltaNetStateStore(numLayers: 3, dims: dims,
                                       isLinearLayer: [true, false, true])
        for layer in [0, 2] {
            for index in store.convState[layer].indices {
                store.convState[layer][index] = 7
            }
            for index in store.recurrentState[layer].indices {
                store.recurrentState[layer][index] = 9
            }
        }
        store.reset()
        #expect(store.convState[0].allSatisfy { $0 == 0 })
        #expect(store.recurrentState[0].allSatisfy { $0 == 0 })
        #expect(store.convState[2].allSatisfy { $0 == 0 })
        #expect(store.recurrentState[2].allSatisfy { $0 == 0 })
        #expect(store.convState[1].isEmpty)
        #expect(store.recurrentState[1].isEmpty)
    }
}
