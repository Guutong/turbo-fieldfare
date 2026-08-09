import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P4-1: the Metal DeltaNet path must reproduce the plain-Swift-fp32
/// `DeltaNetCPUBlock` it replaces.
///
/// ADR-0001 kept the Phase 3 block obviously-correct-over-fast precisely so
/// the Phase 4 port would have an exact diff target; this is that diff, at the
/// unit level (P4-2 does it at whole-model scale). Same layer, same real
/// repacked weights, same tokens, both paths stepped in lockstep so the
/// recurrent and conv state carry forward — a state-plumbing bug shows up as
/// divergence that grows with token index, not just a one-shot mismatch.
///
/// The kernels deliberately keep the reference's sequential reduction order,
/// so the only expected divergence is fp32 FMA contraction and libm-vs-Metal
/// transcendental ULPs. The gate is 1e-5 relative L2 — three orders of
/// magnitude tighter than ADR-0002's model-level budget, and per ADR-0002 it
/// is not to be widened to make a failing kernel pass.
///
/// Measured on first green run (M-series, 6 tokens, layer 0): deltaOut relL2
/// 4.6e-07 -> 1.2e-06 with maxAbs <= 3.7e-07, conv state 2.9e-07, recurrent
/// state 4.7e-07 — i.e. ~10x headroom under the gate, and flat rather than
/// compounding across tokens, which is what confirms the GPU-resident state
/// is not drifting away from the reference.
@Suite struct DeltaNetMetalParityTests {
    private static let modelDir = "scratch/qwen36.gturbo"

    private static var isModelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir)
    }

    /// Deterministic pseudo-random activations in a realistic range, so the
    /// test is reproducible without a fixture of its own.
    private struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: state >> 32)
            return Float(bits) / Float(UInt32.max) * 2 - 1
        }
    }

    private static func relativeL2(_ ours: [Float], _ reference: [Float]) -> Float {
        precondition(ours.count == reference.count)
        var diff: Float = 0
        var norm: Float = 0
        for i in ours.indices {
            let d = ours[i] - reference[i]
            diff += d * d
            norm += reference[i] * reference[i]
        }
        return sqrt(diff) / max(sqrt(norm), 1e-30)
    }

    private static func maxAbsolute(_ ours: [Float], _ reference: [Float]) -> Float {
        var worst: Float = 0
        for i in ours.indices { worst = max(worst, abs(ours[i] - reference[i])) }
        return worst
    }

    @Test(.enabled(if: isModelAvailable))
    func metalBlockMatchesCPUBlockOverMultipleTokens() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: Self.modelDir),
                                   device: device, expecting: .qwen36_35B_A3B)
        let dims = DeltaNetDimensions.qwen36_35B_A3B
        let D = model.config.hiddenSize

        // Layer 0 is a DeltaNet layer in Qwen3.6 and is the layer the P3-3
        // numeric isolation gate covers, so a divergence here is directly
        // comparable to that gate's evidence.
        let layer = 0
        #expect(model.config.layerKindMask[layer] == 2,
                "layer \(layer) must be a linear (DeltaNet) layer")

        let cpuWeights = try DeltaNetCPUBlock.LayerWeights(
            model: model, layer: layer, D: D, dims: dims)
        var cpuConvState = [Float](repeating: 0, count: dims.convStateCount)
        var cpuRecurrentState = [Float](repeating: 0, count: dims.recurrentStateCount)

        let block = try DeltaNetMetalBlock(context: context, hiddenSize: D, dims: dims)
        let gpuWeights = try DeltaNetMetalBlock.LayerWeights(
            device: device, model: model, layer: layer, D: D, dims: dims)
        let state = try DeltaNetMetalBlock.GPUStateStore(
            device: device, numLayers: 1, dims: dims, isLinearLayer: [true])
        let convState = try #require(state.convState[0])
        let recurrentState = try #require(state.recurrentState[0])

        let hidden = device.makeBuffer(length: D * MemoryLayout<Float16>.size,
                                       options: .storageModeShared)!
        let hiddenPtr = hidden.contents().assumingMemoryBound(to: Float16.self)

        // More than the conv width (4) so the conv state actually rotates,
        // and enough steps for recurrent-state drift to show.
        let tokenCount = 6
        var rng = LCG(state: 0x5DEECE66D)
        for token in 0..<tokenCount {
            // Both paths see bit-identical input: the runner's residual stream
            // is FP16, so round through Float16 before either path reads it.
            var x = [Float](repeating: 0, count: D)
            for i in 0..<D {
                let half = Float16(rng.next() * 0.5)
                hiddenPtr[i] = half
                x[i] = Float(half)
            }

            let reference = DeltaNetCPUBlock.forward(
                x: x, weights: cpuWeights, dims: dims,
                convState: &cpuConvState, recurrentState: &cpuRecurrentState)

            let cb = context.queue.makeCommandBuffer()!
            block.encode(commandBuffer: cb, hidden: hidden, weights: gpuWeights,
                         convState: convState, recurrentState: recurrentState)
            cb.commit()
            cb.waitUntilCompleted()
            try checkCommandBufferError(cb.error)

            let ours = block.lastDeltaOut
            let relL2 = Self.relativeL2(ours, reference)
            let maxAbs = Self.maxAbsolute(ours, reference)
            #expect(relL2 <= 1e-5,
                    "token \(token) deltaOut relL2 \(relL2) (maxAbs \(maxAbs))")
            #expect(ours.allSatisfy { $0.isFinite })

            // The residual write-back must match the CPU call site's
            // `Float16(x[i] + deltaOut[i])` contract exactly.
            for i in 0..<D {
                let expected = Float16(x[i] + reference[i])
                let got = hiddenPtr[i]
                let tolerance = max(abs(Float(expected)) * 1e-2, 1e-3)
                #expect(abs(Float(got) - Float(expected)) <= tolerance,
                        "token \(token) hidden[\(i)]: \(got) vs \(expected)")
            }
        }

        // State parity after the full lockstep run — this is the P4-1
        // deliverable: state lives in GPU buffers and still evolves exactly
        // as the Swift reference's arrays do.
        let gpuConv = (0..<dims.convStateCount).map {
            convState.contents().assumingMemoryBound(to: Float.self)[$0]
        }
        let gpuRecurrent = (0..<dims.recurrentStateCount).map {
            recurrentState.contents().assumingMemoryBound(to: Float.self)[$0]
        }
        let convRelL2 = Self.relativeL2(gpuConv, cpuConvState)
        let recurrentRelL2 = Self.relativeL2(gpuRecurrent, cpuRecurrentState)
        #expect(convRelL2 <= 1e-5, "conv state relL2 \(convRelL2)")
        #expect(recurrentRelL2 <= 1e-5, "recurrent state relL2 \(recurrentRelL2)")
    }

    @Test(.enabled(if: isModelAvailable))
    func gpuStateStoreResetZeroesEveryLinearLayer() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let dims = DeltaNetDimensions.qwen36_35B_A3B
        let store = try DeltaNetMetalBlock.GPUStateStore(
            device: device, numLayers: 3, dims: dims,
            isLinearLayer: [true, false, true])

        #expect(store.convState[1] == nil, "full-attention layers hold no state")
        #expect(store.recurrentState[1] == nil)

        let recurrent = try #require(store.recurrentState[2])
        let ptr = recurrent.contents().assumingMemoryBound(to: Float.self)
        #expect(ptr[0] == 0, "state must start zeroed")
        ptr[0] = 1.5
        ptr[dims.recurrentStateCount - 1] = -2.5
        store.reset()
        #expect(ptr[0] == 0)
        #expect(ptr[dims.recurrentStateCount - 1] == 0)
    }
}
