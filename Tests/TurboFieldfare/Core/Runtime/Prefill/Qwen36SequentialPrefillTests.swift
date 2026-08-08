import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P3-3b: Qwen3.6 prompts are prefilled by replaying the prompt through the
/// per-token decode loop (which carries the pre-norm topology and the DeltaNet
/// recurrence), never through the Gemma-shaped chunked path.
///
/// The routing decision is a pure function of `LayerTopology`, so most of this
/// suite runs without a device or weights. The end-to-end case additionally
/// requires the repacked model and is skipped when it is absent, mirroring
/// `Qwen36Layer0NumericIsolationTests`.
@Suite struct Qwen36SequentialPrefillTests {
    private static let modelDir = "scratch/qwen36.gturbo"

    private static var isModelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir)
    }

    @Test func qwen36TopologyRoutesToTheSequentialDecodeLoop() {
        #expect(PrefillRoutePolicy.route(for: .qwen36) == .sequentialDecodeLoop)
        #expect(ArchConfig.qwen36_35B_A3B.topology == .qwen36)
        #expect(PrefillRoutePolicy.route(for: ArchConfig.qwen36_35B_A3B.topology)
                == .sequentialDecodeLoop)
    }

    @Test func gemmaTopologyKeepsTheChunkedPrefillPath() {
        #expect(PrefillRoutePolicy.route(for: .gemma4) == .chunked)
        #expect(ArchConfig.gemma4_26B_A4B.topology == .gemma4)
        #expect(PrefillRoutePolicy.route(for: ArchConfig.gemma4_26B_A4B.topology)
                == .chunked)
    }

    /// The real proof: a multi-token Qwen3.6 prompt goes through
    /// `prefillChunked` (the public entry point `runRawCompletion` uses) and
    /// `executePrefillChunk` — the Gemma-specific chunked machinery — is never
    /// entered, while every prompt token is stepped through the decode loop.
    @Test func qwen36PromptNeverEntersTheChunkedPrefillPath() async throws {
        guard Self.isModelAvailable else { return }

        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: Self.modelDir),
                                   device: context.device,
                                   expecting: .qwen36_35B_A3B)
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: 64)
        #expect(runner.prefillRoute == .sequentialDecodeLoop)

        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let prompt: [Int32] = [151_643, 9707, 1879, 11, 1246]
        runner.reset()
        var progress: [Int] = []
        let result = try await runner.prefillChunked(tokens: prompt[...],
                                                     startPosition: 0,
                                                     outputMode: .greedyIfAvailable,
                                                     config: .defaultChunked,
                                                     into: scratch.logits) { done in
            progress.append(done)
        }

        #expect(runner.chunkedPrefillChunkCount == 0)
        #expect(runner.sequentialPrefillTokenCount == prompt.count)
        #expect(result.newPosition == prompt.count)
        #expect(progress == Array(1...prompt.count))

        // One more decode step on top of the sequential prefill: the KV cursor
        // and the DeltaNet state carried straight through, and still no chunk.
        try await runner.produce(token: prompt.last!,
                                 position: prompt.count,
                                 into: scratch.logits)
        #expect(runner.chunkedPrefillChunkCount == 0)
        #expect(runner.continuationPosition == prompt.count + 1)
    }
}
