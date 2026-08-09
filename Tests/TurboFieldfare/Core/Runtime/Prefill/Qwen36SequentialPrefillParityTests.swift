import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// P5-1: the Qwen3.6 sequential prefill loop must stay numerically identical to
/// a decode-only replay of the same prompt, now that the DeltaNet recurrence
/// runs on Metal (P4-1/P4-2) instead of the plain-Swift-fp32 `DeltaNetCPUBlock`
/// it was originally validated against in P3-3b.
///
/// Both paths walk the same `produceToken` recurrence, so this is not a
/// tolerance question: the two runs must agree exactly, and the assertion is on
/// the observable the CLI itself emits — the greedy token sequence. The only
/// structural difference between them is that prefill emits the lm_head on the
/// last prompt token only, while the decode-only replay emits it on every
/// token; if the GPU-resident DeltaNet conv window / recurrent state or the KV
/// cursor were disturbed by that head-skipping, the divergence would show up in
/// the continuation.
///
/// The continuation check matters more than the first-token check: four greedy
/// steps on top of each prompt replay exercise the carried state, so a
/// state-plumbing regression appears as a diverging token sequence rather than
/// a one-shot mismatch.
///
/// Both paths run the fused greedy head (`RuntimeConfiguration.production`),
/// which is what the CLI uses at temperature 0 — the configuration P3-4, P4-1
/// and P4-2 verified against. Note the `RawCompletionScratch.logits` buffer is
/// Float16, so it must not be read back as Float32.
///
/// This sequential loop is deliberately the *oracle* for the future chunkwise
/// parallel prefill form (plan.md P5-1: "do not attempt the chunkwise parallel
/// form yet — this sequential path becomes the oracle for it later"). When that
/// form lands, it must be diffed against this path, not against a fresh
/// reference.
///
/// Requires the repacked model; skipped when absent, mirroring
/// `Qwen36SequentialPrefillTests`.
@Suite struct Qwen36SequentialPrefillParityTests {
    private static let modelDir = "scratch/qwen36.gturbo"

    private static var isModelAvailable: Bool {
        FileManager.default.fileExists(atPath: modelDir)
    }

    /// The criterion-C prompt used for the P0-7 fixture and the P3-4/P4-1/P4-2
    /// CLI checks. Tokenized through the model's own tokenizer with the same
    /// `addBOS: true` the CLI's raw-completion path uses, so the test sees the
    /// exact id sequence the verified CLI run saw.
    private static let promptText = "The capital of France is"
    private static let continuationSteps = 4

    /// Greedy-decode `continuationSteps` tokens on top of a prompt replay,
    /// starting from `seed`, reading the fused head's argmax each step.
    private static func continue_(runner: RealForwardRunner,
                                  scratch: RawCompletionScratch,
                                  seed: Int32,
                                  from startPosition: Int) async throws -> [Int32] {
        var tokens: [Int32] = [seed]
        var position = startPosition
        for _ in 0..<continuationSteps {
            try await runner.produce(token: tokens[tokens.count - 1],
                                     position: position,
                                     into: scratch.logits)
            position += 1
            tokens.append(Int32(runner.lastGreedyToken))
        }
        return tokens
    }

    @Test func sequentialPrefillMatchesDecodeOnlyReplay() async throws {
        guard Self.isModelAvailable else { return }

        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: URL(fileURLWithPath: Self.modelDir))
        let prompt = tokenizer.encode(Self.promptText, addBOS: true)
        #expect(!prompt.isEmpty)

        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: Self.modelDir),
                                   device: context.device,
                                   expecting: .qwen36_35B_A3B)
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: 64)
        #expect(runner.prefillRoute == .sequentialDecodeLoop)

        let vocab = model.config.vocabSize
        let scratch = try RawCompletionScratch(context: context, vocab: vocab)

        // Both paths run the fused greedy head — the exact configuration the
        // release CLI uses at temperature 0, and the one P3-4/P4-1/P4-2 verified.
        // Path A: prefill on — the whole prompt through `prefillChunked`, which
        // routes to `prefillSequential` for the Qwen3.6 topology.
        runner.reset()
        let prefillResult = try await runner.prefillChunked(tokens: prompt[...],
                                                            startPosition: 0,
                                                            outputMode: .greedyIfAvailable,
                                                            config: .defaultChunked,
                                                            into: scratch.logits) { _ in }
        #expect(runner.chunkedPrefillChunkCount == 0)
        #expect(runner.sequentialPrefillTokenCount == prompt.count)
        #expect(prefillResult.newPosition == prompt.count)
        guard case .greedyToken(let prefillSeed) = prefillResult.seed else {
            Issue.record("expected a fused greedy seed, got \(prefillResult.seed)")
            return
        }
        let prefillContinuation = try await Self.continue_(runner: runner,
                                                           scratch: scratch,
                                                           seed: Int32(prefillSeed),
                                                           from: prompt.count)

        // Path B: prefill off — the same prompt replayed one token at a time
        // through `produce`, exactly what RawCompletion's scalar replay does.
        runner.reset()
        for (offset, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: offset, into: scratch.logits)
        }
        #expect(runner.chunkedPrefillChunkCount == 0)
        let decodeSeed = runner.lastGreedyToken
        let decodeContinuation = try await Self.continue_(runner: runner,
                                                          scratch: scratch,
                                                          seed: Int32(decodeSeed),
                                                          from: prompt.count)

        // Same recurrence, same order of operations: exact equality, no gate.
        #expect(prefillSeed == decodeSeed,
                "first generated token differs: prefill \(prefillSeed) vs decode-only \(decodeSeed)")
        #expect(prefillContinuation == decodeContinuation,
                "prefill continuation \(prefillContinuation) != decode-only \(decodeContinuation)")

        // Ground truth, so an unwritten/degenerate head cannot make the equality
        // checks above pass vacuously: the criterion-C continuation of
        // "The capital of France is" begins with "Paris".
        let text = tokenizer.decode(prefillContinuation)
        #expect(text.trimmingCharacters(in: .whitespaces).hasPrefix("Paris"),
                "expected a 'Paris' continuation, got \(text) ids \(prefillContinuation)")
    }
}
