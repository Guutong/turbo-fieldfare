import Foundation
import Metal
import TurboFieldfare

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        // Validated before tokenizer/weight load so an oversized context fails
        // fast with a message instead of an allocation failure.
        if let reason = ContextCap.rejectionReason(for: args.maxContext) {
            return errored(stderr, reason, 2)
        }
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        if let reason = ContextCap.promptRejectionReason(promptTokens: promptIds.count,
                                                         maxContext: args.maxContext) {
            return errored(stderr, reason, 2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !config.isPureGreedy)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        // Auto-detect model family from manifest arch.
        let manifestURL = modelURL.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let expectedArch: ArchConfig
        if let root = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
           let arch = root["arch"] as? [String: Any] {
            expectedArch = ArchConfig.detect(
                hiddenSize: arch["hiddenSize"] as? Int ?? 0,
                numLayers: arch["numLayers"] as? Int ?? 0,
                numExperts: arch["numExperts"] as? Int ?? 0,
                numKVHeads: arch["numKVHeads"] as? Int ?? 0,
                numFullKVHeads: arch["numFullKVHeads"] as? Int ?? 0)
        } else {
            expectedArch = .gemma4_26B_A4B
        }
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: expectedArch,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        // P6b-4: opt-in n-gram speculation accounting. Purely observational —
        // it never feeds a token to the model, so enabling it cannot change
        // the generated text.
        let speculator = ProcessInfo.processInfo.environment["TFF_SPEC_DECODE"] == "1"
            ? NGramSpeculator()
            : nil
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig,
            speculator: speculator) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        if let speculator {
            let s = speculator.stats
            let line = "[spec-decode rounds=\(s.rounds) committed=\(s.committedTokens) drafts=\(s.draftsProposed) noEvidence=\(s.roundsWithoutEvidence) proposed=\(s.draftTokensProposed) accepted=\(s.draftTokensAccepted) acceptRate=\(String(format: "%.4f", s.acceptanceRate)) tokensPerRound=\(String(format: "%.3f", s.tokensPerRound))]\n"
            stderr.write(Data(line.utf8))
        }
        if ProcessInfo.processInfo.environment["TFF_EXPERT_CACHE_STATS"] == "1" {
            let cache = model.routedExpertCacheStats()
            let rate = String(format: "%.4f", cache.hitRate)
            let line = "[expert-cache slots=\(runtime.expertCacheSlots) policy=\(runtime.modelExpertCachePolicy.rawValue) lookups=\(cache.lookups) hits=\(cache.hits) misses=\(cache.misses) plans=\(cache.plans) hitRate=\(rate) evictions=\(cache.evictions) pinProtect=\(cache.pinnedProtections) pinOverride=\(cache.pinOverrides) ioSec=\(String(format: "%.2f", Double(cache.readNanos) / 1e9)) qdPeak=\(cache.peakInFlightReads)]\n"
            stderr.write(Data(line.utf8))
            // P6b-3: prefill/decode split. `prefillHitRate` is the prefill
            // expert dedup rate, and `ioFactor` is prefill I/O reduction vs a
            // no-reuse engine that pread every routed expert of every token.
            let prefillRate = String(format: "%.4f", cache.prefillHitRate)
            let ioFactor = String(format: "%.2f", cache.prefillIOReductionFactor)
            let split = "[expert-cache prefill lookups=\(cache.prefillLookups) hits=\(cache.prefillHits) misses=\(cache.prefillMisses) plans=\(cache.prefillPlans) hitRate=\(prefillRate) ioReduction=\(ioFactor)x | decode lookups=\(cache.decodeLookups) hits=\(cache.decodeHits) misses=\(cache.decodeMisses)]\n"
            stderr.write(Data(split.utf8))
            let perLayer = model.routedExpertCacheStatsByLayer().enumerated().compactMap {
                index, entry -> String? in
                guard let entry, entry.lookups > 0 else { return nil }
                return "L\(index):\(entry.hits)/\(entry.lookups)/e\(entry.evictions)"
            }
            stderr.write(Data("[expert-cache per-layer \(perLayer.joined(separator: " "))]\n".utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
