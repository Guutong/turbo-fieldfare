import Foundation

/// Tuning for `NGramSpeculator`.
///
/// `matchLengths` is kimi-k3's evidence ladder: try a length-4 suffix match
/// first, fall back to length-3, give up if neither has evidence in the
/// context. Shorter matches are deliberately NOT attempted — a 1- or 2-token
/// suffix matches almost everywhere and its continuation carries no signal.
public struct NGramSpeculatorConfig: Sendable, Equatable {
    /// Maximum tokens drafted per round (K).
    public var maxDraft: Int
    /// Suffix lengths to try, in preference order.
    public var matchLengths: [Int]

    public init(maxDraft: Int = 4, matchLengths: [Int] = [4, 3]) {
        precondition(maxDraft >= 1)
        precondition(!matchLengths.isEmpty)
        precondition(matchLengths.allSatisfy { $0 >= 1 })
        self.maxDraft = maxDraft
        self.matchLengths = matchLengths
    }

    public static let `default` = NGramSpeculatorConfig()
}

/// Measured speculation accounting for one generation.
///
/// A "round" is one verification pass: a real batched implementation proposes
/// a draft, runs ONE forward pass over the draft positions, accepts the longest
/// matching prefix (`a` tokens), and always also commits the model's own token
/// at the first mismatching position. So a round commits `a + 1` tokens, and
/// `tokensPerRound` is exactly plan.md's "tokens per decode step".
public struct SpeculationStats: Sendable, Equatable {
    public var rounds = 0
    public var committedTokens = 0
    public var draftsProposed = 0
    public var draftTokensProposed = 0
    public var draftTokensAccepted = 0
    /// Rounds that had no length-4 and no length-3 evidence at all.
    public var roundsWithoutEvidence = 0

    /// Accepted draft tokens / proposed draft tokens. plan.md target: >= 0.50.
    public var acceptanceRate: Double {
        draftTokensProposed == 0 ? 0 : Double(draftTokensAccepted) / Double(draftTokensProposed)
    }

    /// Committed tokens per verification round. plan.md target: >= 1.5.
    public var tokensPerRound: Double {
        rounds == 0 ? 0 : Double(committedTokens) / Double(rounds)
    }
}

/// n-gram (prompt-lookup) speculative drafter for greedy decoding.
///
/// ## Why this is byte-identical to serial decode by construction
///
/// A draft token is never fed to the model before it is verified. Greedy
/// decode already produces the real argmax `a_n` for position `n` as a side
/// effect of the forward pass over token `t_n`; a drafted `d` for position
/// `n+1` is accepted **iff `d == a_n`**, which is a pure comparison against a
/// number the serial path had computed anyway. Accepting therefore feeds the
/// model exactly the token serial decode would have fed, and rejecting feeds
/// `a_n` — also exactly what serial decode would have fed. No unverified token
/// ever reaches `produceToken`, so the DeltaNet conv/recurrent state and the
/// full-attention KV cache are only ever advanced by committed tokens.
///
/// That is the design decision recorded in PHASE-LOG for P6b-4: **neither
/// snapshot-and-rollback nor stage-and-commit — never-commit-unverified.** It
/// is available because verification here is sequential; see the History entry
/// for why a batched verification pass (which WOULD need rollback) does not
/// exist for the Qwen3.6 topology.
public final class NGramSpeculator {
    public let config: NGramSpeculatorConfig
    public private(set) var stats = SpeculationStats()

    /// Draft tokens proposed for upcoming positions and not yet verified.
    private var pending: [Int32] = []
    /// True between a round's draft proposal and its bonus token.
    private var roundOpen = false

    public init(config: NGramSpeculatorConfig = .default) {
        self.config = config
    }

    public func reset() {
        pending.removeAll(keepingCapacity: true)
        roundOpen = false
        stats = SpeculationStats()
    }

    /// Draft up to `maxDraft` continuation tokens for `context`.
    ///
    /// Evidence-gated longest-suffix matching: for each length in
    /// `matchLengths` (4 then 3), look for the MOST RECENT earlier occurrence
    /// of `context`'s trailing n-gram and return whatever followed it. Empty
    /// when no ladder rung has evidence.
    public func draft(context: [Int32]) -> [Int32] {
        for n in config.matchLengths {
            guard context.count > n else { continue }
            let suffixStart = context.count - n
            // Most recent earlier occurrence wins: scan candidate start
            // indices from high to low.
            var i = suffixStart - 1
            while i >= 0 {
                var matched = true
                for j in 0..<n where context[i + j] != context[suffixStart + j] {
                    matched = false
                    break
                }
                if matched {
                    let start = i + n
                    let end = min(start + config.maxDraft, context.count)
                    if start < end { return Array(context[start..<end]) }
                }
                i -= 1
            }
        }
        return []
    }

    /// Account for one committed token.
    ///
    /// Call once per token the decode loop commits, with `context` being the
    /// full token history (prompt + generated) BEFORE `realToken` is appended.
    /// Drives the round bookkeeping described on `SpeculationStats`.
    public func observe(realToken: Int32, context: [Int32]) {
        if !roundOpen {
            // Start a new verification round.
            roundOpen = true
            stats.rounds += 1
            let proposed = draft(context: context)
            if proposed.isEmpty {
                stats.roundsWithoutEvidence += 1
            } else {
                stats.draftsProposed += 1
                stats.draftTokensProposed += proposed.count
                pending = proposed
            }
        }
        stats.committedTokens += 1
        if let head = pending.first, head == realToken {
            stats.draftTokensAccepted += 1
            pending.removeFirst()
        } else {
            // First mismatch (or the draft ran out): `realToken` is the
            // round's bonus token, the remaining tail is rejected, round ends.
            pending.removeAll(keepingCapacity: true)
            roundOpen = false
        }
    }
}
