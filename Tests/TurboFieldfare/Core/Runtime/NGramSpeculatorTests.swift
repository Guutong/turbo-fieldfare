import XCTest
@testable import TurboFieldfare

/// P6b-4 — n-gram speculative decoding drafter and round accounting.
final class NGramSpeculatorTests: XCTestCase {

    // MARK: - Drafting

    func testNoDraftWithoutEvidence() {
        let spec = NGramSpeculator()
        XCTAssertTrue(spec.draft(context: [1, 2, 3, 4, 5, 6, 7, 8]).isEmpty)
    }

    func testNoDraftWhenContextShorterThanShortestRung() {
        let spec = NGramSpeculator()
        XCTAssertTrue(spec.draft(context: [1, 2, 3]).isEmpty)
    }

    func testLengthFourSuffixMatchDraftsFollowingTokens() {
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 4))
        //            0  1  2  3  4   5   6   7  8  9 10 11
        let context: [Int32] = [1, 2, 3, 4, 90, 91, 92, 93, 1, 2, 3, 4]
        // Trailing 4-gram [1,2,3,4] recurs at index 0; what followed is 90..93.
        XCTAssertEqual(spec.draft(context: context), [90, 91, 92, 93])
    }

    func testDraftIsCappedAtMaxDraft() {
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 2))
        let context: [Int32] = [1, 2, 3, 4, 90, 91, 92, 93, 1, 2, 3, 4]
        XCTAssertEqual(spec.draft(context: context), [90, 91])
    }

    func testFallsBackToLengthThreeWhenNoLengthFourMatch() {
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 3))
        //                       0  1  2   3   4   5  6  7  8
        let context: [Int32] = [2, 3, 4, 70, 71, 72, 9, 2, 3, 4]
        // Trailing 4-gram is [4,9,2,3]... no earlier match. Trailing 3-gram
        // [2,3,4] recurs at index 0, followed by 70,71,72.
        XCTAssertEqual(spec.draft(context: context), [70, 71, 72])
    }

    func testLengthFourMatchPreferredOverLengthThreeMatch() {
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 1))
        // 3-gram [2,3,4] occurs at index 8 (followed by 50) and as part of the
        // 4-gram [1,2,3,4] at index 0 (followed by 40). The length-4 rung must
        // win even though the 3-gram evidence is more recent.
        let context: [Int32] = [1, 2, 3, 4, 40, 99, 98, 97, 2, 3, 4, 50, 1, 2, 3, 4]
        XCTAssertEqual(spec.draft(context: context), [40])
    }

    func testMostRecentOccurrenceWinsWithinARung() {
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 1))
        let context: [Int32] = [1, 2, 3, 4, 40, 1, 2, 3, 4, 50, 1, 2, 3, 4]
        XCTAssertEqual(spec.draft(context: context), [50])
    }

    func testShortSuffixRungsAreNotAttempted() {
        // A 2-gram match must NOT produce a draft: the ladder stops at 3.
        let spec = NGramSpeculator()
        let context: [Int32] = [7, 8, 55, 60, 61, 62, 7, 8]
        XCTAssertTrue(spec.draft(context: context).isEmpty)
    }

    // MARK: - Round accounting

    /// Replays a token stream through `observe`, feeding history as the decode
    /// loop does, and returns the resulting stats.
    private func replay(prompt: [Int32], generated: [Int32],
                        config: NGramSpeculatorConfig = .default) -> SpeculationStats {
        let spec = NGramSpeculator(config: config)
        var history = prompt
        for token in generated {
            spec.observe(realToken: token, context: history)
            history.append(token)
        }
        return spec.stats
    }

    func testNoEvidenceMeansOneTokenPerRound() {
        let stats = replay(prompt: [1, 2, 3, 4, 5], generated: [10, 11, 12, 13])
        XCTAssertEqual(stats.committedTokens, 4)
        XCTAssertEqual(stats.rounds, 4)
        XCTAssertEqual(stats.roundsWithoutEvidence, 4)
        XCTAssertEqual(stats.draftTokensProposed, 0)
        XCTAssertEqual(stats.tokensPerRound, 1.0, accuracy: 1e-9)
        XCTAssertEqual(stats.acceptanceRate, 0.0, accuracy: 1e-9)
    }

    func testFullyAcceptedDraftCommitsDraftPlusBonusInOneRound() {
        // Prompt establishes "1 2 3 4 -> 5 6 7 8"; generation repeats it and
        // then diverges with a bonus token.
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        let stats = replay(prompt: prompt, generated: [5, 6, 7, 8, 42],
                           config: NGramSpeculatorConfig(maxDraft: 4))
        XCTAssertEqual(stats.committedTokens, 5)
        // One round: 4 drafted tokens all accepted + 1 bonus token (42).
        XCTAssertEqual(stats.rounds, 1)
        XCTAssertEqual(stats.draftTokensProposed, 4)
        XCTAssertEqual(stats.draftTokensAccepted, 4)
        XCTAssertEqual(stats.acceptanceRate, 1.0, accuracy: 1e-9)
        XCTAssertEqual(stats.tokensPerRound, 5.0, accuracy: 1e-9)
    }

    func testPartialRejectionEndsTheRoundAtFirstMismatch() {
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        // Draft is [5,6,7,8]; the model really emits 5, 6, then 99.
        let stats = replay(prompt: prompt, generated: [5, 6, 99],
                           config: NGramSpeculatorConfig(maxDraft: 4))
        XCTAssertEqual(stats.committedTokens, 3)
        XCTAssertEqual(stats.rounds, 1)
        XCTAssertEqual(stats.draftTokensProposed, 4)
        // 5 and 6 accepted; 7 mismatched, 8 rejected with the tail.
        XCTAssertEqual(stats.draftTokensAccepted, 2)
        XCTAssertEqual(stats.acceptanceRate, 0.5, accuracy: 1e-9)
        XCTAssertEqual(stats.tokensPerRound, 3.0, accuracy: 1e-9)
    }

    func testRejectedTailIsNotCarriedIntoTheNextRound() {
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        let spec = NGramSpeculator(config: NGramSpeculatorConfig(maxDraft: 4))
        var history = prompt
        for token in [Int32(5), 99] {
            spec.observe(realToken: token, context: history)
            history.append(token)
        }
        XCTAssertEqual(spec.stats.rounds, 1)
        // Next observe must open a fresh round rather than resume [7, 8].
        spec.observe(realToken: 7, context: history)
        XCTAssertEqual(spec.stats.rounds, 2)
        XCTAssertEqual(spec.stats.committedTokens, 3)
    }

    func testRepeatedPatternDrivesTokensPerRoundAboveOne() {
        // A strongly repetitive stream: the classic best case for prompt-lookup.
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        var generated: [Int32] = []
        for _ in 0..<6 { generated.append(contentsOf: [1, 2, 3, 4, 5, 6, 7, 8]) }
        let stats = replay(prompt: prompt, generated: generated,
                           config: NGramSpeculatorConfig(maxDraft: 4))
        XCTAssertEqual(stats.committedTokens, 48)
        XCTAssertGreaterThan(stats.tokensPerRound, 1.5)
        XCTAssertGreaterThan(stats.acceptanceRate, 0.5)
    }

    func testResetClearsPendingDraftAndStats() {
        let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        let spec = NGramSpeculator()
        spec.observe(realToken: 5, context: prompt)
        XCTAssertGreaterThan(spec.stats.committedTokens, 0)
        spec.reset()
        XCTAssertEqual(spec.stats, SpeculationStats())
        // A fresh round must open on the next observe.
        spec.observe(realToken: 5, context: prompt)
        XCTAssertEqual(spec.stats.rounds, 1)
    }

    // MARK: - Byte-identity invariant

    /// The core P6b-4 correctness claim: speculation never changes the emitted
    /// token stream, because `observe` only ever compares against tokens the
    /// serial path already produced. Replaying an arbitrary stream must leave
    /// the committed count equal to the real stream length, whatever the
    /// drafts were.
    func testCommittedStreamAlwaysMatchesRealStreamLength() {
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<50 {
            let prompt = (0..<20).map { _ in Int32.random(in: 0..<6, using: &rng) }
            let generated = (0..<40).map { _ in Int32.random(in: 0..<6, using: &rng) }
            let spec = NGramSpeculator()
            var history = prompt
            var committed: [Int32] = []
            for token in generated {
                spec.observe(realToken: token, context: history)
                committed.append(token)
                history.append(token)
            }
            XCTAssertEqual(committed, generated)
            XCTAssertEqual(spec.stats.committedTokens, generated.count)
            XCTAssertLessThanOrEqual(spec.stats.rounds, generated.count)
            XCTAssertLessThanOrEqual(spec.stats.draftTokensAccepted,
                                     spec.stats.draftTokensProposed)
        }
    }
}
