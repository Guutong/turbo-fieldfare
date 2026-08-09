import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

/// P6b-3: prefill/decode phase attribution of routed-expert cache counters.
///
/// Qwen3.6 prefill is the decode loop replayed one prompt token at a time
/// (P3-3b), so no chunk of tokens is ever routed together and the streamer
/// cannot infer the phase from its own access pattern — the runner declares it.
/// These tests pin the attribution mechanics and the derived dedup metrics.
extension PreadExpertStreamerTests {

  /// Planning never reads the file, but the streamer requires a real regular
  /// file to open, so back these by the same synthetic layer the other suites use.
  static func makePhaseStreamer() throws -> PreadExpertStreamer {
    let device = try MetalContext().device
    let url = try Self.writeSyntheticLayer()
    return try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
  }

  @Test func streamerStartsInDecodePhase() throws {
    let streamer = try Self.makePhaseStreamer()
    #expect(streamer.cachePhase == .decode,
            "decode is the safe default: an unattributed lookup must not inflate prefill")
  }

  @Test func onlyPlansMadeInPrefillPhaseCountAsPrefill() throws {
    let streamer = try Self.makePhaseStreamer()

    // Decode-phase fetch: 4 cold misses, published into slots.
    _ = try streamer.loadExpertsCached(experts: [0, 1, 2, 3])
    var stats = streamer.cacheStats
    #expect(stats.lookups == 4)
    #expect(stats.prefillLookups == 0, "decode-phase lookups must not be attributed to prefill")

    // Prefill-phase plan: the same four experts, now all resident -> 4 hits.
    streamer.setCachePhase(.prefill)
    _ = try streamer.loadExpertsCached(experts: [0, 1, 2, 3])
    stats = streamer.cacheStats
    #expect(stats.lookups == 8)
    #expect(stats.prefillLookups == 4)
    #expect(stats.prefillHits == 4)
    #expect(stats.prefillMisses == 0)
    #expect(stats.prefillPlans == 1)
    // The derived decode view is the complement, never counted separately.
    #expect(stats.decodeLookups == 4)
    #expect(stats.decodeHits == 0)
    #expect(stats.decodeMisses == 4)
  }

  @Test func phaseCanBeSwitchedBackToDecode() throws {
    let streamer = try Self.makePhaseStreamer()
    streamer.setCachePhase(.prefill)
    _ = streamer.planExpertsCached(experts: [0, 1])
    streamer.setCachePhase(.decode)
    _ = streamer.planExpertsCached(experts: [0, 1])
    let stats = streamer.cacheStats
    #expect(stats.lookups == 4)
    #expect(stats.prefillLookups == 2, "post-prefill lookups stop accruing to prefill")
    #expect(stats.decodeLookups == 2)
  }

  @Test func prefillMissesAreCountedSeparatelyFromHits() throws {
    let streamer = try Self.makePhaseStreamer()
    streamer.setCachePhase(.prefill)
    _ = try streamer.loadExpertsCached(experts: [0, 1])  // 2 cold misses
    _ = try streamer.loadExpertsCached(experts: [0, 1])  // 2 hits
    let stats = streamer.cacheStats
    #expect(stats.prefillLookups == 4)
    #expect(stats.prefillMisses == 2)
    #expect(stats.prefillHits == 2)
    #expect(stats.prefillHitRate == 0.5)
    // 4 requests served by 2 disk reads = 2x less expert I/O than a no-reuse
    // engine that pread every routed expert of every prefill token.
    #expect(stats.prefillIOReductionFactor == 2.0)
  }

  @Test func prefillMetricsAreZeroWhenNothingWasAttributed() {
    let stats = ExpertCacheStats()
    #expect(stats.prefillHitRate == 0)
    #expect(stats.prefillIOReductionFactor == 0,
            "no prefill lookups must not report an infinite reduction")
  }

  @Test func prefillCountersSumAcrossLayers() {
    let lhs = ExpertCacheStats(lookups: 10, hits: 6, misses: 4, plans: 2,
                               prefillLookups: 8, prefillHits: 5,
                               prefillMisses: 3, prefillPlans: 1)
    let rhs = ExpertCacheStats(lookups: 4, hits: 1, misses: 3, plans: 1,
                               prefillLookups: 2, prefillHits: 1,
                               prefillMisses: 1, prefillPlans: 1)
    let sum = lhs + rhs
    #expect(sum.prefillLookups == 10)
    #expect(sum.prefillHits == 6)
    #expect(sum.prefillMisses == 4)
    #expect(sum.prefillPlans == 2)
    #expect(sum.decodeLookups == 4, "decode is derived from the totals minus prefill")
  }
}
