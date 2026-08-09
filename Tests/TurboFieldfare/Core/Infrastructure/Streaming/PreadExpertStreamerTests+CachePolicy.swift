import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

/// P6b-1: LRU eviction policy, slot pinning, and per-layer eviction counters.
extension PreadExpertStreamerTests {
  private func makeStreamer(_ url: URL,
                            slots: Int,
                            policy: ExpertCachePolicy) throws -> PreadExpertStreamer {
    let device = try MetalContext().device
    return try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device,
      slotCount: slots, cachePolicy: policy)
  }

  /// `adviseExpertMisses` reports only non-resident experts, so it is a
  /// residency probe that does not mutate cache state the way planning does.
  private func isResident(_ streamer: PreadExpertStreamer, _ expert: Int) -> Bool {
    streamer.adviseExpertMisses(experts: [expert]).requested == 0
  }

  /// Expert 0 is used often but not recently; expert 1 is the most recent but
  /// rare. LFU must evict the recent-rare expert, LRU the stale-popular one.
  /// Same access trace, divergent victims.
  @Test func lruAndLfuPickDifferentVictims() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }

    func survivor(policy: ExpertCachePolicy) throws -> (zero: Bool, one: Bool) {
      let streamer = try makeStreamer(url, slots: 2, policy: policy)
      _ = try streamer.loadExpertsCached(experts: [0])
      _ = try streamer.loadExpertsCached(experts: [0])
      _ = try streamer.loadExpertsCached(experts: [0])
      _ = try streamer.loadExpertsCached(experts: [1])
      _ = try streamer.loadExpertsCached(experts: [2])  // evicts exactly one
      return (isResident(streamer, 0), isResident(streamer, 1))
    }

    let lru = try survivor(policy: .lru)
    #expect(lru.zero == false)  // 0 is least recently used
    #expect(lru.one == true)

    let lfu = try survivor(policy: .lfu)
    #expect(lfu.zero == true)  // 0 is most frequently used
    #expect(lfu.one == false)
  }

  @Test func lruKeepsTheMostRecentlyUsedExpertResident() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try makeStreamer(url, slots: 2, policy: .lru)

    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [1])
    _ = try streamer.loadExpertsCached(experts: [0])  // 0 is now most recent
    _ = try streamer.loadExpertsCached(experts: [2])  // must evict 1

    #expect(isResident(streamer, 0))
    #expect(!isResident(streamer, 1))
  }

  @Test func pinnedSlotSurvivesAnEvictionRoundThatWouldTakeIt() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try makeStreamer(url, slots: 2, policy: .lru)

    let plan0 = streamer.planExpertsCached(experts: [0])
    _ = try streamer.executeExpertCachePlan(plan0)
    _ = try streamer.loadExpertsCached(experts: [1])
    // Expert 0 is now the LRU victim. Pin its slot and it must be spared.
    streamer.pinSlots(plan0.assignedSlots)
    #expect(streamer.pinnedSlotCount == 1)

    let plan2 = streamer.planExpertsCached(experts: [2])
    #expect(plan2.assignedSlots[0] != plan0.assignedSlots[0])
    _ = try streamer.executeExpertCachePlan(plan2)
    #expect(isResident(streamer, 0))
    #expect(streamer.cacheStats.pinOverrides == 0)
    #expect(streamer.cacheStats.pinnedProtections > 0)

    // After unpinning, the same slot becomes evictable again.
    streamer.unpinSlots(plan0.assignedSlots)
    #expect(streamer.pinnedSlotCount == 0)
  }

  @Test func pinningIsOverriddenRatherThanFailingWhenNoSlotIsFree() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try makeStreamer(url, slots: 2, policy: .lru)

    let warm = streamer.planExpertsCached(experts: [0, 1])
    _ = try streamer.executeExpertCachePlan(warm)
    streamer.pinSlots(warm.assignedSlots)  // every slot pinned

    let plan = streamer.planExpertsCachedIfPossible(experts: [2])
    #expect(plan != nil)
    #expect(streamer.cacheStats.pinOverrides == 1)
  }

  @Test func evictionCounterCountsOnlyDisplacedResidents() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try makeStreamer(url, slots: 2, policy: .lru)

    // Two cold fills land on empty slots — misses, but not evictions.
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [1])
    #expect(streamer.cacheStats.misses == 2)
    #expect(streamer.cacheStats.evictions == 0)

    _ = try streamer.loadExpertsCached(experts: [2])
    _ = try streamer.loadExpertsCached(experts: [3])
    #expect(streamer.cacheStats.evictions == 2)
    #expect(streamer.cacheStats.hits == 0)
  }

  @Test func cacheStatsAddCombinesEvictionAndPinCounters() {
    let a = ExpertCacheStats(lookups: 4, hits: 1, misses: 3, plans: 2,
                             evictions: 2, pinnedProtections: 1, pinOverrides: 0)
    let b = ExpertCacheStats(lookups: 2, hits: 2, misses: 0, plans: 1,
                             evictions: 5, pinnedProtections: 3, pinOverrides: 1)
    let sum = a + b
    #expect(sum.evictions == 7)
    #expect(sum.pinnedProtections == 4)
    #expect(sum.pinOverrides == 1)
    #expect(sum.lookups == 6)
  }
}
