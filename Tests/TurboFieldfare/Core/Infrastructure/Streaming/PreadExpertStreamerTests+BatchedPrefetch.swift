import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

/// P6b-2: batched, offset-sorted expert prefetch (3-phase acquire).
extension PreadExpertStreamerTests {

  /// A layout whose expert offsets are deliberately *not* monotonic in expert
  /// index, so "sorted by disk offset" is observably different from "sorted by
  /// expert index" and from router top-K order.
  static func shuffledOffsetLayout(path: String) -> StreamLayout {
    // expert 0 -> last blob, 1 -> second, 2 -> first, 3 -> third.
    let stride = UInt64(expertStride)
    return StreamLayout(
      path: path,
      streamOffset: streamOffset,
      streamSize: streamSize,
      expertsPerLayer: numExperts,
      expertStride: stride,
      expertOffsets: [3 * stride, 1 * stride, 0 * stride, 2 * stride])
  }

  @Test func missOrderIsSortedByAscendingDiskOffset() throws {
    let layout = Self.shuffledOffsetLayout(path: "/dev/null")
    // Router order 0,1,2,3 -> offsets 3,1,0,2 strides. All four are misses.
    let plan = ExpertCachePlan(
      experts: [0, 1, 2, 3], assignedSlots: [0, 1, 2, 3], misses: [0, 1, 2, 3], hits: 0)
    let order = PreadExpertStreamer.offsetSortedMissOrder(plan: plan, layout: layout)
    #expect(order == [2, 1, 3, 0])
    let offsets = order.map { layout.expertOffset(layer: 0, expert: plan.experts[$0]) }
    #expect(offsets == offsets.sorted(), "phase 2 must issue preads in ascending offset order")
  }

  @Test func missOrderSkipsHitsAndStaysDeterministic() throws {
    let layout = Self.shuffledOffsetLayout(path: "/dev/null")
    // Only indices 1 and 3 are misses; hits must not appear in the read order.
    let plan = ExpertCachePlan(
      experts: [0, 1, 2, 3], assignedSlots: [0, 1, 2, 3], misses: [1, 3], hits: 2)
    let order = PreadExpertStreamer.offsetSortedMissOrder(plan: plan, layout: layout)
    // expert 1 is at 1*stride, expert 3 at 2*stride -> already ascending.
    #expect(order == [1, 3])
    #expect(
      PreadExpertStreamer.offsetSortedMissOrder(plan: plan, layout: layout) == order,
      "ordering must be deterministic across calls")
  }

  @Test func emptyMissSetProducesNoReads() throws {
    let layout = Self.shuffledOffsetLayout(path: "/dev/null")
    let plan = ExpertCachePlan(
      experts: [0, 1], assignedSlots: [0, 1], misses: [], hits: 2)
    #expect(PreadExpertStreamer.offsetSortedMissOrder(plan: plan, layout: layout).isEmpty)
  }

  /// Reads issued in offset order must still land in the slot the plan reserved
  /// for each expert — sorting changes issue order, never the mapping.
  @Test func offsetSortedFetchStillPlacesEachExpertInItsReservedSlot() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    // Router order is deliberately descending so offset-sorting reverses it.
    let experts = [3, 2, 1, 0]
    let plan = streamer.planExpertsCached(experts: experts)
    #expect(plan.misses.count == 4)
    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(experts[index]) },
        "expert \(experts[index]) did not land in its reserved slot")
    }
  }

  @Test func queueDepthCapIsRespected() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [3, 2, 1, 0])
    let stats = streamer.cacheStats
    #expect(stats.peakInFlightReads > 0, "phase 2 should have recorded in-flight reads")
    #expect(
      stats.peakInFlightReads <= PreadExpertStreamer.prefetchQueueDepth,
      "queue depth cap exceeded: \(stats.peakInFlightReads)")
    #expect(stats.readNanos > 0, "phase 2 wall time should be recorded")
  }

  /// Phase-3 lock safety: if any read in the batch fails, nothing is published,
  /// so no slot can be observed as resident holding a partially-read expert.
  @Test func failedReadPublishesNothing() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    // Truncate past expert 0 only: expert 0's read succeeds, the rest hit EOF.
    let truncatedLen = off_t(Self.streamOffset) + off_t(Self.expertStride)
    #expect(truncate(url.path, truncatedLen) == 0)

    let experts = [0, 1, 2, 3]
    #expect(throws: StreamerError.self) {
      _ = try streamer.loadExpertsCached(experts: experts)
    }
    // Re-planning the same experts must report them all as misses again: the
    // failed batch published nothing, not even the one read that succeeded.
    let replan = streamer.planExpertsCached(experts: experts)
    #expect(replan.hits == 0, "a failed batch must leave no slot marked resident")
    #expect(replan.misses.count == experts.count)
  }

  @Test func statsSummationCombinesIOTimeAndPeakDepth() {
    let lhs = ExpertCacheStats(readNanos: 100, peakInFlightReads: 3)
    let rhs = ExpertCacheStats(readNanos: 250, peakInFlightReads: 7)
    let sum = lhs + rhs
    #expect(sum.readNanos == 350, "I/O wall time accumulates across layers")
    #expect(sum.peakInFlightReads == 7, "peak depth is a max, not a sum")
  }
}
