import Foundation
import Metal
import TurboFieldfareFormat

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

/// Batched wrapper around a per-layer `BatchedExpertCachePlan`.
/// Carries the unified plan plus a `[K][topK]` slot-index table so the
/// MoE kernel can drive from (tokenIdx, expertTopKIdx) → blob via lookup.
public struct RoutedExpertBatchFetchPlan: Sendable {
    public let layer: Int
    public let batchPlan: BatchedExpertCachePlan

    /// Number of tokens in the batch.
    public var tokenCount: Int { batchPlan.slotIndices.count }
    /// Top-K per token.
    public var topK: Int { batchPlan.slotIndices.first?.count ?? 0 }
    /// Unique experts in the combined resident set (= flat plan size).
    public var experts: [Int] { batchPlan.plan.experts }
    /// Experts that miss in cache (need disk fetch).
    public var misses: [Int] { batchPlan.plan.misses }
    /// Total expert selections across all tokens (= K * topK).
    public var totalLookups: Int { batchPlan.totalLookups }
    /// Distinct expert selections that resolved to a hit.
    public var hits: Int { batchPlan.plan.hits }

    public init(layer: Int, batchPlan: BatchedExpertCachePlan) {
        self.layer = layer
        self.batchPlan = batchPlan
    }
}

extension Model {
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    // MARK: - Batched multi-token expert planning (P7-1)

    /// Plan cached-expert fetch for K tokens' router selections across one layer.
    /// Collapses K×topK candidate experts into a single deduplicated plan.
    public func planRoutedExperts(layer: Int,
                                  tokens: [[Int]],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertBatchFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        let batchPlan = streamer.planExpertsCached(tokens: tokens, avoidingSlots: validSlots)
        return RoutedExpertBatchFetchPlan(layer: layer, batchPlan: batchPlan)
    }

    /// Execute a batched plan and return tensor views keyed by token-index then
    /// expert-topK-index. Returns `[TensorView][tokenIdx][topKIdx]`.
    public func fetchRoutedExperts(plan: RoutedExpertBatchFetchPlan) async throws
        -> [[TensorView]] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        // Execute once on the unified plan — all tokens share the resident set.
        let buffers = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let raw = try streamer.executeExpertCachePlan(plan.batchPlan.plan)
                    continuation.resume(returning: Self.makeBatchedExpertViews(raw, plan: plan))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        return buffers
    }

    private static func makeBatchedExpertViews(
        _ flatBuffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        plan: RoutedExpertBatchFetchPlan
    ) -> [[TensorView]] {
        // flatBuffers[i] ↔ plan.batchPlan.plan.experts[i] ↔ plan.experts[i].
        let viewsPerFlat = flatBuffers.enumerated().map { fi, entry -> TensorView in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(plan.layer), UInt32(plan.experts[fi]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }

        // assignedSlots[flatIdx] = slotIdx. Invert for reverse lookup.
        let assignedSlots = plan.batchPlan.plan.assignedSlots
        var slotToFlat = [Int: Int](minimumCapacity: assignedSlots.count)
        for (flatIdx, slotIdx) in assignedSlots.enumerated() {
            slotToFlat[slotIdx] = flatIdx
        }

        // Per-token: each slotIndex → TensorView via slot→flat lookup.
        return plan.batchPlan.slotIndices.map { slotRow -> [TensorView] in
            slotRow.map { slotIdx -> TensorView in
                guard slotIdx >= 0, let fi = slotToFlat[slotIdx] else {
                    // Invalid/unallocated slot — shouldn't happen if planning is correct.
                    preconditionFailure("invalid slot \(slotIdx) for layer \(plan.layer)")
                }
                return viewsPerFlat[fi]
            }
        }
    }

    /// P6b-1: pin the slots a committed encode is reading so a later plan for
    /// the same layer cannot select them as eviction victims. No-op for layers
    /// with no open streamer (mmap/full-resident modes).
    public func pinRoutedExpertSlots(layer: Int, slots: [Int]) {
        guard !slots.isEmpty else { return }
        streamersQueue.sync {
            guard streamersBox.streamers.indices.contains(layer) else { return }
            streamersBox.streamers[layer]?.pinSlots(slots)
        }
    }

    public func unpinRoutedExpertSlots(layer: Int, slots: [Int]) {
        guard !slots.isEmpty else { return }
        streamersQueue.sync {
            guard streamersBox.streamers.indices.contains(layer) else { return }
            streamersBox.streamers[layer]?.unpinSlots(slots)
        }
    }

    /// P6b-3: declare which generation phase routed-expert cache lookups should
    /// be attributed to. Applies to every already-open layer and is remembered
    /// for layers that open later.
    public func setExpertCachePhase(_ phase: ExpertCachePhase) {
        streamersQueue.sync {
            streamersBox.cachePhase = phase
            for streamer in streamersBox.streamers {
                streamer?.setCachePhase(phase)
            }
        }
    }

    public func expertCachePhase() -> ExpertCachePhase {
        streamersQueue.sync { streamersBox.cachePhase }
    }

    /// P6-1: cumulative routed-expert cache hit/miss counts across all opened layers.
    public func routedExpertCacheStats() -> ExpertCacheStats {
        streamersQueue.sync {
            streamersBox.streamers.compactMap { $0 }
                .reduce(ExpertCacheStats()) { $0 + $1.cacheStats }
        }
    }

    /// P6-1: per-layer cumulative routed-expert cache stats (nil for unopened layers).
    public func routedExpertCacheStatsByLayer() -> [ExpertCacheStats?] {
        streamersQueue.sync { streamersBox.streamers.map { $0?.cacheStats } }
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}
