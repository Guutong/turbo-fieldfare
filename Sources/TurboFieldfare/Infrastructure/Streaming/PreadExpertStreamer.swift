import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

/// Cumulative routed-expert cache lookup counters (P6-1 instrumentation).
public struct ExpertCacheStats: Sendable, Equatable {
    public var lookups: Int
    public var hits: Int
    public var misses: Int
    public var plans: Int
    /// P6b-1: number of resident experts displaced from a slot by a miss.
    /// A miss that lands on a never-used (empty) slot is not an eviction.
    public var evictions: Int
    /// P6b-1: number of times a pinned slot was excluded from the eviction
    /// candidate set while planning a miss.
    public var pinnedProtections: Int
    /// P6b-1: number of times pinning had to be overridden because there were
    /// not enough unpinned slots to place the plan's misses. Should stay 0.
    public var pinOverrides: Int
    /// P6b-2: cumulative wall-clock nanoseconds spent inside phase 2 of the
    /// batched acquire (the parallel, offset-sorted preads). This is the
    /// directly measurable "I/O wall time" the batching is meant to reduce.
    public var readNanos: UInt64
    /// P6b-2: largest number of preads observed in flight simultaneously.
    /// Must never exceed `PreadExpertStreamer.prefetchQueueDepth`.
    public var peakInFlightReads: Int
    /// P6b-3: the subset of `lookups` that occurred while the streamer was in
    /// `.prefill` phase. Qwen3.6 prefill is a per-token replay of the decode
    /// loop (P3-3b), so prefill and decode share every counter above; this
    /// split is what makes a prefill-only reuse (dedup) rate measurable.
    public var prefillLookups: Int
    /// P6b-3: the subset of `hits` that occurred during prefill. Each prefill
    /// hit is one expert blob NOT re-read from disk — the dedup saving.
    public var prefillHits: Int
    /// P6b-3: the subset of `misses` that occurred during prefill, i.e. the
    /// expert blobs prefill actually read from disk.
    public var prefillMisses: Int
    /// P6b-3: the subset of `plans` made during prefill.
    public var prefillPlans: Int

    public init(lookups: Int = 0,
                hits: Int = 0,
                misses: Int = 0,
                plans: Int = 0,
                evictions: Int = 0,
                pinnedProtections: Int = 0,
                pinOverrides: Int = 0,
                readNanos: UInt64 = 0,
                peakInFlightReads: Int = 0,
                prefillLookups: Int = 0,
                prefillHits: Int = 0,
                prefillMisses: Int = 0,
                prefillPlans: Int = 0) {
        self.prefillLookups = prefillLookups
        self.prefillHits = prefillHits
        self.prefillMisses = prefillMisses
        self.prefillPlans = prefillPlans
        self.lookups = lookups
        self.hits = hits
        self.misses = misses
        self.plans = plans
        self.evictions = evictions
        self.pinnedProtections = pinnedProtections
        self.pinOverrides = pinOverrides
        self.readNanos = readNanos
        self.peakInFlightReads = peakInFlightReads
    }

    public var hitRate: Double {
        lookups > 0 ? Double(hits) / Double(lookups) : 0
    }

    /// P6b-3: prefill-only hit rate. This IS the prefill expert dedup rate —
    /// the fraction of prefill expert requests served from an already-resident
    /// blob instead of a fresh pread.
    public var prefillHitRate: Double {
        prefillLookups > 0 ? Double(prefillHits) / Double(prefillLookups) : 0
    }

    /// P6b-3: prefill I/O reduction factor versus a hypothetical no-reuse
    /// engine that pread every routed expert of every prefill token. Equal to
    /// `prefillLookups / prefillMisses`; 1.0 means no reuse at all.
    public var prefillIOReductionFactor: Double {
        prefillMisses > 0 ? Double(prefillLookups) / Double(prefillMisses)
                          : (prefillLookups > 0 ? Double.infinity : 0)
    }

    public var decodeLookups: Int { lookups - prefillLookups }
    public var decodeHits: Int { hits - prefillHits }
    public var decodeMisses: Int { misses - prefillMisses }

    public static func + (lhs: ExpertCacheStats, rhs: ExpertCacheStats) -> ExpertCacheStats {
        ExpertCacheStats(lookups: lhs.lookups + rhs.lookups,
                         hits: lhs.hits + rhs.hits,
                         misses: lhs.misses + rhs.misses,
                         plans: lhs.plans + rhs.plans,
                         evictions: lhs.evictions + rhs.evictions,
                         pinnedProtections: lhs.pinnedProtections + rhs.pinnedProtections,
                         pinOverrides: lhs.pinOverrides + rhs.pinOverrides,
                         readNanos: lhs.readNanos &+ rhs.readNanos,
                         peakInFlightReads: max(lhs.peakInFlightReads, rhs.peakInFlightReads),
                         prefillLookups: lhs.prefillLookups + rhs.prefillLookups,
                         prefillHits: lhs.prefillHits + rhs.prefillHits,
                         prefillMisses: lhs.prefillMisses + rhs.prefillMisses,
                         prefillPlans: lhs.prefillPlans + rhs.prefillPlans)
    }
}

/// P6b-3: which generation phase the engine is currently in, so routed-expert
/// cache counters can be attributed to prompt prefill versus decode. Qwen3.6's
/// prefill runs through the same per-token decode loop (P3-3b), so nothing in
/// the streamer can infer the phase on its own — the runner must declare it.
public enum ExpertCachePhase: String, Sendable {
    case prefill
    case decode
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// A batched expert-cache plan for K tokens' worth of router selections.
///
/// Unlike `ExpertCachePlan` which serves one token's top-K experts, this struct
/// represents one planning call that accepts K × topK candidate expert IDs,
/// deduplicates them across all K tokens, and returns a single fetch plan plus
/// a per-[token][expert] slot-index lookup so the MoE kernel can drive from the
/// right slot per position without CPU-GPU sync.
public struct BatchedExpertCachePlan: Sendable {
    /// The unified fetch plan — deduplicated experts + their slots.
    public let plan: ExpertCachePlan

    /// `slotIndices[tokenIndex][expertInTopK]` → the deduped slot index into
    /// `slotPointers` / `slotBuffers`, or -1 when the deduped plan found no
    /// slot yet (should not happen after execution). Kept parallel to the
    /// original `[token][expert]` topology so the MoE kernel doesn't need
    /// to reshape anything on the CPU side before encoding.
    public let slotIndices: [[Int]]

    /// How many unique experts were fetched across all K tokens.
    public var uniqueFetchCount: Int { plan.misses.count }

    /// Total lookups performed (K × topK per layer).
    public var totalLookups: Int { plan.experts.count }

    /// Hit rate across all tokens.
    public var hitRate: Double {
        guard totalLookups > 0 else { return 0 }
        return Double(plan.hits) / Double(totalLookups)
    }

    public init(plan: ExpertCachePlan, slotIndices: [[Int]]) {
        self.plan = plan
        self.slotIndices = slotIndices
    }
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }
    /// P6b-2: maximum preads in flight at once during phase 2 of a batched
    /// acquire (plan.md's "queue depth 16"). With Qwen3.6's top-K of 8 a single
    /// layer's plan never reaches this cap; it bounds larger batches (prefill
    /// dedup, wider top-K) from swamping the I/O subsystem.
    public static let prefetchQueueDepth = 16
    /// A/B escape hatch for the offset sort so the same binary can measure
    /// sorted vs router-order issue. Default on.
    static let prefetchSortEnabled =
        ProcessInfo.processInfo.environment["TFF_EXPERT_PREFETCH_SORT"] != "0"
    /// P6b-5: A/B escape hatch for F_NOCACHE on the expert weight file
    /// descriptor. Bypassing the OS page cache for these transient,
    /// never-reused expert reads keeps it from evicting the resident
    /// backbone weights' pages. Default on; measure sorted-off to confirm.
    static let noCacheEnabled =
        ProcessInfo.processInfo.environment["TFF_EXPERT_NOCACHE"] != "0"

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    /// P6b-1: per-slot pin depth. A slot with a non-zero pin count is holding
    /// weights an in-flight encode is still reading, so it must not be chosen
    /// as an eviction victim. Nested pins are counted, not booleaned, so
    /// overlapping pin/unpin pairs cannot unpin each other's slot early.
    private var slotPinCount: [Int]
    private var useClock = 0
    private let cacheLock = NSLock()

    /// P6-1 instrumentation: cumulative hit/miss counts over accepted cache plans.
    /// Mutated only under `cacheLock`, on the same critical section the planner
    /// already takes, so it adds no extra synchronization to the hot path.
    private var stats = ExpertCacheStats()

    /// P6b-3: current generation phase. Read and written under `cacheLock`,
    /// the same lock the planner already holds, so attributing a plan to a
    /// phase costs no extra synchronization.
    private var phase: ExpertCachePhase = .decode

    public var cacheStats: ExpertCacheStats {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return stats
    }

    public var cachePhase: ExpertCachePhase {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return phase
    }

    public func setCachePhase(_ newPhase: ExpertCachePhase) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        phase = newPhase
    }

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        if Self.noCacheEnabled {
            _ = fcntl(openedFD, F_NOCACHE, 1)
        }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotPinCount = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    /// P6b-1: mark slots as ineligible for eviction until the matching
    /// `unpinSlots` call. Callers pin the slots a committed GPU encode is
    /// reading, so a concurrently planned fetch cannot pread over them.
    public func pinSlots(_ slots: [Int]) {
        guard !slots.isEmpty else { return }
        cacheLock.lock()
        for slot in slots where slot >= 0 && slot < slotCount {
            slotPinCount[slot] &+= 1
        }
        cacheLock.unlock()
    }

    public func unpinSlots(_ slots: [Int]) {
        guard !slots.isEmpty else { return }
        cacheLock.lock()
        for slot in slots where slot >= 0 && slot < slotCount {
            slotPinCount[slot] = max(0, slotPinCount[slot] - 1)
        }
        cacheLock.unlock()
    }

    public var pinnedSlotCount: Int {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return slotPinCount.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    // MARK: - Batched multi-token planning (P7-1)

    /// Plan a cached expert fetch for K tokens' router selections in ONE call.
    ///
    /// Accepts `tokens` where each element is an array of `topK` expert IDs
    /// selected by the layer's router for that token. All K × topK candidate
    /// IDs are flattened, deduplicated, and fed to the existing cache planner;
    /// the result is a combined `BatchedExpertCachePlan` with:
    ///  - `plan`: standard `ExpertCachePlan` over the deduplicated set
    ///  - `slotIndices[K][topK]`: maps every [token][expert] back to its
    ///    assigned slot so the MoE kernel can drive from the right blob
    ///
    /// This collapses what would be K separate cache plans (and thus K round
    /// trips to disk for distinct misses) into one — enabling the layer-major
    /// execution path Phase 7 targets.
    public func planExpertsCached(tokens: [[Int]],
                                  avoidingSlots: Set<Int> = []) -> BatchedExpertCachePlan {
        // Collect all unique expert IDs in stable order (first occurrence wins).
        let topK = tokens.first?.count ?? 0
        precondition(!tokens.isEmpty && topK > 0, "tokens must be non-empty with topK > 0")
        precondition(tokens.allSatisfy { $0.count == topK },
                     "all tokens must have the same topK count")

        var seenExperts = [Int: Int]() // expertID → flat index
        var flatExperts = [Int]()     // flat index → expertID
        for tokExperts in tokens {
            for eid in tokExperts {
                if seenExperts[eid] == nil {
                    seenExperts[eid] = flatExperts.count
                    flatExperts.append(eid)
                }
            }
        }

        // Build the [flatIndex] → [(tokenIdx, expertInTopK)] lookup.
        var flatToPositions = [Int: [(t: Int, e: Int)]]()
        for (tIdx, tokExperts) in tokens.enumerated() {
            for (eIdx, eid) in tokExperts.enumerated() {
                let fi = seenExperts[eid]!
                flatToPositions[fi, default: []].append((tIdx, eIdx))
            }
        }

        // Plan across ALL unique experts at once — cache hits and misses merged.
        let plan = planExpertsCached(experts: flatExperts, avoidingSlots: avoidingSlots)

        // Look up the slot for each flat expert via plan's assignment map.
        let flatSlotFor = { (flatIdx: Int) -> Int in
            plan.assignedSlots[flatIdx]
        }

        // Wire up per-token slot indices.
        var slotIndices = tokens.map { _ in [Int](repeating: -1, count: topK) }
        for (fi, positions) in flatToPositions {
            let slot = flatSlotFor(fi)
            for (tIdx, eIdx) in positions {
                slotIndices[tIdx][eIdx] = slot
            }
        }

        return BatchedExpertCachePlan(plan: plan, slotIndices: slotIndices)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let unreserved = (0..<slotCount).filter { !reserved[$0] }
        // P6b-1: pinned slots are held by an in-flight encode and are excluded
        // from the victim set. If (and only if) the plan cannot otherwise be
        // placed, pinning is overridden rather than failing the fetch; that is
        // counted so it shows up in the stats instead of silently corrupting a
        // read. With top-K << slotCount it should never happen.
        let unpinned = unreserved.filter { slotPinCount[$0] == 0 }
        let protectedCount = unreserved.count - unpinned.count
        var evictable = unpinned.sorted { shouldEvictSlot($0, before: $1) }
        var pinOverrides = 0
        var pinnedProtections = 0
        if !misses.isEmpty {
            pinnedProtections = protectedCount
        }
        if misses.count > evictable.count {
            let pinned = unreserved
                .filter { slotPinCount[$0] > 0 }
                .sorted { shouldEvictSlot($0, before: $1) }
            pinOverrides = min(misses.count - evictable.count, pinned.count)
            pinnedProtections = max(0, protectedCount - pinOverrides)
            evictable.append(contentsOf: pinned)
        }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        stats.lookups &+= experts.count
        stats.pinnedProtections &+= pinnedProtections
        stats.pinOverrides &+= pinOverrides
        stats.hits &+= experts.count - misses.count
        stats.misses &+= misses.count
        stats.plans &+= 1
        if phase == .prefill {
            // P6b-3: same four adds, attributed to the prompt-prefill phase.
            stats.prefillLookups &+= experts.count
            stats.prefillHits &+= experts.count - misses.count
            stats.prefillMisses &+= misses.count
            stats.prefillPlans &+= 1
        }
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            if slotExpert[slot] >= 0 { stats.evictions &+= 1 }
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        // P6b-2 — 3-phase batched acquire (kimi-k3 `getmany`).
        //
        // Phase 1 (serial, under `cacheLock`) already happened in
        // `makeExpertCachePlan`: each miss has a slot reserved, and that slot's
        // `slotExpert` was set to -1, which *is* the in-flight marker — no other
        // lookup can claim it as a hit, and P6b-1's pinning keeps a concurrent
        // encode's slots out of the victim set. Nothing is published yet.
        //
        // Phase 2 (parallel, sorted by ascending disk offset, bounded queue
        // depth) is below. Phase 3 (serial publish) follows it.
        let order = Self.offsetSortedMissOrder(plan: plan, layout: layout)
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        nonisolated(unsafe) var inFlight = 0
        nonisolated(unsafe) var peakInFlight = 0
        let start = DispatchTime.now().uptimeNanoseconds

        for wave in stride(from: 0, to: order.count, by: Self.prefetchQueueDepth) {
            let waveCount = min(Self.prefetchQueueDepth, order.count - wave)
            DispatchQueue.concurrentPerform(iterations: waveCount) { waveOffset in
                let index = order[wave + waveOffset]
                errorLock.lock()
                inFlight += 1
                peakInFlight = max(peakInFlight, inFlight)
                errorLock.unlock()
                do {
                    _ = try self.loadExpert(
                        layer: 0,
                        expert: plan.experts[index],
                        slot: plan.assignedSlots[index])
                } catch {
                    errorLock.lock()
                    if firstError == nil { firstError = error }
                    errorLock.unlock()
                }
                errorLock.lock()
                inFlight -= 1
                errorLock.unlock()
            }
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds &- start

        // Phase 3 (serial, under `cacheLock`): publish. A slot only becomes
        // visible as a hit after its read has completed; on any read failure
        // nothing is published, so a half-read slot can never be mistaken for
        // resident weights.
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
        }
        stats.readNanos &+= elapsed
        stats.peakInFlightReads = max(stats.peakInFlightReads, peakInFlight)
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    /// P6b-2: the order phase 2 issues a plan's misses in — ascending absolute
    /// file offset, so the read stream walks the layer file forward instead of
    /// jumping around in router top-K order. Ties (impossible for distinct
    /// experts, possible for a degenerate layout) break on plan index so the
    /// order is deterministic. Returns indices into `plan.experts`.
    static func offsetSortedMissOrder(plan: ExpertCachePlan, layout: StreamLayout) -> [Int] {
        guard Self.prefetchSortEnabled else { return plan.misses }
        return plan.misses.sorted { lhs, rhs in
            let lhsOffset = layout.expertOffset(layer: 0, expert: plan.experts[lhs])
            let rhsOffset = layout.expertOffset(layer: 0, expert: plan.experts[rhs])
            if lhsOffset != rhsOffset { return lhsOffset < rhsOffset }
            return lhs < rhs
        }
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
