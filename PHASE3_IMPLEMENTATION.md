# Phase 3 Implementation Plan: Layer-Major Spec Decode

## Goal
Restructure DraftVerifier from token-major to layer-major to enable batched expert fetching.

**Expected gain**: ~342ms saved per K=4 round → 5.5-6 tok/s (from 4.15 baseline)

## Current Structure (Token-Major)

```swift
for tk in 0..<K {
  snapshot DeltaNet state
  
  for L in 0..<numLayers {
    // Encode layer L for token tk
    if isDeltaNet {
      deltaNet.encode(hiddenOffset: tokenOff)
      commit, wait
    } else {
      // Full attention
      RMSNorm, QKV GEMV, attention, o_proj
      commit, wait
    }
    
    // MoE routing
    encode router
    commit, wait, read back expert indices
    append to layerPlans
    
    // Drain MoE (if needed)
    if !layerPlans.isEmpty {
      finishPendingMoE(...)  // fetch experts, encode phase1/phase2
    }
  }
  
  // Final norm + lm_head + argmax for this token
}

restore to snapshot[0]
```

**Problem**: 
- K×40 CB commits (160 for K=4)
- K×40 expert fetches (160 fetches)
- K×40 CPU waits

## Target Structure (Layer-Major)

```swift
// Snapshot DeltaNet state ONCE before layer loop

for L in 0..<numLayers {
  // Encode layer L for all K tokens
  
  if isDeltaNet {
    // Use encodeBatched(K) — processes all K tokens in one call
    deltaNet.encodeBatched(tk: K)
    commit, wait
  } else {
    // Full attention — loop K times in same CB
    let cb = makeCommandBuffer()
    for tk in 0..<K {
      RMSNorm(xOffset: tk*D)
      QKV GEMV(xOffset: tk*D)
      attention
      o_proj(xOffset: tk*D)
    }
    commit, wait
  }
  
  // MoE routing for all K tokens
  let routerCB = makeCommandBuffer()
  for tk in 0..<K {
    encode router(xOffset: tk*D)
  }
  commit, wait
  
  // Read back all K sets of expert indices
  let allKExperts: [[Int]] = readback K sets
  
  // Batched expert fetch (ONE fetch for union of all K experts)
  let batchPlan = model.planRoutedExperts(layer: L, tokens: allKExperts)
  let batchViews = try await model.fetchRoutedExperts(plan: batchPlan)
  
  // Encode MoE phase1/phase2 for all K tokens
  let moeCB = makeCommandBuffer()
  for tk in 0..<K {
    phase1(xOffset: tk*D, experts: batchViews[tk])
    phase2(xOffset: tk*D, experts: batchViews[tk])
  }
  commit, wait
}

// Final norm + lm_head + argmax for all K tokens
for tk in 0..<K {
  finalNorm(xOffset: tk*D)
  lmHead(xOffset: tk*D)
  argmax
}

restore to snapshot[0]
```

**Benefit**:
- 40 CB commits (not 160)
- 40 expert fetches (not 160) — union of K experts per layer
- 40 CPU waits (not 160)

## Implementation Steps

### Step 1: Foundation (DONE)
- ✅ Add xOffset support to FusedQKVGEMV
- ✅ Remove copyScratchRegion workaround

### Step 2: Loop Invert (HIGH RISK)
Restructure the main loop from token-major to layer-major.

**Changes needed**:
1. Move DeltaNet snapshot outside layer loop (line 127-129 → before line 134)
2. Invert loops: `for L in 0..<numLayers { for tk in 0..<K { ... } }`
3. DeltaNet: replace `encode(hiddenOffset: tokenOff)` with `encodeBatched(tk: K)` (line 220)
4. Full attention: wrap lines 233-550 in `for tk in 0..<K` loop, keep same CB
5. MoE routing: collect all K expert sets before drain (line 561-618)
6. MoE drain: call once per layer with all K plans (line 191-194)
7. Final norm + head: move outside layer loop, loop K times (line 622-667)

**Risk**: Very high. 600+ lines of loop body affected. State management, KV cache writes, DeltaNet recurrence all need careful handling.

**Mitigation**: 
- Implement incrementally with build checks after each major change
- Test after each step with `swift test --filter DeltaNet`
- Benchmark after completion to verify CB count drops

### Step 3: Wire Batched Expert APIs (MEDIUM RISK)
Replace per-token expert fetch with union batch fetch.

**Changes needed in finishPendingMoE**:
1. Accept `[(layer: Int, experts: [[Int]])]` instead of `[(layer: Int, experts: [Int])]`
2. Call `model.planRoutedExperts(layer:tokens:)` instead of per-token planning
3. Call `model.fetchRoutedExperts(plan: RoutedExpertBatchFetchPlan)` instead of per-token fetch
4. Pass `batchViews[tk][e]` to phase1/phase2 kernels

**Risk**: Medium. Batched APIs exist but are untested in hot path.

### Step 4: Batch MoE Dispatch (LOW RISK)
Loop K times in same encoder for phase1/phase2.

**Changes needed**:
1. Create one encoder, loop K times dispatching phase1/phase2 with different offsets
2. Same pattern as LMHeadChain encoder merge (P8-2d)

**Risk**: Low. Proven pattern.

## Critical Dependencies

### DeltaNet encodeBatched
- Already exists (DeltaNetMetalBlock.swift:620)
- Processes K tokens sequentially in one call
- Handles state updates correctly (token k depends on k-1)
- Constraint: `tk <= maxTK` (default 128)

### KV Cache Writes
- Each token writes to `startKVPosition + tk` (different ring-buffer slots)
- Kernels support offsets (xOffset, hiddenOffset)
- No conflict between tokens (causal order preserved)

### Batched Expert APIs
- `planRoutedExperts(layer:tokens:avoidingSlots:)` — ModelExpertIO.swift:122
- `fetchRoutedExperts(plan: RoutedExpertBatchFetchPlan)` — ModelExpertIO.swift:134
- Both exist but have ZERO callers today
- Need to verify batchViews indexing matches kernel expectations

## Testing Strategy

After each step:
1. `swift build` — compilation clean
2. `swift test --filter DeltaNet` — parity tests pass
3. Correctness: `TFF_SPEC_DECODE=1` run, compare output token-for-token (must match greedy baseline)
4. Performance: `TFF_PHASE_TIMING=1` — verify CB count drops, io phase drops, tok/s improves

## Estimated Effort

- Step 2 (loop invert): 2-3 days (high complexity, careful testing needed)
- Step 3 (batched expert): 1 day (wire existing APIs)
- Step 4 (batched dispatch): 0.5 day (simple pattern)

**Total**: ~4 days for Phase 3 completion

## Rollback Plan

If Step 2 (loop invert) proves too risky or introduces bugs:
- Keep token-major structure
- Skip Phase 3
- Focus on Phase 4 (true batched kernels) for future speed gains

## Success Criteria

- CB count drops from ~160 to ~40 per K=4 round
- io phase drops from ~104ms×K to ~104ms (union fetch)
- tok/s improves from 4.15 to 5.5+
- Correctness: output matches greedy baseline token-for-token
