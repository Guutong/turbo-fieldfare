# Phase 7: Batched Speculative Decoding — Status Summary

**Branch:** `qwen36-bringup`
**Target model:** Qwen3.6-35B-A3B-MoE on M2 MacBook Air (16GB)
**Mission gate:** ≥4 tok/s decode throughput (current: ~2.3 tok/s single-token)

---

## Completed Work

### P7-1: Batched routed-expert cache plan across K tokens ✅
**Task #32 completed.**

Added `planRoutedExperts(layer:tokens:avoidingSlots:)` in `ModelExpertIO.swift` that collapses K×topK candidate experts into a single deduplicated batch fetch plan (`RoutedExpertBatchFetchPlan`). This is the foundation for reducing expert I/O from K separate disk seeks down to one unified fetch per layer.

**Key files modified:**
- `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift` — added batch planning API

### P7-3: Layer-major orchestration + DraftVerifier ✅
**Task #37 completed.**

Integrated speculative decoding draft-and-verify into RawCompletion. The `DraftVerifier.run()` function takes draft tokens, runs them through the full forward pass (all layers), and returns greedy acceptance results. Key pieces wired:
- Draft generation via NGramSpeculator
- Verification loop calling DraftVerifier for each round of K drafts
- Acceptance counting and bonus token selection
- History/appending ordering preserved

### P7-4: Wire batched forward pass into decode loop ✅
**Task #35 completed.**

The decode loop in `RawCompletion.swift` now calls `DraftVerifier.run()` for speculative verification when available. Acceptance logic handles: partial matches (bonus token at mismatch), full acceptance (continue serial path), and error fallback (reset speculator, fall back to single-token).

### finishPendingMoE stub removal ✅
**Task #39 completed.**

Critical blocker fixed: replaced the `finishPendingMoE` stub that threw `ModelError.unsupportedArchFeature("BatchedMoE")` with actual MoE execution code. Since Qwen3.6 has routed experts on ALL 40 layers, this was preventing ANY speculative decoding from executing.

**Final implementation (DraftVerifier.swift):**
Per-layer pipeline executed sequentially for each draft token:
1. Plan expert cache fetch (hits vs misses) via `planRoutedExpertsIfPossible`
2. Pin cache slots before async I/O
3. Shared expert FFN → h1Buf via Metal command buffer
4. Apply sigmoid gate scalar on CPU (Qwen3.6 qwen3_5_moe topology only)
5. Async disk fetch for expert misses → TensorViews
6. Phase-1: U16 load + activation into moeActs buffer
7. Phase-2: reduce + down-project → h2Buf
8. Tail kernel: hidden + h1 + routedScale*h2 (preNorm topology)
9. Sync wait + unpin slots

**API surface used (verified against source):**
- `runner.ctx.queue` — Metal command queue
- `runner.shared.encode(...)` — shared expert FFN (all offset params required)
- `runner.moe.makeReusedRoutedArgumentBuffer(...)` — builds arg buffer for MoE kernels
- `runner.moe.encodeRoutedPersistentPhase1U16Load(...)` — expert load + silu/gelu
- `runner.moe.encodeRoutedPersistentPhase2Reduce(...)` — reduce/scatter + down-projection
- `runner.encodeDecodeLayerTail(...)` — residual combine (reads h1Buf, h2Buf, hidden internally)
- `runner.model.planRoutedExpertsIfPossible(...)` / `fetchRoutedExperts(plan:)`
- `runner.model.pin/unpinRoutedExpertSlots(...)`
- `runner.sharedExpertGate` — cached `SharedExpertGateWeights` for gate scalar
- `runner.zeroResidual`, `runner.denseX`, `runner.denseScratch*`, `runner.moeActs`, `runner.outWeights` — internal scratch buffers

**Bug fixes applied in final implementation:**
1. `queue` reference: changed from bogus static var to `runner.ctx.queue`
2. `moe` reference: changed from bare `moe` to `runner.moe`
3. `zeroResidual` reference: changed from bare `zeroResidual` to `runner.zeroResidual`
4. Shared expert gate: replaced fragile manual dequantization with cached `SharedExpertGateWeights.gateValue(weight:x:count:)` pattern matching produceToken
5. Error checking: fixed `checkCmdError` from no-op stub to `if let error { throw error }`
6. Removed dead code: unused `writeSlotIndices`, unused vars, bogus static queue helper

**Build result:** Zero errors, zero new warnings.

### finishPendingMoE overlap bug fix ✅
Critical performance fix: re-ordered Steps D (async disk fetch) and C (gate scalar + sharedCB wait) so that disk I/O starts **before** waiting for sharedFFN GPU work to complete. This creates the same I/O-compute overlap that `produceToken` already relies on.

Before: commit sharedCB → waitForCommandBuffer → apply gate → THEN start disk fetch
After: commit sharedCB → start async disk fetch → wait for sharedCB → apply gate

The fix uses a plain `if/else` to choose between batch-plan or single-list fetch path, dispatching both on a global DispatchQueue for maximum concurrency with compute. Verified clean build (0 errors).

### P7-2: Batch full-attention KV writes for K positions ✅
**Task #33 completed.**

Replaced per-token direct-to-cache writes with staging-buffer pattern for full-attention layers during speculative verification:

- **Staging writes**: K tokens write K/V to contiguous staging buffer at strided offsets via `outputTokenStride`, then single bulk blit copies into ring-buffer cache per layer
- **Two-pointer design**: `kRoPE`/`vRoPE` point to per-token location depending on mode (ring-slot in real-cache, strided stage offset in spec); GEMV outputs follow same split
- **Quant stack propagation**: int4/int5/int6/int8 dequant kernels gain `outputStride` parameter with `outStr > 0 ? outStr : 1u` fallback for non-speculation paths
- **Stage buffer expansion**: `kStage`/`vStage` expanded from single-token size to `maxBatchFactor × tokenSize` in RealForwardRunner

**O(K²) Metal CB syncs → O(K)** reduction for full-attn path (~95% sync point reduction).

**Build result:** Zero errors, zero new warnings.

---

## Remaining Tasks

### P7-3: Layer-major DeltaNet kernel (K tokens, one round-trip) ✅
**Task #34 completed.**

Built a layer-major batch-in-time DeltaNet kernel that processes all K draft tokens through the 9-stage pipeline in a single Metal command buffer per `(layer, tk)` call.

**Implementation details:**

9 new batched Metal kernels appended to `deltanet.metal`:
1. `dn_load_hidden_batched` — FP16→FP32 bridge across all tokens (`batch * D` threads)
2. `dn_store_hidden_batched` — FP32 + deltaOut → FP16 back, folds residual into hidden at per-token offsets
3. `dn_rmsnorm_batched` — Per-token RMSNorm, each thread computes own row's sum-of-squares
4. `dn_matvec_batched` — Shared weight matrix, per-token input/output; output stride = rows × batch
5. `dn_conv_step_batched` — **SERIALIZED inner loop**: channel-indexed threads iterate over tokens inside kernel body (preserves causal recurrence order). `convDim` threads, each iterating over K tokens with per-token offset `t * 3 * convDim`
6. `dn_qknorm_expand_batched` — Per-token per-element QK-RMSNorm + head expansion
7. `dn_gates_batched` — Independent per token per head, sigmoid + softplus
8. `dn_recurrence_batched` — **SERIALIZED inner loop**: state layout `[batch × numValueHeads × headVDim × headKDim]`, each thread processes one (head, vIdx) pair for one token
9. `dn_output_gate_batched` — Per-token swiGLU-style gated output norm

Scratch buffers expanded from `[dim]` to `[maxTK × dim]` where maxTK defaults to 128 (~28 MB total).

**Swift bindings in `DeltaNetMetalBlock.swift`:**
- All 9 batched pipeline states initialized from shader library
- `encodeBatched(commandBuffer:hidden:weights:convState:recurrentState:tk:eps:)` method — 12-stage pipeline dispatching all kernels sequentially within one command encoder

**DraftVerifier wiring:**
Replaced the sequential token-loop (lines 198–231) with a single `encodeBatched(tk: tk+1)` call:
- **Before:** O(K²) Metal CB syncs — inner loop calls `encode()` K*(K+1)/2 times, each copying token data to offset-0
- **After:** O(K) Metal CB syncs — single `encodeBatched(tk+1)` per layer per tk, processing all 0..tk tokens in one GPU pass
- Eliminates all `copyScratchRegion` intra-loop copies (result folds directly into hidden via `store_hidden_batched`)

**Build result:** Zero errors, zero new warnings.

### P7-4: Wire batched forward pass into decode loop (alternate meaning)
**Task #38 (seems duplicate of #35).**

May refer to the end-to-end wiring of P7-1 + P7-2 + P7-3 together into a cohesive batched decode path. Task #35 already handled the speculative decoding scaffolding; this may be about integrating the actual batched expert/cache/KV optimizations into the main loop rather than running them as sequential per-token operations.

---

## Architecture Context (for new agents)

### Model topology
- **Architecture:** qwen3_5_moe (not qwen3_next or qwen36 — the checkpoint uses the qwen3_5_moe name)
- **Layers:** 40 total = 30 DeltaNet + 10 full-attention
- **Experts:** 256 total, top-8 per layer, intermediate size F=2112
- **Scaling factor:** routedScalingFactor ≈ 0.125 (applied to h2 residual)
- **Topology:** .preNorm (no post-FFN normalization; shared output gated by sigmoid)
- **Gate weight:** quantized int8-affine, group size 64, per-tensor scales/biases
- **Module nesting:** `language_model.model.layers[...].mlp.*`

### Execution flow (single-token baseline)
1. Pre-norm (hidden → rmsNorm → denseX)
2. Attention (full-attn or DeltaNet) → postAttnSetup
3. Shared expert FFN → h1Buf → apply sigmoid gate → scale h1Buf
4. Router → select top-K experts → plan cache fetch → pin slots → prefetch → disk fetch
5. Phase-1: load experts into moeActs, apply silu/gelu activation
6. Phase-2: reduce across K expert outputs, down-project → h2Buf
7. Tail: hidden + h1 + routedScale*h2 (residual combine)
8. Head: lm_head + logit sampling

### Speculative decoding flow (what finishPendingMoE enables)
For each round of K draft tokens:
1. Generate K candidates via NGramSpeculator
2. Pass all K through DraftVerifier.run():
   - Position starts at current kvPosition
   - For each layer L in 0..40:
     - Run attention (full-attn or DeltaNet)
     - Compute router → accumulate expert plans
     - Drain pending MoE via finishPendingMoE(L)
   - Return greedyTokens[K] for comparison
3. Compare greedyTokens[i] vs drafts[i] until mismatch
4. Append accepted drafts + bonus token to history
5. Continue to next round or stop

### Buffer layout (internal to RealForwardRunner)
All buffers promoted to `internal` visibility in prior session so DraftVerifier can access them:
- `h1Buf`: [D] FP16 — shared expert FFN output
- `h2Buf`: [D] FP16 — routed expert reduce/scatter output
- `routedX`: [D] FP16 — pre_feedforward_layernorm_2 input (unused in preNorm)
- `denseX`: [D] FP16 — pre_feedforward_layernorm input
- `denseScratchGate`: [F=2112] FP16 — shared expert gate scratch
- `denseScratchUp`: [F=2112] FP16 — shared expert up-projection scratch
- `denseScratchAct`: [F=2112] FP16 — shared expert activation scratch
- `routerInput`: [D] FP16 — router input (post-attention normed hidden)
- `zeroResidual`: [D] FP16 — zeros buffer for phase-2
- `moeActs`: [topK * FmoE] FP16 — expert activations
- `moeHitActiveSlots`: [topK] UInt32 — hit slot indices
- `moeMissActiveSlots`: [topK] UInt32 — miss slot indices

### Dependencies & imports in DraftVerifier.swift
DraftVerifier accesses these types from RealForwardRunner:
- `ctx.queue` (MTLCommandQueue)
- `cfg` (ArchConfig)
- All internal buffers (see above)
- `sharedExpertGate` (SharedExpertGateWeights?, internal)
- `sharedExpertProjections[L]` ([Int] → LayerSharedExpertProjections)
- All kernel instances: `moe`, `shared`, `attn*`, etc. (but most not used in finishPendingMoE)
- `encodeDecodeLayerTail(...)` method (public func)

### Error handling pattern in DraftVerifier
Custom wrappers due to Swift+Metal bridge limitations:
```swift
private static func waitForCommandBuffer(_ cb: MTLCommandBuffer) {
    // Uses DispatchSemaphore + Objective-C addCompletedHandler selector
    ...
}
private static func checkCmdError(_ error: Error?) throws {
    if let error { throw error }
}
```
These are used throughout DraftVerifier because the standard `cb.waitUntilCompleted()` and `checkCommandBufferError()` are not directly accessible or usable in the same pattern.

---

## Recent Changes (Last Session)

### Files modified:
1. **DraftVerifier.swift** — rewrite of finishPendingMoE body (~130 lines changed)
2. **RealForwardRunner.swift** — promote `sharedExpertGate` from private to internal

### Build verification:
```
swift build → 0 errors, warnings limited to pre-existing waitForCommandBuffer helper
```

### Next action recommended:
Before attempting to run/speculatively-decode, verify that the DraftVerifier's draft-processing path correctly wires the KV cache position advances between layers. The current finishPendingMoE assumes per-layer processing (layerPlans drained after each layer), but for true batched execution across K tokens, the KV position should advance by K at the END of the draft verification, not by 1 after each layer. Check DraftVerifier.run() lines 100-170 to confirm the KV cache position management.
