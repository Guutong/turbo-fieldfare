# P7-7 — DeltaNet int4 GEMV: fix the decode bottleneck and the memory gate

**Status:** planned, not started
**Owner:** unassigned
**Prereq:** commit `dfc9f46` (P7-6 phase-timing instrumentation + SHA-256 fix)
**Gates this targets:** speed (>=4 tok/s, currently ~2.2) and memory (<=2 GB RSS, currently 4-5 GB)

This document is written so an agent unfamiliar with the codebase can execute
it. Every claim below is backed by a measurement or a source line, and every
task has an explicit acceptance test. Do not skip the measurement steps — the
two previous attempts at this bottleneck failed because they changed code
before measuring.

---

## 1. Executive summary

30 of the 40 layers in Qwen3.6-35B-A3B are DeltaNet ("linear attention")
layers. They currently:

1. **Dequantize their weights from int4 to fp32** at first use and keep them
   resident, and
2. multiply them with a **naive one-thread-per-output-row fp32 GEMV** kernel.

Together this makes DeltaNet consume **99% of decode GPU time** and about
**4 GB of RSS**. Fixing it means keeping the weights int4 and multiplying them
with the SIMD-reduction int4 GEMV kernel the codebase *already has* and already
uses on the 10 full-attention layers.

Expected result: decode GPU time for DeltaNet drops from ~342 ms/tok to
~20 ms/tok, and RSS drops by ~4 GB. Both gates are projected to pass.

---

## 2. Evidence (all measured on the target machine)

Target machine is the **M2 MacBook Air, 16 GB** — this repo's `plan.md` target.
Verify you are on it before trusting any number:

```
sysctl -n hw.model     # must print Mac14,2
sysctl -n hw.memsize   # must print 17179869184
```

(The Mac Studio at `javis@192.168.1.46` is *not* the target and has its own
documented RSS confound. Do not measure gates there.)

Benchmark command used throughout this document:

```
TFF_PHASE_TIMING=1 TFF_TRUST_INSTALL=1 /usr/bin/time -l \
  ./.build/release/TurboFieldfareCLI --model scratch/qwen36.gturbo \
  --prompt "The capital of France is" --temperature 0 --max-new 48 --prefill off
```

Current output (reproducible across runs):

```
[stop=maxTokens prefill=5tok new=48tok decode=21.74s tok/s=2.208]
[phase-timing ms/tok: cb1=73.494 cb1Wait=360.699 cb1GPU=345.352
                      cb1GPUlinear=341.782 cb1GPUfull=3.570
                      plan=14.653 io=105.849 cb2=0.602 head=7.476]
maximum resident set size: ~4.0-5.0 GB
```

Read it as:

| Quantity | Value | Meaning |
|---|---|---|
| `cb1Wait` | 360.7 ms/tok | CPU wall time blocked waiting for cb1 |
| `cb1GPU` | 345.4 ms/tok | of which, GPU was genuinely executing (96%) |
| `cb1GPUlinear` | **341.8 ms/tok** | DeltaNet layers (30 layers) = **11.4 ms/layer** |
| `cb1GPUfull` | 3.6 ms/tok | full-attention layers (10 layers) = **0.36 ms/layer** |

**The decisive comparison:** a DeltaNet layer costs **32x more GPU time than a
full-attention layer**, even though "linear attention" exists to be the cheaper
one. Full attention is not the problem. DeltaNet is, and by a margin that
leaves nothing else worth optimizing first.

`cb1GPU` being 96% of `cb1Wait` also rules out CPU/GPU scheduling gaps: the GPU
is busy, not idle-waiting. This is real executed work, so the fix must reduce
the work, not reschedule it.

---

## 3. Root cause

### 3.1 Weights are dequantized to fp32 and kept resident

`Sources/TurboFieldfare/Runtime/DeltaNet/DeltaNetMetalBlock.swift`, in
`LayerWeights.init` (~line 37-79), calls `CPU.dequantResident(..., bits: 4)`
for every projection and uploads the result through `upload(_ values: [Float])`,
which allocates `values.count * MemoryLayout<Float>.size` — i.e. **fp32, 8x the
stored int4 footprint**. `WeightsCache` (~line 83-103) caches these per layer
and never evicts.

Per-layer fp32 weight bytes, using
`DeltaNetDimensions.qwen36_35B_A3B` (`DeltaNetState.swift:27`: `convDim: 8192,
numKeyHeads: 16, numValueHeads: 32, headKDim: 128, headVDim: 128`, so
`keyDim = 2048`, `valueDim = 4096`) and `hiddenSize D = 2048`:

| Tensor | Shape | Params | fp32 bytes |
|---|---|---|---|
| `qkv` | 8192 x 2048 | 16.78 M | 67.1 MB |
| `z` | 4096 x 2048 | 8.39 M | 33.6 MB |
| `out` | 2048 x 4096 | 8.39 M | 33.6 MB |
| `a`, `b` | 32 x 2048 each | 0.13 M | 0.5 MB |
| `conv` + norms | small | — | 0.2 MB |
| **total** | | **33.7 M** | **~135 MB / layer** |

**x 30 DeltaNet layers = ~4.05 GB.** This matches the observed 4-5 GB RSS, and
it is the memory-gate failure. As int4-with-scales (0.5625 B/param, the format
already on disk) the same weights would be **~19 MB/layer, ~0.57 GB total**.

> Note on RSS history: earlier runs in the same session reported 1.1-1.8 GB.
> The allocation was almost certainly always 4 GB; under memory pressure macOS
> compresses pages, which lowers reported RSS. Treat the low figures as
> measurement artifacts, and re-measure RSS only on an idle machine
> (`vm_stat` "Pages free" above ~150000).

### 3.2 The GEMV kernel is the worst possible access pattern

`Sources/TurboFieldfare/Metal/DeltaNet/deltanet.metal:105-121`:

```metal
[[kernel, max_total_threads_per_threadgroup(256)]]
void dn_matvec(
    device const float* w, device const float* x, device float* y,
    constant uint& rows, constant uint& cols, uint tid [[thread_position_in_grid]]
) {
    if (tid >= rows) return;
    const uint base = tid * cols;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) acc += w[base + c] * x[c];
    y[tid] = acc;
}
```

One thread computes one output row by serially walking that row. Adjacent
threads (`tid`, `tid+1`) read addresses `cols * 4 = 8192` bytes apart, so every
memory transaction in a SIMD group pulls a separate cache line and discards most
of it. There is no `simd_sum` reduction, no threadgroup staging of `x`, and no
vectorized loads.

Measured effective bandwidth: 4.05 GB/token / 341.8 ms = **~11.9 GB/s**, against
roughly 100 GB/s of hardware capability — about **12% of peak**.

### 3.3 The contrast that proves the fix works

Full-attention layers move ~18.9 M params/layer (q/k/v/o) — comparable to
DeltaNet's 33.7 M — but stay int4 and use the SIMD-reduction kernels in
`Sources/TurboFieldfare/Metal/Quant/dequant_int4.metal` (note `simd_sum` at
lines 172 and 251). They cost **0.36 ms/layer** and reach ~29 GB/s effective.

The same treatment applied to DeltaNet predicts:
`33.7 M params x 0.5625 B = 19 MB/layer` at ~29 GB/s -> **~0.65 ms/layer**,
so 30 layers -> **~20 ms/tok** instead of 341.8 ms/tok.

---

## 4. Projected outcome

| Metric | Now | Projected | Gate |
|---|---|---|---|
| DeltaNet GPU time | 341.8 ms/tok | ~20 ms/tok | — |
| Total decode | ~452 ms/tok | ~130 ms/tok | — |
| Throughput | 2.21 tok/s | **~7.7 tok/s** | >=4 PASS |
| RSS | 4-5 GB | **~1 GB** | <=2 GB PASS |

Both gates from one change. Treat the numbers as projections until Task 5
measures them; the ratio argument (section 3.3) is the load-bearing part, not
the exact figure.

---

## 5. Design

Keep DeltaNet's structure and math identical. Change only how the five
projections (`qkv`, `z`, `a`, `b`, `out`) are stored and multiplied.

**Before:** `TensorView` (int4 on disk) -> `dequantResident` -> `[Float]` ->
`MTLBuffer` (fp32) -> `dn_matvec`.

**After:** `TensorView` (int4) -> `MTLBuffer` holding the original packed int4
weights + scales + biases -> int4 SIMD GEMV kernel.

Two implementation options. **Prefer Option A.**

### Option A — reuse `DequantInt4GEMV` (recommended)

`Sources/TurboFieldfare/Kernels/Quant/DequantInt4GEMV.swift` already exposes:

```swift
func encode(commandBuffer: MTLCommandBuffer,
            weights: MTLBuffer, weightsOffset: Int = 0,
            scales: MTLBuffer,  scalesOffset: Int = 0,
            biases: MTLBuffer,  biasesOffset: Int = 0,
            x: MTLBuffer, xOffset: Int = 0,
            y: MTLBuffer, yOffset: Int = 0,
            m: UInt32, n: UInt32, ...)
```

`m` = rows, `n` = cols. This is a proven, tested path (see
`Tests/.../Quant/DequantInt4GEMVTests.swift`).

Two frictions to resolve before coding:

1. **Encoder type.** `DeltaNetMetalBlock.encode` opens one
   `MTLComputeCommandEncoder` and encodes every stage into it, while
   `DequantInt4GEMV.encode` takes a `MTLCommandBuffer` and makes its own
   encoder. Read `DequantInt4GEMV.swift` and check whether it exposes an
   encoder-level entry point. If it does not, add one
   (`func encode(encoder: MTLComputeCommandEncoder, ...)`) and have the
   existing command-buffer method call it — a pure refactor, no behaviour
   change, and the existing tests must keep passing.
2. **Precision.** DeltaNet buffers are fp32; these kernels dequantize to and
   accumulate in fp16. Do not treat that as a problem to work around — see
   section 5.1. Follow the precision policy there rather than converting fp16
   results back to fp32 at every stage boundary.

### 5.1 Precision policy: fp16 math, fp32 state

M1/M2 GPU ALUs execute float16 natively at roughly twice the fp32 rate, so
fp16 is the right arithmetic precision here, not a compromise. The existing
int4 kernels already dequantize to `half` and accumulate in fp16, which means
**Option A delivers the bandwidth win (int4 storage) and the ALU win (fp16
math) in the same change** — there is nothing extra to do to get the second one.

Apply this split:

| Data | Precision | Why |
|---|---|---|
| Projection weights (`qkv`, `z`, `a`, `b`, `out`) | **int4** + BF16 scales/biases | The bandwidth win; format already on disk |
| Projection outputs, `q`/`k`/`v`, `beta`, `g`, conv output | **fp16** | Native ALU rate; single-step values, no error accumulation |
| **Recurrent state** (`recurrentState`, 32x128x128 fp32) | **fp32 — do not change** | Accumulates across every token of the sequence; fp16 here drifts |
| Conv state | fp32 (leave as is) | Tiny (98 KB/layer); not worth the risk |

This is the same split `mpsops/mps-linear-attention` uses for gated delta rule
on Apple Silicon (state fp32, Q/K/V/beta fp16), which is useful independent
precedent that fp16 is safe for the non-recurrent parts of this exact
algorithm.

The `deltaOut relL2` parity check in Task 4 is what proves it on our weights.
If parity fails, suspect the state or the recurrence accumulator first, not
the projections.

### 5.2 If int4 integration stalls: fp16-only as an interim step

If Option A's encoder refactor turns into a multi-day detour, a smaller change
is available: keep the weights dense but upload them as **fp16 instead of
fp32** and make `dn_matvec` a `half` kernel. That alone halves the weight
bytes (135 -> 67 MB/layer, ~2 GB saved) and doubles ALU throughput.

Be clear about what it does *not* fix: at 12% of peak bandwidth, the dominant
loss is the uncoalesced one-thread-per-row access pattern (section 3.2), which
is dtype-independent. Expect roughly 2x, not the ~17x that int4 plus a
SIMD-reduction kernel projects. Treat this as a checkpoint on the way to
Option A, not a destination — and if you stop here, say so explicitly in the
PHASE-LOG entry so the next agent knows the main win is still unclaimed.

### Option B — new int4 kernel inside `deltanet.metal`

Only if Option A's frictions prove structural. Write `dn_matvec_int4`
modelled on the `simd_sum` kernels in `dequant_int4.metal`: one SIMD group per
output row, each lane striding across the row, `simd_sum` to reduce, lane 0
writes. Stage `x` in threadgroup memory once per threadgroup. This duplicates
logic the repo already has, which is why it is the fallback.

### Out of scope for P7-7

- The `io` phase (105 ms/tok) — the next target, but not this task.
- `encodeBatched` / `DraftVerifier` — only touch it in Task 6 if you changed a
  shared signature.
- Any change to DeltaNet math, layer topology, or the MoE path.

---

## 6. Tasks

Execute in order. Do not begin a task until the previous one's acceptance test
passes.

### Task 1 — baseline and provenance

1. Confirm the target machine (section 2).
2. Confirm the tree is clean and at or after `dfc9f46` (`git log --oneline -3`).
3. Build: `swift build -c release --product TurboFieldfareCLI`.
4. Run the benchmark **three times on an idle machine**, recording `tok/s`,
   every `phase-timing` field, and `maximum resident set size`.

**Acceptance:** three runs recorded; `cb1GPUlinear` is 300-360 ms/tok in all
three. If it is not, stop — the premise of this document does not hold on your
machine, and you must re-diagnose before changing code.

### Task 2 — carry int4 weights to the GPU

In `DeltaNetMetalBlock.LayerWeights`:

1. For `qkv`, `z`, `a`, `b`, `out`, stop calling `CPU.dequantResident`. Upload
   the packed int4 bytes plus scales and biases instead. Model the upload on
   how resident int4 tensors are handed to `DequantInt4GEMV` elsewhere —
   `TensorView` already carries `offset`, `scaleOffset`, and `biasOffset`, so
   prefer referencing the existing resident buffer over copying.
2. Leave `conv`, `deltaNorm`, `aLog`, `dtBias`, `inputNorm` as they are — they
   are BF16 vectors, tiny, and not part of this bottleneck.
3. Keep the fp32 path reachable behind a flag (e.g. `TFF_DELTANET_FP32=1`) so
   Task 5 can A/B on one binary and Task 7 has a rollback.

**Acceptance:** builds clean; `swift test --filter DeltaNet` passes with the
flag set to the fp32 path (the new path is not wired up yet).

### Task 3 — int4 GEMV in the DeltaNet encode path

Replace the five `matVec(...)` calls in `DeltaNetMetalBlock.encode`
(lines ~343-346 and ~415) with the int4 GEMV per Option A. Keep every other
stage — `psoLoadHidden`, `psoRMSNorm`, `psoConvStep`, `psoQKNormExpand`,
`psoGates`, `psoRecurrence`, `psoOutputGate`, `psoStoreHidden` — untouched, in
the same order. The dependency chain relies on serial dispatch order within the
encoder; preserve it.

**Acceptance:** builds clean; the runner produces output for a short prompt
without a Metal validation error
(`MTL_DEBUG_LAYER=1 ./.build/release/TurboFieldfareCLI ... --max-new 4`).

### Task 4 — correctness before speed

Run, in this order:

```
swift test --filter DeltaNetParity     # Metal block vs CPU reference, per layer
swift test --filter DeltaNet           # conv, gates, recurrence, state
swift test --filter Qwen36             # end-to-end topology and prefill parity
```

`DeltaNetParityTests.metalBlockMatchesCPUBlockOnEveryDeltaNetLayer` is the
decisive one: it compares all 30 layers against the CPU reference. Its current
baseline is `worst deltaOut relL2 1.289676e-06`.

Then confirm text equivalence — greedy decoding is deterministic, so with
`--temperature 0` the output text must be **byte-identical** to Task 1's:

```
Paris, a city renowned for its iconic landmarks such as the Eiffel Tower, ...
```

**Acceptance:** all three suites pass; int4 relL2 stays within one order of
magnitude of the fp32 baseline (int4 has less headroom than fp32 — a modest
increase is expected and acceptable, a jump to 1e-3 or a text change is not);
generated text is byte-identical. **If the text changes, stop and diagnose.**

### Task 5 — measure

Repeat Task 1's protocol (idle machine, three runs) on the new path, then run
three more with `TFF_DELTANET_FP32=1` to confirm the flag still reproduces the
old numbers on the same binary.

**Acceptance:** record all six runs. Expect `cb1GPUlinear` at or below
40 ms/tok and RSS at or below 1.5 GB. If `cb1GPUlinear` lands far above 40 but
well below 342, the weights are int4 but the kernel is still inefficient —
proceed to Option B rather than declaring victory.

### Task 6 — batched path

`DeltaNetMetalBlock.encodeBatched` (~line 433) is a parallel implementation used
by `DraftVerifier`. If Task 2 changed the `LayerWeights` field types, this will
fail to compile — apply the same treatment.

**Acceptance:** `swift build` clean; `swift test --filter DraftVerifier` and
`swift test --filter DeltaNet` pass.

### Task 7 — full suite, log, commit

1. `swift test` (about 20 minutes; `DeltaNetParityTests` alone takes ~19).
   Compare failures against the pre-existing set: **8 failures out of 853** were
   present before this work and are not yet attributed. Your job is to show you
   added none, not to fix those.
2. Append a dated PHASE-LOG.md entry: measurements before and after, what
   changed, what you did not touch, and the gate verdicts.
3. Commit with a `P7-7:` prefix. Do not push.

**Acceptance:** no new test failures; PHASE-LOG entry states both gate results
explicitly.

---

## 7. Risks

| Risk | Signal | Response |
|---|---|---|
| int4 precision degrades output | relL2 jumps, or text differs | Stop. Compare per-layer relL2 to find which projection is sensitive; consider keeping that one tensor fp32. |
| fp16 vs fp32 dtype mismatch | Metal validation error, or NaN | Follow the precision policy in section 5.1; settle it before Task 3, not during. |
| fp16 applied to the recurrent state | parity drifts, worse on later tokens than early ones | The state must stay fp32 (section 5.1). Error growing with token index is the signature. |
| Speedup below projection | `cb1GPUlinear` well above 40 ms/tok | Weights are int4 but the kernel is weak — go to Option B. |
| Memory does not drop | RSS still 4+ GB | The fp32 arrays are still being materialized — check that `dequantResident` is gone from the hot path and no host copy is retained. |
| A gate passes but text changed | Task 4 text check | Treat as a failure, not a win. Correctness first. |

---

## 8. Method note

Three hypotheses were tried on this bottleneck before the real cause was found:
`DispatchQueue` overhead (disproven — swapping to `NSLock` changed nothing),
thermal/memory pressure (disproven — idling the machine changed nothing), and
SHA-256 re-hashing (**correct, fixed in `dfc9f46`**, but it was not the dominant
cost). Only Instruments (`xcrun xctrace record --template 'Time Profiler'`) and
targeted counters found the truth.

The lesson for whoever executes this plan: **measure, then change.** The
`TFF_PHASE_TIMING=1` counters exist now precisely so the next person does not
have to guess. If a number in this document does not reproduce on your machine,
that discrepancy is the most interesting thing available to you — chase it
before writing code.
