# Handoff: generalizing TurboFieldfare beyond Gemma

Branch: `generalize-arch-for-moe`

## Goal

TurboFieldfare was written around exactly one checkpoint — Gemma 4 26B-A4B, 4-bit,
top-8 routing, 128 experts, group size 64. The architecture's constants were spread
across the config parser, the Metal kernels, and the app UI as literals. The goal is
to make a *second* architecture representable, using poolside/Laguna-S-2.1 as the
forcing function.

Laguna differs from Gemma in almost every dimension that was hardcoded:

| | Gemma 4 26B-A4B | Laguna-S-2.1 |
|---|---|---|
| Params | 26B | 117B |
| Layers / hidden | 30 / 2560 | 48 / 3072 |
| Experts / top-K | 128 / 8 | 256 / 10 |
| MoE intermediate | 704 | 1024 |
| KV heads / head dim | — | 8 / 128 |
| Vocab / sliding window | — | 100352 / 512 |
| Routed scaling | — | 2.5 |
| Attention gating | none | per-head (softplus, pre-o_proj) |
| RoPE | uniform | YaRoP on full-attention layers only |
| Heads | uniform | 48 on full layers, 72 on sliding |
| Layer 0 MLP | MoE | dense |
| Quantization | 4-bit, group 64 | mixed 4/5/6/8-bit, group 128 |

Laguna is **not runnable yet**. See "What still blocks Laguna" below. It is listed in
the model catalog with `isInstallable: false` so the gap is visible in one place
rather than implied by absence.

## What is done

**Phase 0a — config generalization.** `ArchConfig` gained per-layer accessors
(`numHeads(atLayer:)`, `isDenseMLP(layer:)`, `ropeScaling(atLayer:)`) plus
`RopeScaling` and `AttentionGating`. The repack-side parser
(`Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift`) handles both Gemma's
nested `text_config` and Laguna's flat config, and rejects `linear_attention` layer
types outright rather than silently mis-decoding them.

All new fields default to Gemma's values, and `GTurboJSON` only writes them when they
differ. That keeps existing `.gturbo` artifacts byte-identical — deliberate, so this
work could not regress the one model that already worked.

**Phase 0b — 5/6-bit affine dequant.** MLX-compatible quantize/dequantize for the
sub-byte codes Laguna's conversion mixes in, CPU and Metal
(`Sources/TurboFieldfare/Metal/Quant/dequant_subbyte.metal`). 5- and 6-bit codes
straddle byte boundaries, hence the templated `extract_subbyte_code<N>`.

**Phase 0d — attention from config.** Pipelines are built from config rather than
Gemma's literal shapes, and specialized pipelines are now built lazily and cached.
Attention scratch is sized from `maxNumHeads` (the widest layer) rather than the
scalar `numHeads` — with per-layer head counts the scalar is not an upper bound, and
using it under-allocates.

**Phase 0c — top-K generalization.** Just completed; details below.

## Phase 0c in detail — read this before touching MoE

The MoE kernels assumed K=8 structurally, not just numerically. Generalizing to
K ∈ 1...16:

- `kMoEMaxTopK = 16` is a **compile-time cap**. Metal has no variable-length arrays,
  so `top_idx`/`top_score`/`exps`/`partial` are all sized at the cap and the runtime
  K only bounds the loops. An earlier attempt declared these as `device uint x[K]`
  and `threadgroup float partial[K]` — that does not compile. Do not reintroduce it.
- `router_topk_select_k8` → `router_topk_select`, taking K via
  `constant uint& top_k [[buffer(5)]]`, with function-constant override
  `router_fc_top_k` mirroring the existing `moe_fc_top_k`.
- `moe_phase2_down_reduce_k8` → `moe_phase2_down_reduce`, taking K via
  `constant uint& top_k [[buffer(8)]]`. The unrolled 8-term sum became a loop over
  `partial[0..<K]`.
- Phase-2 now dispatches `topK * 32` threads — one simdgroup per routed expert,
  replacing the fixed 256 (which *was* exactly 8 simdgroups; the coupling was
  implicit and easy to miss).
- Host preconditions moved from `topK == 8` to `1 <= topK <= maxStreamedExperts`.

Two defects from the earlier pass were fixed here:

1. **Real bug (host).** The host called `commandBuffer.makeBuffer(...)`, which is not
   an `MTLCommandBuffer` API, and bound the result to the router GEMV encoder rather
   than the select encoder that actually reads K. Removed; K now goes to the correct
   encoder via `setBytes`.
2. **Tidiness, not a bug (kernel).** The shift-down ran `for (uint i = K; i > pos; --i)`
   writing `top_idx[i]` at index K. Since the arrays are sized at `kMoEMaxTopK = 16`
   and K <= 16, that write stayed in bounds and only touched a slot the softmax loop
   never reads. It is now `K - 1`, which is clearer, but be aware it was *not* the
   correctness problem it looks like — a mutation test confirmed the old form passes
   every K=2/10/16 assertion. Correcting an earlier note in this file that called it
   an out-of-range write.

`swift build` is green and `moe.metal` compiles standalone:

```
xcrun -sdk macosx metal -c Sources/TurboFieldfare/Metal/MoE/moe.metal -o /tmp/moe.air
```

**Status: verified.** `Tests/TurboFieldfare/Core/Kernels/MoE/` now covers K ∈ {2, 10,
16} against CPU references: exact expert-set match, distinctness (catches drops and
duplicates), softmax weights summing to one, bit-exact tie-breaks, and the full
phase1+phase2 routed pipeline. Plus a check that the phase-2 pipeline's
`maxTotalThreadsPerThreadgroup` covers the 16 × 32 = 512-thread worst case. Full suite:
551 tests green on M2.

The phase-2 tests were mutation-checked and do have teeth — dropping one expert from
the reduce loop fails all three K values with ~20-29% relative error. The router
selection tests are weaker than they look: both an off-by-one in the shift-down and a
broken tie-break rule survive mutation, because the `s <= top_score[K-1]` early-out
masks them on these fixtures. If you change the selection logic, do not trust those
tests alone.

## Group size 128 — partially done, read this before continuing

Laguna quantizes its routed experts at group size 128; every kernel here was written
around 64. The **generic INT4 GEMV path now supports any group size that is a multiple
of 32**, but the MoE kernels do not yet, and the manifest guard encodes exactly that
split.

What landed:

- `dequant_int4_gemv_generic` in `Metal/Quant/dequant_int4.metal`, taking `groupSize`
  at `[[buffer(7)]]`. The int4 file's diff is **purely additive** — the group-64 fast
  path was not touched, which is the regression argument.
- It deliberately does *not* copy the group-64 block structure. That path packs four
  64-element groups into a 128-byte block and splits lanes 8-per-group via
  `lane >> 3`; at group 128 a block spans only two groups and the mapping would have
  to become `lane >> 4`. Rather than re-derive that, the generic path uses the
  strided-lane loop from `dequant_subbyte_gemv_body` (`for elem = lane; elem <
  groupSize; elem += 32`), which is group-size agnostic by construction. Slower, but
  nothing can run this model end to end yet, so correctness wins.
- `DequantInt4GEMV.encode` takes `groupSize` defaulting to 64; at 64 it dispatches the
  original pipeline with the original buffer wiring, so existing callers are untouched.
- `Quantization.quantizeInt4Affine` is now two **overloads** rather than one defaulted
  parameter, because a bare function reference (`values.map(Quantization.quantizeInt4Affine)`)
  does not see default arguments and stopped compiling.

The same treatment then landed for the **MoE routed-expert kernels**, which is what
Laguna actually needs. `Metal/MoE/moe.metal` gained generic variants — 327 insertions,
**zero deletions**, so Gemma's group-64 block path is untouched by construction:

- `moe_int4_gemv_row_simd_dev_vec_generic` (phase-2 down projection)
- `moe_int4_gate_up_rows_simd_dev_vec_u16load_generic` (phase-1 gate/up)
- kernels `moe_phase1_gate_up_act_u16load_generic`,
  `moe_phase1_gate_up_act_subset_u16load_generic`, `moe_phase2_down_reduce_generic`

The `MoE` encoders take `groupSize` defaulting to `Quantization.groupSize`; at 64 they
dispatch the original specialized pipelines with the original wiring, so Gemma's
function-constant specialization still applies.

**The manifest guard is per-slot, and that is deliberate.** `validateQuant` now allows
128 for `embedding`, `attention`, and `routedExpert` — the three whose kernels actually
handle it. `router` (8-bit, `router_gemv_gemma4_r4`) and `sharedExpert` (still on the
group-64 block path) remain 64-only. An early version allowed 128 for *any* 4-bit slot,
which would let a group-128 checkpoint validate and then decode to silently wrong
numbers — a clean load-time rejection traded for a silent correctness bug. Tests pin
each remaining rejection. **Widen a slot only in the same change that teaches its
kernel group 128.**

## A note on memory, because it is easy to get wrong

Do not judge feasibility by comparing checkpoint size to installed RAM. This runtime
streams routed experts from SSD: Gemma's routed experts are **12.01 GiB on disk** but
reserve only about **1.50 GiB** of slot capacity (16 slots per layer, LFU eviction),
and the whole 14.3 GB model runs on a 16 GB machine. Total size is not the constraint.

What actually has to be resident is common weights + KV + slots. For Laguna the numbers
that matter are therefore its **common-weight size** (not streamed), its **top-10**
routing (more slots per layer than Gemma's 8), and **256 experts vs 128** (lower slot
hit rate, so more SSD reads per token). None of these have been measured — there is no
Laguna `.gturbo` to measure yet. Treat feasibility as *unknown*, not as *ruled out*.

See "Resource split" in `docs/SYSTEM_DESIGN.md` for the real table.

### A first-cut estimate for Laguna

Arithmetic from `config.json` shapes and the quant table above — **not measured**, and
it ignores allocator overhead, alignment, and activation scratch. The total lands at
58.7 GiB ≈ 63 GB, which matches the repo's ~64 GB, so the shape assumptions are at
least self-consistent.

| | |
|---|---|
| routed experts (streamed) | 56.2 GiB — one expert is 4.78 MiB |
| common weights (resident) | ~2.5 GiB |
| slots @ 16/layer × 47 layers | 3.5 GiB |
| slots @ 10/layer (= top-K, the floor) | 2.2 GiB |
| KV @ 4096 ctx | 0.75 GiB |
| **resident total @ 16 slots** | **~6.8 GiB** |
| **resident total @ 10 slots** | **~5.4 GiB** |

So memory is probably *not* the wall — this sits in the same range as Gemma on a 16 GB
machine. **Throughput is the open risk.** Gemma's 16 slots cover 12.5% of its 128-expert
pool; Laguna's 16 cover 6.2% of 256. Halving the hit rate at top-10 instead of top-8
means materially more SSD reads per token, and no amount of kernel work changes that —
it is a property of the routing. Measure tokens/sec before assuming the slot count is
tunable enough to fix it.

## Repack side — tensor naming and model selection (done)

The runtime (consumer) and the repack pipeline (producer) are separate problems.
Kernels can decode group-128 experts, but until this landed the repacker could not
*produce* a Laguna `.gturbo` at all: `classify()` only knew Gemma's names, so every
Laguna tensor fell through to `.unknown`.

Ground truth, fetched from the real repo rather than guessed:

| | Gemma | Laguna |
|---|---|---|
| routed experts | `.experts.switch_glu.` | `.mlp.switch_mlp.` |
| router | `.router.proj.weight` | `.mlp.gate.proj.weight` |
| shared expert | `.mlp.gate_proj.weight` | `.mlp.shared_expert.gate_proj.weight` |
| per-head gate | — | `.self_attn.g_proj.weight` |
| layer 0 MLP | MoE | dense `.mlp.{gate,up,down}_proj` |

Both share the `language_model.` prefix. Naming variants live in ordered tables
(`routedExpertMarkers`, `sharedExpertProbeSuffixes`, `routerProbeSuffixes`) so a third
family is one entry, not another `if name.contains(...)`.

**One subtlety worth keeping.** Laguna's *dense layer-0* MLP ends in
`.mlp.gate_proj.weight` — byte-identical to the generic shared-expert probe. Probing
per entry would let layer 0's bit width land in the manifest's `sharedExpert` slot: a
manifest that loads cleanly and describes the wrong quantization. `writeManifest` now
resolves that slot by scanning the probe table **in order across all entries** and
stopping at the first suffix any entry matches, most-qualified first. A test pins the
ordering; reversing it fails.

`TurboFieldfareRepack` also gained `--model <id>`. No flag still means Gemma exactly.
An unknown id is rejected with the valid list; a non-installable one is rejected with
its `installBlockedReason`. Laguna stays `isInstallable: false`.

## Laguna's real quantization recipe — measured, not assumed

An earlier version of this file listed "mixed precision in the routed path" as the
largest remaining item. **That was wrong**, and it mattered, because it pointed the
next person at the MoE kernels when the actual obstacle is the manifest format.

Pulled from `mlx-community/Laguna-S-2.1-oQ4e/config.json` (defaults: 4-bit, group 128,
affine; the table below is the explicit per-tensor override list):

| Tensors | Bits | Group | Count |
|---|---|---|---|
| routed experts (`mlp.switch_mlp.*`) | **4** | **128** | *(no override — the default)* |
| `self_attn.{q,k,v,o,g}_proj` | **5** | 64 | 20 layers |
| `self_attn.{q,k,v,o,g}_proj` | **8** | 64 | 28 layers |
| layer 0 `mlp.{gate,up}_proj` | **5** | 64 | 1 |
| layer 0 `mlp.down_proj` | **6** | 64 | 1 |
| `mlp.shared_expert.*` | **8** | **128** | 47 layers |
| `embed_tokens`, `lm_head` | **8** | 64 | 1 each |

Consequences, in order of how much they change the plan:

- **The routed path is uniformly 4-bit group-128 — already supported.** Nothing in the
  MoE kernels needs 5/6-bit. The 5- and 6-bit tensors are attention projections and
  layer 0's dense MLP, which are *not* routed. Phase 0b's dequant is still the right
  tool, just for a different consumer than this file previously claimed.
- **The manifest cannot express this checkpoint.** `ManifestQuant` holds one
  `weightBits` + `groupSize` per slot. Laguna's attention slot is 5-bit on 20 layers
  and 8-bit on 28, so no single value is correct. Quantization is **per-layer** here,
  not per-role, and that is a format change, not a kernel change.
- Three slots also fall outside the current load-time guard on their own:
  `embedding` at 8-bit (guard allows only 4), `sharedExpert` at 8-bit *group 128*
  (guard allows only group 64), and `router` at the 4-bit default (guard requires 8).

### The uniformity guard (done)

`writeManifest` probed each slot with a bare assignment inside a loop over entries —
last-write-wins. On Laguna it would have recorded whichever attention layer the plan
visited last and written a manifest that loads cleanly while misdescribing 20 layers.
This is the same *class* of bug as the shared-expert probe, but ordering cannot fix it:
there, one candidate was right; here both observations are correct about their own
layers and the format is what cannot hold them.

`RepackPlanner.uniformBits(slot:observations:)` now collapses each slot across **all**
matching tensors and throws `RepackError.quantSlotNotUniform` when they disagree,
naming one tensor per distinct width. Applied to `embedding`, `attention`, `router`,
`sharedExpert`, and `routedExpert` (the last across every layer and role, not the first
slice of the first layer). Gemma is uniform in every slot, so this is inert for it.

Mutation-checked both directions: forcing it to always throw, and forcing it to never
throw (exactly the old behavior), each fail the suite.

### Per-layer quantization in the manifest (format done, producer not)

`ManifestQuantSlot` gained an optional `perLayer: [ManifestQuantLayerOverride]?` —
sparse `(layer, weightBits, groupSize)` exceptions to the slot's scalar values. The
shape mirrors the upstream `config.json` (defaults plus a sparse override table), so
the uniform case costs nothing and Laguna's attention slot is 20 entries, not 48.

Only `weightBits` and `groupSize` are overridable. `scheme`/`scaleType`/`biasType`
stay on the slot because no known checkpoint varies them per layer, and every field
that *can* differ is a field every reader has to check.

Backward compatibility is the whole design constraint, and it holds two ways: the key
is optional, so manifests written before this decode with `perLayer == nil`; and the
repack side does not emit the key, so freshly written Gemma manifests are byte-identical.
`manifestWithoutPerLayerKeyDecodesAsUniform` pins the first; every other test in
`ManifestReaderTests` pins the second by construction, since the fixture helper omits
the key unless asked.

Two reader APIs:

- `resolved(atLayer:) -> (weightBits, groupSize)` — the slot's own values unless an
  override names that layer.
- `distinctConfigurations` — every `(bits, group)` pair the slot can present.

**`validateQuant` gates `distinctConfigurations`, not the scalar.** Checking only the
default would let an override smuggle in a width no kernel implements — the same
load-cleanly-then-decode-wrong trade the per-slot guard exists to prevent. Duplicate
layer entries are rejected too, because they are two answers to one question and
`resolved(atLayer:)` would silently take the first. All three paths mutation-checked.

So Laguna's attention slot is now **expressible but still correctly rejected**, and the
error names the width (`unsupported quantization for attention: 5-bit group 64`) rather
than the format shrugging.

**The repacker still refuses instead of emitting overrides, on purpose.** Emitting them
would make a 64 GB repack succeed and then fail at load, when `uniformBits` can refuse
during planning instead. Teach the producer to emit `perLayer` in the same change that
teaches a kernel to decode the widths involved — not before.

### 5- and 8-bit fused QKV (kernels done, not yet wired into the forward pass)

`dequant_int5_qkv_gemv_simd` and `dequant_int8_qkv_gemv_simd` in
`Metal/Quant/dequant_subbyte.metal`, with `FusedQKVGEMVGeneric` as their host wrapper. It mirrors `dequant_int4_qkv_gemv_simd`'s
row partitioning — one dispatch covers Q's `Mq` rows then K's and V's `Mkv` each, so
the three projections share one pass over `x` — but delegates the arithmetic to the
existing `dequant_subbyte_gemv_body<5>`.

`FusedQKVGEMVGeneric` is a **sibling** of `FusedQKVGEMV`, not a `bits:` parameter on it.
`FusedQKVGEMV` specializes on Gemma's two known decode shapes via function constants
and sits on the only path that runs end to end; leaving it untouched is the same
regression argument that shaped the group-128 work. A model needing several widths — as
Laguna does, 5-bit attention on 20 layers and 8-bit on 28 — picks per layer from
`ManifestQuantSlot.resolved(atLayer:)`.

Two deliberate differences from the int4 version: shapes are runtime arguments rather
than function constants (there is no measured shape set worth baking in yet), and
`groupSize` is runtime, because that is what the subbyte body already takes.

**Why 8-bit lives in the sub-byte file.** At `BITS=8` `extract_subbyte_code` degenerates
exactly to a plain byte load — `bit_in_byte` is always 0, the mask is 0xFF, and the
straddle test `bit_in_byte + BITS > 8` is never true, so the second byte is never read
and it cannot over-read past a group. That was verified exhaustively against a direct
byte load before being relied on.

The payoff is that it is **group-size generic**. `dequant_int8_gemv_simd` in
`dequant_int8.metal` hardcodes group 64 and maps two elements per lane (`lane * 2`),
which produces silently wrong numbers at group 128 — and Laguna's 8-bit *shared
experts* are group 128. So a group-generic 8-bit path was needed regardless; building
the fused QKV on the strided-lane loop got it without a second bespoke kernel.

The cost is speed: a strided-lane loop is slower than the two-per-lane load. Nothing
runs this checkpoint end to end yet, so correctness wins — the same trade the group-128
int4 work made. If profiling later says it is hot, add a group-64 fast path *beside*
it rather than reshaping this one.

`QuantizationSubByte.quantizeInt8Affine(_:groupSize:)` was added as the group-generic
CPU reference (`Quantization.quantizeInt8Affine` is group-64-only). The two agree
byte-for-byte at group 64, and `int8ReferencesAgreeAtGroup64` pins that exactly, with
no tolerance — otherwise the new kernel would be validated against a different format
than the rest of the codebase writes.

**The risk in this kernel is the partitioning, not the arithmetic** — an off-by-one at
a boundary or a swapped weight/output pairing still yields numbers of the right
magnitude. So the tests draw Q, K and V from independent fixtures and check each
against its own reference. Mutation-checked: off-by-one on the K/V boundary, V reading
K's weights, V using K's row index, and a 2% arithmetic drift are all caught.

Covered shapes, at both widths: Laguna's full-attention layer (48 q-heads × 128,
8 kv-heads, N=3072, group 64), its sliding layer (72 q-heads), group 128, and ragged
row counts that leave the last threadgroup partial so the `global_row >= total_rows`
bound is exercised. Two further mutations confirm the 8-bit path is really reached and
really group-generic: instantiating the 8-bit kernel at `BITS=5`, and hardcoding the
element loop to 64, both fail.

Not yet done: `RealForwardRunner` still calls `FusedQKVGEMV` unconditionally.

## What still blocks Laguna

In rough dependency order:

1. **Per-layer dispatch in the forward pass.** Both widths Laguna's attention needs now
   exist as kernels, but `RealForwardRunner` picks `FusedQKVGEMV` unconditionally.
   It has to select per layer from `ManifestQuantSlot.resolved(atLayer:)` instead, and
   prefill (`PrefillAttention`) needs the same treatment. This is now wiring, not
   kernel work — and it is the first item where Gemma's hot path gets touched, so keep
   the 4-bit dispatch bit-identical when `perLayer` is nil.
2. **8-bit at group 128 for `sharedExpert`**, plus 8-bit `embedding` and the 4-bit
   `router` default. Each is a manifest-guard widening that must land with its kernel.
   With `perLayer` in place these can also be expressed per layer where they vary.
3. **Per-head attention gating.** `AttentionGating.perHead` is parsed and carried
   through config, but no kernel applies it yet.
4. **YaRoP.** `RopeScaling` is parsed and per-layer; the RoPE kernels do not yet
   consume it.
5. **Dense layer 0.** `isDenseMLP(layer:)` exists; the forward pass does not branch
   on it.
6. **Catalog fingerprint.** `--model laguna-s-2-1` now exists but is refused while
   `isInstallable` is false. The real upstream revision is
   `d785a9349850807a34ac0ac1c22c66b718e77881`, recorded only in the catalog's doc
   comment — see the note below on why it is not in the `revision` field yet.
7. **Catalog completion.** `lagunaS2_1` has an empty `revision` and
   `sourceIndexSHA256`. These are **deliberately empty** — the fingerprint check that
   guards against fetching the wrong revision passes vacuously on an empty string, so
   filling them with plausible-looking values is worse than leaving them blank. Pin
   them from the real repo, and only then consider `isInstallable`.

## App / UI layer (done)

Model identity across `TurboFieldfareApp` now derives from the `ModelSource` catalog
instead of restating literals, so a revision bump in one place can no longer leave the
app pinning the previous one. `ModelSource` gained `installBlockedReason`, and
`ModelInstallView` lists non-installable entries as informational text — visible, with
the reason, but with no install affordance. That is what `isInstallable` guards.

Note the App UI target links only `TurboFieldfareAppCore`, not `TurboFieldfareRepackCore`,
so the catalog is reached through a thin projection
(`AppModelInstallDescriptor.unavailableCatalogEntries`) rather than importing
`ModelSource` directly.

## Working notes

- Baseline discipline: Gemma must stay bit-identical. Every phase so far has been
  shaped by that, and it is what makes the refactor safe to continue.
- **Check the checkpoint, don't infer it.** The "mixed precision in the routed path"
  item sat in this file as the top priority until someone read the actual
  `config.json`; the routed path had been uniformly 4-bit the whole time. One `curl`
  of a 106 KB config would have caught it at any point. Fetching config/index metadata
  is cheap and does not require downloading the 64 GB of weights.
- The repo has a real test suite (611 tests as of this writing, all green). Run
  `swift test`, and do not reach green by relaxing an assertion — if a test genuinely
  encodes the old single-model assumption, change it deliberately and say so.
- Metal kernels do not fail loudly on the errors that matter here. Out-of-range
  writes, wrong buffer indices, and threadgroup/simdgroup mismatches all compile and
  run; they just produce wrong numbers. Verify against a CPU reference rather than
  eyeballing plausible output.
- **Mutation-test kernel work before believing it.** Copy the `.metal` aside, break
  one thing the tests claim to cover, confirm the suite actually fails, restore, then
  `grep` to confirm zero mutations survive. This caught two things on this branch:
  it proved the group-128 and phase-2 tests are load-bearing, and it proved the
  router selection tests are *not* (see above). A green suite is not evidence the
  tests are checking anything.
- Prefer widening a guard and its kernel in the same change. Validation that admits
  data the kernels cannot decode is worse than no validation, because it trades a
  clear load-time error for wrong numbers at inference.
