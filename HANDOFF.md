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

Laguna is **not runnable yet**. It now repacks to `.gturbo` and loads, and the
decoder-block difference described below is implemented; it stops at
`RealForwardRunner.init` on two deliberate refusals (sigmoid routing, silu) that
have no kernel behind them yet.

**Start at "Read this first: a gate was opened without a kernel behind it."** There
is an open validation gate with no kernel behind it and four failing tests that are
correctly objecting to it. Everything else in this file is less urgent than that.

## Read this first: a gate was opened without a kernel behind it

Commit `1c40a81` ("Allow group 128 for router in manifest validation") is **wrong and
must be reverted or backed by a kernel.** It was made while chasing a load error
without checking whether a kernel existed, which is the exact trap the comment above
`validateQuant` and the tests below both warn about.

Laguna's router is 8-bit **group 128** (`manifest.quant.router`). `router_gemv_gemma4_r4`
is group-**64**-only, and not by omission — `router_gemv_gemma4_body`
(`Metal/MoE/moe.metal:86`) hardcodes it structurally:

```
const uint n_groups = DD / kMoEGroupSize;              // 64
const uint idx = g * kMoEGroupSize + lane * 2u;        // 32 lanes x 2 = 64/group
```

With the gate open, a group-128 router manifest loads cleanly and the kernel reads
scales and biases at the wrong stride. It does not crash. It produces a plausible
routing distribution that is wrong for every token. Today Laguna does not reach it
only because the `.sigmoidTopK` guard in `RealForwardRunner.init` throws first —
that is luck, not protection, and it does not protect any other model.

The sibling commit `569bdeb` ("Allow 5-bit and 8-bit attention") is, by contrast,
**legitimate — checked, not assumed.** `FusedQKVGEMVGeneric`
(`Kernels/Fusions/FusedQKVGEMVGeneric.swift:24`) wraps `dequant_int5_qkv_gemv_simd`
and `dequant_int8_qkv_gemv_simd` and takes `groupSize` at runtime, and
`RealForwardRunner` dispatches per layer from `attentionQuantByLayer` in both decode
(`:1654`) and prefill (`:870`, `:887`). That gate has a kernel behind it.

So the four failing tests split two and two, and the two halves need opposite
treatment:

| test | site | verdict |
|---|---|---|
| `productionManifestRejectsGroup128ForSlotsWhoseKernelIsGroup64` [router] | `:269` | **correct** — no kernel; fix the gate or the kernel |
| `perLayerOverrideCannotBypassAGroupSizeGate` (router 4-bit g128) | `:339` | **correct** — same |
| `perLayerOverrideWithUnsupportedWidthIsRejected` (attention 5-bit) | `:324` | **obsolete** — kernel now exists |
| `lagunaMixedAttentionIsExpressibleButStillRejectedByTheKernelGate` | `:375` | **obsolete** — its own name records the assumption that expired |

For the first two: do not delete or relax them to get green. Their comments say it
outright — *"Accepting 128 would load cleanly and then produce silently wrong
numbers, so the rejection is the feature. Relax a slot only in the same change that
teaches its kernel group 128."* Either teach `router_gemv_gemma4_body` a runtime
group size (the generic MoE kernels at `moe.metal:272` and `:409` show the pattern:
take `groupSize` as a parameter and stride by lane), or revert `1c40a81` and let
Laguna fail at load until the kernel exists.

For the second two: update them deliberately and say so in the commit message, which
is what the "Working notes" rule below asks for. They encode "5-bit attention has no
kernel", which stopped being true when `FusedQKVGEMVGeneric` landed. Rewrite them to
assert what is still true — that a width with *no* kernel is rejected — rather than
deleting the coverage. A width nobody implemented (6-bit, say) is the natural
replacement subject.

Sequencing note: the sigmoid router kernel (item 1 below) has to be written anyway,
and it needs a group-size-generic GEMV front half. Doing that first and giving the
Gemma router the same treatment is less work than doing them separately.

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

### 5- and 8-bit fused QKV (done — kernels and forward-pass dispatch)

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

### Per-layer QKV dispatch in decode (done)

This is the first change on this branch that touches Gemma's hot path, so the shape of
it matters more than usual.

- `Model.attentionQuant(atLayer:)` resolves a layer's `(weightBits, groupSize)` from
  the manifest, falling back to 4-bit group 64 when there is no `quant` block at all.
  The rule itself is a static `Model.attentionQuant(_:atLayer:)` so it can be tested
  without standing up a whole `Model` — the no-quant-block case is awkward to construct
  and it is easy to write a test that only asserts on the fixture. An earlier version
  of that test did exactly that and was rewritten.
- `RealForwardRunner` resolves all layers once at init into `attentionQuantByLayer`,
  rather than re-walking the override table inside the decode loop.
- It builds one `FusedQKVGEMVGeneric` per **distinct non-4-bit width** the model uses.
  A uniformly 4-bit checkpoint builds none, so it pays nothing for the capability; a
  mixed one gets its PSOs up front instead of stalling mid-decode to compile one.
  An unimplemented width throws at init with a named error, not at first use.
- The dispatch branches on `weightBits != 4`, and **the 4-bit branch is the original
  call unchanged** — so a uniformly 4-bit checkpoint encodes exactly what it did before.

**One hazard worth knowing about, because the first version had it.** Writing the
branch as `if bits != 4, let wide = table[bits]` falls through to the 4-bit encoder
when the lookup misses, which would read 5-bit bytes as nibbles and produce numbers
that look plausible and are entirely wrong. It is now a `guard ... else {
preconditionFailure }`. The invariant that makes the miss unreachable lives in `init`;
relying on it silently is the trap this codebase keeps setting.

`AttentionQuantSelectionTests` covers uniform slots, Laguna's real 20/28 split, the
no-quant-block fallback, and that `distinctConfigurations` (what init builds) is a
superset of what `resolved(atLayer:)` (what dispatch asks for) can return — a
disagreement between those two is precisely what would trip the precondition at decode
time. Mutation-checked: ignoring per-layer overrides, and a wrong fallback width, both
fail.

## What still blocks Laguna

All seven items below are done. They are **not** the whole list — see "The real
remaining blocker" after them, which is what actually stops Laguna today.

In rough dependency order:

1. **Prefill (done).** `RealForwardRunner` prefill attention projections (`Q`, `K`, `V`, `O`) now resolve quantization `(weightBits, groupSize)` per layer via `attentionQuantByLayer[L]`. Standard 4-bit group-64 layers retain the 4-bit fast path (`prefillQMM` / `DequantInt4GEMV`), while non-4-bit or group-128 layers (e.g., 5-bit or 8-bit attention projections) dispatch via per-row sub-byte GEMV encoders (`DequantInt5GEMV` / `DequantInt8GEMV` / `DequantInt4GEMVGeneric`). Verified with `PrefillAttentionQuantTests`.
2. **8-bit at group 128 for `sharedExpert` (done).** Widened `validateQuant` in `ManifestReader.swift` and `IndexLoader.swift` to allow 8-bit/4-bit group 128/64 for `sharedExpert`, `embedding`, `attention`, and `router`. Implemented sub-byte 8-bit group 128 GEMV encoder (`DequantInt8GEMV` generic path & `dequant_int8_gemv_generic`) and verified in `SharedExpertInt8Tests.swift` and `ManifestReaderLagunaTests.swift`.
3. **Per-head attention gating (done).** Added `apply_attention_gating_per_head` Metal kernel in `attention.metal` (calculating `softplus(g) = (g > 20.0f) ? g : log(1.0f + exp(g))` on `g_out[t, h]` and scaling `attn_out[t, h, d]` in-place), `AttentionGatingKernel` host wrapper in `AttentionGating.swift`, and verified against CPU reference in `AttentionGatingTests.swift`.
4. **YaRoP (done).** Implemented YaRN RoPE frequency scaling in `rope.metal`, `prefill.metal`, `fused.metal`, and `RoPE.swift` CPU reference, passing per-layer `ropeScaling` and `attentionFactor` score scaling in `RealForwardRunner`. Verified with `YaRoPTests.swift`.
5. **Dense layer 0 (done).** Forward pass in `RealForwardRunner` branches on `cfg.isDenseMLP(L)`, dispatching dense MLP projections (`gate_proj`, `up_proj`, `down_proj`) for dense layers, sizing scratch buffers to `max(config.intermediateSize, config.denseMLPIntermediateSize)`, and skipping MoE routing. Verified with `DenseMLPLayerTests.swift`.
6 & 7. **Catalog completion & fingerprinting (done).** Catalog entry `lagunaS2_1` in `SupportedModelSource.swift` is now pinned with revision `d785a9349850807a34ac0ac1c22c66b718e77881` and `sourceIndexSHA256` (`45709bf61be0398b4b34ed68845c80f8d2bab75f1f16d19e63c95e15571f0cc2`), with `isInstallable: true`. Verified with `SupportedModelSourceTests.swift`.

## The real remaining blocker: Laguna's decoder block is a different shape

The seven items above were the whole list, and were all marked done — but treat those
"verified with X" claims as the previous author's, not as checked. Item 5 cites
`DenseMLPLayerTests.swift`, and that test currently fails on its own fixture
(`ffnIntermediate = 64; expected 256`). The list was also written before anyone got a
Laguna checkpoint far enough to *load*. Doing that
(see "Repacking from a local checkpoint" below) surfaced a blocker that was never on
it: the runtime's per-layer topology is Gemma's sandwich-norm block, and Laguna is a
plain pre-norm block. This is not a naming mismatch that an alias can paper over —
the two blocks compute different things.

Gemma 4, as `RealForwardRunner` implements it:

```
h1 = rmsnorm(dense(pre_feedforward_layernorm(x)),   post_feedforward_layernorm_1)
h2 = rmsnorm(routed(pre_feedforward_layernorm_2(x)), post_feedforward_layernorm_2)
h  = x + layer_scalar * rmsnorm(h1 + h2, post_feedforward_layernorm)
```

Laguna (`LagunaDecoderLayer.forward`, `modeling_laguna.py:490`):

```
h = x + attn(input_layernorm(x))
h = h + mlp(post_attention_layernorm(h))
```

where `mlp` is `shared_expert(u) + 2.5 * routed(u)` on a *single* normed input `u`,
or a plain dense MLP on layer 0. Note that Laguna's `post_attention_layernorm` is
Qwen-convention: it normalizes the *residual before the MLP*, not the attention
output. Gemma's same-named tensor normalizes the attention output. Do not assume the
names line up.

Eight per-layer tensors the runtime demands do not exist in the checkpoint at all —
verified against `model.safetensors.index.json`, not inferred:

```
pre_feedforward_layernorm[.weight]     post_feedforward_layernorm[_1,_2][.weight]
router.scale                           router.per_expert_scale
layer_scalar
```

`Model.postFFN1` is the accessor that actually throws first
(`no IndexEntry named ...post_feedforward_layernorm_1.weight`), from
`RealForwardRunner.init` building `sharedExpertProjections`.

Two more divergences sit behind that one and will not announce themselves — they
produce plausible wrong numbers rather than an error:

- **Router scoring.** `router_topk_gemma4` does softmax over top-k and applies
  `per_expert_scale`. `LagunaTopKRouter` (`modeling_laguna.py:169`) does **sigmoid**
  over all experts, adds `e_score_correction_bias` *for selection only*, then
  renormalizes the unbiased gathered weights (`norm_topk_prob: true`). The repacker
  already packs `mlp.gate.e_score_correction_bias` (`RepackPlanner.swift:618`); the
  runtime has no consumer for it and no sigmoid path.
- **Activation.** Every activation in `Sources/TurboFieldfare/Metal/` is hardcoded
  `gelu_pytorch_tanh` (`moe.metal`, `prefill.metal`, `dequant_int8.metal`,
  `utility.metal`). Laguna uses **silu**. Worse, the manifest currently *says* gelu:
  `ArchInfo.swift:204` reads `hidden_activation` then `hidden_act`, and Laguna's
  `config.json` carries neither — `silu` is a Python-side default in
  `configuration_laguna.py:147`. So `hiddenActivation` is carried and validated but
  never reaches a kernel. (The manifest side of this is now fixed: `ArchInfo.load`
  supplies the right default per `model_type`. Nothing reads it yet.)

### Norm topology — done, but unverified

Laguna's block maps onto the existing kernel slots cleanly once the norms can be
*skipped* rather than substituted — a unit-weight tensor is not equivalent, since
RMSNorm still divides by the RMS:

| Gemma slot | Laguna |
|---|---|
| `post_attention_layernorm` (on attn out) | skip |
| `pre_feedforward_layernorm`, `_2` | both = `post_attention_layernorm` |
| `post_feedforward_layernorm`, `_1`, `_2` | skip |
| `layer_scalar` | 1.0 |
| routed branch | scale by `routedScalingFactor` (2.5, already in arch) |

This is implemented end-to-end and compiles. **No test covers it and no model has
run through it** — treat every claim below as "written, not verified".

`ArchConfig` gained `normTopology: NormTopology` (`.sandwich` default / `.preNorm`)
and `routerScoring: RouterScoring` (`.softmaxTopK` default / `.sigmoidTopK`), both
defined in `ModelTypes.swift` with the reference formulas in their doc comments.
They round-trip through `manifest.json -> arch` (`ManifestReader`, `GTurboJSON`)
and are only written when they differ from Gemma, so a Gemma repack still produces
a byte-identical `arch` block. `ArchInfo.load` derives them from `model_type`.

Four new Metal kernels, deliberately *separate* from the sandwich ones rather than
function-constant branches inside them, so Gemma's path is literally the code it
was before:

| kernel | file | host wrapper |
|---|---|---|
| `fused_post_attn_setup_prenorm` | `Metal/Fusions/fused.metal` | `FusedPostAttentionSetup.encodePreNorm` |
| `fused_layer_tail_prenorm` | `Metal/Fusions/fused.metal` | `FusedLayerTail.encodePreNorm` |
| `prefill_post_attn_setup_prenorm_block` | `Metal/Prefill/prefill.metal` | `PrefillPostAttentionSetup.encodePreNorm` |
| `prefill_layer_tail_prenorm_block` | `Metal/Prefill/prefill.metal` | `PrefillLayerTail.encodePreNorm` |

Accessors that name a tensor `.preNorm` does not have (`postFFN`, `postFFN1`,
`postFFN2`, `layerScalar`, `routerScale`, `routerPerExpertScale`) now return
`TensorView?` and are nil under that topology, so the forward pass can ask without
knowing which model it holds. `Model.routerSelectionBias` was added for
`e_score_correction_bias`, which the repacker already packs.

Dispatch happens in two helpers so no call site branches twice:
`RealForwardRunner.encodeDecodeLayerTail` (decode, both dense and routed sites) and
the local `encodePrefillLayerTail` (prefill, likewise). `gPostAttnSetup` branches
inline. The shared-expert output norm (`postF1`) is skipped at all four of its call
sites when nil.

One subtlety worth not re-deriving: under `.preNorm` the router input is the *same*
`u = rmsnorm(h) * post_attention_layernorm` that both FFN branches consume, not
Gemma's unweighted `rmsnorm_no_scale(h)`. The prenorm setup kernel therefore writes
`u` to `dense_x`, `routed_x`, and `router_x` alike. It is redundant stores, and it
keeps every downstream consumer topology-agnostic. Relatedly, `.sigmoidTopK` has no
`router.scale` and no `1/sqrt(D)` factor, so `effectiveScaleBuffers` is filled with
BF16 1.0 for those models (`RealForwardRunner.init`).

## Remaining kernel work

Two things are wired everywhere *except* the kernel. `RealForwardRunner.init`
refuses to construct when either is required, throwing
`ModelError.unsupportedArchFeature`, so Laguna fails loudly at load instead of
running Gemma's kernel and returning plausible wrong numbers. **Deleting those two
guards is the last step of each item, not the first.**

### 1. Sigmoid router

Reference: `LagunaTopKRouter.forward`, `modeling_laguna.py:169`.

```python
router_logits   = F.linear(u, W)                 # [T, E], u already normed
routing_scores  = sigmoid(router_logits)         # over ALL experts
scores_for_sel  = routing_scores + e_score_correction_bias
_, sel          = topk(scores_for_sel, top_k)
w               = routing_scores.gather(-1, sel)  # UNBIASED scores
w               = w / w.sum(-1, keepdim=True)     # norm_topk_prob = true
```

Three details that are easy to get wrong and impossible to see in the output:

- The bias shifts *selection only*. The gathered weights come from
  `routing_scores`, not `scores_for_selection`. Folding the bias into the weights
  is a plausible-looking bug that changes every token slightly.
- Renormalization is over the K gathered weights, gated on `norm_topk_prob`
  (true for Laguna). It is not a softmax — sigmoid outputs are independent.
- `router_logit_softcapping` exists in the reference (`tanh(x/c)*c`) but Laguna's
  config does not set it, so it is 0.0 / disabled. Implement it or assert it is
  absent; do not silently ignore a nonzero value.

Sites: `moe.encodeRouterGemma4` (decode, `Kernels/MoE/MoE.swift`, kernel
`router_topk_gemma4`) and `prefillRouter.encodeGemma4Block` (prefill,
`prefill.metal:478` `prefill_router_gemma4_block`). Add `..._sigmoid` twins rather
than editing these. The GEMV front half is identical and can be shared; only the
scoring tail differs. The host side already has `views.routerSelectionBias` /
`model.routerSelectionBias(layer:)` plumbed to the call sites and unused.

### 2. silu activation

Every FFN activation in the tree is hardcoded `gelu_pytorch_tanh`:

```
Metal/MoE/moe.metal:492, 528, 605, 642    (routed experts, four dispatch shapes)
Metal/Prefill/prefill.metal:636           (prefill grouped experts)
Metal/Quant/dequant_int8.metal:152        (8-bit shared expert)
Metal/Primitives/utility.metal:17         (gelu_mul_fp16)
```

`arch.hiddenActivation` is carried and validated but reaches no kernel. Laguna is
silu (`x * sigmoid(x)`); note its `config.json` omits `hidden_act` entirely and
`configuration_laguna.py:147` defaults it to silu — `ArchInfo.load` now supplies
that per `model_type`, so the manifest is correct, but nothing reads it.

A function constant selecting the activation is the cheaper option than doubling
seven kernels; the activation is one inlined call in each. Whichever way, the
selection must be visible in `ArchConfig` rather than inferred at the call site.

### 3. Then

Delete both guards in `RealForwardRunner.init`, repack, and run. Note that
`--local-checkpoint` skips re-downloading; see below.

### Suggested order

The dependencies are mostly one-way, so this order avoids rework:

1. **Router group size** (the section above). Make `router_gemv_gemma4_body` take a
   runtime `groupSize`, and settle the four `ManifestReaderTests` — two by fixing
   the kernel, two by deliberately rewriting an expired assumption. Nothing else
   can be trusted until the validation gates mean something again.
2. **Sigmoid router**, reusing the now-generic GEMV front half. Ends with the
   `routerScoring` guard in `RealForwardRunner.init` deleted.
3. **silu**, which is independent of both and could be done first if you want a
   quick win. Ends with the `hiddenActivation` guard deleted.
4. **Tests for the norm topology**, against a CPU reference, before running Laguna.
   A pre-norm forward pass that is subtly wrong looks exactly like a model that
   needs more sampling tuning.
5. **Laguna end-to-end.**

Steps 1 and 4 are the ones most likely to be skipped under pressure and the ones
that most determine whether the result can be trusted.

### Verification, in the order that catches the most

1. **Gemma first, every time.** It must stay bit-identical. If a Gemma generation
   changes, something in the shared path moved. This is the cheapest signal
   available and it is worth running before looking at Laguna at all. Note there
   is **no automated end-to-end Gemma baseline in the suite** — the golden-value
   tests that exist (`DequantInt4GEMVTests`, `MoEFusedFFNTests`) are per-kernel.
   Comparing a full Gemma generation before and after is a manual step, and
   capturing that baseline as a test would be worth more than most of the work
   below.
2. **CPU reference per kernel, not end-to-end.** The handoff note below about
   Metal failing silently applies to all of this. A wrong buffer index in the
   sigmoid router produces a valid-looking distribution.
3. **Mutation-test before believing green.** Copy the `.metal` aside, break one
   thing each new test claims to cover, confirm the suite fails, restore, `grep`
   for survivors. This has already caught tests on this branch that checked
   nothing (see the router-selection note in "Working notes").
4. Only then a Laguna end-to-end generation, checked for coherent text rather
   than for "it ran".

### State of the test suite

`swift build` is clean. `swift test` is 629 tests with **7 failing**:

| test | file | status |
|---|---|---|
| `productionManifestRejectsGroup128ForSlotsWhoseKernelIsGroup64` [router] | `ManifestReaderTests:269` | **regression** — `1c40a81`, no kernel |
| `perLayerOverrideCannotBypassAGroupSizeGate` | `ManifestReaderTests:339` | **regression** — `1c40a81`, no kernel |
| `perLayerOverrideWithUnsupportedWidthIsRejected` | `ManifestReaderTests:324` | **obsolete test** — `569bdeb`, kernel exists |
| `lagunaMixedAttentionIsExpressibleButStillRejectedByTheKernelGate` | `ManifestReaderTests:375` | **obsolete test** — `569bdeb`, kernel exists |
| `denseLayer0ForwardPassInPrefillAndDecode` | `DenseMLPLayerTests:470` | pre-existing fixture bug (`ffnIntermediate = 64; expected 256`) |
| `nonInstallableSourceIsExcludedFromCatalogInstallOffers` | `AppModelInstallDescriptorTests:63` | pre-existing — expects Laguna non-installable |
| `unavailableCatalogEntriesSurfacesNonInstallableSourcesWithAReason` | `AppModelInstallDescriptorTests:68,72` | pre-existing — same |
| `notYetInstallableModelIDIsRejectedWithItsBlockedReason` | `RepackCLITests:55,56` | pre-existing — disk-space-dependent path |

The four regressions are the subject of the section above. The earlier claim in this
file of "6 pre-existing failures" counted issues, not tests; the real pre-existing
count is 4 tests / 6 issues.

**Nothing that was added this session has a test.** Not the norm topology, not the
local-checkpoint repacker, not the per-slot `groupSize` plumbing, not the dense-layer
layout handling. The four new Metal kernels have never executed — the build compiles
them, which proves only that they parse. That is the largest gap in this branch, and
per the note below a green suite would not have told you otherwise.

## Upstream sync

This branch is merged up to `origin/main` (`fcd8f78`, drumih/turbo-fieldfare) as of
2026-08-03 — zero commits behind. The six upstream commits since the `f8abc44`
branch point touched the OpenAI server, the Mac app, the decode service, the
detokenizer, and benchmark docs; this branch touches runtime kernels, the repacker,
and model IO. **The two sets share no files**, so the merge was automatic and is
likely to stay easy. Re-merge often rather than letting it drift.

The suite went 629 -> 674 tests across the merge with the same 7 failures, so
upstream's 45 new tests pass here and nothing on this branch regressed.

## Repacking from a local checkpoint

`TurboFieldfareRepack --local-checkpoint <dir>` repacks from an already-downloaded HF
snapshot instead of streaming from the Hub, which is what made the above reachable
without re-downloading 65 GB per attempt.

`LocalCheckpointRepacker` supplies the two things the pipeline gets from HTTP: a
`RemoteSnapshot` (metadata copied from disk, shard headers parsed with `pread`) and a
`SourceByteProvider` that `pread`s ranges from local shards. `RepackPlanner`,
`RangeCopyPlanner`, and `WriterCore` are untouched — `SourceByteProvider` was already
the right seam. `resolvedCommit` is synthesized as 40 zeros because
`RemoteInstallCheckpoint.validate` requires exactly 40 hex chars.

Getting Laguna through the repacker turned up a run of format bugs, all fixed:

- `QuantSpec` carried only `bits`, so `writeManifest` fell back to
  `plan.baseGroupSize` (128) for every slot. Laguna's attention, embedding, and router
  tensors are group **64** while the base is 128. `QuantSpec` now carries `groupSize`
  and the manifest emits it per slot.
- Laguna's attention bit width varies by layer (5-bit and 8-bit). The uniformity
  refusal became `perLayer` override emission.
- `validateQuant` admitted only 4-bit attention and 4-bit group-64 router.
- Layer 0 is dense, so it has no expert file. `layout.json` keeps a 0-expert entry so
  `layers` stays indexable by layer number, and a 0-byte placeholder file is written
  beside the real ones; readers and both validators treat an empty expert list as the
  dense marker.
- `layout.json` is 47 x 256 x 8 entries and outgrew the 16 MB metadata cap. It now has
  its own 128 MB cap and is written compact (it is machine-read only), and its
  top-level `expertsPerLayer` is taken from the first layer that has experts rather
  than from layer 0, which has none.

### Reclaiming disk space between attempts

Each failed repack leaves an APFS Time Machine local snapshot that `df` does not
report, so `df` will claim tens of GB less free than the Finder does. `diskutil info`
lists them; `sudo tmutil deletelocalsnapshots /` reclaims them. This bit several
attempts in a row before it was diagnosed.

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
- The repo has a real test suite (629 tests as of this writing, 6 pre-existing failures
  in `DenseMLPLayerTests`, `AppModelInstallDescriptorTests`, and `RepackCLITests` —
  fixture mismatches and disk-space-dependent error paths, not regressions). Run
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
