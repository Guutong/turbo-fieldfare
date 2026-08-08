# Handoff: Qwen3.6-35B-A3B on TurboFieldfare

Branch: `generalize-arch-for-moe`
Plan written: 2026-08-07
Supersedes the Laguna-S-2.1 handoff (preserved in Appendix A).

## Goal

Run **Qwen3.6-35B-A3B** (`mlx-community/Qwen3.6-35B-A3B-4bit`) through TurboFieldfare
in ~2 GB of resident RAM on a **16 GB M2 MacBook Air**, by streaming MoE experts from
SSD — the same trick that runs Gemma 4 26B-A4B today.

**Done means:** `TurboFieldfareCLI --model qwen36.gturbo --prompt ...` emits coherent,
correct text at a usable token rate. Chat template, Qwen tool-calling, the OpenAI
server, and the Mac app are explicitly out of scope for this milestone.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Target model | Qwen3.6-35B-A3B, fixed | Not negotiable; accept the Gated DeltaNet build |
| Laguna garbled-output bug | **Skip it** | Do one bring-up, not two. Cost: the pre-norm / sigmoid-router / group-128 paths stay unvalidated and Qwen3.6 inherits them unproven — see Risk R1 |
| Weight source | `mlx-community/Qwen3.6-35B-A3B-4bit` | Uniform group-64 affine = the exact format the Gemma path preserves without requantizing. Same bits the mlx-lm oracle runs, so a parity mismatch is provably our bug, not quantization drift |
| DeltaNet prefill | **Sequential first, chunkwise later** | The sequential path becomes the oracle for the chunkwise kernel. Starting chunked leaves nothing to diff against |
| Vision tower | Dropped | `qwen3_5_moe_text` is a supported text-only config; mlx-lm's `sanitize()` already filters `vision_tower` / `model.visual` |

## Target architecture — verified from config.json

```
model_type            qwen3_5_moe   (text-only variant: qwen3_5_moe_text)
architectures         Qwen3_5MoeForConditionalGeneration
num_hidden_layers     40          hidden_size    2048
vocab_size            248320      tie_word_embeddings  false   ← untied LM head
rms_norm_eps          1e-6        hidden_act     silu
```

**Layer layout** — `full_attention_interval: 4`, i.e. 10 × [linear, linear, linear, full].
Full-attention layers are indices **3, 7, 11, 15, 19, 23, 27, 31, 35, 39**. All other
30 layers are Gated DeltaNet. `sliding_window` **does not exist** in this config.

```
Full attention (10 layers)
  num_attention_heads   16      num_key_value_heads  2      head_dim  256
  partial_rotary_factor 0.25 → rotary dim 64        rope_theta  10_000_000
  rope_type "default", mrope_interleaved true, mrope_section [11,11,10]

Gated DeltaNet (30 layers)
  linear_num_key_heads   16     linear_key_head_dim    128   → q,k: 2048→2048
  linear_num_value_heads 32     linear_value_head_dim  128   → v:   2048→4096
  linear_conv_kernel_dim 4      (causal conv1d over q/k/v before the delta rule)
  32 value heads over 16 key heads → each key head serves 2 value heads

MoE (all 40 layers — mlp_only_layers is ABSENT, so there are no dense layers)
  num_experts 256    num_experts_per_tok 8
  moe_intermediate_size 512    shared_expert_intermediate_size 512
```

**Unverified — must be read out of mlx-lm before writing the router:** `scoring_func`,
`norm_topk_prob`, and `moe_routed_scaling_factor` are all ABSENT from config.json, so
they fall back to defaults in the modeling code. Qwen3-MoE historically uses
softmax + `norm_topk_prob`, but the Qwen3-Next/DeltaNet family may differ. **Getting
this wrong produces exactly the Laguna failure mode: fluent words, incoherent text.**

## Memory budget (4-bit, estimated)

| Resident | Size |
|---|---|
| Token embeddings (248320 × 2048) | ~286 MB |
| LM head — **untied, a second full copy** | ~286 MB |
| Full attention weights, 10 layers | ~106 MB |
| DeltaNet weights, 30 layers | ~426 MB |
| Shared experts, 40 layers | ~71 MB |
| Routers, 40 × 2048×256 | ~21–42 MB |
| **Weights subtotal** | **~1.22 GB** |
| DeltaNet recurrent state (30 × 32 heads × 128 × 128 fp32) | ~63 MB — **context-independent** |
| Conv1d state | ~2 MB |
| KV cache @ 8K ctx (only 10 layers carry one) | ~164 MB |
| **Total before expert cache** | **~1.45 GB** |

Leaves roughly **550 MB for the hot-expert cache** inside a 2 GB budget (~310 of 10,240
experts). Tight but workable.

**Context must be capped.** KV is ~20 KB/token across the 10 full-attention layers:
8K → 164 MB, 32K → 655 MB, and the model's native 262K → **5.4 GB**, which blows the
budget on its own. Ship with a cap around 8–16K.

**Expert streaming:** each expert is ~1.77 MB (3 × 2048×512 at 4-bit + group-64 scales).
Worst case with zero cache reuse is 8 × 40 × 1.77 MB ≈ **566 MB/token**; at ~2.5–3 GB/s
NVMe that is ~4–5 tok/s, in the same range as Gemma's 5–6. On-disk expert total
≈ **18.1 GB**, ~19.2 GB installed.

## Blocker inventory

| # | Blocker | Location |
|---|---|---|
| B1 | Throws on any `layer_type` outside `full_attention`/`sliding_attention` | `ArchInfo.swift:76` |
| B2 | `sliding_window` read as required; key absent in this config | `ArchInfo.swift` (`try i("sliding_window")`) |
| B3 | `model_type` table has no `qwen3_5_moe` entry — silently falls back to Gemma's sandwich norm + softmax router + gelu | `ArchInfo.swift`, `switch modelType` |
| B4 | `fullAttentionLayerMask` is binary (1=full, 0=sliding). Needs a **third** state for linear | `ArchInfo.swift`, `GTurboManifestV1.swift`, `ManifestReader.swift` |
| B5 | No DeltaNet kernels, no recurrent state, no conv1d state anywhere in the runtime | `Kernels/`, `Metal/` |
| B6 | Checkpoint stores **fused `gate_up_proj`**; repacker expects separate gate/up | `RepackPlanner.swift`, `TensorMetadata.swift` |
| B7 | KV cache allocated for all layers; must allocate for the 10 full layers only | `RealForwardRunner.swift` |
| B8 | Packed-experts layout untested at 256 experts × 40 layers = 10,240 entries | `PackedExpertsLayout.swift`, `GTurboPackedExpertsLayoutV1.swift` |
| B9 | MRoPE — for text-only, all three sections share one position, so it should reduce to plain partial RoPE (rotary dim 64). **Verify, do not assume** | `Kernels/.../RoPE` |
| B10 | Vision tensors must be filtered during repack | `RepackPlanner.swift` |

## Who runs each phase

The `fable-method/opencode/` local-model layer (`spec-gate.js` + `judge-gate.js`) is a
**process** fix, not a capability fix — Round 12 concluded the bottleneck "was never raw
capability; it was synthesis and normative framing, and both move into code." So it fits
phases that have a written spec and a mechanically checkable result, and does not fit the
kernel math, where correctness is an activation tolerance rather than a command exiting
zero. `Qwen3.6-35B-A3B` in particular was rated 7/10 and "avoid for agent work" in the pi
eval — the judge-gate compensates for its honesty failure, not for novel kernel authorship.

| Phase | Runner | Why |
|---|---|---|
| 0 — oracle dump | **opencode + local model** | Scripted mlx-lm hooks; success is a file with the right shapes |
| 1 — format & repack | **opencode + local model** | Narrow diffs against existing tests; exactly what the two gates were built for |
| 2 — attention in isolation | **mixed** | Wiring is mechanical; diagnosing a tolerance failure is not |
| 3 — DeltaNet in Swift | **frontier only** | Novel numerical code with no local reference to imitate |
| 4 — DeltaNet in Metal | **frontier only** | Hardest category in the plan |
| 5 — prefill | **mixed** | Measurable, but the fixes are judgement calls |
| 6 — memory & throughput | **mixed** | Measurement is scriptable; tuning is not |

Phases 3 and 4 are the ones to protect. If they get handed to a local model because the
earlier phases went well, the failure will look like plausible Metal that produces fluent
garbage — indistinguishable from R1 and R2, and it will cost days to separate.

**Do not pull server work forward.** TurboFieldfare already ships opencode fixtures
(`Tests/TurboFieldfareServer/Fixtures/opencode-1.15.11-*.json`) and an opencode config in
`docs/OPENAI_SERVER.md`, so the eventual state is opencode ← TurboFieldfare ← Qwen3.6 on
this machine. That is downstream of this milestone: the server needed to dogfood the build
does not exist until the architecture works, and chasing it early stalls the bring-up.

## Phase plan

### Phase 0 — Oracle first, no Swift yet · *local-model opencode*

The single highest-leverage step. Everything downstream is diffed against this.

- **The oracle cannot run on the Air.** 19 GB of 4-bit weights will not fit in 16 GB of
  unified memory. Run mlx-lm on the **omlx box (`100.71.235.112`)**, which already serves
  this model, and copy the dumps back.
- Read `mlx_lm/models/qwen3_5_moe.py` end to end. Extract, verbatim: router scoring,
  `norm_topk_prob`, routed scaling, whether attention applies q_norm/k_norm, the exact
  DeltaNet projection set (q, k, v, a/beta, gate), conv1d placement, and the delta-rule
  recurrence.
- For one fixed short prompt, dump to safetensors: token ids, embedding output,
  **per-layer hidden states for all 40 layers**, plus router logits and selected expert
  ids for a couple of layers. Commit as a test fixture.

**Exit:** fixture committed; router and DeltaNet math written down in prose with the
mlx-lm line references, in `docs/IMPLEMENTATION_REFERENCES.md`.

### Phase 1 — Format and repack · *local-model opencode*

- B1/B2/B3: accept `linear_attention`, make `sliding_window` optional, add the
  `qwen3_5_moe` entry (preNorm / silu / router-scoring-from-Phase-0).
- B4: replace the binary mask with a 3-way layer-kind array. Additive manifest field,
  defaulted so existing Gemma installs still validate byte-identically.
- B6 fused `gate_up_proj` split, B10 vision filtering, B8 layout at 10,240 experts.

**Exit:** `qwen36.gturbo` on disk; layout validator passes; manifest `arch` block
matches config.json field by field.

### Phase 2 — Full-attention layers, in isolation · *mixed*

Do **not** try to run all 40 layers. Use the fixture to **inject the oracle's hidden
state at the input of layer 3** and check only that layer's output.

- Wire loading, expert streaming, router, shared expert, MoE combine.
- Full attention: 16q/2kv, head_dim 256, rotary dim 64, theta 1e7.
- B7: allocate KV for the 10 full layers only.

**Exit:** layer 3 output matches the fixture within tolerance, with correct expert
selection. This validates streaming + MoE + router *before* DeltaNet exists.

### Phase 3 — Gated DeltaNet in plain Swift · *frontier only*

CPU, fp32, sequential, no Metal. Slow and obviously correct.

- conv1d state, delta-rule recurrence, gating, 16 key heads → 32 value heads.
- Test layer 0 in isolation against the fixture, same injection trick as Phase 2.

**Exit:** layer 0 matches the fixture; then a full 40-layer forward (Swift for the 30
linear layers, Metal for the rest) produces **coherent text**. Slow is fine. This is
the milestone that proves the architecture — everything after is optimization.

### Phase 4 — DeltaNet decode in Metal · *frontier only*

Port the recurrence to Metal with state resident in GPU buffers. Diff every layer
against the Phase 3 Swift path — same machine, same weights, so any divergence is the
kernel.

**Exit:** logits match the Swift path; DeltaNet no longer dominates decode time.

### Phase 5 — Prefill · *mixed*

Sequential recurrence over prompt tokens, sharing the decode step. Chunkwise parallel
scan is deliberately deferred; when it lands, diff it against this.

**Exit:** coherent output for multi-token prompts.

### Phase 6 — Memory and throughput · *mixed*

Tune the hot-expert cache for 256-way granularity, enforce the context cap, measure.

**Exit:** ≤2 GB resident at 8–16K context, ≥4 tok/s decode.

## Risks

- **R1 — Inherited unproven paths.** Skipping Laguna means pre-norm, sigmoid routing and
  group-128 reach Qwen3.6 unvalidated. *Mitigation:* Phase 2's layer-isolation test
  exercises all three against a real oracle, which Laguna never had.
- **R2 — Wrong router scoring.** Silently produces fluent-but-incoherent text — the exact
  Laguna symptom. *Mitigation:* Phase 0 reads it from source; Phase 2 asserts on expert ids.
- **R3 — Expert cache thrash.** 256 fine-grained experts reuse worse across tokens than
  Gemma's fatter ones, and 320 reads/token of 1.77 MB is a fragmented I/O pattern.
  *Mitigation:* measure hit rate in Phase 6 before optimizing.
- **R4 — Oracle lives on another machine.** Adds a copy step to every parity check.
  *Mitigation:* dump once, commit the fixture, work offline from it.
- **R5 — Disk.** 80 GB free; ~19 GB needed. Fits, but `scratch/` already holds 73 GB
  (Laguna 60 GB, Gemma 13 GB). Keep Laguna — re-downloading 60 GB costs more than the disk.
- **R6 — MRoPE assumption.** If text-only MRoPE does *not* reduce to plain partial RoPE,
  the 10 attention layers are subtly wrong. B9 verifies before Phase 2.

## Carried over from the Laguna session

Still live, and Qwen3.6 runs the same code:

1. **BF16 router hack.** The repacker stores router weights as raw BF16 while the manifest
   claims 8-bit affine; `router_gemv_bf16_r4` works around it. Fix properly in
   `RepackPlanner.planResidentFile` by quantizing router tensors to INT8 affine group-128,
   then delete the BF16 kernels.
2. **`kPrefillMoEGroupSize` hardcoded to 128** in `prefill.metal`. Make it a runtime
   parameter like the decode path (buffer index 7/9).
3. **ManifestReaderTests** — 4 tests need updating for group-128, 5/8-bit attention, and
   sigmoid routing.
4. **Prefill shared expert** — verify `prefillSharedExpert.encodeBlock` handles group-128.
5. Test baseline at time of writing: 674 tests, 668 pass, 6 pre-existing failures
   (3 × AppModelInstallDescriptorTests, 1 × DenseMLPLayerTests, 2 × RepackCLITests).

---

## Appendix A — previous handoff: generalizing beyond Gemma (Laguna-S-2.1)

Status as of 2026-08-04: Laguna-S-2.1 ran end to end at ~1 tok/s on a 117B MoE, emitting
recognizable English words that did not form coherent sentences. Diagnosed as a likely
tokenizer-mapping or numeric-precision issue rather than a kernel bug. **Not fixed** —
superseded by the decision above.

Fixes landed in that session:

1. **Pre-norm prefill post-attention setup** — `RealForwardRunner.swift` called the
   sandwich variant unconditionally in prefill; added a `switch cfg.normTopology` branch
   (~line 1134) calling `prefillPostAttention.encodePreNorm(...)` for `.preNorm`.
2. **BF16 router kernel** — added `router_gemv_bf16_r4` and
   `prefill_router_sigmoid_bf16_block`, dispatched via `useBF16: routerW.scaleLength == 0`.
   A workaround; see carried-over item 1.
3. **Prefill MoE group-128** — added `kPrefillMoEGroupSize = 128`; the QMM attention path
   stays on `kPrefillGroupSize = 64`.
4. **Decode MoE group-128** — added `model.routedExpertGroupSize`, passed explicitly at all
   `encodeRoutedPersistentPhase1*` / `Phase2Reduce` call sites.
5. **Sigmoid router, silu activation, pre-norm topology** — verified working.
6. **Mac app model selector** — model picker plus `switchModelSource(toID:)`.

Files touched: `Metal/MoE/moe.metal`, `Metal/Prefill/prefill.metal`, `Kernels/MoE/MoE.swift`,
`Kernels/Prefill/MoE/PrefillRouter.swift`, `Runtime/Inference/RealForwardRunner.swift`,
`Runtime/Inference/Model.swift`, and the App install/state files.
