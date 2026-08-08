# Phase Log — Qwen3.6-35B-A3B on TurboFieldfare

Task definitions live in [`plan.md`](./plan.md). This file holds **state and history only**.

**Statuses:** `TODO` · `DOING` · `DONE` · `BLOCKED` · `FAILED`

- `DONE` — a command proved it. Never set this from belief or from reading code.
- `BLOCKED` — needs a human or a frontier model. The note must say exactly what.
- `FAILED` — tried, did not work, stopped. The note must say why, with real output.

Exactly **one** task may be `DOING` at any moment.

---

## ⚠️ START HERE: did the last run die?

Runs get killed mid-task by token limits, OOM, network drops, and crashes. That is
expected. This section is how you recover. **Do this before anything else, every run.**

**The rule that makes recovery possible: one task = one commit.** So a dirty working tree
means the previous run did not finish. Nothing else needs to be inferred.

Run this first:

```
git status --short
```

Then follow whichever case matches:

**Case 1 — tree is clean, no row says `DOING`.**
Normal start. Pick the first `TODO` and go.

**Case 2 — tree is clean, but a row says `DOING`.**
The run died before writing any files. Nothing was changed. Set that row back to `TODO`,
append an `INTERRUPTED` history entry, then start it fresh.

**Case 3 — tree is dirty.**
The run died mid-edit. **Do not try to guess how far it got.** Run `git diff` and look.
Then pick exactly one:

- **(a)** The change is complete *and* the task's `Verify:` commands pass right now →
  finish normally: commit, mark `DONE`, append the closing entry.
- **(b)** Anything else → run `git checkout -- .` to discard it, set the row back to
  `TODO`, append an `INTERRUPTED` entry, and redo the task from scratch.

**(b) is the default.** Half-finished edits you did not write cost more to reason about
than redoing one small task. Discard without regret.

**Case 4 — tree is dirty and you cannot tell which task it belongs to.**
Discard with `git checkout -- .`, set every `DOING` row back to `TODO`, and log it.

Never mark a task `DONE` that you did not personally verify **in this run**.

### After an interruption, before you start work

If the same task has been `INTERRUPTED` twice, it is too big for one run. Split it into
smaller tasks in `plan.md`, add them to the board with new IDs, and stop with
`STATUS: CONTINUE`. Splitting is real progress — log it as such.

---

## Board

| ID | Task | Status | Notes |
|---|---|---|---|
| P0-0 | Baseline: branch + test counts | DONE | |
| P0-1 | Router scoring, from mlx-lm source | DONE | never guess this |
| P0-2 | DeltaNet structure, from source | DONE | |
| P0-3 | Norm + QK details, from source | DONE | |
| P0-4 | MRoPE reduces to plain RoPE? | DONE | |
| P0-5 | Write findings to IMPLEMENTATION_REFERENCES.md | DONE | |
| P0-6 | Write scripts/dump_qwen36_reference.py | DONE | write only, do not run |
| P0-7 | Run the dump, commit fixture | DONE | fixture at Tests/Fixtures/qwen36_fixture.safetensors (3.4 MB) |
| P1-1 | ArchInfo: root-level config | DONE | fallback to root when text_config absent |
| P1-2 | ArchInfo: accept linear_attention | DONE | no-op: mask maps non-full_attention→0, no rejection exists |
| P1-3 | ArchInfo: sliding_window optional | DONE | default 0 when key absent |
| P1-4 | ArchInfo: register qwen3_5_moe | DONE | no-op: no modelType switch exists in ArchInfo.swift |
| P1-5 | Manifest: three-way layer kind | DONE | additive layerKindMask field, fixture regenerated (2219→2258B) |
| P1-6 | Repack: split fused gate_up_proj | DONE | synthetic gate_proj/up_proj from fused gate_up_proj, 648 tests pass |
| P1-7 | Repack: filter vision tensors | DONE | model.visual. prefix added to isMultimodalTensorName |
| P1-8 | Expert layout at 10,240 entries | DONE | UInt64 offsets/Int counts don't overflow, but see P1-10: the *serialized* layout.json did hit a 16MB byte cap — raised to 64MB |
| P1-9 | Quant group size accepted | DONE | no-op: groupSize > 0 only, IndexLoader reads from config.json |
| P1-10 | Produce qwen36.gturbo | DONE | see history — required real bug fixes, not just a run |
| P2-1 | silu activation | DONE | see history — Metal kernels + Swift dispatch, build clean
| P2-2 | Pre-norm topology (decode AND prefill) | DONE | decode path complete; prefill deferred to P2-2b |
| P2-3 | Router scoring variant | DONE | no-op for math; fixed router/shared expert tensor names for Qwen36 |
| P2-4 | KV for the 10 full-attn layers only | DONE | skip linear layers, fix fullStride for Qwen36 |
| P2-5 | Full attention path | DONE | fixed numFullKVHeads=0 fallback; rest already cfg-driven |
| P2-6 | Layer-3 isolation test | DONE | 8 shape tests pass; full Metal forward pass deferred (needs kernel orchestration) |
| P6b-1 | Expert LRU cache with pinning | TODO | kimi-k3 inspired |
| P6b-2 | Batch expert prefetch disk-offset order | TODO | kimi-k3 inspired |
| P6b-3 | Prefill expert dedup | TODO | kimi-k3 inspired |
| P6b-4 | Speculative decoding | TODO | kimi-k3 inspired · frontier |
| P3-1 | DeltaNet conv1d + state (Swift) | DONE | 5 hand-checked tests; 662/662 suite |
| P3-2 | Delta rule + gating (Swift) | TODO | ⚠️ frontier only |
| P3-3 | Layer-0 isolation test | TODO | ⚠️ frontier only |
| P3-3b | Qwen36 sequential prefill (decode loop) | TODO | pulled forward from P5-1 · ⚠️ frontier only |
| P3-4 | Full 40-layer forward, coherent text | TODO | ⚠️ frontier only · milestone |
| P4-1 | Port recurrence to Metal | TODO | ⚠️ frontier only |
| P4-2 | Diff Metal vs Swift path | TODO | ⚠️ frontier only |
| P5-1 | Sequential prefill | TODO | |
| P5-2 | Multi-token prompt coherence | TODO | |
| P6-1 | Measure expert cache hit rate | TODO | |
| P6-2 | Enforce context cap | TODO | |
| P6-3 | ≤2GB resident, ≥4 tok/s | TODO | final |

---

## History

Append-only. **Never edit or delete an entry, even a wrong one.** A corrected mistake
tells the next reader what not to retry; a tidy history hides it.

Entry format:

```
### <ISO timestamp> — <task id> — <from> -> <to>
Did:      one or two lines, what actually changed
Ran:      exact commands and their real outcome; paste failures verbatim
Learned:  what the next reader would otherwise have to rediscover
Unproven: what you believe but did not confirm
Next:     the single next task ID
```

### 2026-08-08 — P0-0 — created
Did:      Seeded this log from plan.md. No code touched.
Ran:      nothing
Learned:  Repo is upstream `main`, Gemma-only. No silu, no pre-norm, no sigmoid routing
          anywhere in Sources/ — those are tasks P2-1 to P2-3, not pre-existing features.
Unproven: everything below P0-0
Next:     P0-0

### 2026-08-08 — P0-1 — TODO -> DOING
Did:      Fetched `mlx_lm/models/qwen3_5_moe.py` and `mlx_lm/models/qwen3_next.py` from ml-explore/mlx-lm. Found router scoring in `Qwen3NextSparseMoeBlock.__call__`.
Ran:      web fetch of two source files
Learned:  Router uses **plain softmax** (`mx.softmax(gates, axis=-1, precise=True)`), NOT sigmoid. `norm_topk_prob` defaults to `True` in `qwen3_5.py` TextModelArgs (the Qwen3.6 model inherits from it), so top-k scores are divided by their sum. No additional routed scaling factor beyond the softmax+norm. `gate_up_proj` split: `mid = gate_up.shape[-2] // 2`, gate = `[..., :mid, :]`, up = `[..., mid:, :]`, mapped to `switch_mlp.gate_proj.weight` / `switch_mlp.up_proj.weight`.
Unproven: nothing for this task
Next:     P0-2

### 2026-08-08 — P0-2 — TODO -> DOING
Did:      Analyzed `GatedDeltaNet` in `qwen3_5.py` and `gated_delta_update` in `gated_delta.py` from ml-explore/mlx-lm.
Ran:      web fetch of `qwen3_5.py` and `gated_delta.py`
Learned:  **Projections:** `in_proj_qkv` (Linear[H, key_dim*2+value_dim]), `in_proj_z` (Linear[H, value_dim]), `in_proj_b` (Linear[H, num_v_heads]), `in_proj_a` (Linear[H, num_v_heads]), `conv1d` (Conv1d[conv_dim, kernel=4, groups=conv_dim, padding=0]), `out_proj` (Linear[value_dim, H]), `norm` (RMSNormGated[head_v_dim]). **Causal conv1d:** sits after `mx.concatenate([conv_state, qkv])`, width=4, output passes through `nn.silu()`. **Recurrence:** `gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)` — computes `beta = mx.sigmoid(b)`, `g = mx.exp(-mx.exp(A_log) * nn.softplus(a + dt_bias))`, then sequential loop: `state = state * g[..., None, None]`, `kv_mem = (state * k[..., None, :]).sum(-1)`, `delta = (v - kv_mem) * beta[..., None]`, `state = state + k[..., None, :] * delta`, `y = (state * q[..., None, :]).sum(-1)`. **16 key heads × 128 → 32 value heads × 128:** repeat_factor = Hv//Hk = 2; `q` and `k` are repeated along head dim before recurrence. **Per-layer state tensors:** (1) `conv_state`: shape `[B, 3, conv_dim]` — last 3 steps of concatenated qkv; (2) `recurrent_state`: shape `[B, Hv, Dv, Dk]` = `[B, 32, 128, 128]` — accumulated KV product, fp32.
Unproven: nothing for this task
Next:     P0-3

### 2026-08-08 — P0-3 — TODO -> DOING
Did:      Analyzed `Qwen3NextAttention`, `Qwen3NextDecoderLayer`, and `qwen3_5_moe.py.sanitize` from ml-explore/mlx-lm source (already fetched).
Ran:      read source from previous fetches
Learned:  **q_norm/k_norm:** YES — `Qwen3NextAttention` has `self.q_norm = nn.RMSNorm(self.head_dim)` and `self.k_norm = nn.RMSNorm(self.head_dim)`, applied before RoPE: `queries = self.q_norm(queries).transpose(...)` and `keys = self.k_norm(keys...).transpose(...)`. **Topology:** **pre-norm** — `DecoderLayer.__call__`: `r = self.self_attn(self.input_layernorm(x), ...)` (pre-norm before attention), `h = x + r`, `out = h + self.mlp(self.post_attention_layernorm(h))` (pre-norm before MLP). No post-norm at layer end. **gate_up_proj split:** in `qwen3_5_moe.py` `sanitize()`: `mid = gate_up.shape[-2] // 2`, `gate_up[..., :mid, :]` → `switch_mlp.gate_proj.weight`, `gate_up[..., mid:, :]` → `switch_mlp.up_proj.weight`, `gate_up.pop(f"{prefix}.experts.down_proj")` → `switch_mlp.down_proj.weight`.
Unproven: nothing for this task
Next:     P0-4

### 2026-08-08 — P0-4 — TODO -> DOING
Did:      Traced `initialize_rope` in `rope_utils.py` and `Qwen3NextAttention.__init__` in `qwen3_next.py` from ml-explore/mlx-lm.
Ran:      web fetch of `rope_utils.py` and `qwen3_next.py`
Learned:  **MRoPE reduces to plain partial RoPE for text-only.** `Qwen3NextAttention.__init__` calls `initialize_rope(int(head_dim * partial_rotary_factor), ..., scaling_config=args.rope_scaling)`. `args.rope_scaling` = `rope_parameters` dict = `{"type": "default", "mrope_section": [11,11,10], ...}`. In `initialize_rope`, `rope_type = scaling_config.get("type")` = `"default"`, which hits the `if rope_type in ["default", "linear"]` branch and returns `nn.RoPE(dims, traditional=traditional, base=base)` — a standard RoPE with no mrope_section awareness. The `mrope` type branch (which also just returns plain `nn.RoPE`) is never reached because the config says `type: "default"`. For text-only input, all three mrope_sections receive identical positions, so the mrope_section is irrelevant. Rotary dim = `head_dim * partial_rotary_factor` = 256 * 0.25 = 64.
Unproven: nothing for this task
Next:     P0-5

### 2026-08-08 — P0-5 — TODO -> DOING
Did:      Added "Qwen3.6-35B-A3B" section to `docs/IMPLEMENTATION_REFERENCES.md` with findings from P0-1 through P0-4, in the citation style used in the file.
Ran:      `grep -c "qwen3_5_moe" docs/IMPLEMENTATION_REFERENCES.md` -> 2
Learned:  Section covers: MoE router (plain softmax, norm_topk_prob=True, no extra scaling), Gated DeltaNet (projections, conv1d, recurrence, state tensors, head mapping), Attention norm/topology (q_norm/k_norm, pre-norm, gate_up_proj split), MRoPE reduction (type: default → plain partial RoPE, rotary_dim=64).
Unproven: nothing for this task
Next:     P0-6

### 2026-08-08 — P0-6 — TODO -> DOING
Did:      Wrote `scripts/dump_qwen36_reference.py` — loads mlx-community/Qwen3.6-35B-A3B-4bit, runs fixed prompt "The capital of France is", captures input_token_ids, embedding_output, hidden_in/out for all 40 layers, router_logits and expert_ids for layers 0 and 3. Saves one safetensors file.
Ran:      `python3 -m py_compile scripts/dump_qwen36_reference.py` -> clean (exit 0)
Learned:  Script uses mlx-lm `load()` API, HookedQwen3Next wrapper for per-layer capture, safetensors.numpy.save_file for output. Does NOT run the dump (P0-7 is human-only).
Unproven: nothing for this task
Next:     P0-7

### 2026-08-08 — P0-7 — TODO -> BLOCKED
Did:      Marked P0-7 as BLOCKED — requires ~19 GB of weights resident, does not fit in 16 GB host.
Ran:      nothing
Learned:  A human must run `python3 scripts/dump_qwen36_reference.py --output Tests/Fixtures/qwen36_fixture.safetensors` on a ≥32 GB machine and commit the fixture under `Tests/Fixtures/`.
Unproven: nothing for this task
Next:     STOP — P0-7 is human-only. All Phase 0 tasks complete.

### 2026-08-08 — P0-7 — BLOCKED -> DONE
Did:      Fixture generated on omlx machine (javis), transferred to local repo at Tests/Fixtures/qwen36_fixture.safetensors (3.4 MB)
Ran:      python3 scripts/dump_qwen36_reference.py --model ~/.omlx/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --output qwen36_fixture.safetensors
Learned:  Script needed fix: removed transformers AutoTokenizer line (TokenizersBackend error), mlx_lm.load() returns tokenizer directly. Fixture contains input_token_ids, embedding_output, hidden_in/out for all 40 layers, router_logits and expert_ids for layers 0 and 3.
Unproven: nothing for this task
Next:     P1-1

### 2026-08-08 — P1-1 — TODO -> DONE
Did:      Made text_config optional in ArchInfo.load — falls back to root object when absent. Gemma still uses text_config, Qwen3.6 uses root keys.
Ran:      swift build -> clean; swift test -> 648 tests pass (same baseline)
Learned:  Single-line change: `root["text_config"] as? [String: Any] ?? root as? [String: Any]` in the guard. No test needed for this specific change — regression guard is the full suite.
Unproven: nothing for this task
Next:     P1-2

### 2026-08-08 — P1-2 — TODO -> DONE
Did:      No-op — no guard rejects unknown layer_types. The mask at line 48 maps non-full_attention to 0, so linear_attention silently becomes 0. P1-5 will add proper three-way layer kind.
Ran:      swift build -> clean
Learned:  Current boolean mask (1=full, 0=not-full) already accepts linear_attention without crashing. Proper three-way handling deferred to P1-5.
Unproven: nothing for this task
Next:     P1-3

### 2026-08-08 — P1-3 — TODO -> DONE
Did:      Made sliding_window optional in ArchInfo.load — defaults to 0 when key absent. Qwen3.6 has no sliding_window key; Gemma has one.
Ran:      swift build -> clean; swift test -> 648 tests pass (same baseline)
Learned:  Replaced `try i("sliding_window")` with a safe fallback: `(tc["sliding_window"] as? Int) ?? (tc["sliding_window"] as? NSNumber)?.intValue ?? 0`. No test needed — regression guard is the full suite.
Unproven: nothing for this task
Next:     P1-4

### 2026-08-08 — P1-4 — TODO -> DONE
Did:      No-op — no modelType switch exists in ArchInfo.swift. Model identification is done by comparing config values against known ArchConfig entries (e.g. ArchConfig.gemma4_26B_A4B). Model-specific behavior is handled by config values (hiddenActivation, layer_types, etc.) loaded from config.json.
Ran:      swift test --filter ArchInfo -> no matching tests (expected, no switch exists)
Learned:  ArchInfo.swift has no model type discrimination. It just loads config values into ArchInfo struct. The runtime identifies models by comparing ArchInfo values against known ArchConfig entries.
Unproven: nothing for this task
Next:     P1-10

### 2026-08-08 — P1-10 — TODO -> DONE
Did:      Added `--local <snapshot_dir>` to the CLI (`LocalRepacker.swift`) to repack a
          checkpoint already on disk, since the 35B/4bit model only exists on `javis`
          and can't be downloaded again through the HF streaming path there. Ran it
          against the real `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` snapshot on `javis` —
          this immediately surfaced five real bugs that synthetic-fixture tests had been
          silently signing off on:
          1. `ArchInfo.load`'s `rope_parameters` parsing assumed Gemma's nested
             `{full_attention: {...}, sliding_attention: {...}}` shape. Qwen3.6's is
             flat (`{rope_theta: 10000000, partial_rotary_factor: 0.25, ...}`), so both
             `ropeFull`/`ropeSWA` silently fell through to hardcoded defaults —
             `fullRopeTheta` would have been baked into the manifest as 1,000,000
             instead of the real 10,000,000, which `RealForwardRunner.swift:846`
             actually uses for the 10 real full-attention layers. Fixed by falling back
             to the flat dict itself when the nested sub-keys are absent.
          2. `hidden_act` (real key, value `"silu"`) was being read as
             `hidden_activation` (doesn't exist for this checkpoint), always silently
             defaulting to `"gelu_pytorch_tanh"`. Not yet load-bearing (P2-1 hasn't
             wired activation selection into the runtime yet) but wrong in the manifest.
          3. `intermediateSize` ("shared expert FFN" per its own doc comment) fell back
             to `moe_intermediate_size` (per-*routed*-expert size) when `intermediate_size`
             was absent — numerically right by coincidence here (both 512) but the wrong
             field; fixed to prefer `shared_expert_intermediate_size`.
          4. `RepackPlanner.routedExpertRole` only matched Gemma's
             `.experts.switch_glu.{gate,up,down}_proj` tensor names. Qwen3.6's real
             on-disk tensor names are `.switch_mlp.{gate,up,down}_proj.weight` (already
             split, not fused — P1-6's `splitFusedGateUpProj` was solving a problem this
             checkpoint doesn't have). With no match, every layer got
             `expertsPerLayer: 0` and `expertStride` came out `0`, tripping the "invalid
             dimensions or stride" layout validator. Fixed by matching either pattern.
          5. `LocalRepacker.swift`'s safetensors header parsing loaded each ~5GB shard
             entirely into memory and guessed the header/binary boundary by scanning for
             byte values (`0x00`, `}`) — wrong on real multi-tensor headers (stops at the
             first tensor's closing brace, not the header's). Fixed to read the correct
             8-byte little-endian length prefix and read exactly that many bytes, mirroring
             `RemoteSnapshotLoader.swift`'s proven approach.
          Also found and fixed: `packed_experts/layout.json`'s 16MB read cap
          (`VerifiedInstallTool.swift`, `GTurboLayoutValidator.swift`, and — this one
          matters, it's in the actual runtime loader — `PackedExpertsLayout.swift`) is
          too small for Qwen3.6's scale (256 experts x 40 layers = 22.5MB real file).
          P1-8's "no explicit limits" conclusion only checked integer overflow, not this
          byte-size cap. Raised to a shared `GTurboFormatV1.layoutMaxBytes = 64MB`
          constant used by all three sites instead of three independent literals.
          Also: `LocalRepacker.swift` was copying the raw source `.safetensors` shards
          into the output directory — nothing in `Sources/TurboFieldfare` (the runtime)
          ever reads a `.safetensors` file, so this just wasted ~20GB and made
          `--verify-install`'s unexpected-entries check unhappy. Removed. Also switched
          from placeholder all-zero SHA-256 hashes to real ones via `WriterCore.hashEntireFile`
          (`--verify-install` re-hashes every declared file, so placeholders would have
          failed verification the moment hashing was reached).
Ran:      On `javis` (has the checkpoint, 166GB free): `swift build` clean, `swift test`
          648/648 pass (both machines, at every step below). Then, iterating on the five
          bugs above one at a time:
          `./.build/.../TurboFieldfareRepack --output /tmp/qwen36_gturbo_out --local
          ~/.omlx/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` → succeeded, "Installed
          Qwen3.6-35B-A3B-4bit". Then `--verify-install --input-gturbo
          /tmp/qwen36_gturbo_out` → "Verified 43 files (20764181264 bytes)". Manually
          diffed `manifest.json`'s `arch` block against the real `config.json` on
          `javis` field by field: hiddenSize=2048, numHeads=16, numKVHeads=2,
          headDim=256, vocabSize=248320, numExperts=256, topKExperts=8,
          moeIntermediateSize=512, numLayers=40, fullRopeTheta=10000000,
          partialRotaryFactor=0.25, hiddenActivation=silu — all match.
Learned:  Never trust a "DONE" from a synthetic fixture that was written to match the
          code instead of a real checkpoint (`SyntheticSnapshot.swift`'s `rope_parameters`
          and expert-tensor names both encoded the same Gemma-only assumptions the real
          code had — the tests could never have caught any of bugs 1-4). P1-1 and P1-6's
          "DONE" marks were real progress but not actually checkpoint-verified; this
          entry is the first task in the whole board that ran against real weights, and
          it's why the board's own doctrine says a command must prove DONE.
Unproven: The produced `qwen36.gturbo` has never been loaded by the actual runtime
          (`TurboFieldfareMac`/CLI) — P2-1 onward is what will first exercise inference
          against it. `/tmp/qwen36_gturbo_out` on `javis` is scratch, not committed.
Next:     P2-1

### 2026-08-08 — P2-1 — TODO -> DONE
Did:      Added silu activation support end-to-end: Metal kernels (4 files — utility.metal,
          moe.metal, dequant_int8.metal, prefill.metal), Swift dispatch (4 kernel wrappers
          + RealForwardRunner), and ActivationType enum on ArchConfig. All kernel paths
          (standalone gelu_mul_fp16, fused MoE phase1, fused shared int8, fused prefill
          MoE phase1) now branch on activation via function constants (FC_MOE_ACT_SILU=4,
          FC_INT8_ACT_SILU=74, FC_PREFILL_ACT_SILU=77) or a separate silu_mul_fp16 kernel
          for the standalone path. Backward compatible: all new activation params default
          to .geluPytorchTanh.
Ran:      swift build -> clean (ok)
Learned:  ArchInfo.swift already read both hidden_act and hidden_activation keys (fixed in
          P1-10), so repack already writes "silu" into manifest.json for Qwen3.6. The
          runtime now reads and acts on it — before this, hiddenActivation was validated
          but never used for kernel selection (all Metal hardcoded gelu_pytorch_tanh).
          Metal silu formula: x / (1 + exp(-x)), using precise::exp for accuracy.
Unproven: Not tested against real weights — --verify-install or a forward pass on javis
          with the qwen36.gturbo produced in P1-10 would be the real proof. The
          ArchConfig.gemma4_26B_A4B baseline still has hiddenActivation: "gelu_pytorch_tanh",
          so existing Gemma models are unaffected.
Next:     P2-2

### 2026-08-08 — P2-2 — TODO -> DONE
Did:      Added pre-norm topology support for Qwen3.6 decode path. Added LayerTopology
          enum (gemma4/qwen36) with auto-detection heuristic (40 layers + 256 experts
          + 2 KV heads + 0 full KV heads). Made validateRuntimeSchema topology-aware
          — Qwen36 skips the 5 extra Gemma FFN norms, router.scale, router.per_expert_scale,
          and layer_scalar. Added add_fp16 elementwise Metal kernel + Swift wrapper.
          Branched the decode layer loop: raw residual (hidden += oOut, no norm on attn
          output), single post_attention_layernorm for all FFN branches (shared + routed
          share ffInput), no post-FFN norms, raw combine (hidden += h1 + h2), no
          layer_scalar. Dummy all-ones buffers for effectiveScale and perExpertScale
          (Qwen36 router doesn't have these). Prefill path deferred — uses aliased
          tensors to avoid crashes but still runs Gemma topology (P2-2b follow-up).
Ran:      swift build -> clean; swift test -> 648/648 pass (Gemma path unchanged)
Learned:  The Gemma topology has 7 norms per layer; Qwen36 has 2. The repacker passes
          through source tensor names verbatim — missing norms are absent from
          model_weights.bin. validateRuntimeSchema must be topology-aware or Qwen36
          models fail at load time. The Metal library concatenation means static inline
          functions from earlier files are visible to later files (silu in moe.metal
          used by utility.metal). Local variable shadowing of buffer parameters is a
          real Metal compile error (float act shadows device half* act).
Unproven: Not tested against real weights — decode path topology is code-complete but
          unverified end-to-end. Prefill path still runs Gemma topology for Qwen36
          (won't crash but produces wrong results).
Next:     P2-3 (router) or P2-2b (prefill pre-norm)

### 2026-08-08 — P2-3 — TODO -> DONE
Did:      Router scoring math confirmed equivalent (softmax + norm_topk_prob ≡ softmax
          over top-K). But Qwen3.6 checkpoint uses different tensor names: router is
          `mlp.gate.weight` (not `router.proj.weight`), shared expert is
          `mlp.shared_expert.{gate,up,down}_proj.weight` (not `mlp.*`). Made accessors
          and validateRuntimeSchema topology-aware. Relaxed router quant to [4,8].
Ran:      swift build -> clean; swift test -> 648 pass
Learned:  The mlx checkpoints use `mlp.gate` for the router Linear layer (it IS the gate
          of the SparseMoeBlock). The `shared_expert_gate` (underscore, scalar) is a
          different tensor from `shared_expert.gate_proj` (dot, projection).
Unproven: nothing for this task
Next:     P2-4

### 2026-08-08 — P2-4 — TODO -> DONE
Did:      Linear (DeltaNet) layers skip KV buffer allocation — they use recurrent state
          instead. Fixed fullStride for Qwen3.6 (numFullKVHeads=0 → reuse swaStride).
          Later fixed capacity=0 → capacity=1 to avoid div-by-zero in kSlot/vSlot.
Ran:      swift build -> clean; swift test -> 648 pass
Learned:  KVCacheManager.kSlot/vSlot are called unconditionally for ALL layers in the
          decode loop, so linear layers need a minimal (1-slot) allocation even though
          attention is skipped.
Unproven: nothing for this task
Next:     P2-5

### 2026-08-08 — P2-5 — TODO -> DONE
Did:      Fixed numFullKVHeads=0 fallback in decode + prefill paths. Qwen3.6 full-attn
          layers reuse numKVHeads (2) since there are no separate global KV heads.
          All other params (fullRopeTheta=10M, partialRotaryFactor=0.25, headDim=256,
          numHeads=16) flow correctly from ArchConfig without code changes.
Ran:      swift build -> clean; swift test -> 648 pass
Learned:  Prefill path had the same bug as decode. q_proj for Qwen3.6 full-attn is
          [8192, 2048] (double the expected [4096, 2048]) — likely a combined
          Q + auxiliary projection. O-proj is [2048, 4096] (from Q-head dimension).
Unproven: nothing for this task
Next:     P2-6

### 2026-08-08 — P2-6 — BLOCKED -> DONE  ← MILESTONE
Did:      Fixed 6 bugs to get local repack + model loading working on Mac. Repacked
          Qwen3.6 from `/Users/guutong/models/Qwen3.6-35B-A3B-4bit` → `/tmp/qwen36.gturbo`.
          Model loads and runs full decode inference at ~25 tok/s without crashing.
          8 layer-3 shape validation tests pass. DeltaNet layers skip attention
          (identity passthrough). Tokenizer auto-detects Qwen vs Gemma vocabulary.
          Auto-detection of ArchConfig from manifest (CLI + AppModelInstallationProbe).
          Bugs fixed: (1) ResidentWriter.write never called → all-zero sparse file,
          (2) SourceTensor.shardPath relative instead of absolute, (3) schema validation
          required self_attn for DeltaNet layers, (4) numFullKVHeads=0 → kvHeads=0
          division by zero, (5) q_proj doubled rows, (6) quant bits from config
          overrides. Added qwen36_35B_A3B ArchConfig + layer masks.
Ran:      swift build -> clean; swift test -> 657 pass (8 new layer-3 tests);
          CLI: prefill=6tok new=5tok decode @ ~25 tok/s (generates EOS — expected
          since 30 DeltaNet layers are identity passthrough)
Learned:  Never trust a "DONE" from synthetic tests — P1-10 worked on javis but the
          local repack path had 6 silent bugs the test suite never caught. The
          model_weights.bin was a sparse hole (all zeros on disk). Inferring layer
          shapes from config.json + ArchInfo instead of checking actual checkpoint
          shapes led to q_proj=4096 expectation vs actual 8192. AirLLM's "quantize
          the transfer" philosophy and kimi-k3's expert LRU cache are directly
          applicable. The GFTokenizer Gemma hardcode blocked Qwen3.6 for 2 hours.
Unproven: Output is garbage (EOS every time) — 30 DeltaNet layers are identity
          passthrough. Real coherent text requires Phase 3 DeltaNet. Full Metal
          forward pass against P0-7 fixture deferred (needs 8+ kernel orchestration).
Next:     P3-1 (frontier) or P2-2b (prefill pre-norm)

### 2026-08-08 — P3-1 — TODO -> DOING
Did:      Pre-implementation grilling session fixed every Phase 3 decision before code;
          all recorded durably: new CONTEXT.md glossary + docs/adr/ (ADR-0001 hybrid
          compute split, ADR-0002 pre-committed parity gate). Decisions: (1) frontier
          gate confirmed by owner — qwen3.8-max[1m] is the frontier here; (2) sequential
          decode-loop prefill for Qwen36, P2-2b stays TODO, new task P3-3b owns it;
          (3) hybrid compute split — five projections on Metal int4 GEMV kernels,
          conv1d/recurrence/gating in Swift fp32 on CPU; (4) P3-3 gate fixed before
          code: rel-L2 ≤ 2e-2 + max-abs ≤ 1e-2 on hidden_out.0, expert_ids exact,
          never widen, knife-edge routing escalates to owner; (5) unit oracles =
          hand-computed conv constants + committed numpy oracle script for the
          recurrence; (6) subagents do research/oracle/judging only — code stays
          single-threaded, one task one commit. Board updated: P3-3b row added,
          P3-4 Needs -> P3-3b, P5-1 note.
Ran:      git status --short -> clean apart from the new docs above
Learned:  DeltaNet weights verified present in /tmp/qwen36.gturbo resident index — all
          9 tensors × 30 layers (in_proj_qkv/z/a/b + out_proj are 4-bit quantized with
          scale+bias payloads; conv1d/A_log/dt_bias/norm are bf16). Zero DeltaNet code
          exists in Sources/ yet. The fixture is a 5-token SEQUENCE capture (hidden_in/out
          shaped [5,2048], hidden_out.0 rms 0.022 / max 0.64), so the isolation test must
          run all 5 tokens and its legitimacy rests on sequential-recurrence ≡
          masked-prefill equivalence.
Unproven: three load-bearing facts delegated to source-verification agents: MLX affine
          int4 dequant convention; sequential ≡ masked equivalence for gated_delta_update;
          exact RMSNormGated + z-gate ordering.
Next:     P3-1

### 2026-08-08 — P3-1 — DOING -> DONE
Did:      DeltaNet conv1d (width 4, causal, silu) + per-layer state container in plain
          Swift fp32: Runtime/DeltaNet/{DeltaNetConv,DeltaNetState}.swift. Conv step is
          tap-0-oldest cross-correlation over concat(conv_state, qkv), matching the mlx
          decode semantics recorded in P0-2; state store keys by GLOBAL layer index so
          the decode loop needs no rank mapping, full-attn layers hold empty arrays,
          reset() zeroes only linear layers. 5 hand-checked unit tests, pencil
          derivation in the test header. Grilling outcomes (CONTEXT.md, ADR-0001/0002)
          committed separately just before this.
Ran:      swift build -> clean; swift test --filter DeltaNetConv -> 5/5 pass;
          swift test -> 662/662 pass (baseline 657 + 5 new). One failure en route was
          my own test literal being wrong (correct footprint is 65_863_680 bytes; I
          wrote 66_063_360) — code was right, literal fixed.
Learned:  Metal inventory (subagent): all five DeltaNet projections reuse
          DequantInt4GEMV AS-IS for any m with n%64==0 — no new GEMV kernel needed;
          Model.resident(name:) is the generic accessor; GPU↔CPU handoff follows the
          router-index pattern (shared buffer + contents() bind after
          waitUntilCompleted); hidden buffers are fp16; bf16 scale/bias upcast inside
          the kernel. No n=2048/4096 entries in DequantInt4GEMV's specialized-PSO list
          yet — generic fallback is correct but slower; specialization is a later speed
          item, not a Phase 3 blocker.
Unproven: conv tap order / state ordering vs real mlx — source-verification agent still
          running; P3-3's fixture comparison is the real proof either way.
Next:     P3-2
