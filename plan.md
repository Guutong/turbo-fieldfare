# Plan — Qwen3.6-35B-A3B on TurboFieldfare

## Goal

Make Qwen3.6-35B-A3B produce correct text from TurboFieldfare's CLI in ~2 GB of RAM,
by streaming MoE experts from SSD. Target machine: 16 GB M2 MacBook Air.

## How to use this file

1. Open `PHASE-LOG.md`. Find the first task whose Status is `TODO`.
2. Come back here and read **only that one task block**. Ignore every other task.
3. Do exactly what its `Do:` line says. Nothing else.
4. Run its `Verify:` commands. They decide the outcome — not your judgement.
5. Commit, then update `PHASE-LOG.md`.

One task per run. Never two. If a task feels too big, split it into new tasks and stop.

## Facts you must not re-derive

Weights: `mlx-community/Qwen3.6-35B-A3B-4bit` — group-64 affine 4-bit.

| Fact | Value |
|---|---|
| Layers | 40, hidden 2048, vocab 248320 |
| LM head | **Untied** (`tie_word_embeddings: false`) — a second full copy |
| Activation | `silu` — this repo currently has gelu only |
| Norm topology | **pre-norm** — this repo currently has sandwich only |
| Config shape | Keys at **root**. There is **no `text_config`** wrapper |
| `sliding_window` | **Key does not exist** |
| Layer kinds | `full_attention_interval: 4` → 10× [linear, linear, linear, full] |
| Full-attn layers | indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 |
| Other 30 layers | Gated DeltaNet (linear attention) |
| Full attention | 16 q heads, 2 kv heads, head_dim 256, rotary dim 64, rope_theta 10000000 |
| Gated DeltaNet | 16 key heads ×128, 32 value heads ×128, causal conv1d width 4 |
| MoE | all 40 layers, 256 experts, top-8, moe_inter 512, shared_inter 512 |
| Checkpoint quirk | `gate_up_proj` is **fused** — must be split when repacking |
| Vision tower | Present. Filter it out; this is text-only |

**Unknown — read from mlx-lm source, never guess:** router scoring (sigmoid vs softmax),
`norm_topk_prob`, routed scaling factor. `config.json` omits all three. A wrong router
produces fluent but incoherent text, which is very expensive to diagnose later.

Memory: ~1.45 GB resident before the expert cache. Only the 10 full-attention layers
carry a KV cache (~20 KB/token). DeltaNet state is ~63 MB and does not grow with context.
Cap context at 8–16K; the native 262K would need 5.4 GB of KV alone.

## Rules that apply to every task

- **One task = one commit.** This is what makes crash recovery work.
- Never weaken, skip, or delete a test to make something pass.
- Never widen a numeric tolerance to make a parity check pass. Mark `BLOCKED` instead.
- **Gemma 4 is the regression guard.** Its tests must still pass after every task. If
  they break, you changed shared behaviour instead of adding a branch for the new model.
- Out of scope entirely: OpenAI server, Mac app, tool-calling, chat template.
- If a `Verify:` command fails and you cannot fix it in this run, mark the task `FAILED`
  with the real output pasted in. Do not mark it `DONE`.

## Success

The mission is over when all three hold:

- **A.** All 40 layers match the reference fixture within tolerance, via a committed test.
- **B.** `swift test` shows no new failures versus the P0-0 baseline.
- **C.** Greedy decoding (`--temperature 0`, mandatory) answers all three correctly:
  `"The capital of France is"` → Paris · `"2 + 2 ="` → 4 ·
  `"The largest planet in our solar system is"` → Jupiter

---

# Phase 0 — Reference oracle

No Swift changes in this phase.

### P0-0 — Baseline
Needs: nothing · Runner: local-ok
Do: Create branch `qwen36-bringup` off `main`. Run the test suite and record the exact
pass/fail counts in the log — every later task compares against this number.
Verify: `git branch --show-current` prints `qwen36-bringup`; `swift test 2>&1 | tail -20`
Done when: the branch exists and the counts are written into `PHASE-LOG.md`.

### P0-1 — Router scoring, from source
Needs: P0-0 · Runner: local-ok
Do: Fetch `mlx_lm/models/qwen3_5_moe.py` from ml-explore/mlx-lm. Find how the MoE router
scores experts: sigmoid or softmax, whether `norm_topk_prob` is applied, and any routed
scaling factor. Write the answers with file+line citations into the log.
Verify: quote the actual lines of code in your log entry.
Done when: all three answered with citations.
If you cannot find it: mark `BLOCKED`. Do not guess, and do not copy the answer from a
different Qwen model.

### P0-2 — DeltaNet structure, from source
Needs: P0-1 · Runner: local-ok
Do: From the same file, write down: the exact list of projections, where the causal
conv1d sits and its width, the recurrence itself, the gating terms, how 16 key heads map
onto 32 value heads, and exactly which tensors make up the per-layer state.
Verify: quote the code in your log entry.
Done when: all six answered with citations.

### P0-3 — Norm and QK details, from source
Needs: P0-2 · Runner: local-ok
Do: From the same file, answer: do attention layers apply q_norm/k_norm to heads? Is the
topology pre-norm or sandwich? How does `sanitize()` split the fused `gate_up_proj`?
Verify: quote the code.
Done when: all three answered.

### P0-4 — MRoPE reduction
Needs: P0-3 · Runner: local-ok
Do: Confirm that for text-only input (no image or video tokens) the interleaved MRoPE
with `mrope_section [11,11,10]` reduces to plain partial RoPE with rotary dim 64. All
three sections should receive the same position for text.
Verify: quote the mlx-lm code path that shows it.
Done when: confirmed, or `BLOCKED` if it does not reduce — that would change Phase 2.

### P0-5 — Write it all down
Needs: P0-4 · Runner: local-ok
Do: Add a "Qwen3.6-35B-A3B" section to `docs/IMPLEMENTATION_REFERENCES.md` collecting
the findings from P0-1 through P0-4, in the citation style already used in that file.
Verify: `grep -c "qwen3_5_moe" docs/IMPLEMENTATION_REFERENCES.md` returns > 0
Done when: the section exists and every claim carries a citation.

### P0-6 — Dump script
Needs: P0-5 · Runner: local-ok
Do: Write `scripts/dump_qwen36_reference.py`. It loads `mlx-community/Qwen3.6-35B-A3B-4bit`
and writes one safetensors file containing: input token ids, embedding output, the hidden
state at **both the input and output of all 40 layers**, and router logits plus selected
expert ids for layers 0 and 3. Fixed prompt, greedy, deterministic.
Verify: `python3 -m py_compile scripts/dump_qwen36_reference.py`
Done when: it compiles. **Do not run it here** — see P0-7.

### P0-7 — Run the dump  ⚠️ NEEDS A HUMAN
Needs: P0-6 · Runner: human
Do: Nothing yourself. This needs ~19 GB of weights resident, which does not fit in this
machine's 16 GB. Mark this task `BLOCKED`, state that it must run on a host with ≥32 GB,
and stop the run.
Done when: a human has run it and committed the fixture under `Tests/Fixtures/`.

---

# Phase 1 — Repack path

Goal: turn the checkpoint into a `.gturbo` on disk. No inference yet.

### P1-1 — Root-level config
Needs: P0-7 · Runner: local-ok
File: `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift` (~line 32)
Do: It currently requires a `text_config` wrapper and throws without one. Make it fall
back to the root object when `text_config` is absent. Gemma must keep working.
Verify: `swift build` succeeds; `swift test --filter ArchInfo` passes.
Done when: both pass.

### P1-2 — Accept linear_attention
Needs: P1-1 · Runner: local-ok
File: same file, the guard rejecting unknown `layer_types`
Do: Allow `linear_attention` alongside `full_attention` and `sliding_attention`.
Verify: `swift build`; `swift test --filter ArchInfo`
Done when: both pass and a config containing `linear_attention` no longer throws.

### P1-3 — sliding_window optional
Needs: P1-2 · Runner: local-ok
File: same file, `try i("sliding_window")`
Do: Make it optional with a safe default. This config has no such key.
Verify: `swift build`; `swift test --filter ArchInfo`
Done when: both pass.

### P1-4 — Register the model type
Needs: P1-3, P0-1 · Runner: local-ok
File: same file, the `switch modelType` table
Do: Add a `qwen3_5_moe` case setting pre-norm topology, silu activation, and the router
scoring you established in P0-1. Do not change the `default` branch — Gemma depends on it.
Verify: `swift build`; `swift test --filter ArchInfo`
Done when: both pass.

### P1-5 — Three-way layer kind
Needs: P1-4 · Runner: local-ok
Files: `Sources/TurboFieldfareFormat/GTurboManifestV1.swift`, and the manifest reader
(find it: `grep -rln "fullAttentionLayerMask" Sources/`)
Do: The layer mask is a boolean (full vs sliding). Qwen3.6 needs three kinds. Add this as
a **new additive field** so an existing Gemma repack still produces a byte-identical
manifest. Do not repurpose the existing field.
Verify: `swift build`; `swift test --filter Manifest`
Done when: both pass and Gemma's manifest output is unchanged.

### P1-6 — Split fused gate_up_proj
Needs: P1-5 · Runner: local-ok
Files: `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift`,
`Sources/TurboFieldfareRepack/Core/Format/TensorMetadata.swift`
Do: The checkpoint stores gate and up as one fused tensor; the repacker expects two.
Split it during planning. See P0-3 for how mlx-lm does it.
Verify: `swift build`; `swift test --filter Repack`
Done when: both pass.

### P1-7 — Filter vision tensors
Needs: P1-6 · Runner: local-ok
File: `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift`
Do: Skip any tensor whose name starts with `vision_tower` or `model.visual`.
Verify: `swift build`; `swift test --filter Repack`
Done when: both pass.

### P1-8 — Expert layout at scale
Needs: P1-7 · Runner: local-ok
Files: `grep -rln "PackedExperts" Sources/`
Do: This model has 256 experts × 40 layers = 10,240 expert entries, far more than Gemma.
Confirm the packed layout and its validator handle that count. Fix if not.
Verify: `swift build`; `swift test --filter PackedExperts`
Done when: both pass.

### P1-9 — Quant group size
Needs: P1-8 · Runner: local-ok
Do: Determine the quantization group size this checkpoint actually uses (expected: 64,
affine). Confirm the repacker and manifest validator accept it for every tensor slot.
Verify: `swift build`; `swift test`
Done when: no new failures versus the P0-0 baseline.

### P1-10 — Produce the model
Needs: P1-9 · Runner: local-ok
Do: Run the repacker against the checkpoint to produce `qwen36.gturbo`. Needs ~20 GB free.
Verify: the layout validator passes, and the manifest `arch` block matches `config.json`
field by field.
Done when: the file exists and validates.
If the download or disk fails: mark `BLOCKED` with the real error.

---

# Phase 2 — Full-attention layers

Goal: prove one attention layer against the fixture. Do not attempt all 40.

### P2-1 — silu activation
Needs: P1-10 · Runner: local-ok
Do: Add silu alongside the existing gelu, selected by the arch's activation field.
Verify: `swift build`; `swift test`
Done when: no new failures, and Gemma still uses gelu.

### P2-2 — Pre-norm topology
Needs: P2-1 · Runner: local-ok
Do: Add a pre-norm decoder path alongside the existing sandwich path, selected by the
arch's norm topology. **Both the decode path and the prefill path need it** — missing the
prefill half is a known trap that produces plausible but wrong output.
Verify: `swift build`; `swift test`
Done when: no new failures, and Gemma still uses sandwich.

### P2-3 — Router scoring variant
Needs: P2-2 · Runner: local-ok
Do: If P0-1 found the router is not plain softmax top-k, implement that variant, selected
by the arch field. If it *is* plain softmax, mark this `DONE` with a note saying so and
change nothing.
Verify: `swift build`; `swift test`
Done when: no new failures.

### P2-4 — KV only for full-attention layers
Needs: P2-3 · Runner: local-ok
File: `grep -rln "RealForwardRunner" Sources/`
Do: Allocate a KV cache only for the 10 full-attention layers. The other 30 must not get
one — allocating 40 wastes roughly 3× the KV budget.
Verify: `swift build`; `swift test`
Done when: no new failures.

### P2-5 — Full attention path
Needs: P2-4 · Runner: local-ok
Do: Wire the full-attention layer for this model: 16 q heads, 2 kv heads, head_dim 256,
rotary dim 64, rope_theta 10000000.
Verify: `swift build`; `swift test`
Done when: no new failures.

### P2-6 — Layer-3 isolation test  ← the first real proof
Needs: P2-5 · Runner: mixed
Do: Add a test that loads the P0-7 fixture, injects the recorded hidden state at the
**input** of layer 3, runs only that layer, and compares against the recorded **output**
of layer 3. Assert the selected expert ids match too.
Verify: `swift test --filter Layer3`
Done when: it passes within tolerance.
Do not: widen the tolerance to make it pass. If it fails, that is a real bug — log the
actual numeric divergence and mark `FAILED`.

---

# Phase 3 — Gated DeltaNet in Swift  ⚠️ FRONTIER MODEL ONLY

Plain Swift, fp32, sequential. Slow and obviously correct beats fast and subtly wrong.

**If you are a small or local model, mark the task `BLOCKED NEEDS-FRONTIER` and stop.**
Novel numerical kernels are the one thing here that fails silently, and wrong-but-plausible
Metal is the most expensive outcome available in this project.

### P3-1 — conv1d and state
Needs: P2-6 · Runner: frontier
Do: Implement the causal conv1d (width 4) and the per-layer state container in plain Swift.
Verify: `swift test --filter DeltaNetConv`
Done when: a unit test over a small hand-checked input passes.

### P3-2 — Delta rule and gating
Needs: P3-1 · Runner: frontier
Do: Implement the recurrence and gating from P0-2, including the 16 key heads → 32 value
heads mapping.
Verify: `swift test --filter DeltaNetRule`
Done when: passes.

### P3-3 — Layer-0 isolation test
Needs: P3-2 · Runner: frontier
Do: Same injection trick as P2-6, but on layer 0 (a DeltaNet layer).
Verify: `swift test --filter Layer0`
Done when: passes within tolerance.

### P3-4 — Full forward, coherent text  ← the milestone
Needs: P3-3 · Runner: frontier
Do: Run all 40 layers — Swift for the 30 linear layers, Metal for the rest. Slow is fine.
Verify: the three greedy prompts in Success criterion C.
Done when: all three answer correctly.

---

# Phase 4 — DeltaNet in Metal  ⚠️ FRONTIER MODEL ONLY

### P4-1 — Port the recurrence
Needs: P3-4 · Runner: frontier
Do: Move the recurrence to Metal with state resident in GPU buffers.
Verify: `swift test --filter DeltaNet`
Done when: passes.

### P4-2 — Diff against Swift
Needs: P4-1 · Runner: frontier
Do: Compare the Metal path against the Phase 3 Swift path layer by layer, same machine,
same weights. Any divergence is the kernel.
Verify: `swift test --filter DeltaNetParity`
Done when: all 40 layers agree and criterion C still passes.

---

# Phase 5 — Prefill

### P5-1 — Sequential prefill
Needs: P4-2 · Runner: mixed
Do: Prefill by looping the decode recurrence over prompt tokens. Do **not** attempt the
chunkwise parallel form yet — this sequential path becomes the oracle for it later.
Verify: `swift test`
Done when: no new failures.

### P5-2 — Multi-token prompts
Needs: P5-1 · Runner: mixed
Do: Confirm coherence on longer prompts, not just the three short ones.
Verify: criterion C plus one paragraph-length prompt.
Done when: output is coherent.

---

# Phase 6 — Memory and speed

### P6-1 — Expert cache hit rate
Needs: P5-2 · Runner: mixed
Do: Measure the hot-expert cache hit rate. 256 fine-grained experts reuse worse than
Gemma's fatter ones. Report the number before changing anything.
Verify: paste the measured hit rate into the log.
Done when: measured.

### P6-2 — Context cap
Needs: P6-1 · Runner: mixed
Do: Enforce a context cap of 8–16K, with a clear error past it.
Verify: `swift test`
Done when: no new failures.

### P6-3 — Final numbers
Needs: P6-2 · Runner: mixed
Do: Measure resident memory and decode speed.
Verify: ≤2 GB resident at 8–16K context, ≥4 tok/s.
Done when: both met — and then the mission is complete.

---

# Phase 6b — Speed (inspired by kimi-k3-in-c)

Ideas sourced from `FareedKhan-dev/kimi-k3-in-c` — an existence proof that streaming-MoE
on one machine can hit usable tokens-per-second with careful I/O engineering.

### P6b-1 — Expert LRU cache with pinning
Needs: P6-1 · Runner: mixed
Do: Add an LRU cache for recently-used routed experts. Experts that are selected again
within a short window stay resident and skip disk I/O entirely. Pinning prevents the
expert currently being computed from being evicted mid-encode. Track hit rate and
eviction count per layer.
Why: kimi-k3's `k3_cache.c` does exactly this — LRU with pinning + INFLIGHT slot state.
Repeated expert selections (common for consecutive tokens on similar topics) save
~17.55 MB per expert pread.
Verify: `swift test --filter ExpertCache`; re-measure hit rate vs P6-1 baseline.

### P6b-2 — Batch expert prefetch in disk-offset order
Needs: P6b-1 · Runner: mixed
Do: When the router selects top-K experts, issue a single batched pread for all misses
sorted by disk offset (seek minimization). kimi-k3's `getmany` uses 3-phase acquire:
serial reserve slots → parallel preads sorted by offset → serial publish. Queue depth 16.
Why: Sorting preads by disk offset cuts rotational/seek latency vs issuing them in
expert-index order. kimi-k3 measures this as the difference between usable and
unusable streaming throughput.
Verify: benchmark showing reduced I/O wall time per token.

### P6b-3 — Prefill expert dedup
Needs: P5-1 · Runner: mixed
Do: During prefill, route all tokens in a chunk → collect unique expert IDs → fetch
each expert once → reuse across all tokens that need it. kimi-k3 does this in
`k3_moe_prefill` with a 64-token chunk, getting 3–4× less expert I/O.
Verify: prefill I/O bytes reduced by ≥2× vs per-token fetch on a 64-token prompt.

### P6b-4 — Speculative decoding
Needs: P5-2 · Runner: frontier
Do: Implement n-gram speculative decoding: find repeated token sequences in prior
context, draft up to K tokens, verify in one batched forward pass. kimi-k3 uses
evidence-gated longest-suffix matching (length ≥4→3), verified greedily in one sweep.
Output is byte-identical to serial decode by construction. Measured: +22% cost per
extra verified token. Target: ≥1.5 tokens per decode step.
Verify: output matches serial decode exactly; measured acceptance rate ≥50%.
