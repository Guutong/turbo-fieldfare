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
| P0-7 | Run the dump, commit fixture | BLOCKED | ⚠️ needs ≥32GB host, not this Air |
| P1-1 | ArchInfo: root-level config | TODO | |
| P1-2 | ArchInfo: accept linear_attention | TODO | |
| P1-3 | ArchInfo: sliding_window optional | TODO | |
| P1-4 | ArchInfo: register qwen3_5_moe | TODO | needs P0-1 |
| P1-5 | Manifest: three-way layer kind | TODO | additive only |
| P1-6 | Repack: split fused gate_up_proj | TODO | |
| P1-7 | Repack: filter vision tensors | TODO | |
| P1-8 | Expert layout at 10,240 entries | TODO | |
| P1-9 | Quant group size accepted | TODO | |
| P1-10 | Produce qwen36.gturbo | TODO | needs ~20GB free |
| P2-1 | silu activation | TODO | |
| P2-2 | Pre-norm topology (decode AND prefill) | TODO | prefill half is a known trap |
| P2-3 | Router scoring variant | TODO | no-op if plain softmax |
| P2-4 | KV for the 10 full-attn layers only | TODO | |
| P2-5 | Full attention path | TODO | |
| P2-6 | Layer-3 isolation test | TODO | first real proof |
| P3-1 | DeltaNet conv1d + state (Swift) | TODO | ⚠️ frontier only |
| P3-2 | Delta rule + gating (Swift) | TODO | ⚠️ frontier only |
| P3-3 | Layer-0 isolation test | TODO | ⚠️ frontier only |
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

### 2026-08-08 — P0-1 — TODO -> DOING
Did:      Fetched `mlx_lm/models/qwen3_5_moe.py` and `mlx_lm/models/qwen3_next.py` from ml-explore/mlx-lm. Found router scoring in `Qwen3NextSparseMoeBlock.__call__`.
Ran:      web fetch of two source files
Learned:  Router uses **plain softmax** (`mx.softmax(gates, axis=-1, precise=True)`), NOT sigmoid. `norm_topk_prob` defaults to `True` in `qwen3_5.py` TextModelArgs (the Qwen3.6 model inherits from it), so top-k scores are divided by their sum. No additional routed scaling factor beyond the softmax+norm. `gate_up_proj` split: `mid = gate_up.shape[-2] // 2`, gate = `[..., :mid, :]`, up = `[..., mid:, :]`, mapped to `switch_mlp.gate_proj.weight` / `switch_mlp.up_proj.weight`.
Unproven: nothing for this task
Next:     P0-2
