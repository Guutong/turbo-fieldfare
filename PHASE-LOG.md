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
| P0-1 | Router scoring, from mlx-lm source | TODO | never guess this |
| P0-2 | DeltaNet structure, from source | TODO | |
| P0-3 | Norm + QK details, from source | TODO | |
| P0-4 | MRoPE reduces to plain RoPE? | TODO | |
| P0-5 | Write findings to IMPLEMENTATION_REFERENCES.md | TODO | |
| P0-6 | Write scripts/dump_qwen36_reference.py | TODO | write only, do not run |
| P0-7 | Run the dump, commit fixture | TODO | ⚠️ needs ≥32GB host, not this Air |
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

### 2026-08-08 — P0-0 — TODO -> DOING
Did:      Created branch `qwen36-bringup` off `main`. Ran full test suite.
Ran:      `swift test 2>&1 | tail -30` -> `Test run with 648 tests in 122 suites passed after 39.381 seconds.`
Learned:  Baseline: 648 tests, 0 failures, 122 suites. All green on Gemma-only code.
Unproven: nothing for this task
Next:     P0-1
