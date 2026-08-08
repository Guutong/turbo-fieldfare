# Pre-committed P3-3 parity gate and independent oracle strategy

The layer-0 isolation test (P3-3) compares our full layer output against the fixture's
`hidden_out.0`. Its gate was fixed **before any DeltaNet code exists**, so it cannot be
quietly widened to make a failing test pass:

1. Relative L2 error `‖ours − ref‖₂ / ‖ref‖₂ ≤ 2e-2` on `hidden_out.0` (5×2048).
2. Max absolute element error ≤ 1e-2 (one badly-wrong element may not hide in L2).
3. `expert_ids.0` must match exactly (all 5 tokens × 8 selections) — routing is
   integer-correct or the math upstream of the router is wrong. A single knife-edge
   token whose 8th/9th expert boundary sits inside bf16 noise is escalated to the
   project owner, never silently relaxed.

The 2e-2 budget is the reference side's own noise floor: mlx rounds ~8 intermediate
values to bf16 through layer 0 (~0.2–0.4% each); our fp16-GEMV + fp32-recurrence path
is in places more precise. If the gate fails, the task is marked FAILED/BLOCKED with
the measured divergence pasted into PHASE-LOG — or the gate itself is renegotiated
with the owner, never edited unilaterally.

Unit-test oracles follow the same independence principle: conv1d tests use
hand-computed constants with the derivation in comments (pencil-checkable at tiny
dimensions); the recurrence tests use constants printed by `scripts/deltanet_oracle.py`
— a committed plain-numpy transcription of the delta-rule equations that imports
nothing from mlx. Real-dimension tests check shape/state plumbing only; the numeric
oracle at real scale is the fixture itself.

Decision made 2026-08-08 with the project owner, before implementation.
