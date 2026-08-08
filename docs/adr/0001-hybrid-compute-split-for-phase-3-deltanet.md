# Hybrid compute split for the Phase 3 DeltaNet path

Phase 3 implements Gated DeltaNet to make Qwen3.6's 30 linear layers produce real
output. The plan's doctrine reads "plain Swift, fp32, sequential … Swift for the 30
linear layers, Metal for the rest," which literally implies CPU-side dequant + GEMV
for the five quantized projections too. We decided on **hybrid instead**: the five
projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) run on
Metal using the runtime's existing int4 GEMV kernels; only conv1d, the delta
recurrence, gating, and the RMSNormGated-z output gate run in Swift fp32 on CPU, with
GPU↔CPU transfers per layer per token.

The trade-off is attribution vs speed. Full-CPU would make any P3-3 parity failure
point at exactly one suspect (the new DeltaNet code), but costs ~1 s/token — minutes
per milestone verification run. Hybrid buys ~10× back; the cost is that P3-3 becomes
an orchestration test (Metal hand-offs are in the loop). We accepted that because the
Metal GEMV kernels already ran correctly against this checkpoint's 4-bit weights in
P2-6, so they are not an untested variable, and the recurrence itself — the genuinely
novel math — still gets independent hand-checked and scripted-oracle unit tests
(see ADR-0002).

**Considered options**: full-CPU naive Swift (rejected on speed; its clean-attribution
property is largely preserved by testing projections and recurrence separately);
full-CPU with Accelerate `cblas_sgemv` (the reopen lever if CPU ever becomes the
bottleneck again); hybrid (chosen). The Phase 4 Metal port still gets its oracle: the
CPU conv/recurrence/gating code stays as the diff target for P4-2.

Decision made 2026-08-08 with the project owner, before implementation, during the
Phase 3 grilling session.
