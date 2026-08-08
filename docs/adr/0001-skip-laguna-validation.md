# Skip Laguna validation and bring up Qwen3.6 directly

Status: accepted (2026-08-07)

The `generalize-arch-for-moe` branch added pre-norm topology, sigmoid routing, silu, and
group-128 quantization in order to run Laguna-S-2.1, which reached end-to-end generation
but emits fluent-sounding incoherent text — a defect never diagnosed. Rather than fix
Laguna first, we are leaving it broken and bringing up Qwen3.6-35B-A3B directly, because
Qwen3.6 is the model we actually want and doing both means paying for two bring-ups.

## Considered options

Fixing Laguna first was the safer sequence: it exercises the pre-norm, sigmoid-router and
group-128 paths that Qwen3.6 also runs, so a bug there is a bug we inherit. We rejected it
because Laguna is a 117B model we have no other use for, and because its debugging was
stalled precisely for the reason recorded below.

## Consequences

Those three code paths reach Qwen3.6 unvalidated. The mitigation is not optimism — it is
that Laguna was being debugged by reading its output, which cannot localize a fault, while
Qwen3.6 has something Laguna never had: a **usable oracle**. mlx-lm implements this
architecture and can run the identical 4-bit weights, so bring-up diffs activations layer
by layer and injects reference activations to test a layer in isolation.

If Qwen3.6 also produces fluent-incoherent output, suspect the inherited paths first —
router scoring above all, since `scoring_func` and `norm_topk_prob` are absent from the
config and fall back to modeling-code defaults. Wrong routing produces exactly this
symptom in both models.

The oracle cannot run on the 16 GB development machine; ~19 GB of weights require dumping
reference activations on a larger host.
