# TurboFieldfare

A Swift/Metal inference engine that runs large MoE models in ~2 GB of RAM by keeping
only norms, projections, and embeddings resident and streaming routed-expert weights
from SSD at decode time. This glossary covers the bring-up domain shared by the Gemma 4
and Qwen3.6 model families.

## Language

### Model format

**.gturbo**:
The installed-model directory format: `manifest.json`, `model_weights.bin` (resident
index + resident weights), `packed_experts/`, and `tokenizer/`.
_Avoid_: checkpoint, model file

**Repack**:
The one-way conversion of an MLX safetensors checkpoint into a `.gturbo` directory.
_Avoid_: convert, install (install includes verification)

**Resident weights**:
Non-expert tensors (embeddings, norms, projections, LM head) packed into
`model_weights.bin` and held in memory for the whole run.
_Avoid_: dense weights, static weights

**Resident index**:
The leading table of `model_weights.bin` mapping each resident tensor name to its
dtype, shape, and file offsets (weights, scales, biases).

**Packed experts**:
Routed MoE expert weights stored as per-expert blobs in `packed_experts/`, streamed
via `pread` at decode time and never resident as a group.
_Avoid_: expert shards

### Layer structure

**Topology**:
A decoder layer's norm-and-residual layout. *Sandwich* (Gemma 4): norms before and
after each sublayer, residual adds the normed output, scaled by `layer_scalar`.
*Pre-norm* (Qwen3.6): norms only before each sublayer, raw residuals.
_Avoid_: norm scheme, architecture

**Layer kind**:
One of the three per-layer attention structures recorded in `layerKindMask`:
full attention (1), sliding attention (0), linear attention / DeltaNet (2).

**DeltaNet layer**:
A Gated DeltaNet linear-attention layer. Summarizes history into fixed-size state
instead of a growing KV cache. Qwen3.6 has 30 of them; the 10 remaining layers are
full attention.
_Avoid_: linear layer (ambiguous with Linear projections), SSM layer, Mamba

**Recurrent state**:
The per-DeltaNet-layer fp32 matrix `[Hv, Dv, Dk]` (32×128×128 for Qwen3.6) that
replaces a KV cache. Updated by the delta rule every token; constant in size.
_Avoid_: hidden state (means the residual stream), SSM state

**Conv state**:
The per-DeltaNet-layer rolling buffer holding the last 3 pre-convolution qkv vectors,
so the width-4 causal conv1d can run token-by-token at decode time.

**Sequential prefill**:
Processing prompt tokens one at a time through the decode loop. Slow; the reference
behavior for correctness.
_Avoid_: serial prefill, token-loop prefill

**Chunked prefill**:
Batched prompt processing through dedicated prefill kernels (the Gemma path). Fast;
not yet correct for Qwen3.6 (task P2-2b).

### Verification

**Fixture**:
Committed reference activations dumped once from the real model
(`Tests/Fixtures/qwen36_fixture.safetensors`): token IDs, embedding output, per-layer
hidden in/out, router logits and expert IDs for layers 0 and 3. Ground truth for
parity tests; never regenerated casually.

**Oracle**:
A slower implementation trusted to be correct, against which a faster one is diffed.
The Phase 3 Swift DeltaNet path is the permanent oracle for the Phase 4 Metal port.

**Isolation test**:
Injecting a fixture's recorded hidden state at one layer's input, running only that
layer, and comparing against its recorded output.

**Parity gate**:
The pre-committed numeric tolerance an isolation test must meet. Fixed before
implementation; never widened to make a test pass.

**Frontier gate**:
The board rule that tasks marked `FRONTIER MODEL ONLY` must be executed by a frontier
model or marked `BLOCKED NEEDS-FRONTIER` — novel numerical kernels fail silently on
weaker models.
