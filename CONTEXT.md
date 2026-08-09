# TurboFieldfare

Out-of-core inference for Mixture-of-Experts language models on Apple Silicon. The
defining constraint is that the model does not fit in RAM: a small set of weights stays
resident and the rest is read from SSD as each token needs it. A Swift/Metal engine that
runs large MoE models in ~2 GB of RAM by keeping only norms, projections, and embeddings
resident and streaming routed-expert weights from SSD at decode time. This glossary
covers the bring-up domain shared across the Gemma 4, Qwen3.6, and Laguna model families.

## Language

Two overlapping vocabularies are in active use as the project generalizes beyond the
original Gemma 4 / Qwen3.6 scope to also cover the Laguna architecture. Both are kept
below rather than force-merged, since the terms don't map 1:1 and losing either would
lose information about how each work stream talks about the system.

### Model format (Qwen3.6 bring-up)

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

### Layer structure (Qwen3.6 bring-up)

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

### Verification (Qwen3.6 bring-up)

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

### Memory model (generalized MoE / Laguna)

**Resident core**:
The weights held in RAM for the whole session — embeddings, LM head, attention,
routers, shared experts, and norms. Everything that is not a routed expert.
_Avoid_: shared core, base model, hot weights

**Streamed expert**:
A routed expert's weights, read from SSD on demand when the router selects it. Never
guaranteed to be in RAM.
_Avoid_: offloaded expert, cold expert

**Hot-expert cache**:
The bounded pool of recently used streamed experts kept in RAM to avoid re-reading them.
Its size is what remains of the memory budget after the resident core.
_Avoid_: expert LRU, expert pool

**Memory budget**:
The target ceiling for total resident bytes — the number the project exists to hit.
Distinct from the model's on-disk size.

### Model shape (generalized MoE / Laguna)

**Layer kind**:
Which attention mechanism a decoder layer uses: `full`, `sliding`, or `linear`. A model
may interleave them. This is a three-way property; describing it as a boolean mask is
wrong for any model with linear layers.
_Avoid_: attention mask, full-attention mask (both imply two states)

**Gated DeltaNet layer**:
A layer whose attention is a linear-time recurrence over a fixed-size state rather than
softmax attention over a growing cache. The `linear` layer kind.
_Avoid_: linear attention layer (ambiguous — names the family, not the mechanism)

**Recurrent state**:
The fixed-size tensor a Gated DeltaNet layer carries between tokens. Its size does not
grow with context length. Not a cache: nothing is being avoided by keeping it, it *is*
the layer's memory.
_Avoid_: linear KV cache, DeltaNet cache

**KV cache**:
The per-token keys and values retained by a full-attention layer. Grows linearly with
context, and only layers of kind `full` have one.

**Expert granularity**:
How wide each routed expert is. Many narrow experts and few wide experts can have the
same parameter count but very different streaming behaviour, because granularity sets
the size of each SSD read and how often a cached expert is reused.

### Pipeline (generalized MoE / Laguna)

**Repack**:
Converting a published checkpoint into the project's on-disk format — regrouping tensors
so that a token's experts can be read as few contiguous ranges.
_Avoid_: convert, quantize (repacking preserves the source quantization; it does not
introduce it)

**`.gturbo`**:
The repacked on-disk model: a manifest, the resident core, and the packed expert store.

**Oracle**:
A trusted implementation of the same model, on the same weights, whose intermediate
activations define correct. Bring-up of a new architecture is diffing against the oracle
layer by layer, not reading the output and judging it.
_Avoid_: reference model, ground truth
