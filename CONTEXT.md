# TurboFieldfare

Out-of-core inference for Mixture-of-Experts language models on Apple Silicon. The
defining constraint is that the model does not fit in RAM: a small set of weights stays
resident and the rest is read from SSD as each token needs it.

## Language

### Memory model

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

### Model shape

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

### Pipeline

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
