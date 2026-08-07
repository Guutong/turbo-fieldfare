# Implementation references

TurboFieldfare builds on work by other developers and researchers. These are
the sources that materially shaped its Gemma 4 implementation, Metal kernels,
out-of-core runtime, and experiments under the 8 GB memory constraint.

Upstream code references whose exact behavior mattered are pinned to commits
checked on 2026-07-16. Project home pages remain branch-level when they are
included for broader design context rather than a line-level claim.

## Model and weights

- The official [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
  defines the model family, the 26B A4B mixture-of-experts shape, hybrid
  attention, and intended capabilities.
- Hugging Face Transformers provides executable decoder cross-checks for
  [Gemma 4 math](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/models/gemma4/modular_gemma4.py),
  [configuration](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/models/gemma4/configuration_gemma4.py),
  and [RoPE utilities](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/modeling_rope_utils.py).
  TurboFieldfare used them to verify normalization, K/V derivation, routing,
  MoE combination, layer scaling, and final logits.
- [MLX-LM Gemma 4](https://github.com/ml-explore/mlx-lm/blob/15b522f593b7ca5fbc0cac6f7572d40859d2d8fe/mlx_lm/models/gemma4_text.py)
  and [MLX-VLM Gemma 4](https://github.com/Blaizzy/mlx-vlm/blob/84f43753380355c0455a2bafb291d4b7cbcf81d1/mlx_vlm/models/gemma4/language.py)
  supplied independent implementation checks. MLX-LM also served as the
  bounded logit and quality reference.
- [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit),
  pinned at
  [`0d77464e`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/tree/0d77464eeb233a2da68ebf9d7dc4edaac7db956d),
  is the source of the weights, configuration, tokenizer, and chat-template
  sidecars. The repacker preserves its group-64 MLX affine values rather than
  requantizing them.
- [Hugging Face swift-transformers](https://github.com/huggingface/swift-transformers)
  is the direct tokenizer dependency. TurboFieldfare adds bounded streaming
  detokenization around it.

## Metal and kernels

- Pinned [MLX Metal kernels](https://github.com/ml-explore/mlx/tree/4367c73b60541ddd5a266ce4644fd93d20223b6e/mlx/backend/metal/kernels)
  were the main reference for quantized QMV/QMM, RMSNorm, RoPE, and attention
  geometry. The tagged [v0.32.0 vector SDPA](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/sdpa_vector.h)
  inspired the D512 one-pass attention variant.
- Pinned [llama.cpp/ggml Metal](https://github.com/ggml-org/llama.cpp/tree/79bba02a6741de194912d370015866414faa83ad/ggml/src/ggml-metal)
  informed row-SIMD quantized matvec, register-resident decode,
  capability-gated kernels, memory mappings, and resource hazards.
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) supplied transferable
  patterns for SIMD-per-row GEMV, packed loads, reductions, online softmax, and
  split-KV attention.

<a id="apple-metal"></a>
### Apple platform contracts

- Apple's [Metal Performance Primitives guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
  and [Metal tensor session](https://developer.apple.com/videos/play/wwdc2026/330/)
  define the platform contract for `MTLTensor`, cooperative inputs,
  `mpp::tensor_ops::matmul2d`, execution scopes, data types, and alignment.
  They guided the staged affine MPP prefill path.

### Apple shader operations

- Apple's [inline Metal 4 operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
  documents shader-local tensors and inline cooperative operations considered
  in the Metal 4 experiments.

## Out-of-core inference

- [`danveloper/flash-moe`](https://github.com/danveloper/flash-moe),
  [`Anemll/flash-moe`](https://github.com/Anemll/flash-moe), and
  [SwiftLM](https://github.com/SharpAI/SwiftLM) informed Apple-Silicon
  SSD-backed MoE, positional reads, reusable expert buffers, I/O workers, and
  GPU synchronization.
- Apple's [LLM in a Flash](https://arxiv.org/abs/2312.11514) framed the
  out-of-core problem around transferred bytes, useful read size, and flash
  scheduling.
- Apple's Darwin [`pread(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/pread.2.html),
  [`mmap(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html),
  and [`fcntl(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
  document concurrent offset reads, page alignment, and file-advice APIs such
  as `F_RDADVISE`.

## KV-cache and attention research

- [FlashAttention](https://arxiv.org/abs/2205.14135),
  [online normalizer calculation](https://arxiv.org/abs/1805.02867), and
  [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
  supplied the tiled-attention and associative online-softmax principles behind
  TurboFieldfare's split-KV kernels.
- Pinned [vLLM attention operators](https://github.com/vllm-project/vllm/tree/530852f9591a822ff4065908778a58fa015f0e69/vllm/v1/attention/ops)
  and [vLLM Metal kernels v2](https://github.com/vllm-project/vllm-metal/tree/11f1b453b74c60d113d67f9a5e7fda41500fd5b5/vllm_metal/metal/kernels_v2)
  were implementation references for transformed and quantized KV caches,
  fused dequantization, and online-softmax reduction.
- [TurboQuant](https://arxiv.org/abs/2504.19874) and
  [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) informed the rejected K4/V4
  KV-cache experiments.

## Qwen3.6-35B-A3B

Pinned [MLX-LM Qwen3.5/Qwen3-Next](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py),
[Qwen3-Next](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py),
[Gated DeltaNet](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py),
and [RoPE utilities](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py)
from `ml-explore/mlx-lm` supplied the bounded reference for the Qwen3.6 bringup.

### MoE router

`Qwen3NextSparseMoeBlock.__call__` in `qwen3_next.py` scores experts with **plain
softmax** (`mx.softmax(gates, axis=-1, precise=True)`), not sigmoid. Top-k indices
are selected via `mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]`. When
`norm_topk_prob` is `True` (the default in the Qwen3.5 `TextModelArgs` base class),
the selected scores are divided by their sum before being multiplied into the expert
outputs. No additional routed scaling factor exists beyond the softmax + norm_topk_prob.

### Gated DeltaNet

`Qwen3NextGatedDeltaNet` in `qwen3_next.py` with `gated_delta_update` in
`gated_delta.py`: projections are `in_proj_qkvz` (Linear[H, key_dim*2+value_dim*2]),
`in_proj_ba` (Linear[H, num_v_heads*2]), `conv1d` (Conv1d[conv_dim, kernel=4,
groups=conv_dim, padding=0]), `out_proj` (Linear[value_dim, H]), and `norm`
(RMSNormGated per head_v_dim). The causal conv1d sits after
`mx.concatenate([conv_state, mixed_qkv])` with width 4; output passes through
`nn.silu()`. Recurrence: `beta = mx.sigmoid(b)`, `g = mx.exp(-mx.exp(A_log) *
nn.softplus(a + dt_bias))`, sequential loop over sequence: `state = state * g`,
`kv_mem = (state * k).sum(-1)`, `delta = (v - kv_mem) * beta`, `state = state +
k * delta`, `y = (state * q).sum(-1)`. 16 key heads × 128 map to 32 value heads ×
128 via repeat_factor=2; `q` and `k` are repeated along the head dimension before
recurrence. Per-layer state: (1) `conv_state` — shape `[B, 3, conv_dim]`, last 3
steps of concatenated qkv; (2) `recurrent_state` — shape `[B, 32, 128, 128]`,
fp32 accumulated KV product.

### Attention norm and topology

`Qwen3NextAttention` in `qwen3_next.py` applies **q_norm/k_norm** via
`nn.RMSNorm(self.head_dim)` before RoPE: `queries = self.q_norm(queries).transpose(...)`
and `keys = self.k_norm(keys...).transpose(...)`. The decoder layer uses **pre-norm**
topology: `r = self.self_attn(self.input_layernorm(x), ...)` then `h = x + r`, then
`out = h + self.mlp(self.post_attention_layernorm(h))`. No post-norm at layer end.

The `sanitize()` method in `qwen3_5_moe.py` splits the fused `gate_up_proj`:
`mid = gate_up.shape[-2] // 2`, gate = `gate_up[..., :mid, :]` →
`switch_mlp.gate_proj.weight`, up = `gate_up[..., mid:, :]` →
`switch_mlp.up_proj.weight`, down = `gate_up.pop(down_proj)` →
`switch_mlp.down_proj.weight`.

### MRoPE reduction

`Qwen3NextAttention.__init__` calls `initialize_rope(int(head_dim *
partial_rotary_factor), ..., scaling_config=args.rope_scaling)` where
`args.rope_scaling` = `rope_parameters` dict = `{"type": "default",
"mrope_section": [11,11,10], ...}`. In `initialize_rope` (from `rope_utils.py`),
`rope_type = scaling_config.get("type")` = `"default"`, which hits the
`if rope_type in ["default", "linear"]` branch and returns
`nn.RoPE(dims, traditional=traditional, base=base)` — a standard RoPE with no
mrope_section awareness. The `mrope` type branch is never reached because the
config specifies `type: "default"`. For text-only input, all three mrope_sections
receive identical positions, making the mrope_section irrelevant. Rotary dim =
`head_dim * partial_rotary_factor` = 256 × 0.25 = 64.
