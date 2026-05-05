# vLLM Memory Estimation Skill

## Purpose
Estimate the GPU VRAM footprint for Large Language Models served via vLLM, accounting for model weights, KV cache, speculative decoding, MoE architectures, LoRA adapters, beam search, and PagedAttention block rounding.

## When to Use
- User asks about GPU memory requirements for running an LLM with vLLM
- User needs to determine the optimal `gpu_memory_utilization` setting
- User wants to size hardware for a specific model and workload
- User is planning multi-model concurrency on a shared GPU
- User needs to understand why vLLM is running out of memory (OOM)

## Input Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `vram` | number | 128 | Total GPU VRAM in GB |
| `overhead` | number | 1.5 | Static overhead in GB (CUDA context, tokenizer, block tables, Python runtime) |
| `baseParams` | number | 27 | Base model parameters in billions |
| `weightBytes` | number | 1 | Bytes per weight: 2 (FP16/BF16), 1 (FP8), 0.5 (INT4) |
| `layers` | number | 64 | Number of transformer layers |
| `kvHeads` | number | 4 | Number of KV attention heads (from `num_key_value_heads` in model config) |
| `headDim` | number | 256 | Head dimension size (hidden_size / num_attention_heads) |
| `contextLen` | number | 262144 | Maximum context length in tokens |
| `maxSeqs` | number | 1 | Maximum concurrent sequences (vLLM `--max-num-seqs`) |
| `kvBytes` | number | 2 | Bytes per KV cache element: 2 (FP16/BF16), 1 (FP8) |
| `moeExperts` | number | 1 | Number of MoE experts (1 = dense model) |
| `loraParams` | number | 0 | LoRA adapter parameters in billions |
| `numBeams` | number | 1 | Beam search width (1 = greedy/sampling) |
| `blockSize` | number | 16 | PagedAttention block size in tokens (8, 16, or 32) |
| `enableSpecDec` | boolean | false | Enable speculative decoding with a draft model |
| `draftParams` | number | 3 | Draft model parameters in billions |
| `draftWeightBytes` | number | 1 | Bytes per draft weight: 2 (FP16/BF16), 1 (FP8), 0.5 (INT4) |
| `draftMoEExperts` | number | 1 | Number of MoE experts in draft model (1 = dense) |
| `nPredict` | number | 5 | Draft tokens per step (informational, does not affect VRAM) |

## Calculation Steps

Perform the following steps in order. Use exact floating-point arithmetic.

### Step 1: MoE Multiplier & Effective Parameters
```
moeMultiplier = 0.333 + (0.667 * moeExperts)
effectiveParams = (baseParams * moeMultiplier) + loraParams
weightsGB = effectiveParams * weightBytes
```
- ~33% of weights are in attention layers (shared across experts)
- ~67% of weights are in FFN layers (replicated per expert)

### Step 2: Draft Model Weights (if `enableSpecDec` is true)
```
draftMoeMultiplier = 0.333 + (0.667 * draftMoEExperts)
draftEffParams = draftParams * draftMoeMultiplier
draftWeightsGB = draftEffParams * draftWeightBytes
```
- If speculative decoding is disabled, `draftWeightsGB = 0`
- Draft KV cache is transient (discarded after verification) — not included

### Step 3: Effective Context (PagedAttention Block Rounding)
```
effectiveContext = ceil(contextLen / blockSize) * blockSize
```
- vLLM rounds context up to the nearest block boundary

### Step 4: Effective Sequences (Beam Search Multiplier)
```
effectiveSeqs = maxSeqs * numBeams
```

### Step 5: KV Cache
```
kvCacheBytes = 2 * layers * kvHeads * headDim * effectiveContext * effectiveSeqs * kvBytes
kvCacheGB = kvCacheBytes / 1024^3
```
- The `2` accounts for both Key and Value tensors

### Step 6: Totals
```
totalGB = weightsGB + draftWeightsGB + kvCacheGB + overhead
utilization = totalGB / vram
```

## Output Format

Return the following breakdown:

```
Memory Breakdown:
  Model Weights:        XX.XX GB  (effective params: XX.XX B)
  Draft Model Weights:  XX.XX GB  (effective params: XX.XX B)  [if enabled]
  KV Cache:             XX.XX GB  (effective seqs: X)
  Static Overhead:      XX.XX GB
  ─────────────────────────────────
  Total Required VRAM:  XX.XX GB
  VRAM Usage:           XX.X%

Target gpu_memory_utilization: X.XXX
  vLLM flag: --gpu-memory-utilization X.XXX
```

If `totalGB > vram`, indicate that the configuration exceeds available VRAM.

## Worked Example

**Input:** Qwen3.6-27B defaults (FP8, 64 layers, 4 KV heads, 256 head dim, 262K context, 128 GB VRAM)

| Variable | Value |
|---|---|
| vram | 128 |
| overhead | 1.5 |
| baseParams | 27 |
| weightBytes | 1 (FP8) |
| layers | 64 |
| kvHeads | 4 |
| headDim | 256 |
| contextLen | 262144 |
| maxSeqs | 1 |
| kvBytes | 2 (FP16) |
| moeExperts | 1 |
| loraParams | 0 |
| numBeams | 1 |
| blockSize | 16 |
| enableSpecDec | false |

**Step 1:** MoE multiplier = 0.333 + (0.667 × 1) = 1.0
Effective params = (27 × 1.0) + 0 = 27 B
Weights = 27 × 1 = **27.00 GB**

**Step 2:** Speculative decoding disabled → draft weights = **0.00 GB**

**Step 3:** Effective context = ceil(262144 / 16) × 16 = **262144** (exact multiple, no rounding)

**Step 4:** Effective seqs = 1 × 1 = **1**

**Step 5:** KV cache = (2 × 64 × 4 × 256 × 262144 × 1 × 2) / 1024³
= 4,294,967,296 / 1,073,741,824
= **4.00 GB**

**Step 6:** Total = 27.00 + 0.00 + 4.00 + 1.50 = **32.50 GB**
Utilization = 32.50 / 128 = **0.254**

**Output:**
```
Memory Breakdown:
  Model Weights:        27.00 GB  (effective params: 27.00 B)
  KV Cache:             4.00 GB  (effective seqs: 1)
  Static Overhead:      1.50 GB
  ─────────────────────────────────
  Total Required VRAM:  32.50 GB
  VRAM Usage:           25.4%

Target gpu_memory_utilization: 0.254
  vLLM flag: --gpu-memory-utilization 0.254
```

## Notes

- **Estimates are approximate.** Actual vLLM usage may vary slightly due to internal alignment, CUDA allocator fragmentation, and runtime overhead.
- **MTP heads are included in base parameters.** Multi-Token Prediction (MTP) architectures like Grok already include MTP head weights in their reported parameter count — no separate calculation is needed.
- **Draft KV cache is transient.** In speculative decoding, the draft model's KV cache is discarded after each verification step and does not contribute to static VRAM usage.
- **`n_predict` does not affect VRAM.** This parameter controls throughput (how many tokens the draft model proposes per step) but has no impact on static memory.
- **KV cache quantization** (FP8 KV cache) requires a compatible vLLM build and GPU hardware.
- **INT4 weight quantization** has a noticeable quality trade-off and may not be supported by all vLLM builds.
- For **multi-GPU / tensor parallelism**, the total VRAM requirement is distributed across GPUs. Divide `totalGB` by the number of GPUs for a per-GPU estimate (overhead and small structures may still be duplicated).
