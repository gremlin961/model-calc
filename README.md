# Advanced vLLM Memory Estimator & Optimization Calculator

## Overview
This repository contains a lightweight, interactive, single-page web application designed to calculate the GPU VRAM footprint for Large Language Models (LLMs) served via [vLLM](https://github.com/vllm-project/vllm).

It specifically helps you determine the optimal `gpu_memory_utilization` setting to maximize context length and concurrency without encountering Out-Of-Memory (OOM) errors, or to carve out space for running multiple models concurrently.

## Features
* **Real-time Computation:** Automatically recalculates VRAM requirements as you adjust parameters.
* **Comprehensive Metrics:** Breaks down memory usage into Model Weights, KV Cache, and Static Overhead, with sub-metrics for effective parameters and effective concurrent sequences.
* **vLLM Specific:** Calculates the exact `gpu_memory_utilization` fraction required for your specific hardware configuration.
* **Quantization Support:** Adjust weight and KV cache precision (FP16/BF16, FP8, INT4) to see the immediate impact on memory.
* **MoE Support:** Enter the total loaded parameters (what's actually in VRAM). For example, Qwen3.6-35B-A3B → enter 35, Mixtral 8x7B → enter 46.7.
* **LoRA Support:** Adds LoRA adapter parameter overhead to the total weight footprint.
* **Speculative Decoding:** Adds draft model weight VRAM (including MoE scaling for draft model). The draft model's KV cache is transient and not included in static footprint.
* **Beam Search:** Multiplies KV cache by the number of beams.
* **PagedAttention Block Size:** Rounds up context length to the nearest block boundary (8, 16, or 32 tokens), reflecting vLLM's actual memory allocation.
* **Architecture Aware:** Factors in deep model configurations like Layers, KV Heads, and Head Dimension (crucial for models with Grouped Query Attention).

## Usage
This tool is a self-contained HTML file. No build steps or web servers are required.
1.  Download or copy the `index.html` file.
2.  Open it directly in any modern web browser.
3.  Input your total GPU VRAM and model parameters to get your optimization targets.

## The Math Behind It
The calculator uses the following formula to estimate vLLM's memory footprint:
`Total Memory = Model Weights + KV Cache Pool + Static Overhead`

### Weight Calculation
* **Base Parameters:** Enter the TOTAL loaded parameters (what's actually in VRAM). For MoE models, this is the total parameter count, not the active count.
* **Effective Params:** `base_params + lora_params`
* **Weights VRAM:** `effective_params * bytes_per_weight`

### Speculative Decoding (Draft Model)
* The draft model's weights load alongside the target model, adding permanent VRAM overhead.
* Draft model MoE multiplier uses the same formula as the target model.
* Draft model effective params: `draft_params * draft_moe_multiplier`
* Draft model weights VRAM: `draft_eff_params * draft_bytes_per_weight`
* Draft model KV cache is transient (verification tokens are discarded), so it's not included in the static footprint.
* `n_predict` (draft tokens per step) is informational — it affects throughput but not static VRAM.

### MTP (Multi-Token Prediction) Note
* MTP heads are already part of the base model's parameter count. No separate calculation is needed — set `baseParams` to the model's total parameter count.

### KV Cache Calculation
* **PagedAttention Block Rounding:** `effective_context = ceil(context_len / block_size) * block_size`
* **Beam Search Multiplier:** `effective_seqs = max_seqs * num_beams`
* **KV Cache (GB):** `(2 * layers * kv_heads * head_dim * effective_context * effective_seqs * kv_bytes) / 1024^3`
* The `2` accounts for both Key and Value tensors.
