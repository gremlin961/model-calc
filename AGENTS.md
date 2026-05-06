# AGENTS.md

## Structure
- Single-file app: everything lives in `index.html` (HTML + Tailwind via CDN + vanilla JS).
- No build step, no local dependencies, no server. Open `index.html` directly in a browser.
- Tailwind CSS is loaded from `cdn.tailwindcss.com` — requires internet connection on first load.

## Formulas
- **Effective Params:** `baseParams + loraParams` — `baseParams` is the TOTAL loaded parameters (what's actually in VRAM). For MoE models, enter total params, not active count.
- **Weights GB:** `effectiveParams * weightBytes`
- **Draft Model (Speculative Decoding):** `draftParams` is total loaded parameters. Draft KV cache is transient — not included.
- **Effective Context:** `Math.ceil(contextLen / blockSize) * blockSize` (PagedAttention block rounding)
- **Effective Seqs:** `maxSeqs * numBeams`
- **KV Cache (GB):** `(2 * layers * kvHeads * headDim * effectiveContext * effectiveSeqs * kvBytes) / 1024^3`

## Defaults
- Pre-loaded for Qwen3.6-27B (FP8 weights, 64 layers, 4 KV heads, 256 head dim, 262k context) on 128 GB VRAM.
- LoRA params: 0, beams: 1, block size: 16 tokens.
- Speculative decoding: disabled. Draft model defaults: 3B params, FP8, n_predict=5.
