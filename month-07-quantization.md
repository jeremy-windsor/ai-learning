# Month 07 — Quantization (Days 181–210)

**Goal of this month:** Understand how a 140 GB model fits onto a 24 GB consumer GPU. By Day 210 you'll be able to read filenames like `qwen3-14b-instruct-Q4_K_M.gguf` or `Llama-3.1-70B-Instruct-AWQ` and know exactly what they mean.

---

## Week 1 — Number Formats

### Day 181 — Why Number Format Matters
A model parameter is just a number. Stored in 32 bits (FP32) it takes 4 bytes. Stored in 4 bits (INT4) it takes 0.5 bytes — **8× smaller**. Smaller = fits in less VRAM, reads from memory faster, runs faster (inference is bandwidth-bound — Day 160). The cost: precision.

### Day 182 — FP32 (Single Precision)
The classical default — 1 sign bit, 8 exponent bits, 23 mantissa bits. Range ~10⁻³⁸ to 10³⁸, ~7 decimal digits of precision. Used historically for training; rarely used in 2026.

### Day 183 — FP16 (Half Precision)
1 sign, 5 exponent, 10 mantissa. Half the memory, ~2× the speed on tensor cores. **Problem:** narrow range — overflows easily during training.

### Day 184 — BF16 (Brain Float 16)
Google's invention. 1 sign, **8 exponent** (same as FP32), 7 mantissa. Same dynamic range as FP32, lower precision. **Trains stably without loss scaling.** Now the default for both training and inference. Supported on A100/H100/B100 and modern AMD/Intel.

### Day 185 — FP8
1 sign, 4–5 exponent, 2–3 mantissa. Two variants: **E4M3** (more precision) for forward, **E5M2** (more range) for gradients. H100 added native FP8 tensor cores. Used for both training (DeepSeek-V3) and inference (vLLM, TensorRT-LLM).

### Day 186 — INT8 / INT4 / INT2
Pure integers, no exponent. INT8 = 256 levels, INT4 = 16 levels. To use them for weights, you store a per-group **scale factor** (FP16) so the integers represent a range like [-2.5, 2.5] in steps of ~0.3. This is the foundation of "post-training quantization" — Day 191.

### Day 187 — FP4 (NVFP4, MXFP4)
The 2024–2025 frontier. 1 sign, 2 exponent, 1 mantissa = **4 bits per weight, native tensor-core support on Blackwell**. With small per-block scale factors (microscaling), accuracy holds up impressively. Expect FP4 to become the default inference format on new hardware.

### Day 188 — Quick Reference Table
| Format | Bits | Bytes/param | 7B model size | 70B model size |
|--------|------|-------------|---------------|----------------|
| FP32   | 32   | 4           | 28 GB         | 280 GB         |
| BF16/FP16 | 16 | 2          | 14 GB         | 140 GB         |
| FP8    | 8    | 1           | 7 GB          | 70 GB          |
| INT8   | 8    | 1           | 7 GB          | 70 GB          |
| Q5_K_M | ~5.5 | ~0.7        | 5 GB          | 50 GB          |
| INT4 / Q4 | 4 | 0.5        | 3.5 GB        | 35 GB          |
| FP4 / Q3 | 3-4 | ~0.4       | 3 GB          | 30 GB          |
| Q2_K   | ~2.5 | ~0.3        | 2.5 GB        | 25 GB          |

---

## Week 2 — Quantization Methods

### Day 189 — Two Worlds: Training-Time vs Post-Training
- **Quantization-Aware Training (QAT)** — train (or fine-tune) the model with quantization simulated in the loop. Best quality. Expensive.
- **Post-Training Quantization (PTQ)** — take an already-trained BF16 model and crunch its weights down. Cheaper, faster, slightly lower quality. **PTQ dominates in practice.**

### Day 190 — Round-To-Nearest (RTN) — The Naïve Baseline
Just round each weight to the nearest representable value. Works okay for INT8. Falls apart at INT4 — the rounding errors compound through 80 layers and degrade the model noticeably.

### Day 191 — GPTQ
A smarter PTQ method: process weights one column at a time, and after rounding each column, *adjust the remaining weights* to compensate for the rounding error. Uses a small calibration dataset. Result: surprisingly little quality loss at INT4. The first widely-used 4-bit method. File extension: `-GPTQ`.

### Day 192 — AWQ (Activation-aware Weight Quantization)
Observation: not all weights matter equally — those that interact with **large-magnitude activations** matter most. AWQ identifies and **protects the top ~1% of "salient" weight channels** (keeps them in higher precision, scales the rest accordingly). Often beats GPTQ in quality at the same bit-width. File extension: `-AWQ`.

### Day 193 — GGUF: The Llama.cpp Format
**GGUF** (GPT-Generated Unified Format) is the file format used by llama.cpp / Ollama / LM Studio. It contains the quantized weights, tokenizer, chat template, metadata — everything needed to run, all in one file. Successor to the older GGML format.

### Day 194 — Decoding GGUF Quantization Names
Names like `Q4_K_M`, `Q5_K_S`, `Q8_0` look cryptic. Reading guide:
- **Q\<n\>** — n-bit quantization (Q4 = ~4 bits/weight)
- **_K** — k-quants (better than older `Q4_0`, uses superblocks with multiple scales)
- **_S / _M / _L** — small / medium / large variants — slightly more bits in critical layers
- **Q8_0** — legacy 8-bit, very high quality, large file
**Practical recommendation**: `Q4_K_M` is the sweet spot for most use cases. `Q5_K_M` if you have spare VRAM. `Q6_K` for near-lossless. `Q8_0` if you want overkill.

### Day 195 — EXL2 / ExLlamaV2
A format optimized for fast GPU-only inference (no CPU offload). Allows **mixed precision** — different layers/columns at different bitwidths chosen by importance. Often faster than GPTQ on consumer NVIDIA GPUs.

### Day 196 — bitsandbytes
The library used by Hugging Face for on-the-fly INT8 / NF4 (4-bit normal-float) quantization, especially during QLoRA fine-tuning. Less optimized than GPTQ/AWQ for pure inference, but very convenient.

### Day 197 — SmoothQuant, ZeroQuant, OmniQuant
Research methods that smooth the activation magnitudes (rebalance "hard to quantize" tensors) before quantizing. Used inside FP8 production stacks like TensorRT-LLM.

---

## Week 3 — KV Cache Quantization (preview of Month 9)

### Day 198 — Weights Aren't the Only Big Thing
The **KV cache** (the model's memory of the prompt so far — Month 9) can dwarf the weights for long contexts. A 70B model at 32k context has a KV cache of ~10 GB *per request*. Quantizing the KV cache (KV-int8, KV-fp8) is now standard in vLLM/TensorRT-LLM.

### Day 199 — Activation Quantization
For **W8A8** (weight-int8 + activation-int8) inference, you also quantize the *runtime activations*. Harder than weight quantization because activations can have outliers. SmoothQuant and FP8 fix this. Required to actually get the speed benefit on tensor cores.

### Day 200 — End-to-End Quantization Recipes
Modern production typically uses one of:
- **W4A16** (weights INT4, activations FP16) — GPTQ/AWQ; consumer-friendly.
- **W8A8** (INT8 or FP8 both) — best speed on H100/Blackwell, used at scale.
- **FP4 weights + FP8 KV** — bleeding edge on Blackwell.

---

## Week 4 — Tradeoffs and Practical Choices

### Day 201 — Quality Cliff at Low Bits
Across most models, INT8 is essentially indistinguishable from BF16. INT4 loses ~1–3% on most benchmarks. INT3 starts to feel noticeably worse. INT2 is broken for general tasks (still usable for *very* large models — Q2 of a 405B can outperform Q8 of a 70B at the same VRAM).

### Day 202 — The "Bigger Quantized Beats Smaller Full-Precision" Rule
A rule of thumb confirmed across many benchmarks: at a given VRAM budget, you almost always get better quality from a **larger model at lower precision** than a smaller model at higher precision. A Q4 70B beats a BF16 13B easily.

### Day 203 — Speed Impact of Quantization
Quantization helps inference speed in two ways:
1. **Memory bandwidth** — less to read = faster (the bigger win for single-user).
2. **Compute** — INT8/FP8/FP4 tensor cores are 2–4× faster than BF16 on supported hardware.
On older GPUs without INT4 tensor cores, INT4 weights get *dequantized to FP16* on the fly — you save VRAM but not necessarily compute.

### Day 204 — Imatrix / Importance Matrix Quantization
A llama.cpp feature: use a calibration dataset to compute per-weight importance, then allocate bits more carefully. Imatrix-quantized GGUFs (`-imat`) are noticeably better at very low bits (Q2, Q3).

### Day 205 — When Quantization Hurts: Long Context, Math, Code
Reasoning, math, and code generation are more sensitive to quantization noise. If your task is one of these, prefer Q5/Q6 over Q4. Reasoning models (R1, o1-style) are especially sensitive.

### Day 206 — Quantization for Embeddings
Embedding models also get quantized — and additionally, you can **quantize the embedding vectors themselves** (e.g., binary embeddings, scalar-quantized embeddings) to shrink vector databases by 32×. Small accuracy hit, huge storage win.

### Day 207 — Mixed Precision Everywhere
Production stacks today routinely mix: BF16 for embeddings, FP8 for matmuls, FP16 for softmax/normalization, INT4 weights, INT8 KV cache. The framework manages it all. You just pick a recipe.

### Day 208 — How to Choose, in Practice
- **Local laptop / 8 GB VRAM**: Q4_K_M of a 7B model.
- **Local 24 GB**: Q5_K_M of a 14B, or Q4_K_M of a 32B.
- **Mac with 64–128 GB**: Q4–Q6 of 70B.
- **Cloud single H100**: FP8 of 70B.
- **Cloud 8× H100**: FP8 of 405B with tensor parallelism.

### Day 209 — How to Verify Quality After Quantizing
Run a small benchmark suite (MMLU, GSM8K, HumanEval, or your own evals) on both BF16 and quantized versions. Look at score deltas. Spot-check outputs subjectively. *Don't* trust perplexity alone — it can hide capability degradation.

### 🛠️ Build It — Month 7 Hands-On
1. With Ollama, pull the same model at three quants: `qwen3:8b-q4_K_M`, `qwen3:8b-q5_K_M`, `qwen3:8b-q8_0`. Note the file sizes.
2. Run identical prompts through each. Use `time` to measure generation speed (tok/s). Use a stronger model as a judge to score quality.
3. Plot it: x-axis = bits-per-weight, y-axis = quality (and tokens/sec). You've just produced your first quantization tradeoff curve — exactly what real teams use to pick a deployment quant.
4. Bonus: download the BF16 version from Hugging Face and use `bitsandbytes` to load it in 4-bit on the fly. Compare to the GGUF Q4.

**Deliverable:** a one-page write-up with the curve and a recommended quant for your hardware.

### Day 210 — Recap & "Explain it Back"
**Quiz yourself:**
1. Why does quantization make inference faster?
2. What does `Q4_K_M` mean?
3. What's the difference between GPTQ and AWQ?
4. Why is "bigger model at lower precision" usually better than the reverse?
5. When would you avoid INT4?

You now understand the math format that lets a 70B model run on your gaming PC. Next month: the engines that *actually run it.*
