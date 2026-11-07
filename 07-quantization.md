# Module 07 — Quantization

**Mastery: 43% → 50%** · Prerequisites: Module 06

**Goal of this module:** Understand how a 140 GB model fits onto a 24 GB consumer GPU. By the end you'll be able to read filenames like `qwen3-14b-instruct-Q4_K_M.gguf`, `Llama-3.1-70B-Instruct-AWQ`, or `IQ3_XXS` and know what they mean. You'll also be able to estimate a quantized model's size and speed on the back of an envelope, and choose a quantization for your hardware.

**How to read this module:** **Bold** marks a definition. This module builds on parameters and bytes (§1.30), the 2-bytes-per-parameter rule (§4.17), and memory bandwidth (§6.12). **Quantization** means storing a model's numbers with fewer bits, trading a little precision for a lot of memory and speed.

---

## Chapter 1 — Number Formats

### §7.1 — Why Number Format Matters
A model is billions of numbers (§1.30). How many bits you spend per number decides almost everything about running it:

| Format | Bytes per parameter | Llama 3 8B weights |
|--------|---------------------|--------------------|
| FP32 | 4 | 32 GB |
| BF16 | 2 | 16 GB |
| 8-bit | 1 | 8 GB |
| 4-bit | 0.5 | 4 GB (plus a little overhead, §7.18) |

Smaller weights mean three wins:
1. **It fits.** A 4-bit 8B model runs on an 8 GB laptop GPU. The BF16 version doesn't.
2. **It's faster.** Generating each token means reading *every* weight from memory once (§3.29, §6.12). Half the bytes means roughly twice the tokens per second.
3. **It's cheaper.** You need fewer or smaller GPUs.

The cost is precision: each number gets rounded, and too much rounding damages the model.

**Network-engineer analogy:** quantization is like route summarization. You shrink the table by giving up some specificity. Done carefully, nobody notices. Done carelessly, you blackhole traffic.

### §7.2 — How Floating-Point Numbers Work
Most AI formats are **floating point**, the binary version of scientific notation. `6.02 × 10²³` has a sign, digits (`6.02`), and an exponent (`23`). A float stores the same three parts in bits:
- **Sign** (1 bit): positive or negative.
- **Exponent** bits: *how big or small*. More exponent bits = wider **range**.
- **Mantissa** bits: *how many significant digits*. More mantissa bits = finer **precision**.

Every 16-bit or 8-bit format is a choice about how to split a small bit budget between range and precision. Keep that trade-off in mind for the next five sections.

### §7.3 — FP32 (Single Precision)
1 sign + 8 exponent + 23 mantissa bits. Range up to about 3.4 × 10³⁸, about 7 significant decimal digits. It was the classic default for training. Today it's used where precision matters most: the master copy of weights and the optimizer state during training (§4.17), and for accumulating sums inside matrix multiplies. It's almost never used to *store* weights for inference.

### §7.4 — FP16 (Half Precision)
1 sign + 5 exponent + 10 mantissa bits. Half the memory of FP32 and much faster on GPU tensor cores (§6.5). The problem is **range**: the largest FP16 number is 65,504, and anything bigger overflows to infinity. Tiny numbers below about 6 × 10⁻⁸ round to zero. During training that's dangerous, which is why FP16 needed loss scaling (§4.18).

### §7.5 — BF16 (Brain Float 16)
Invented at Google Brain. 1 sign + **8 exponent** (the same as FP32) + 7 mantissa bits. It's literally the top 16 bits of an FP32 number. Same range as FP32 with less precision (about 2–3 significant digits). It **trains stably without loss scaling** and is now the default format for training and for releasing weights. It's supported on NVIDIA A100/H100/Blackwell and RTX 30-series and newer, plus modern AMD, Intel, and Apple chips. Older GPUs like the T4 and V100 lack native BF16 and fall back to FP16.

### §7.6 — FP8
Two 8-bit float variants, standardized by NVIDIA, Arm, and Intel:
- **E4M3** (4 exponent, 3 mantissa bits): more precision, used for weights and activations in the forward pass.
- **E5M2** (5 exponent, 2 mantissa bits): more range, used for gradients in training.

H100 and RTX 40-series GPUs have native FP8 tensor cores, about twice BF16's speed. FP8 is used for training (DeepSeek-V3, §4.18) and is a top choice for serving on H100-class hardware (vLLM, TensorRT-LLM).

### §7.7 — Integers Plus a Scale Factor (INT8, INT4)
Integers have no exponent at all, just evenly spaced steps. INT8 has 256 levels (−128 to 127). INT4 has only 16 (−8 to 7). Weights are small decimals like 0.031 or −0.47, so you can't store them as integers directly. Instead you store:
- the integers, plus
- one **scale factor** (usually FP16) shared by a group of weights, telling you what one integer step is worth.

Worked example: a group's largest weight is 2.5, and you use INT4's range of ±7. Then the scale is `2.5 ÷ 7 ≈ 0.357`, so the integer 3 means `3 × 0.357 ≈ 1.07`. Any weight gets snapped to the nearest multiple of 0.357. Chapter 2 builds on this.

### §7.8 — FP4 (NVFP4, MXFP4)
The newest frontier: a 4-bit float (1 sign, 2 exponent, 1 mantissa, called **E2M1**). It can only represent 0, 0.5, 1, 1.5, 2, 3, 4, and 6 (plus negatives), so it always comes with small per-block scale factors (**microscaling**):
- **MXFP4** (an open industry standard): one shared 8-bit scale per block of 32 values, about 4.25 bits per weight.
- **NVFP4** (NVIDIA): one FP8 scale per block of 16 values plus a per-tensor scale, about 4.5 bits per weight and more accurate.

NVIDIA's Blackwell GPUs (B200, RTX 50-series) run FP4 natively in their tensor cores. OpenAI's open-weight **gpt-oss** models (2025) shipped their MoE weights in MXFP4. FP4 looks likely to become a standard inference format on new hardware, but it's still early.

### §7.9 — Quick Reference Table
Approximate sizes for weights only. KV cache and runtime overhead come on top (§7.23).

| Format | Bits per weight | 8B model | 70B model |
|--------|-----------------|----------|-----------|
| FP32 | 32 | 32 GB | 280 GB |
| BF16 / FP16 | 16 | 16 GB | 140 GB |
| FP8 / INT8 / Q8_0 | 8–8.5 | 8–8.5 GB | 70–75 GB |
| Q6_K | ~6.6 | 6.6 GB | 58 GB |
| Q5_K_M | ~5.7 | 5.7 GB | 50 GB |
| Q4_K_M | ~4.85 | 4.9 GB | 43 GB |
| INT4 / FP4 (with scales) | ~4.25–4.5 | 4.3–4.5 GB | 37–39 GB |
| Q3_K_M | ~3.9 | 3.9 GB | 34 GB |
| IQ3_XXS | ~3.1 | 3.1 GB | 27 GB |
| Q2_K | ~3.0 | 3.0 GB | 26 GB |

Bits per weight for GGUF types are approximate; they include scale overhead and vary a little by model.

---

## Chapter 2 — How Quantization Works

### §7.10 — Quantize, Store, Dequantize
The basic recipe, called **absmax** quantization, for one group of weights:
1. Find the largest absolute value in the group (say 2.5).
2. **Scale** = largest value ÷ biggest integer level (7 for INT4, 127 for INT8).
3. **Quantize:** divide each weight by the scale and round to the nearest integer. Store those integers plus the scale.
4. **Dequantize** at inference: integer × scale ≈ the original weight.

The difference between the original and the dequantized weight is the **quantization error**. Every method in this chapter is a smarter way to keep that error from hurting the model's *outputs*.

### §7.11 — Granularity: Why Outliers Ruin Everything
One scale for a whole tensor (**per-tensor**) is simplest, but it has a fatal weakness: **outliers**. Real LLMs have a few weights and activations far larger than the rest. One weight of 6.0 among weights near ±0.5 forces a big scale, so with INT4 the step size becomes 0.86, and nearly every normal weight rounds to **zero**. You'll see exactly that in Lab 2.

The fix is finer **granularity**:
- **Per-channel:** one scale per row of the weight matrix.
- **Per-group / per-block:** one scale per small block, typically 16–256 weights. An outlier then only hurts its own block.

Smaller blocks are more accurate but store more scales. That overhead is why "4-bit" formats are really about 4.25–4.9 bits per weight. In Lab 2, switching from one scale to one per 8 weights cuts INT4's error about 3×.

### §7.12 — Two Worlds: Training-Time vs. Post-Training Quantization
- **Quantization-aware training (QAT):** train or fine-tune the model with quantization simulated, so it learns weights that survive rounding. Best quality, but costly, and only the model's creator usually does it. Google's QAT versions of Gemma 3 and the MXFP4 gpt-oss weights are examples.
- **Post-training quantization (PTQ):** take a finished BF16 model and compress it, using minutes to hours on one GPU and often a small **calibration set** (a few hundred sample texts) to see which weights matter. **PTQ is what almost every GGUF, GPTQ, and AWQ file you download is.**

### §7.13 — Round-to-Nearest (RTN): The Naive Baseline
Just apply §7.10 to every block, with no cleverness. It works fine at 8 bits. At 4 bits and below, errors add up across dozens of layers, and quality drops noticeably. Every method below beats it.

### §7.14 — GPTQ
**GPTQ** (2022) quantizes one column of a weight matrix at a time. After rounding each column, it *adjusts the not-yet-quantized columns* to cancel out the error just introduced, guided by calibration data that shows how the inputs to that layer actually behave. It's like rounding a list of invoice amounts while carrying each rounding error into the next line so the total stays right. It was the first widely used method to make 4-bit quantization good. Look for `-GPTQ` in model names, and it's served by vLLM and others.

### §7.15 — AWQ (Activation-Aware Weight Quantization)
**AWQ** (2023) starts from an observation: about 1% of weight channels matter far more than the rest, namely the ones multiplied by large activations. Keeping those in high precision would fix most of the error, but mixed formats are awkward for fast kernels. So AWQ instead **scales those important channels up before quantizing** (and scales the matching activations down by the same factor, so the math is unchanged). They get finer effective resolution while *everything* stays INT4. It often matches or beats GPTQ in quality and is quick to apply. Look for `-AWQ`.

### §7.16 — Taming Activation Outliers: SmoothQuant and Friends
Weights are easy compared with **activations**, the values flowing *through* the network at runtime, which contain huge outlier channels (§7.24). **SmoothQuant** (2022) moves that difficulty from activations into weights with a per-channel rescaling, so both can be stored as INT8 (**W8A8**). ZeroQuant, OmniQuant, and later rotation-based methods (QuaRot, SpinQuant) push the same idea further. You rarely run these yourself. They're built into engines like TensorRT-LLM and the quantization tools for vLLM.

---

## Chapter 3 — File Formats and Tools

### §7.17 — GGUF: The llama.cpp Format
**GGUF** (2023, successor to the older GGML format) is the file format of llama.cpp, Ollama, and LM Studio (§8.10–§8.12). The "GG" comes from creator Georgi Gerganov's initials. One file contains everything needed to run: the quantized weights, the tokenizer (§3.1), the chat template (§5.3), and metadata like context length and RoPE settings (§3.24). That single-file convenience is a big part of why local AI took off. GGUF can also mix quantization types *per tensor* inside one file, which is what the `_S`/`_M`/`_L` suffixes control.

### §7.18 — Decoding GGUF Quantization Names
How to read `Q4_K_M`:
- **`Q4`**: about 4 bits per weight for most tensors.
- **`_K`**: **k-quants**. Weights are grouped into **super-blocks** of 256, split into sub-blocks of 16 or 32, each with its own small scale. That two-level scaling is much more accurate than the older single-level types (`Q4_0`, `Q5_1`).
- **`_S` / `_M` / `_L`**: small, medium, or large mix. The more sensitive tensors (parts of attention and the FFN output) get more bits, such as Q6_K inside a Q4_K_M file. That's why Q4_K_M averages ~4.85 bits per weight instead of 4.5.
- **`Q8_0`**: the simple 8-bit type (blocks of 32 weights plus one FP16 scale = 8.5 bits per weight). Nearly lossless, but big.

**Bits per weight (bpw)** is the honest number to compare: file size in bits ÷ parameter count.

| Name | ~bpw | Typical use |
|------|------|-------------|
| Q8_0 | 8.5 | Nearly lossless; when you have the VRAM |
| Q6_K | 6.6 | Nearly lossless at a smaller size |
| Q5_K_M | 5.7 | High quality, a good choice for code and math |
| **Q4_K_M** | **4.85** | **The default sweet spot** |
| Q3_K_M | 3.9 | Fitting a bigger model; noticeable loss |
| Q2_K | ~3.0 | Last resort |

### §7.19 — I-Quants and the Importance Matrix (imatrix)
Two llama.cpp upgrades, mostly for very low bit rates:
- **Importance matrix (imatrix):** run calibration text through the model and record which weights matter most. The quantizer then spends its error budget where it hurts least. It works with k-quants too. Uploaders often mark these files with `imatrix`, `imat`, or `i1`.
- **I-quants (`IQ2_XXS`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, ...):** newer types that snap *groups* of weights to the nearest entry in a carefully designed codebook (a list of allowed patterns), instead of rounding each weight alone. They use an imatrix and give noticeably better quality per bit below about 4 bits. **IQ4_XS** (~4.25 bpw) is often about as good as Q4_K_M in a smaller file.

Trade-off: i-quants can be slower to decode on some CPUs and older GPUs, because unpacking codebooks takes more compute. Measure on your own hardware.

### §7.20 — EXL2 / EXL3 (ExLlamaV2 and V3)
Formats for NVIDIA GPU-only inference with the ExLlama engines (§8.28). **EXL2** lets you pick *any* average bit rate (say 4.65 bpw) and assigns bits per layer by measured importance. Its successor **EXL3** (2025) uses a newer trellis-based method aimed at better quality at very low bit rates. They're fast on consumer NVIDIA cards, but have no CPU offload and a smaller ecosystem than GGUF.

### §7.21 — bitsandbytes
The Hugging Face library that quantizes **on the fly as a model loads**: INT8 (the LLM.int8() method, which keeps outlier features in 16-bit) or 4-bit **NF4** (§5.18). There's no pre-quantized file to find. Add one argument to `from_pretrained`. It's the standard for QLoRA fine-tuning, but slower than GPTQ, AWQ, or GGUF kernels for serving.

### §7.22 — Which Format Goes With Which Engine
| Format | Typical engines | Best for |
|--------|-----------------|----------|
| GGUF (Q*_K, IQ*) | llama.cpp, Ollama, LM Studio | Laptops, Macs, CPU+GPU mixes, single users |
| GPTQ / AWQ (INT4) | vLLM, SGLang, TGI | GPU servers, many users, consumer or datacenter GPUs |
| FP8 | vLLM, SGLang, TensorRT-LLM | H100 / RTX 40-class and newer; production serving |
| NVFP4 / MXFP4 | TensorRT-LLM, vLLM (newer versions) | Blackwell-generation GPUs |
| EXL2 / EXL3 | ExLlamaV2 / V3, TabbyAPI | Fast single-user NVIDIA setups |
| bitsandbytes | Transformers, PEFT | Fine-tuning and experiments |

---

## Chapter 4 — Beyond Weights: KV Cache and Activations

### §7.23 — Weights Aren't the Only Big Thing: The KV Cache
The **KV cache** (§2.37, §9.5) stores each previous token's Key and Value vectors. Its size per token is:

```
layers × KV heads × head size × 2 (K and V) × bytes per number
```

For Llama 3 70B (80 layers, 8 KV heads, 128 per head) in BF16: `80 × 8 × 128 × 2 × 2 = 327,680 bytes ≈ 0.33 MB per token`. At 32,000 tokens of context that's **~10.7 GB per conversation**. Ten simultaneous users is over 100 GB, more than the 4-bit weights themselves.

Storing the cache in **FP8 or INT8** halves it (~5.4 GB at 32K) with little quality loss. It's standard in vLLM and TensorRT-LLM, and available in llama.cpp (`--cache-type-k q8_0 --cache-type-v q8_0`, §9.9). Quantizing the cache harder (4-bit) saves more but starts to hurt long-context accuracy.

### §7.24 — Activation Quantization
**Weight-only** quantization (like Q4_K_M or AWQ) stores weights small, but converts them back to 16-bit to do the math. It saves memory and bandwidth, not compute. To use the fast INT8, FP8, or FP4 tensor-core math, the **activations** must be low-precision too.

That's harder: around 6–7B parameters and up, LLMs develop a few activation channels up to 100× larger than the rest (the 2022 LLM.int8() paper found this). Naive INT8 rounding of those wrecks the output. FP8 copes better thanks to its exponent, and SmoothQuant-style tricks (§7.16) handle INT8.

### §7.25 — End-to-End Recipes
The naming convention is **W**(weight bits)**A**(activation bits):

| Recipe | What it means | Where it fits |
|--------|---------------|---------------|
| **W4A16** | INT4 weights, 16-bit math (GPTQ, AWQ, GGUF) | Consumer GPUs and single users: big memory and bandwidth savings |
| **W8A8** (INT8 or FP8) | 8-bit weights *and* math | H100-class serving at scale: faster compute too |
| **W4A8 / FP4** | 4-bit weights with 8-bit or 4-bit math | Newest hardware (Blackwell), the cutting edge |
| **+ FP8 KV cache** | Added to any of the above | Long contexts and many users |

### §7.26 — Mixed Precision Everywhere
Real inference stacks mix formats inside one model: 4-bit or FP8 for the big weight matrices, higher precision (BF16 or FP32) for sensitive pieces like normalization, softmax, and the running sums inside matrix multiplies, often a higher-precision embedding table and output head, and an 8-bit KV cache. The engine handles all of it. You pick a recipe.

---

## Chapter 5 — Trade-offs and Practical Choices

### §7.27 — The Quality Cliff at Low Bits
Rough patterns that hold for most models (always check your own model and task):
- **8-bit:** practically indistinguishable from BF16.
- **5–6 bit:** very hard to tell apart.
- **~4.5–5 bit (Q4_K_M, AWQ):** small losses, often around 1–3% on benchmarks; fine for most uses.
- **~3–4 bit:** noticeably worse, especially on reasoning, code, and long outputs.
- **Below ~3 bit:** the cliff. Small models break down; very large models degrade more gracefully.

Bigger models generally tolerate quantization better than small ones. There's more redundancy to absorb the rounding.

### §7.28 — "Bigger Model, Fewer Bits" Usually Wins
At a *fixed memory budget*, a larger model at lower precision usually beats a smaller model at higher precision. Example with ~20 GB to spend:

| Option | Size | Usually |
|--------|------|---------|
| 8B at BF16 | 16 GB | Weaker: wastes bits on precision it barely needs |
| 32B at Q4_K_M | ~20 GB | **Stronger**: 4× the parameters for the same memory |

The rule weakens below about 3 bits per weight, where the larger model's damage can outweigh its size advantage. It also compares models of the same family and generation; a newer small model can beat an older big one. And remember the KV cache needs room too (§7.23).

### §7.29 — Speed: The Bandwidth Math
For one user, decode speed is mostly limited by how fast the GPU can read the weights (§3.29, §6.12):

```
max tokens/second ≈ memory bandwidth ÷ bytes of weights read per token
```

On an RTX 4090 (~1 TB/s), Llama 3 8B gives:

| Quant | Size | Best-case tok/s |
|-------|------|-----------------|
| BF16 | 16.1 GB | ~63 |
| Q8_0 | 8.5 GB | ~118 |
| Q4_K_M | 4.9 GB | ~207 |

Real numbers come in lower, often around 60–75% of this, because of other overheads. But the *ratio* holds: halve the bits, nearly double the speed. (For MoE models, only the *active* experts are read per token (§2.36), which is why big MoE models can be fast.)

Two caveats:
- **Compute gains need hardware support.** On GPUs without fast INT4/FP8 math, low-bit weights are converted back to 16-bit for every multiply. You save memory and bandwidth, but not compute.
- **Prefill is compute-bound** (§3.29), so quantization speeds up reading a long prompt less than it speeds up generation, unless the math itself is low precision (§7.24).

### §7.30 — When Quantization Hurts: Reasoning, Math, Code, Long Context
Tasks with long chains of exact steps are more sensitive to quantization noise. One small error in step 3 of a proof or a function compounds. Reasoning models that think for thousands of tokens (§2.27) are especially sensitive. For these tasks, prefer Q5/Q6 or FP8 over Q4 if you can, and evaluate before committing. Long contexts also suffer more from aggressive KV-cache quantization (§7.23).

### §7.31 — Quantizing Embeddings for Search
Embedding models (§3.18) can be quantized like any model. Separately, you can quantize the **stored vectors** in your vector database (§3.20):
- **INT8 (scalar) vectors:** 4× smaller than FP32, with a tiny accuracy loss.
- **Binary vectors** (1 bit per number, keeping only the sign): **32× smaller**, with a bigger loss. A common trick is to search with binary vectors first, then re-score the top results with full-precision vectors (**rescoring**).

### §7.32 — How to Choose, in Practice
Starting points. Leave room for the KV cache and context you need.

| Hardware | Good starting choice |
|----------|----------------------|
| 8 GB GPU / laptop | 7–8B at Q4_K_M (~5 GB) |
| 12–16 GB GPU | 14B at Q4_K_M–Q5_K_M (~9–11 GB) |
| 24 GB GPU | 32B at Q4_K_M or IQ4_XS (~18–20 GB), or 14B at Q8_0 |
| Mac with 64–128 GB unified memory | 70B at Q4–Q6 (~43–58 GB), or a large MoE |
| 1× H100 80 GB | 70B at FP8 (~70 GB) leaves little room for KV cache; AWQ INT4 leaves plenty |
| 8× H100 | 405B at FP8 with tensor parallelism (§8.21) |

### §7.33 — How to Verify Quality After Quantizing
Don't assume. Measure:
1. **Your own task evals** (§5.24): the same test prompts through the BF16 and quantized models, scored the same way.
2. **Standard benchmarks** for a broad check (MMLU, GSM8K, HumanEval).
3. **KL divergence from the original model:** compare the quantized model's next-token probability lists (§2.17) with the BF16 model's on the same text. Lower is closer. llama.cpp's `llama-perplexity` tool can report it. It's more sensitive than perplexity alone.
4. **Spot-check by hand**, especially long and tricky outputs.

Don't rely on perplexity (§4.5) alone. A quant can keep perplexity nearly unchanged while losing reasoning or instruction-following ability.

### 🛠️ Build It — Hands-On Lab
**Lab 1: see range vs. precision (§7.2–§7.5).** Pure Python, nothing to install.

```python
import struct

def to_fp16(x):
    try:
        value = struct.unpack("<e", struct.pack("<e", x))[0]   # "e" = IEEE half precision
        return f"{value:.8g}"
    except OverflowError:
        return "overflow!"

def to_bf16(x):
    # BF16 is just the top 16 bits of FP32: same exponent, shorter mantissa.
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]

def to_fp32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

print(f"{'value':>14} {'FP32':>14} {'FP16':>14} {'BF16':>14}")
for x in (0.1, 3.14159265, 1234.5678, 70000.0, 0.00000001):
    print(f"{x:>14.8g} {to_fp32(x):>14.8g} {to_fp16(x):>14} {to_bf16(x):>14.8g}")
```

What you should see: FP16 keeps more digits than BF16 for normal numbers (0.099976 vs 0.099609), but FP16 **overflows** at 70,000 and rounds 0.00000001 to **0**. BF16 handles both, just less precisely (70000 becomes 69632). That's exactly why training moved to BF16.

(For simplicity, this BF16 conversion chops off the extra bits instead of rounding, so its results are slightly lower than real hardware's. The lesson is the same.)

**Lab 2: quantize a layer and watch the outlier (§7.10–§7.11).** Pure Python.

```python
import random

random.seed(7)

# 32 "weights" like a real layer: mostly small numbers near zero...
weights = [round(random.gauss(0, 0.5), 3) for _ in range(32)]
weights[5] = 6.0          # ...plus one outlier, which real models have (§7.11)

def quantize(block, bits):
    levels = 2 ** (bits - 1) - 1              # INT8 -> 127, INT4 -> 7
    scale = max(abs(w) for w in block) / levels   # "absmax" scale for this block
    ints = [round(w / scale) for w in block]      # what gets stored
    return [q * scale for q in ints], ints, scale # dequantized values

def mean_error(original, restored):
    return sum(abs(a - b) for a, b in zip(original, restored)) / len(original)

print("first 8 weights:   ", weights[:8])
for bits in (8, 4):
    # One scale for the whole tensor
    restored, ints, scale = quantize(weights, bits)
    err_tensor = mean_error(weights, restored)
    # One scale per group of 8 weights (what GGUF / GPTQ / AWQ actually do)
    restored_g = []
    for i in range(0, len(weights), 8):
        restored_g += quantize(weights[i:i + 8], bits)[0]
    err_group = mean_error(weights, restored_g)
    print(f"\nINT{bits}: stored ints (first 8) = {ints[:8]}, scale = {scale:.4f}")
    print(f"  mean error, one scale per tensor:   {err_tensor:.4f}")
    print(f"  mean error, one scale per 8 weights: {err_group:.4f}")
```

What you should see: with one INT4 scale for the whole tensor, the outlier forces a step size of 0.857, and the stored integers are mostly **0**. The layer's information is mostly erased. One scale per 8 weights cuts the error about 3× (0.206 → 0.068). INT8 is far gentler either way (0.012 → 0.004). Things to try: delete the outlier line and rerun. Try groups of 4 and 32.

**Lab 3: size and speed calculator (§7.18, §7.29).**

```python
# File size and best-case decode speed for different quantizations (§7.18, §7.29).
BPW = {             # approximate bits per weight, including scale overhead
    "BF16":   16.0,
    "Q8_0":    8.5,
    "Q6_K":    6.56,
    "Q5_K_M":  5.7,
    "Q4_K_M":  4.85,
    "IQ4_XS":  4.25,
    "Q3_K_M":  3.9,
    "IQ3_XXS": 3.06,
    "Q2_K":    2.96,
}
PARAMS = 8.03e9            # Llama 3 8B
BANDWIDTH = 1008e9         # bytes/second: RTX 4090 memory bandwidth (§6.12)

print(f"{'quant':<8} {'size GB':>8} {'max tok/s':>10}")
for name, bpw in BPW.items():
    size = PARAMS * bpw / 8                 # bytes
    tok_s = BANDWIDTH / size                # each token reads every weight once
    print(f"{name:<8} {size / 1e9:>8.1f} {tok_s:>10.0f}")
```

Change `PARAMS` and `BANDWIDTH` to match your model and GPU (`nvidia-smi` and your card's spec sheet, §6.15). Then compare the predictions with real measurements from Lab 4.

**Lab 4: measure real trade-offs (needs Ollama).**
1. Pull the same model at several quantizations. Check the model's "Tags" page on ollama.com for exact names, such as `qwen3:8b-q4_K_M`, `qwen3:8b-q8_0`, and `qwen3:8b-fp16`.
2. Run the same prompts with `ollama run <model> --verbose`, which prints the **eval rate** (decode tokens/second) after each answer. Compare it with Lab 3's prediction.
3. Score the answers on 10–20 of your own test prompts (§5.24), using a stronger model as a judge with the order shuffled.
4. Plot bits per weight against quality and against tokens/second. That's your first quantization trade-off curve, exactly what teams use to pick a deployment format.

**Deliverable:** a one-page note with your curve and the quantization you'd pick for your hardware.

### §7.34 — Checkpoint — Explain It Back
1. Why does quantization make inference faster, not just smaller?
2. What's the difference between exponent bits and mantissa bits, and why did training move from FP16 to BF16?
3. How does integer quantization with a scale factor work?
4. Why do outliers make per-tensor quantization bad, and what fixes it?
5. What does `Q4_K_M` mean, and why is it ~4.85 bits per weight rather than 4?
6. What's the difference between GPTQ and AWQ?
7. What are i-quants and the imatrix, and when do they matter most?
8. Roughly how big is Llama 3 70B's KV cache at 32K context in BF16, and how do you shrink it?
9. Why is "bigger model at lower precision" usually better at the same memory, and when does that break?
10. What's the difference between W4A16 and W8A8?

<details>
<summary>Answer key (try first!)</summary>

1. Each generated token reads every weight from memory, and decoding is limited by memory bandwidth, so fewer bytes per weight means more tokens per second.
2. Exponent bits set the range (how big or small); mantissa bits set the precision (significant digits). FP16's narrow range overflowed or underflowed during training. BF16 has FP32's range.
3. Store small integers plus a shared scale per group. The real value ≈ integer × scale, where the scale is the group's largest absolute value ÷ the top integer level.
4. One huge value forces a huge scale, so normal weights round to zero. Use small groups or blocks, each with its own scale, so an outlier only hurts its own block.
5. About 4-bit k-quants (super-blocks of 256 weights with sub-block scales), medium mix. The scales add overhead, and the more sensitive tensors are stored at higher bit rates (like Q6_K).
6. GPTQ rounds column by column and adjusts the remaining weights to cancel the error. AWQ finds the channels hit by large activations and scales them up before quantizing, so they keep more resolution.
7. The imatrix records which weights matter most, using calibration text. I-quants snap groups of weights to codebook patterns. Both matter most below about 4 bits per weight.
8. About 0.33 MB per token × 32,000 ≈ 10.7 GB per conversation. Store the cache in FP8/INT8 (about half), use shorter contexts, or use models with fewer KV heads (GQA).
9. More parameters hold more knowledge and skill than extra precision does, down to about 4 bits per weight. Below about 3 bits the damage outweighs the size advantage.
10. W4A16 stores 4-bit weights but does the math in 16-bit (it saves memory and bandwidth). W8A8 does the math in 8-bit too (it also saves compute on supporting hardware).
</details>

You now understand the number formats that let a 70B model run on a gaming PC or a Mac. Next module: the engines that *actually run it.*
