# Module 06 — Hardware & Compute

**Mastery: 42% → 50%** · Prerequisites: Module 04 (helps to know what training/inference each demand)

**Goal of this module:** Understand the silicon and the memory that AI runs on. By the end you'll know why VRAM is the bottleneck, why memory bandwidth (not FLOPS) often matters most for inference, and what each line of `nvidia-smi` is telling you.

---

## Chapter 1 — CPU vs GPU vs TPU vs NPU

### §6.1 — The CPU: Few Cores, Smart, Serial
A modern CPU has 8–128 cores. Each is sophisticated — branch prediction, large caches, out-of-order execution. Optimized for **latency on a single task**. Great for spreadsheets, web servers, OS work. Terrible at "do this same simple math 10 million times in parallel."

### §6.2 — The GPU: Thousands of Cores, Dumb, Parallel
An H100 has ~17,000 CUDA cores plus specialized tensor cores. Each core is much weaker than a CPU core — but there are thousands of them, and they all execute the same instruction on different data simultaneously (**SIMT** — single instruction, multiple threads). Perfect for matrix multiply, which *is* what neural networks do.

### §6.3 — Why Matrix Multiplication Is the Workload
Every layer of a transformer is, fundamentally, a few large matrix multiplies (attention's QK, attention's softmax-times-V, the FFN's two matmuls). Together they account for >90% of compute. Specialize the silicon for matmul → win at AI. That's the entire premise of GPUs and TPUs in this era.

### §6.4 — Tensor Cores
Starting with NVIDIA Volta (2017), GPUs added **tensor cores** — small fixed-function units that do a 4×4 matrix multiply-accumulate in one cycle. An H100 has 528 tensor cores delivering ~1,000 TFLOPS BF16 *with structured sparsity* (~500 TFLOPS dense — the number real workloads see) — vs ~67 TFLOPS from the regular CUDA cores. Tensor cores are why modern training/inference is fast.

### §6.5 — Google TPUs
Google's custom AI chip. Built around a **systolic array** — a grid of multipliers where data flows through rhythmically. TPUs power Gemini training and Google's internal inference. Comparable to NVIDIA in performance but only available via Google Cloud.

### §6.6 — Apple Silicon, AMD, and the NPU
- **Apple M-series** has a unified memory architecture — CPU, GPU, and Neural Engine share one big pool of fast RAM. This is why MacBooks can run surprisingly large models locally.
- **AMD MI300X** competes with H100 in raw specs; software (ROCm) is still catching up.
- **NPUs** (Neural Processing Units) in laptops/phones — small, low-power chips for on-device inference (Copilot+ PCs, iPhone Neural Engine).

### §6.7 — The NVIDIA Lineup
- **A100** (2020): 40/80 GB HBM2e, ~312 TF BF16. Workhorse of the GPT-3/4 era.
- **H100** (2022): 80 GB HBM3, ~1,000 TF BF16 with sparsity (~500 dense). The gold standard 2023–2025.
- **H200** (2024): same compute, 141 GB HBM3e — more memory matters more than more FLOPS for big LLMs.
- **B100/B200 (Blackwell)** (2024–2025): ~5× H100 perf, FP4 native support.
- **GB200 NVL72**: 72 Blackwell GPUs as one giant logical accelerator.
- Consumer: **RTX 4090** (24 GB), **RTX 5090** (32 GB) — mostly used for local inference and small fine-tunes.

---

## Chapter 2 — Memory: The Real Bottleneck

### §6.8 — VRAM vs System RAM
**VRAM** (Video RAM, on the GPU package) is what holds model weights, KV cache, and activations during compute. **System RAM** is on the motherboard. PCIe between them is ~30 GB/s — *glacial* compared to VRAM. Anything spilling out of VRAM destroys performance.

### §6.9 — HBM: High-Bandwidth Memory
GPU VRAM today is **HBM** — DRAM stacks mounted directly next to the GPU die via silicon interposer. H100 HBM3 hits ~3.35 TB/s of bandwidth. For comparison, top-tier DDR5 system RAM is ~80 GB/s. **40× difference.** This is the moat NVIDIA / TSMC / SK Hynix protect.

### §6.10 — Memory Bandwidth Is The Inference Bottleneck
A surprising fact: for *single-stream* LLM inference, you are usually **memory-bandwidth-bound**, not compute-bound. Every token generation requires reading the full model weights from VRAM. Tokens/sec ≈ memory_bandwidth / model_size_in_bytes. This is why a 70B model in BF16 (140 GB) maxes out at ~24 tok/s on an H100 (3.35 TB/s ÷ 140 GB).

### §6.11 — Sizing Rule of Thumb
For inference VRAM: **model_size_in_bytes ≈ 2 × params** for FP16/BF16. Plus **20–50%** overhead for KV cache, activations, framework overhead. So a 7B BF16 model needs ~14 GB → really ~17 GB to be safe. A 4-bit quantized 7B needs ~4 GB → ~6 GB safe.

### §6.12 — Why Bigger Batches = Better GPU Utilization
With one user, you read the full weights to produce one token. With 32 users batched together, you read the weights *once* and produce 32 tokens. Throughput goes up ~30×, latency per user stays roughly the same. **This is why production servers always batch** (Module 09, continuous batching).

### §6.13 — Roofline Model (Conceptual)
Plot performance vs arithmetic intensity (FLOPs per byte read). Two ceilings: peak FLOPS (the "roof") and peak bandwidth × intensity (the "slope"). Workloads either hit the FLOPS roof (compute-bound) or the bandwidth slope (memory-bound). LLM inference: bandwidth slope. LLM training with big batches: closer to the FLOPS roof.

### §6.14 — `nvidia-smi` Decoded
- **GPU-Util %** — fraction of time *any* kernel was running. Misleading for memory-bound workloads.
- **Memory-Used / Memory-Total** — VRAM consumption.
- **Pwr** — current draw vs limit. Throttling is real.
- **Temp** — sustained 85 °C+ throttles clocks.
For real perf insight use `nsight-compute`, `dcgm`, or `nvidia-smi dmon`.

---

## Chapter 3 — Interconnect: Beyond a Single GPU

### §6.15 — PCIe
Standard motherboard interconnect. PCIe Gen4 x16 = ~32 GB/s, Gen5 x16 = ~64 GB/s. Fine for one GPU. Bottleneck for multi-GPU training where gradients must sync.

### §6.16 — NVLink and NVSwitch
NVIDIA's high-bandwidth GPU-to-GPU interconnect. H100 NVLink = 900 GB/s per GPU pair. **NVSwitch** is a chip that lets 8 (or 72, in NVL72) GPUs all talk to each other at NVLink speed simultaneously. This is what makes a DGX/HGX server a "single big GPU" for tensor-parallel inference.

### §6.17 — InfiniBand
For talking *between* servers. Mellanox/NVIDIA InfiniBand HDR/NDR delivers 200/400 Gbps per port. Frontier training clusters use **fat-tree** topologies with InfiniBand so any GPU can talk to any other at full speed. Ethernet at 400/800 Gbps (RoCE) is now competitive and used by Meta and others.

### §6.18 — Why Interconnect Limits Cluster Size
Distributed training requires synchronizing gradients every step. As you add GPUs, communication grows faster than compute. At some point, adding GPUs *slows training down* — until you redesign the parallelism strategy (Module 10).

### §6.19 — The Cost of a Cluster
A single H100 SXM costs ~$30k. An 8-GPU HGX server: ~$300k. Add networking, storage, power, cooling: ~$400k. A 1,000-GPU cluster: ~$50–80M. A 100,000-GPU cluster (xAI Colossus, Meta's clusters): billions. This is why frontier AI is concentrated in a few hands.

---

## Chapter 4 — The CUDA Software Stack

### §6.20 — CUDA: Why NVIDIA Wins
**CUDA** is NVIDIA's programming model and ecosystem for GPU compute. Released 2007. Every major ML framework targets CUDA first. Competitors (AMD's ROCm, Intel's oneAPI) have struggled for years to catch up. The "CUDA moat" is software, not silicon.

### §6.21 — cuDNN, cuBLAS, NCCL
- **cuBLAS** — optimized matrix-multiply kernels.
- **cuDNN** — optimized deep learning primitives (convolution, attention, normalization).
- **NCCL** — collective communication (all-reduce, broadcast) for multi-GPU.
PyTorch and TensorFlow call into these. You almost never write CUDA directly unless you're optimizing a hot kernel like FlashAttention.

### §6.22 — PyTorch
By 2026, PyTorch is the dominant deep learning framework. Used by Meta, OpenAI, Anthropic, and everyone publishing research. **TensorFlow** survives mostly inside Google. **JAX** is gaining ground for research and Google internal use.

### §6.23 — Compilers: torch.compile, XLA, MLIR
Just-in-time compilers that fuse operations and generate optimized GPU kernels. **`torch.compile`** can give 1.5–3× speedups for free. **TensorRT-LLM** (Module 08) is essentially a heavily-optimized compiler for inference.

### §6.24 — Triton (the Language, Not the Server)
OpenAI's Python-like DSL for writing GPU kernels. Used by FlashAttention and many recent perf wins. Easier than raw CUDA, nearly as fast. Different from NVIDIA Triton **Inference Server** (Module 11) — confusingly same name.

### §6.25 — Operating Above the Stack
For 99% of users (you, probably), you call high-level APIs (Hugging Face, vLLM, Ollama) and never touch CUDA. But knowing what's underneath helps you debug "why is this slow?" — usually the answer is memory bandwidth, not compute.

---

## Chapter 5 — Practical Sizing

### §6.26 — Estimating Inference VRAM
Quick formula: **VRAM ≈ params × bytes_per_param × 1.3 + KV cache**
- 7B model in FP16: 7e9 × 2 × 1.3 ≈ 18 GB
- 7B model in INT4: 7e9 × 0.5 × 1.3 ≈ 5 GB
- KV cache for 4k context, 32 layers, 4096 hidden: ~2 GB
Add ~1 GB for framework overhead. Always benchmark.

### §6.27 — Estimating Training VRAM
Roughly **16–20× inference** for full fine-tuning with AdamW (weights × 2 + gradients × 2 + optimizer states × 8 + activations). LoRA cuts this to ~3–5×. QLoRA cuts it further by quantizing the frozen base.

### §6.28 — Choosing a GPU for Your Workload
- **Local hobbyist**: RTX 4090 (24 GB) — runs 7B–14B comfortably, 30B quantized.
- **Mac dev**: M3/M4 Max with 64+ GB unified memory — runs 70B quantized.
- **Single-server inference**: H100 80 GB or H200 141 GB.
- **Multi-GPU serving**: 8× H100 with NVLink.
- **Pretraining**: stop reading and call your sales rep.

### §6.29 — The Ladder of Capability vs Cost
| Tier | Hardware | What runs | Cost |
|------|----------|-----------|------|
| Phone | NPU + 8 GB | 1B–3B quantized | $0 |
| Laptop | 16 GB unified | 7B quantized | $0 |
| Workstation | RTX 4090 | 14B FP16, 30B Q4 | $2k |
| Mac Studio | 192 GB unified | 70B Q4–Q8 | $7k |
| Single H100 | 80 GB HBM3 | 70B FP8 | $30k or $2/hr |
| 8× H100 | 640 GB | 405B FP8, 70B FP16 | $300k or $25/hr |
| 100k H100 cluster | — | Pretrain GPT-4 class | $3B+ |

### §6.30 — Checkpoint — Explain it Back
**Quiz yourself:**
1. Why is GPU memory **bandwidth** more important than FLOPS for single-user inference?
2. What is the difference between VRAM and system RAM, and why does spilling hurt?
3. Why is CUDA a moat?
4. What does NVLink do that PCIe doesn't?
5. Estimate the VRAM needed to run Llama 3 70B in 4-bit. (~40 GB + KV cache.)

You now know the metal. Next module: how we squeeze big models into small VRAM.
