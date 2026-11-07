# Module 06 — Hardware & Compute

**Mastery: 36% → 43%** · Prerequisites: Module 04 (helps to know what training and inference each demand)

**Goal of this module:** Understand the silicon and memory that AI runs on. By the end you'll know why VRAM is the first thing that limits what you can run, why memory *bandwidth* (not raw FLOPS) usually sets your tokens per second, how to estimate whether a model fits and how fast it can possibly go, how GPUs talk to each other, and what `nvidia-smi` is telling you.

**How to read this module:** **Bold** marks a definition. You'll reuse a few earlier ideas constantly: a matrix multiply is many dot products (§1.24), FLOPs and FLOP/s (§4.15), parameters as stored numbers (§1.30), and prefill vs. decode (§3.29). Number formats like BF16 and 4-bit are covered in Module 07. Here, just know they set how many bytes each weight takes.

---

## Chapter 1 — CPU vs GPU vs TPU vs NPU

### §6.1 — The CPU: Few Cores, Smart, Serial
A modern CPU has roughly 8–128 **cores** (independent processors on one chip). Each core is sophisticated: it predicts which way `if` statements will go, reorders instructions to avoid waiting, and has large caches. It's optimized for **latency**: finishing *one* task as fast as possible. Great for operating systems, databases, and web servers. Poor at "do this same simple multiply-add ten billion times."

### §6.2 — The GPU: Thousands of Simple Cores in Lockstep
An NVIDIA H100 has about **17,000 CUDA cores** plus specialized tensor cores (§6.5). Each core is far weaker than a CPU core, but groups of them execute the *same instruction* on different data at the same moment. NVIDIA calls this **SIMT** (single instruction, multiple threads). A GPU is optimized for **throughput**: total work per second across a huge parallel job. Originally that was shading millions of pixels; now it's matrix math.

**Network-engineer analogy:** a CPU is the router's control plane, smart and flexible and handling one complex thing at a time. A GPU is the forwarding ASIC: it does one simple operation on an enormous number of items in parallel, at line rate.

### §6.3 — Inside a GPU: SMs, Warps, and Kernels
A few words you'll see in GPU discussions:
- **SM (streaming multiprocessor):** the GPU's building block, a cluster of cores with its own small, fast memory. An H100 has 132 of them.
- **Warp:** a group of 32 threads that execute in lockstep. If threads in a warp take different branches of an `if`, they take turns, which wastes time. That's why GPU code avoids branching.
- **Kernel:** a program that runs on the GPU, launched across thousands of threads at once. "Matrix multiply" is a kernel; so is "softmax." Frameworks like PyTorch launch long sequences of kernels. Making them faster or **fusing** several into one (§9.33) is most of GPU performance engineering.

### §6.4 — Why Matrix Multiplication Is the Workload
Every transformer layer is mostly a handful of large matrix multiplies: the Q, K, V and output projections, attention's `Q·K` and `× V`, and the FFN's matrices (§2.12–§2.15). Together they're well over 90% of the FLOPs.

A handy rule: **running a model costs about 2 FLOPs per parameter per token** (one multiply and one add per weight). An 8B model needs about 16 GFLOPs to produce each token. (Training costs ~6 per parameter per token: §4.15.) Specialize silicon for matrix multiply and you win at AI. That's the premise of GPUs and TPUs in this era.

### §6.5 — Tensor Cores
Starting with NVIDIA's Volta generation (2017), GPUs added **tensor cores**: hardware units that each perform a small matrix multiply-and-accumulate (a tile like 4×4 times 4×4) as a single operation, instead of one multiply-add at a time. They're also built for low-precision formats (FP16, BF16, FP8, and on Blackwell, FP4).

The difference is dramatic on an H100 SXM:
- Regular CUDA cores: ~67 TFLOP/s (FP32)
- Tensor cores: **~989 TFLOP/s dense BF16**, ~1,979 for FP8

Spec sheets often quote a number **twice** these "with sparsity." That only applies to a special pattern of zeroed weights that real LLM workloads rarely use. **Always use the dense number for estimates.** (Module 04's 6ND math used ~1 PFLOP/s dense: §4.15.)

### §6.6 — Google TPUs
Google's custom AI chip (**Tensor Processing Unit**), in production since 2015. Its core is a **systolic array**: a grid of multiply-add units where data pulses through in rhythm, each unit passing partial results to its neighbor, so values are reused without returning to memory. TPUs train and serve Gemini, are rented out only through Google Cloud, and are used by some other labs (Anthropic has announced large TPU deployments). They're programmed mainly through JAX and XLA (§6.25) rather than CUDA.

### §6.7 — Apple Silicon, AMD, Unified Memory, and NPUs
- **Apple M-series** chips use **unified memory**: CPU, GPU, and Neural Engine share one big pool of RAM, so the GPU can use most of it. A Mac with 128 GB can hold models that no single consumer NVIDIA card can. The trade-off is bandwidth: an M4 Max is about 546 GB/s versus ~1,000–1,800 GB/s on high-end NVIDIA cards, so it fits more but generates tokens more slowly (§6.12).
- **Other unified-memory boxes** (2025): AMD's Ryzen AI Max ("Strix Halo") and NVIDIA's DGX Spark offer around 128 GB of shared memory at roughly 250–275 GB/s. They're popular for running big models slowly but affordably at home.
- **AMD Instinct MI300X** (192 GB HBM3, ~5.3 TB/s) competes with H100 and H200 on specs. Its software stack, **ROCm**, has improved a lot but still trails CUDA in polish and coverage (§6.27).
- **NPUs** (neural processing units) in phones and laptops (Apple Neural Engine, Qualcomm, Intel, AMD) are small, low-power accelerators for on-device features. They're great for efficiency, but most local-LLM tools still run mainly on the GPU or CPU.

### §6.8 — The NVIDIA Lineup
| GPU | Year | Memory | Bandwidth | Dense BF16 | Notes |
|-----|------|--------|-----------|------------|-------|
| **A100** | 2020 | 40/80 GB HBM2e | ~2.0 TB/s | ~312 TF | Workhorse of the GPT-3/GPT-4 era |
| **H100 SXM** | 2022 | 80 GB HBM3 | 3.35 TB/s | ~989 TF | The standard of 2023–2025 |
| **H200** | 2024 | 141 GB HBM3e | 4.8 TB/s | ~989 TF | Same compute as H100, much more memory and bandwidth |
| **B200** (Blackwell) | 2024–25 | ~180–192 GB HBM3e | ~8 TB/s | ~2,250 TF | Native FP4; "up to 5× H100" claims are FP4 inference marketing |
| **GB200 NVL72** | 2025 | 72 GPUs per rack | — | — | 72 Blackwell GPUs joined by NVLink into one big system |
| **RTX 3090 / 4090** | 2020 / 2022 | 24 GB GDDR6X | ~0.94 / ~1.0 TB/s | — | Consumer cards; used 3090s are a local-LLM favorite |
| **RTX 5090** | 2025 | 32 GB GDDR7 | ~1.8 TB/s | — | Current consumer flagship |

The H100 → H200 step is the lesson: **for big LLMs, more memory and bandwidth often matter more than more FLOPS.** NVIDIA's next generations (Blackwell Ultra, then Rubin) continue the pattern. Check current specs before you buy anything; this table ages fast.

---

## Chapter 2 — Memory: The Real Bottleneck

### §6.9 — The Memory Hierarchy
Computers use layers of memory, from tiny and blazing fast to huge and slow:

| Layer | Typical size | Speed (rough) |
|-------|-------------|---------------|
| Registers and on-chip SRAM per SM | KB to a few hundred KB | Fastest, essentially instant |
| GPU L2 cache | ~50 MB (H100) | Very fast |
| **VRAM / HBM** | 24–192 GB | ~1–8 TB/s |
| System RAM (DDR5) | 32 GB–2 TB | ~90 GB/s (desktop) to several hundred (servers) |
| PCIe link from GPU to system | — | ~32–64 GB/s |
| NVMe SSD | TBs | ~3–14 GB/s |

**Network-engineer analogy:** it's TCAM versus DRAM in a router. The fast table is small and expensive, the big table is slow. Good performance means keeping the hot working set in the fast layer. Much GPU optimization, including FlashAttention (§9.15), is about keeping data in on-chip SRAM instead of round-tripping to HBM.

### §6.10 — VRAM vs. System RAM
**VRAM** (video RAM) is the memory on the GPU board itself. It holds the model weights, the KV cache (§2.37), and activations while the GPU computes. **System RAM** sits on the motherboard, reachable by the GPU only across the PCIe link, which is roughly 30–100× slower than VRAM.

If the model doesn't fit in VRAM, engines like llama.cpp and Ollama can **offload** some layers to the CPU and system RAM (§8.11). It works, but speed drops sharply, because the slowest part sets the pace. One partial exception: in MoE models (§2.36), only a few experts are used per token, so keeping rarely used experts in system RAM hurts less than offloading dense layers.

### §6.11 — HBM: High-Bandwidth Memory
Data-center GPUs use **HBM**: stacks of memory chips mounted right beside the GPU on a silicon base (an **interposer**), connected by thousands of short wires. An H100's HBM3 delivers **3.35 TB/s**. A desktop with dual-channel DDR5 gets about 90 GB/s, roughly **37× less**. Consumer cards use **GDDR** memory, which sits on the board around the chip: slower than HBM (~1–1.8 TB/s), but much cheaper. HBM is made by only a few companies (SK hynix, Samsung, Micron) and packaged with TSMC's advanced packaging, and that supply is a major bottleneck for the whole industry.

### §6.12 — Memory Bandwidth Sets Your Decode Speed
The single most useful fact in this module: **for one user generating text, the GPU is waiting on memory, not math.** Each decode step (§3.29) produces one token, but has to read *every* weight from VRAM to do it. So:

```
max tokens/sec  ≈  memory bandwidth  ÷  bytes of weights read per token
```

Worked examples (theoretical ceilings; real speeds are typically 50–80% of these):

| Model | Weight bytes | GPU | Ceiling |
|-------|-------------|-----|---------|
| 8B at BF16 | 16 GB | RTX 4090 (1.0 TB/s) | ~63 tok/s |
| 8B at 4-bit | ~4.5 GB | RTX 4090 | ~224 tok/s |
| 8B at BF16 | 16 GB | H100 (3.35 TB/s) | ~209 tok/s |
| 70B at 4-bit | ~39 GB | M4 Max (546 GB/s) | ~14 tok/s |

Two big consequences:
- **Quantization (Module 07) speeds up decoding,** not just saves memory: fewer bytes to read per token means more tokens per second.
- **MoE models read only their *active* parameters** (§2.36). Qwen3-235B-A22B at 4-bit has ~132 GB of weights in total, but each token reads only about 12 GB, so it can type like a ~22B dense model, *if* you have memory to hold all 132 GB.

### §6.13 — Arithmetic Intensity and the Roofline
To tell whether a job is limited by math or by memory, compute its **arithmetic intensity**: FLOPs done per byte read from memory.

Every chip has a **ridge point** = peak FLOP/s ÷ bandwidth. For the H100: `989 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOPs per byte`.
- Below the ridge point, the job is **memory-bound**: the cores sit idle waiting for data.
- Above it, the job is **compute-bound**: the math units are the limit.

Plot "achievable performance" against intensity and you get the **roofline model**: a sloped line (limited by bandwidth) that flattens into a flat roof (limited by FLOPS).

Where LLM work lands:
- **Decode with one user:** each weight (2 bytes in BF16) is used for 2 FLOPs, so the intensity is about **1 FLOP/byte**, far below 295. Hopelessly memory-bound.
- **Prefill:** a 2,000-token prompt reuses each weight 2,000 times in one pass, so it's **compute-bound**. That's why prompt processing ("read" speed) is far faster per token than generation ("write" speed).
- **Training** with big batches: compute-bound, near the roof.

### §6.14 — Why Batching Multiplies Throughput
If one user's decode wastes most of the GPU, serve many users at once. With a batch of 32, the GPU reads the weights **once** and uses them for 32 tokens, one per user. Intensity rises to ~32 FLOPs/byte, and total throughput rises ~32×, while each user's speed barely changes. You'll see in the lab that an H100 with an 8B model stays memory-bound until a batch of roughly 300. (That simplified math ignores the KV cache, which also has to be read for every user and grows with context, so real batches hit limits sooner.)

This is *the* economic fact of LLM serving. It's why production servers always batch (continuous batching: §8.19, §9.26), and why a provider can sell tokens far cheaper than you can generate them alone at home.

### §6.15 — `nvidia-smi` Decoded
`nvidia-smi` is the first command to run on any NVIDIA box. What the fields mean:
- **Memory-Usage:** VRAM used versus total. Engines like vLLM **pre-allocate** most of the VRAM at startup (for the KV cache), so "90% used" at idle is normal.
- **GPU-Util:** the percentage of time *any* kernel was running. It is **not** "how much of the GPU's math is in use." A memory-bound decode can show 100% while the tensor cores are mostly idle. Don't read it as efficiency.
- **Pwr: Usage / Cap:** current power draw versus the limit. Hitting the cap lowers clock speeds.
- **Temp:** sustained high temperatures (typically above ~85 °C) cause **thermal throttling**.
- **Perf (P0–P12):** performance state. P0 is full speed; higher numbers are power-saving states.
- **Processes:** which programs hold VRAM. Useful when "out of memory" appears and something else is squatting on the card.

Useful variants: `nvidia-smi dmon` (a live, scrolling per-second view), `watch -n 1 nvidia-smi`, and `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv`. For deeper profiling: NVIDIA Nsight Systems / Nsight Compute, or DCGM for monitoring a fleet.

---

## Chapter 3 — Interconnect: Beyond a Single GPU

### §6.16 — Bits vs. Bytes, and PCIe
Network links are quoted in **bits** per second (Gb/s); memory and GPU buses usually in **bytes** per second (GB/s). Divide by 8: a 400 Gb/s port is 50 GB/s. Keep that straight and the numbers below line up.

**PCIe** is the standard slot connecting GPUs to the motherboard. A ×16 slot moves about **32 GB/s per direction at Gen4** and **~64 GB/s at Gen5**. That's fine for one GPU that loads its model once. But it's a bottleneck when GPUs must constantly exchange data, as in tensor-parallel inference (§9.29) or training gradient sync (§10.4). Consumer multi-GPU rigs talk over PCIe, which is why splitting one model across two gaming cards scales imperfectly.

### §6.17 — NVLink and NVSwitch
**NVLink** is NVIDIA's direct GPU-to-GPU link. An H100 has **900 GB/s of total NVLink bandwidth** (both directions, summed across its links), about 14× a PCIe Gen5 slot. Blackwell doubles it to 1.8 TB/s.

**NVSwitch** is a switch chip for NVLink. Inside an 8-GPU HGX/DGX server, it lets every GPU talk to every other at full NVLink speed simultaneously: a **non-blocking** fabric, in network terms. The GB200 **NVL72** extends one NVLink domain across a whole rack of 72 GPUs. This is what lets 8 or 72 GPUs act like "one big GPU" for tensor parallelism (§8.21).

### §6.18 — InfiniBand and RoCE: The Cluster Network
Between servers, GPUs talk over a dedicated **back-end network**, separate from the normal front-end network. This part should feel familiar:
- **InfiniBand** (NVIDIA, formerly Mellanox): 400 Gb/s (NDR) per port is common, and 800 Gb/s (XDR) is arriving. It natively supports **RDMA** (remote direct memory access): one GPU writes straight into another server's GPU memory without involving either CPU.
- **RoCE** (RDMA over Converged Ethernet): RDMA over Ethernet at 400/800 Gb/s. It needs a **lossless** fabric (PFC, ECN, careful buffer tuning), because RDMA reacts very badly to drops. Meta trained Llama 3 on both an InfiniBand cluster and a RoCE cluster of ~24,000 GPUs each.
- **Topology:** usually a non-blocking **fat-tree / Clos** fabric, typically **rail-optimized**: GPU #1 of every server connects to the same leaf switch ("rail 1"), GPU #2 to rail 2, and so on, so the most common traffic patterns cross the fewest hops.
- Each GPU typically gets its own 400 Gb/s NIC, so an 8-GPU server has ~3.2 Tb/s of back-end bandwidth.

Training traffic is bursty, synchronized, and full of large flows (all-reduce, §10.4). One slow link or congested port stalls *every* GPU waiting on it. Tail latency in the fabric directly becomes idle GPU time, and network engineers who understand ECMP hashing, congestion control, and lossless Ethernet are in demand here.

### §6.19 — Why Interconnect Limits Cluster Size
Data-parallel training (§4.25) synchronizes gradients every step. As you add GPUs, each one's compute share shrinks while communication doesn't shrink as fast. Eventually GPUs spend more time talking than computing, and adding more makes training *slower*. The fixes are smarter parallelism strategies and overlapping communication with compute (Module 10).

### §6.20 — Power and Cooling
AI hardware is limited by electricity and heat as much as by chips:
- An H100 SXM draws up to **700 W**, and a B200 about **1,000 W**. An 8-GPU server draws ~10 kW, roughly what several homes use.
- A GB200 NVL72 rack draws on the order of **120 kW** and requires **liquid cooling**. Traditional datacenter racks were designed for 5–15 kW.
- Frontier clusters are measured in **hundreds of megawatts to gigawatts**, which is why AI companies now sign power deals and build next to power plants (§12.36).
- At home: an RTX 4090 is ~450 W and a 5090 ~575 W. Plan for the power supply, the heat, and the noise.

### §6.21 — The Cost of a Cluster
Rough figures, which shift with supply and demand:
- One H100 SXM: ~$25–40k. An 8-GPU HGX server: ~$300k.
- Add networking, storage, power, and cooling: ~$400k+ per server.
- A 1,000-GPU cluster: ~$50–80M. A 100,000+ GPU cluster (xAI's Colossus, Meta's and OpenAI's largest): billions.

Cloud rental (~$2–4 per H100-hour) turns that capital cost into an hourly one. This is why frontier AI is concentrated in a few hands (§4.33).

---

## Chapter 4 — The Software Stack

### §6.22 — CUDA: Why NVIDIA Wins
**CUDA** (2007) is NVIDIA's platform for programming its GPUs: a C/C++ extension, a compiler, drivers, and a huge library ecosystem. Nearly every ML framework, kernel, and research paper targets CUDA first. Competitors have strong chips, but code written and tuned over 15+ years for CUDA doesn't move easily. The "CUDA moat" is software and developer habit, not just silicon.

### §6.23 — cuBLAS, cuDNN, NCCL
- **cuBLAS:** hand-tuned matrix-multiply kernels.
- **cuDNN:** deep-learning building blocks (convolutions, normalization, attention).
- **NCCL** ("nickel"): GPU-to-GPU communication collectives like all-reduce (§10.4), tuned for NVLink and InfiniBand topologies.

PyTorch calls these for you. You almost never write CUDA yourself, unless you're optimizing a hot kernel like FlashAttention.

### §6.24 — PyTorch (and JAX)
**PyTorch** is the dominant deep-learning framework in 2026, used by Meta, OpenAI, and most published research. You write normal-looking Python, and it runs kernels on the GPU. **JAX** (Google) is the main alternative, especially on TPUs and inside Google DeepMind. **TensorFlow** survives mostly in older production systems. Hugging Face Transformers (§5.23) sits on top of PyTorch.

### §6.25 — Compilers: torch.compile, XLA, TensorRT
Frameworks normally run operations one kernel at a time. A **compiler** looks at the whole computation, **fuses** steps together (§9.33), and generates faster kernels.
- **`torch.compile`** (PyTorch 2.x) is often 1.3–2× faster for a one-line change, depending on the model.
- **XLA** is the compiler behind JAX and TPUs.
- **TensorRT / TensorRT-LLM** is NVIDIA's heavily optimized inference compiler (§8.25).

### §6.26 — Triton (the Language, Not the Server)
OpenAI's **Triton** lets you write GPU kernels in Python-like code. It's much easier than raw CUDA and often nearly as fast. `torch.compile` generates Triton kernels under the hood, and tools like Unsloth (§5.23) use hand-written ones. Don't confuse it with **NVIDIA Triton Inference Server** (§11.3), an unrelated serving product with the same name.

### §6.27 — Beyond NVIDIA: ROCm, Metal/MLX, Vulkan
- **ROCm:** AMD's CUDA equivalent. PyTorch and vLLM support it, and it's steadily improving.
- **Metal / MLX:** Apple's GPU API and Apple's own array framework. llama.cpp's Metal backend and **MLX** are how Macs run LLMs fast.
- **Vulkan:** a cross-vendor graphics/compute API. llama.cpp can use it on almost any GPU (AMD, Intel, NVIDIA), usually slower than the vendor-specific backend.
- **CPU:** llama.cpp runs anywhere, using wide vector instructions (AVX2/AVX-512 on x86, NEON on ARM). Slow, but universal.

| Platform | Pros | Cons |
|----------|------|------|
| NVIDIA + CUDA | Everything supports it first; fastest kernels | Expensive; limited VRAM on consumer cards |
| AMD + ROCm | More VRAM per dollar | Fewer tools work out of the box; more debugging |
| Apple Silicon | Huge unified memory, quiet, efficient | Lower bandwidth; not for training at scale |
| CPU only | Runs anywhere, uses cheap RAM | Slowest by far |

### §6.28 — Operating Above the Stack
Most users call high-level tools (Ollama, vLLM, Hugging Face) and never touch CUDA. That's fine. But knowing what's underneath lets you answer "why is this slow?" The answer is usually **memory** (doesn't fit, spilled to system RAM, or bandwidth-bound), then **interconnect**, and only rarely raw compute.

---

## Chapter 5 — Practical Sizing

### §6.29 — The Sizing Rule of Thumb
**Weights in GB ≈ parameters (in billions) × bytes per parameter.**
- BF16/FP16: 2 bytes. An 8B model is ~16 GB.
- 8-bit: 1 byte. ~8 GB.
- 4-bit: ~0.5–0.6 bytes in practice (formats store extra scale values: §7.18). ~4.5–5 GB.

Then add the **KV cache** (§6.30) and roughly **1–2 GB of runtime overhead**. If the total is near your VRAM limit, it won't fit comfortably: shorten the context or pick a smaller quantization.

### §6.30 — Estimating Inference VRAM, Including the KV Cache
```
VRAM ≈ weights + KV cache + overhead
KV cache bytes = 2 × layers × KV heads × head size × bytes per value × tokens
```
(The 2 is one K and one V per token. KV heads are the shared K/V heads from GQA, §2.37.)

Worked example, **Llama 3 8B** (32 layers, 8 KV heads, head size 128, BF16 cache):
- Per token: `2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes ≈ 128 KB`
- 8K context: ~1.1 GB. 128K context: ~17 GB, *more than the BF16 weights themselves*.
- Without GQA (32 KV heads, like Llama 2 7B) it would be 4× larger: ~512 KB per token.

The cache also grows with the number of simultaneous users, since each needs its own. That's why serving engines manage it so carefully (§8.18, §9.6) and why KV-cache quantization exists (§9.9).

### §6.31 — Estimating Training VRAM
- **Full fine-tuning** with AdamW: ~16 bytes per parameter plus activations (§4.17), so **about 8× the BF16 inference weights**. An 8B model needs ~128 GB before activations: multiple GPUs.
- **LoRA:** the frozen base stays at 2 bytes per parameter, and only the tiny adapter needs gradients and optimizer state (§5.16). An 8B model fits in ~20–24 GB with modest context.
- **QLoRA:** the frozen base is stored at 4-bit (~0.5 bytes per parameter, §5.18). An 8B model fits in ~8–12 GB, and a 70B model on a single 48 GB card.

Activations grow with batch size × sequence length. **Gradient checkpointing** (§10.10) trades extra compute for much less activation memory.

### §6.32 — Choosing Hardware for Local Inference
| Option | Typical memory | Pros | Cons |
|--------|---------------|------|------|
| Used RTX 3090 | 24 GB | Cheapest 24 GB of fast VRAM | Power-hungry; older; used market risk |
| RTX 4090 / 5090 | 24 / 32 GB | Fast decode; full CUDA support | Expensive; 32 GB caps dense models at ~30B (4-bit) |
| Two consumer GPUs | 48–64 GB | Fits 70B at 4-bit | PCIe split slows things; power and cooling |
| Mac (M-series Max/Ultra) | 64–512 GB unified | Huge models, quiet, efficient | 2–4× lower bandwidth than top GPUs; slower prefill |
| Strix Halo / DGX Spark | ~128 GB unified | Large MoE models at low cost | ~250–275 GB/s: slow on big dense models |
| CPU + lots of RAM | 128 GB–1 TB+ | Fits anything | Very slow on dense models; OK for MoE with GPU help |
| Cloud GPU (H100 rental) | 80 GB+ | No up-front cost; real serving engines | Pay hourly; data leaves your network |

Rule of thumb: **VRAM decides *what* fits; bandwidth decides *how fast* it types.** For one user at home, MoE models plus lots of memory are often the sweet spot.

### §6.33 — The Ladder of Capability vs. Cost
| Tier | Hardware | What runs comfortably | Rough cost |
|------|----------|----------------------|------------|
| Phone | NPU + 8–16 GB | 1B–4B quantized | You already own it |
| Laptop | 16–32 GB | 7B–14B quantized | You already own it |
| Workstation | RTX 4090 / 5090 | 8B–14B at BF16; ~30B at 4-bit | $2–4k |
| Big-memory Mac | 128–512 GB unified | 70B at 4–8-bit; big MoE | $4–10k+ |
| Single H100 | 80 GB HBM3 | 70B at FP8 (tight); 8B for many users | ~$30k, or ~$2–4/hr |
| 8× H100 server | 640 GB | 405B at FP8; 70B at BF16 for many users | ~$300k, or ~$20–30/hr |
| 100k-GPU cluster | — | Pretraining frontier models | Billions |

### 🛠️ Build It — Hands-On Lab
**Lab 1: will it fit, and how fast can it go? (§6.12, §6.29–§6.30)** Pure Python.

```python
# Will it fit, and how fast can it possibly type?  (§6.12, §6.30)
GB = 1e9

def weights_gb(params_b, bits):
    return params_b * 1e9 * bits / 8 / GB

def kv_cache_gb(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    # 2 = one K and one V vector per layer, per KV head, per token
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / GB

gpus = {                       # name: (VRAM GB, memory bandwidth GB/s)
    "RTX 3090":     (24, 936),
    "RTX 4090":     (24, 1008),
    "RTX 5090":     (32, 1792),
    "M4 Max (Mac)": (128, 546),
    "H100 SXM":     (80, 3350),
}

# Llama 3 8B: 32 layers, 8 KV heads (GQA), head_dim 128
kv = kv_cache_gb(layers=32, kv_heads=8, head_dim=128, tokens=8192)
print(f"Llama 3 8B KV cache at 8K context: {kv:.2f} GB "
      f"({kv * GB / 8192 / 1024:.0f} KB per token)")

for label, params_b, bits in [("8B @ BF16", 8, 16), ("8B @ 4-bit", 8, 4.5), ("70B @ 4-bit", 70, 4.5)]:
    w = weights_gb(params_b, bits)
    need = w + kv + 1.0          # + ~1 GB runtime overhead
    print(f"\n{label}: weights {w:.1f} GB, ~{need:.1f} GB total with 8K context")
    for name, (vram, bw) in gpus.items():
        if need > vram:
            print(f"  {name:<13} does NOT fit ({vram} GB)")
            continue
        ceiling = bw / w         # every token reads every weight once
        print(f"  {name:<13} fits   decode ceiling ~{ceiling:4.0f} tok/s")
```

What you should see: the 8B model at BF16 fits everywhere, with a ceiling of ~63 tok/s on a 4090 and ~209 on an H100. At 4-bit, every ceiling rises about 3.5× (fewer bytes per token). The 70B model at 4-bit (~41 GB) doesn't fit any single consumer card, but fits the Mac at only ~14 tok/s. Things to try: add your own GPU, or a 128K-token context. (For the 8B model, the KV cache alone becomes ~17 GB.)

**Lab 2: the roofline in numbers (§6.13–§6.14).**

```python
# Roofline: when does a GPU stop waiting on memory?  (§6.13–§6.14)
peak_flops = 989e12      # H100 SXM, BF16 dense
bandwidth  = 3.35e12     # bytes/s
ridge = peak_flops / bandwidth
print(f"H100 ridge point: {ridge:.0f} FLOPs per byte")

params = 8e9             # an 8B model in BF16 (2 bytes per weight)
for batch in (1, 8, 32, 128, 512):
    flops = 2 * params * batch          # ~2 FLOPs per weight per token (§6.4)
    bytes_read = 2 * params             # weights are read once per step, shared by the batch
    intensity = flops / bytes_read
    time_compute = flops / peak_flops
    time_memory = bytes_read / bandwidth
    step = max(time_compute, time_memory)
    bound = "memory-bound " if time_memory > time_compute else "compute-bound"
    print(f"batch {batch:>3}: {intensity:5.0f} FLOPs/byte  {bound}  "
          f"~{batch / step:7,.0f} tok/s total, ~{1 / step:4.0f} tok/s per user")
```

What you should see: the ridge point is ~295. From batch 1 to batch 128, per-user speed stays at ~209 tok/s while *total* throughput grows 128×: free throughput from batching. At batch 512 the job becomes compute-bound and per-user speed finally drops. (This ignores KV-cache reads, which make real servers hit the limit sooner.)

**Lab 3: measure your real hardware (optional).**
1. On an NVIDIA machine, run `nvidia-smi` and find each field from §6.15. Then run `nvidia-smi dmon` in one terminal while a model generates in another.
2. With Ollama, run `ollama run llama3.1:8b --verbose` (or any model you have), ask a question, and read the **eval rate** (decode speed) and **prompt eval rate** (prefill speed) it prints. Compare the eval rate with your Lab 1 ceiling. Getting 50–80% of it is normal. Notice that prompt eval is much faster, because prefill is compute-bound (§6.13).

### §6.34 — Checkpoint — Explain It Back
1. What is a CPU optimized for versus a GPU?
2. What are tensor cores, and why should you ignore "with sparsity" FLOP numbers?
3. Why does unified memory let a Mac run bigger models, and what's the trade-off?
4. What's the difference between VRAM and system RAM, and what happens when a model spills over?
5. Why is single-user decoding memory-bandwidth-bound? Estimate the ceiling for an 8B BF16 model on a 1 TB/s GPU.
6. What is arithmetic intensity, and why is prefill compute-bound while decode is memory-bound?
7. Why does batching raise total throughput without slowing each user much?
8. What does NVLink give you that PCIe doesn't, and what is NVSwitch?
9. Why do RoCE training fabrics need to be lossless?
10. Estimate the VRAM for Llama 3 70B at 4-bit with an 8K context.

<details>
<summary>Answer key (try first!)</summary>

1. CPU: finishing one complex task fast (latency). GPU: huge amounts of simple parallel work (throughput).
2. Hardware units that do small matrix multiplies as one operation, at low precision. "Sparsity" figures assume a special zero pattern real LLMs rarely use, so they double the number you'll actually get.
3. The GPU can use most of the system's large RAM pool. But that memory has lower bandwidth than high-end VRAM, so tokens come out more slowly.
4. VRAM is on the GPU and very fast; system RAM is across PCIe and far slower. Offloaded layers run at the slower speed, so generation slows sharply.
5. Every token reads all the weights once, and the math per byte is tiny. 1,000 GB/s ÷ 16 GB ≈ 62 tok/s.
6. FLOPs per byte read from memory. Prefill reuses each weight across thousands of prompt tokens (high intensity); decode uses each weight once per token (~1 FLOP/byte).
7. The weights are read once per step and shared by every request in the batch, so many tokens come out for the same memory traffic.
8. Much higher GPU-to-GPU bandwidth (900 GB/s on H100 vs ~64 GB/s for PCIe Gen5). NVSwitch connects many GPUs so all of them can talk at full NVLink speed at once.
9. RDMA reacts very badly to packet loss, and training traffic is synchronized, so one lossy or congested link stalls every GPU.
10. ~70 × 0.56 ≈ 39 GB of weights, plus ~2.7 GB of KV cache (80 layers, 8 KV heads, head size 128, 8K tokens), plus overhead: roughly **43 GB**.
</details>

You now know the metal. Next module: how we squeeze big models into small VRAM.
