# Module 10 — Distributed Training

**Mastery: 75% → 83%** · Prerequisites: Modules 04, 06

**Goal of this module:** Understand how a 70B–600B model is trained across thousands of GPUs without falling over. By the end you'll be able to explain the collective operations (all-reduce and friends) that everything is built on, why a ring all-reduce is bandwidth-optimal, DDP, ZeRO stages and FSDP, tensor/pipeline/expert/context parallelism, and how a real run maps those onto the network's bandwidth tiers. This is the most "networking" module in the course.

**How to read this module:** **Bold** marks a definition. It builds directly on §4.17 (16 bytes per parameter), §4.24–§4.26 (the parallelism preview), and Module 06's interconnect sections (§6.16–§6.19). If "gradient" or "optimizer step" feel fuzzy, reread §1.33–§1.35 first.

---

## Chapter 1 — Why Distribute, and the Network Underneath

### §10.1 — The Two Reasons We Distribute Training
1. **Speed:** N GPUs can train up to ~N× faster. Pretraining Llama 3 70B took ~6.4M GPU-hours (§4.15); on one GPU that's 730 years.
2. **Capacity:** the model plus its training state (§4.17) doesn't fit on one GPU.

Frontier runs need both at once. A few terms used throughout:
- **Node:** one server, typically 8 GPUs joined by NVLink.
- **Rank:** the ID number of one GPU process in the job (0, 1, 2, ...). **World size:** the total number of ranks.
- **Global batch:** all examples used for one optimizer step, across every GPU. **Micro-batch:** the slice one GPU processes at a time.

### §10.2 — The Memory Anatomy of Training (Recap)
From §4.17: AdamW in mixed precision needs **16 bytes per parameter** of state (2 BF16 weights + 2 BF16 gradients + 4 FP32 master weights + 4 + 4 Adam moments), *plus activations*. Activations are the intermediate values saved during the forward pass for backprop (§1.34). They grow with micro-batch size × sequence length × layers × hidden size, and at long context they are often the biggest item.

So a 7B model needs ~112 GB of state before a single activation. That's already more than an 80 GB H100. Every technique in this module is a different way to **split state, activations, or compute across GPUs**, and each one pays for it in network traffic.

### §10.3 — The Fabric: Bandwidth Tiers
Training clusters have a strict bandwidth hierarchy. Each parallelism strategy is placed on the tier it can afford:

| Tier | Typical link (H100 era) | Bandwidth per GPU | Notes |
|------|-------------------------|-------------------|-------|
| Inside a GPU | HBM memory (§6.11) | ~3,350 GB/s | Not the network, but sets the scale |
| Inside a node | NVLink + NVSwitch (§6.17) | ~450 GB/s each way (900 GB/s total) | All 8 GPUs fully connected |
| Between nodes | InfiniBand NDR or RoCE Ethernet, 400 Gb/s (§6.18) | ~50 GB/s | Usually one NIC per GPU |
| Between pods / buildings | Spine and cross-datacenter links | Often oversubscribed | Avoid heavy traffic here |

Rough rule: **leaving the node costs ~10× in bandwidth**, and latency goes up too.

For a network engineer, the scale-out fabric will look familiar: a **non-blocking Clos (fat-tree)** of leaf and spine switches, usually **rail-optimized**. GPU 0 of every node plugs into "rail 0" leaf switches, GPU 1 into rail 1, and so on, so same-numbered GPUs across nodes are one switch hop apart. Transport is **RDMA**, where the NIC reads and writes GPU memory directly, bypassing the CPU (GPUDirect RDMA). It runs over InfiniBand, or over Ethernet as **RoCE** (RDMA over Converged Ethernet), which needs a lossless or near-lossless configuration (PFC, ECN). Meta trained Llama 3 on both: one 24,000-GPU cluster with RoCE and one with InfiniBand.

### §10.4 — Collective Communication 101
Distributed training is built on a handful of group operations called **collectives**, implemented by **NCCL** (NVIDIA's collective library, §6.23; pronounced "nickel"). Every rank calls the same operation at the same time:

| Collective | What happens | Networking analogy |
|------------|--------------|--------------------|
| **broadcast** | One rank's tensor is copied to every rank | Multicast from one source |
| **reduce** | Every rank's tensors are summed onto one rank | Aggregating counters to one collector |
| **all-reduce** | Every rank's tensors are summed, and *every* rank gets the total | Every router ends up with the same converged table |
| **all-gather** | Each rank contributes a slice, and every rank gets the full concatenation | Everyone floods their piece of the LSDB until all have the whole thing |
| **reduce-scatter** | Sum everything, but each rank keeps only *its slice* of the result | Each collector keeps totals for its own region only |
| **all-to-all** | Every rank sends a *different* slice to every other rank | Full-mesh unicast |

A key identity: **all-reduce = reduce-scatter + all-gather.** ZeRO and FSDP (Chapter 3) are built on that fact.

The art of distributed training is choosing *which* collective runs *when*, on *which* bandwidth tier, and hiding it behind compute.

### §10.5 — Ring All-Reduce: Why It Scales
The naive all-reduce sends every gradient to one "parameter server," which sums them and sends the result back. That server's link becomes the bottleneck. With N GPUs it handles about `2 × (N−1) ×` the gradient size, and gets worse as you add GPUs.

**Ring all-reduce** arranges the GPUs in a logical ring and splits each gradient into N chunks:
1. **Reduce-scatter phase** (N−1 steps): each GPU sends one chunk to its right-hand neighbor, which adds it to its own copy of that chunk. After N−1 steps, each GPU holds the *complete sum* for one chunk.
2. **All-gather phase** (N−1 more steps): pass the finished chunks around the ring until everyone has all of them.

Each GPU sends `2 × (N−1)/N ×` the gradient size, which is **under 2× the gradient size no matter how many GPUs**. Every link is busy and none is a hot spot. That's why it's called bandwidth-optimal. You'll simulate it in the lab and count the bytes.

NCCL actually picks among rings, **trees** (fewer steps, better for latency at large N), and in-network reduction (**SHARP**, where InfiniBand switches do the summing) depending on message size and topology.

### §10.6 — Latency vs. Bandwidth: The α-β Model
The standard cost model for a message: `time = α + n / β`
- **α (alpha):** fixed latency per message (software, NIC, switch hops). Microseconds.
- **n / β:** message size divided by bandwidth.

Small messages are **latency-bound** (α dominates), and large ones are **bandwidth-bound**. A ring has 2(N−1) steps, so at large N its total α adds up. That's why trees and hierarchical schemes exist. Training frameworks also **bucket** gradients (group many small tensors into ~25 MB messages) to move from the α regime into the β regime.

Worked numbers for one all-reduce of a 7B model's BF16 gradients (14 GB; a ring sends ~2× that per GPU):

| Link | Time |
|------|------|
| NVLink (~450 GB/s) | ~0.06 s |
| InfiniBand NDR (~50 GB/s per GPU) | ~0.56 s |
| 100 GbE (~12.5 GB/s) | ~2.2 s |

If a training step's compute takes ~1 second, that last row means more time talking than working. **Communication overhead is the gravity distributed training fights.**

---

## Chapter 2 — Data Parallelism

### §10.7 — Distributed Data Parallel (DDP): The Easy Mode
Put a full copy of the model and its optimizer on each GPU. Each GPU takes a different micro-batch, runs forward and backward, then **all-reduces the gradients** (summing, then dividing by N to average) so every copy holds identical gradients. Every GPU takes the same optimizer step, so the copies stay in sync. Repeat.

DDP also **overlaps** communication with compute. Backprop finishes the last layers first, so DDP starts all-reducing those gradient buckets while earlier layers are still computing. Done well, much of the network time hides behind the math.

### §10.8 — DDP Pros and Cons
| Pros | Cons |
|------|------|
| Simple: barely changes your training code (`torch.nn.parallel.DistributedDataParallel`) | Every GPU holds the *entire* 16 bytes/param of state plus activations |
| Throughput scales nearly linearly until the network saturates | Memory per GPU doesn't shrink at all when you add GPUs |
| Only one collective (all-reduce) per step, easily overlapped | Plain DDP tops out at a few billion parameters on an 80 GB GPU (80 GB ÷ 16 bytes ≈ 5B, minus activations) |

### §10.9 — Global Batch, Micro-Batches, and Gradient Accumulation
`global batch = micro-batch × gradient-accumulation steps × data-parallel GPUs`

**Gradient accumulation:** run several micro-batches forward and backward, add their gradients together, and only *then* communicate and step. This gives a big effective batch without the activation memory of a big micro-batch, and with fewer all-reduces per example. LLM runs use global batches of millions of tokens (Llama 3 405B grew its batch during training, to 16M tokens). Batch size is often **ramped up** in stages, alongside the learning-rate schedule (§4.27).

### §10.10 — Activation Checkpointing (Recomputation)
Activations can dominate memory. **Activation checkpointing** (also called gradient checkpointing; not the same as saving checkpoints to disk, §10.31) trades compute for memory. During the forward pass, keep only a few activations (say, one per transformer block) and throw the rest away. During backward, **recompute** the missing ones from the nearest saved one.

Cost: roughly one extra forward pass, about **+33% compute** if you recompute everything. Benefit: activation memory drops several-fold. Modern frameworks recompute *selectively*, only the cheap-to-recompute but memory-heavy parts (like attention internals), to get most of the savings for a few percent of extra compute.

---

## Chapter 3 — ZeRO and FSDP: Sharding the State

### §10.11 — The Insight Behind ZeRO
DDP is wildly redundant. With 64 GPUs, there are 64 identical copies of the optimizer state, gradients, and weights. **ZeRO** (Zero Redundancy Optimizer, Microsoft DeepSpeed, 2019) **shards** that state: each GPU permanently owns only 1/N of it, and pieces are fetched over the network just when needed. It's still data parallelism, since each GPU still processes different data. It just stops wasting memory.

### §10.12 — ZeRO Stage 1: Shard the Optimizer State
The 12 bytes per parameter of optimizer state (FP32 master weights + two Adam moments) are split across GPUs. Each GPU updates only *its* slice of the weights, then the updated weights are shared. Per-parameter state drops from 16 bytes to `4 + 12/N`, about **4× less** for large N, with **the same communication volume as DDP**. A free win.

### §10.13 — ZeRO Stage 2: + Shard the Gradients
Instead of all-reducing gradients (where everyone gets everything), **reduce-scatter** them, so each GPU receives only the summed gradients for the slice it owns. That's all it needs to update its slice. State drops to `2 + 14/N` bytes per parameter, about **8× less**, still with about the same traffic as DDP (it's half of an all-reduce, plus the all-gather of updated weights).

### §10.14 — ZeRO Stage 3: + Shard the Weights
Now even the BF16 weights are split. Before each layer runs (forward *and* backward), its full weights are **all-gathered** from all GPUs, used, then freed. State drops to `16/N` bytes per parameter: **memory per GPU shrinks linearly as you add GPUs**. The price is about **1.5× DDP's communication volume**, mostly hidden by **prefetching** the next layer's weights while the current layer computes. This is what makes 100B+ models trainable with "just" data parallelism.

### §10.15 — ZeRO by the Numbers
The ZeRO paper's own example, a 7.5B model on 64 GPUs (state only, no activations). You'll reproduce this in the lab:

| Strategy | State per GPU | Communication vs. DDP |
|----------|---------------|-----------------------|
| DDP | 120 GB | 1× |
| ZeRO-1 | 31.4 GB | 1× |
| ZeRO-2 | 16.6 GB | ~1× |
| ZeRO-3 / FSDP | 1.9 GB | ~1.5× |

And a 70B model under ZeRO-3: 140 GB per GPU on 8 GPUs (too much), 17.5 GB on 64 GPUs, 2.2 GB on 512.

### §10.16 — FSDP: PyTorch's Native ZeRO-3
**Fully Sharded Data Parallel** (Meta, 2021, built into PyTorch since 2022) is essentially ZeRO-3 inside PyTorch, typically sharded unit by unit (for example, one transformer block per unit). A newer redesign, **FSDP2** (2024), shards each parameter individually and works more cleanly with `torch.compile` and tensor parallelism. **HSDP** (hybrid sharding) shards *within* a node over fast NVLink and replicates *across* nodes, so only the cheaper DDP-style all-reduce crosses the slower network.

| | FSDP | DeepSpeed ZeRO (§10.26) |
|---|---|---|
| **Pros** | Built into PyTorch, composes with `torch.compile` and TP | Mature, rich features (offload, MoE, pipeline) |
| **Cons** | Fewer extras (offload is basic) | Separate library and config system |

### §10.17 — Offloading and Activation Sharding
Two more ways to squeeze memory:
- **Offload:** keep optimizer state (or even weights) in CPU RAM or on NVMe SSD, and stream them to the GPU as needed (**ZeRO-Offload**, **ZeRO-Infinity**). This lets you fine-tune huge models on few GPUs, but it's slow because PCIe (§6.16) is ~10× slower than NVLink. It's mostly a fine-tuning and research tool, not used for frontier pretraining.
- **Activation sharding:** split activations across GPUs too (sequence parallelism §10.22, context parallelism §10.23).

---

## Chapter 4 — Model Parallelism: Splitting the Model Itself

### §10.18 — When Data Parallelism Isn't Enough
ZeRO-3 shrinks *state*, but every GPU still runs whole layers on its own micro-batch, so activations and per-layer compute stay large. At thousands of GPUs, the global batch would also get too big to train well. At that point, you split the model's *computation* across GPUs. There are three axes: *within* a layer (tensor), *between* layers (pipeline), and *between* experts (expert). There's also a fourth for long sequences (context).

### §10.19 — Tensor Parallelism (TP): Split Within a Layer
Split each big weight matrix across GPUs so each computes part of every matrix multiply (§1.24). In the **Megatron-style** recipe (NVIDIA, 2019), the FFN's first matrix is split by *columns* (each GPU gets a slice of the 4×-wide hidden layer) and the second by *rows*. Then each GPU's partial results are summed with **one all-reduce**. Attention splits naturally by heads (§2.8). The result: **two all-reduces per transformer layer in the forward pass, and two more in the backward pass**.

Those are on the critical path (nothing can proceed until they finish), every layer, every micro-batch. So TP is **only practical over NVLink, within a node**, typically TP = 8 or less. Pros: cuts weights, compute, *and* activations per GPU. Cons: very chatty, and it doesn't scale beyond the fast domain. Inference engines use the same trick to fit big models (§9.29).

### §10.20 — Pipeline Parallelism (PP): Split Between Layers
Put layers 1–20 on stage 0, layers 21–40 on stage 1, and so on, like an assembly line. Only the activations at stage boundaries cross the network: point-to-point sends to the next stage, which are far lighter than TP's all-reduces. That makes PP **workable across nodes**.

The catch is the **pipeline bubble**. When a batch starts, stage 3 waits for stages 0–2. At the end, early stages sit idle while late ones finish. Splitting the batch into many micro-batches fills the pipe. With `p` stages and `m` micro-batches, the idle fraction is about:

```
bubble ≈ (p − 1) / (m + p − 1)
```

8 stages with 8 micro-batches wastes ~47%. With 32 micro-batches, ~18%. Schedules like **GPipe**, **1F1B** (one forward, one backward, interleaved to save memory), and **interleaved/virtual stages** (Megatron; DeepSeek's **DualPipe**) shrink the bubble further but never fully eliminate it.

### §10.21 — Expert Parallelism (EP) for MoE
In a Mixture-of-Experts model (§2.36), place different experts on different GPUs. After the router picks experts for each token, tokens are shipped to the GPUs that hold their experts, processed, and shipped back. That's two **all-to-all** collectives per MoE layer.
- All-to-all is the hardest pattern on a network: every rank sends to every other rank, with uneven sizes that depend on routing.
- **Load balancing** is critical. If the router sends too many tokens to one expert, that GPU becomes a straggler and everyone waits. Training adds balancing losses or bias tricks to spread the load.

DeepSeek-V3 used EP across nodes on 2,048 H800 GPUs, carefully overlapping all-to-all traffic with compute. GPT-4 is *rumored* to be MoE, but that isn't confirmed. The same idea is used to serve MoE models (§9.31).

### §10.22 — Sequence Parallelism (SP)
With TP, some operations (LayerNorm, dropout) aren't split, so every TP rank holds the full activations for them, which is redundant. **Sequence parallelism** (Megatron, 2022) splits *those* operations along the sequence dimension instead, swapping TP's all-reduce for an equivalent reduce-scatter + all-gather pair. It saves activation memory with no extra traffic. It's standard alongside TP.

### §10.23 — Context Parallelism and Ring Attention
For very long sequences (128K to 1M+ tokens), even one sequence's activations don't fit on one GPU. **Context parallelism (CP)** splits the *sequence itself* across GPUs: each holds a contiguous chunk of tokens. Attention needs every token to see every earlier token, so the K/V chunks are passed around the GPUs in a ring (**ring attention**, 2023) while each GPU computes attention for its own queries, overlapping the transfers with the math. Meta's Llama 3 paper describes using CP for its long-context training stage. Other labs (Google for Gemini's 1M-token context, for example) are believed to use similar techniques, but details aren't published.

---

## Chapter 5 — Putting It Together

### §10.24 — 3D (and 4D) Parallelism: Mapping Axes to Bandwidth Tiers
Real runs combine the axes, and each goes on the network tier it can afford:

| Axis | Traffic pattern | Where it runs |
|------|-----------------|---------------|
| Tensor (TP) | All-reduce in every layer, on the critical path | Inside a node (NVLink) |
| Context (CP) | Ring passing of K/V | Inside a node or nearby nodes |
| Pipeline (PP) | Point-to-point activations between stages | Across nodes |
| Data (DP / FSDP) | Gradient reduce-scatter/all-reduce, overlapped | Across nodes and pods, the outermost axis |

Meta's **Llama 3 405B** run is a published example: **TP = 8 × PP = 16 × DP = 128 = 16,384 H100 GPUs** (with CP = 16 added for the long-context stage, trading against DP). They called it **4D parallelism**. The world size is always the product of the axis sizes.

### §10.25 — Choosing a Strategy
A practical decision guide, from simplest to most complex:

| Situation | Use | Why |
|-----------|-----|-----|
| Model state fits on one GPU (≲ 3B params on 80 GB) | DDP | Simplest, fastest |
| Fine-tuning 7–70B on one node | FSDP or ZeRO-3 (or LoRA/QLoRA, §5.16–§5.18) | Shards state, little code change |
| Pretraining 10–100B on many nodes | FSDP/HSDP + TP=8 inside nodes | Keeps chatty traffic on NVLink |
| 100B+ dense, thousands of GPUs | TP + PP + DP (+ CP for long context) | Every axis needed; PP crosses nodes |
| MoE at scale | Add EP; watch all-to-all and load balance | Experts don't fit together |

The rule of thumb: **add complexity only when memory or batch size forces you to**, and keep the most communication-heavy axis on the fastest links.

### §10.26 — DeepSpeed
Microsoft's training library and the origin of ZeRO. It offers ZeRO-1/2/3, ZeRO-Offload/Infinity, pipeline parallelism, MoE training, and more, and is widely used through Hugging Face Accelerate. FSDP overlaps much of it and is preferred when "PyTorch-native" matters.

### §10.27 — Megatron-LM (and Megatron-Core)
NVIDIA's training framework, the source of tensor and sequence parallelism, interleaved pipeline schedules, and the distributed optimizer. Its building blocks are packaged as **Megatron-Core**, used inside NVIDIA NeMo and many lab stacks, and its kernels and ideas also show up in inference engines like vLLM and TensorRT-LLM. It was historically combined with DeepSpeed (Megatron-DeepSpeed).

### §10.28 — Higher-Level Frameworks
These wrap PyTorch with FSDP, DeepSpeed, or Megatron and provide sensible defaults:
- **Axolotl, LLaMA-Factory, Unsloth** (§5.23): the open-source fine-tuning world.
- **NVIDIA NeMo:** NVIDIA's enterprise stack.
- **TorchTitan:** PyTorch's reference for large-scale pretraining with FSDP2 + TP + PP + CP.
- **Colossal-AI:** another open option.

---

## Chapter 6 — The Real World

### §10.29 — How a Frontier Training Run Actually Looks
- 10,000–100,000+ GPUs for weeks to months (§4.16).
- **Checkpoints** saved regularly. The interval is set by how often failures happen. With optimizer state, a 405B checkpoint is several terabytes (16 bytes × 405B ≈ 6.5 TB), so saving must be fast and parallel.
- Loss curves and per-GPU health metrics on dashboards, watched around the clock (§4.28).
- A pre-planned data order, learning-rate schedule, and batch-size ramp (§4.27, §10.9).
- Spike detected → roll back → skip the bad data → resume.

It's closer to running a particle accelerator than typical software work.

### §10.30 — The Network at Scale: Congestion, ECMP, and Stragglers
Training traffic is unlike normal datacenter traffic, and that breaks some common network assumptions:
- **Elephant flows, low entropy.** A collective is a few huge, long-lived flows between the same endpoints, all starting at the same instant. Hash-based **ECMP** spreads flows, not bytes, so with few flows two elephants often land on the same uplink while others sit idle. Meta reported ECMP collisions as a problem and moved to schemes with more flows and smarter, collective-aware routing. Packet spraying and adaptive routing are other common fixes.
- **Synchronized incast.** Thousands of GPUs finish a step and hit the network at once, so buffers, PFC pauses, and ECN tuning matter.
- **Stragglers.** A collective is only as fast as the *slowest* rank. One GPU with thermal throttling, one flapping link, or one bad cable slows *all* 16,000 GPUs. Finding the straggler is a major operations job.
- **Sensitivity to tail latency.** Average link utilization means little. The p99 completion time of each collective sets the step time.

### §10.31 — The Engineering of Reliability
At this scale something always fails (§4.29: 419 unexpected interruptions in Llama 3's 54-day run). Common failures: GPU memory errors, NVLink or NIC faults, NCCL **hangs** (one rank stops responding and everyone waits), NaN losses, and **silent data corruption (SDC)**, where a chip computes wrong numbers without raising an error. Tools and practices:
- **Elastic launch** (`torchrun` / PyTorch Elastic): detect a failed rank and restart the job.
- **NCCL watchdogs** and timeouts, so a hang becomes a fast, diagnosable crash instead of a silent stall.
- **Hot-spare nodes**, automatic health checks before nodes rejoin, and SDC detectors that rerun sample computations.
- **Fast, asynchronous checkpointing** to shrink the work lost per failure.

### §10.32 — Distillation as a Training Strategy
Once you have a giant trained model, you can train smaller ones to imitate it (§5.30). Distillation is now a major stage of many pipelines. Llama 3.2 1B and 3B learned from larger Llama models' output probabilities, Qwen3's small sizes were distilled from larger Qwen3 models, and the DeepSeek-R1-Distill models learned from R1's reasoning outputs. From an infrastructure view, it adds a big *inference* job (running the teacher over billions of tokens) to the training pipeline. Labs rarely confirm whether their small API models (such as "mini" variants) are distilled, though it's widely assumed.

### §10.33 — RL Training Infrastructure
Post-training with RL (§5.7–§5.14) is a harder distributed problem than pretraining. Every step has two very different phases:
1. **Generation (rollouts):** the current model writes many answers, which is an *inference* workload. It's usually run with a fast inference engine (vLLM or SGLang, Module 08) inside the training loop.
2. **Training:** score the answers (reward model or verifier) and update the weights.

Then the new weights must be **synced** from the trainer to the inference engine, often across GPUs, every step. PPO keeps up to four models live (§5.9). GRPO drops the value model (§5.14). Long reasoning rollouts make generation the bottleneck, so frameworks increasingly run the two phases **asynchronously**. Frameworks include **TRL**, **OpenRLHF**, **veRL**, and **NeMo-RL/NeMo-Aligner**.

### §10.34 — Continuous / Online Training
Some labs are reported to update models frequently with new data, A/B test variants, and ship regular updates, so training infrastructure starts to look like a CI/CD pipeline with evaluation gates (§12.31). How continuous it really is at each lab isn't public, but the direction is clear: training is becoming a recurring operation, not a one-time event.

### 🛠️ Build It — Hands-On Lab
**Lab 1: simulate a ring all-reduce (§10.5).** Pure Python, nothing to install. Four "GPUs" each hold a different gradient. After the two phases, all four must hold the identical sum.

```python
# Simulate ring all-reduce across N "GPUs" (§10.5). Pure Python.
N = 4                       # number of GPUs in the ring
SIZE = 8                    # gradient length per GPU (must divide by N)
CHUNK = SIZE // N

# Each GPU starts with its own, different gradient vector.
grads = [[float(10 * g + i) for i in range(SIZE)] for g in range(N)]
expected = [sum(col) for col in zip(*grads)]     # what a correct all-reduce must give
sent = [0] * N              # numbers each GPU puts on the wire

def chunk(c):
    return range(c * CHUNK, (c + 1) * CHUNK)

# Phase 1: reduce-scatter. N-1 steps; each GPU passes one chunk to its right
# neighbor, which adds it in. Afterwards GPU g owns the full sum of one chunk.
for step in range(N - 1):
    msgs = []
    for g in range(N):
        c = (g - step) % N
        msgs.append(((g + 1) % N, c, [grads[g][i] for i in chunk(c)]))
        sent[g] += CHUNK
    for dst, c, data in msgs:
        for i, v in zip(chunk(c), data):
            grads[dst][i] += v

# Phase 2: all-gather. N-1 more steps passing the finished chunks around.
for step in range(N - 1):
    msgs = []
    for g in range(N):
        c = (g + 1 - step) % N
        msgs.append(((g + 1) % N, c, [grads[g][i] for i in chunk(c)]))
        sent[g] += CHUNK
    for dst, c, data in msgs:
        for i, v in zip(chunk(c), data):
            grads[dst][i] = v

print("every GPU now holds:", grads[0])
print("all GPUs identical and correct:", all(g == expected for g in grads))
print("numbers sent per GPU:", sent, f"= 2*(N-1)/N * SIZE = {2 * (N - 1) / N * SIZE:.0f}")
print("naive 'everyone sends to GPU 0, GPU 0 sends back' -> GPU 0 handles",
      2 * (N - 1) * SIZE, "numbers")

for n in (4, 8, 64, 1024):
    print(f"N={n:>4}: ring traffic per GPU = {2 * (n - 1) / n:.3f} x gradient size")
```

What you should see: every GPU holds `[60.0, 64.0, ..., 88.0]`, "all GPUs identical and correct: True," each GPU sent 12 numbers versus 48 through the single naive hub, and ring traffic per GPU levels off just under 2× the gradient size even at 1,024 GPUs. Things to try: set `N = 8` and `SIZE = 16`. Print `grads` after phase 1 to see each GPU owning one finished chunk (that intermediate state is exactly a reduce-scatter).

**Lab 2: a planner for memory, network time, and bubbles (§10.6, §10.15, §10.20).**

```python
# Back-of-the-envelope distributed-training planner (§10.6, §10.15, §10.20).

# 1) Per-GPU memory for model *state* (no activations) under DDP and ZeRO.
def state_gb(params, gpus, stage):
    w, g, opt = 2.0, 2.0, 12.0          # bytes/param: BF16 weights, BF16 grads, FP32 master+Adam m+v
    if stage >= 1: opt /= gpus           # ZeRO-1: shard optimizer state
    if stage >= 2: g /= gpus             # ZeRO-2: + shard gradients
    if stage >= 3: w /= gpus             # ZeRO-3 / FSDP: + shard weights
    return params * (w + g + opt) / 1e9

print("Per-GPU state memory, 7.5B model on 64 GPUs (the ZeRO paper's example):")
for stage, name in enumerate(["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]):
    print(f"  {name:<7} {state_gb(7.5e9, 64, stage):6.1f} GB")

print("\n70B model, ZeRO-3 / FSDP, state per GPU:")
for gpus in (8, 64, 512):
    print(f"  {gpus:>4} GPUs: {state_gb(70e9, gpus, 3):6.1f} GB")

# 2) How long does one gradient all-reduce take? (ring: each GPU sends ~2x the data)
def allreduce_seconds(params, gbytes_per_sec, bytes_per_grad=2):
    return 2 * params * bytes_per_grad / (gbytes_per_sec * 1e9)

print("\nAll-reduce of 7B BF16 gradients (14 GB):")
for link, bw in (("NVLink (H100, ~450 GB/s each way)", 450),
                 ("InfiniBand NDR 400 Gb/s (~50 GB/s)", 50),
                 ("100 GbE (~12.5 GB/s)", 12.5)):
    print(f"  {link:<36} {allreduce_seconds(7e9, bw):6.2f} s")

# 3) Pipeline bubble: idle fraction with p stages and m micro-batches (GPipe/1F1B).
print("\nPipeline bubble (fraction of time GPUs sit idle):")
for p, m in ((4, 4), (8, 8), (8, 32), (16, 128)):
    print(f"  {p:>2} stages, {m:>3} micro-batches: {(p - 1) / (m + p - 1):5.1%}")
```

You should get 120.0 / 31.4 / 16.6 / 1.9 GB for the ZeRO stages (matching §10.15), 0.06 s vs. 0.56 s vs. 2.24 s for the all-reduce, and a bubble that falls from 46.7% to 17.9% when 8 stages get 32 micro-batches instead of 8. Things to try: plan a 405B model. How many GPUs does ZeRO-3 need to get state under 20 GB per GPU? What happens to the all-reduce time over 25 GbE?

**Lab 3 (optional, real PyTorch on CPU):** a real all-reduce with no GPU needed. Install CPU-only PyTorch in a virtual environment (`pip install torch --index-url https://download.pytorch.org/whl/cpu`, a ~200 MB download), save this as `allreduce.py`, and run `torchrun --nproc_per_node=4 allreduce.py`:

```python
# Real all-reduce with PyTorch on CPU (no GPU needed).
# Run with:  torchrun --nproc_per_node=4 allreduce.py
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo")      # "nccl" on real GPUs
rank, world = dist.get_rank(), dist.get_world_size()

grad = torch.full((4,), float(rank + 1))     # each process has a different "gradient"
print(f"rank {rank} before: {grad.tolist()}")
dist.all_reduce(grad, op=dist.ReduceOp.SUM)  # sum across all processes
grad /= world                                 # DDP averages the gradients
print(f"rank {rank} after:  {grad.tolist()}")
dist.destroy_process_group()
```

Each rank starts with 1, 2, 3, or 4 and ends with the average, `[2.5, 2.5, 2.5, 2.5]`. (Lines from the four processes may print interleaved.) That one `dist.all_reduce` call is the operation DDP runs on every training step. Things to try: replace `all_reduce` with `dist.all_gather` or `dist.reduce_scatter_tensor` (check the PyTorch docs for their arguments).

### §10.35 — Checkpoint — Explain It Back
1. What are the two reasons to distribute training?
2. Why does leaving the node cost so much, and what does "rail-optimized" mean?
3. What's the difference between all-reduce, all-gather, and reduce-scatter, and how do they relate?
4. Why does a ring all-reduce send under 2× the gradient size per GPU no matter how many GPUs there are?
5. What does DDP replicate, and why does it stop scaling in memory?
6. Walk through ZeRO stages 1, 2, and 3. What does each shard, and what does stage 3 cost?
7. Why is tensor parallelism kept inside a node, while pipeline parallelism can cross nodes?
8. What is a pipeline bubble, and how do micro-batches shrink it?
9. Why are all-to-all and load balancing the hard parts of expert parallelism?
10. Why can one slow GPU or one ECMP collision slow down a 16,000-GPU run?

<details>
<summary>Answer key (try first!)</summary>

1. Speed (more GPUs finish the work sooner) and capacity (the training state doesn't fit on one GPU).
2. NVLink gives ~450 GB/s per GPU inside a node, but the scale-out network gives ~50 GB/s per GPU, about 10× less. Rail-optimized means GPU *k* of every node connects to the same "rail k" leaf switches, so same-numbered GPUs are one hop apart.
3. All-reduce sums and gives everyone the full result. All-gather concatenates everyone's slices for everyone. Reduce-scatter sums but leaves each rank only its slice. All-reduce = reduce-scatter + all-gather.
4. Each GPU sends N−1 chunks of size 1/N in each of two phases: 2(N−1)/N of the gradient, which approaches but never exceeds 2×. All links work in parallel with no hub.
5. The entire model, gradients, and optimizer state (16 bytes per parameter) on every GPU. Adding GPUs adds speed but never reduces per-GPU memory.
6. Stage 1 shards optimizer state (~4× less), stage 2 also shards gradients via reduce-scatter (~8× less), and stage 3 also shards weights, all-gathering each layer just in time (memory falls as 1/N). Stage 3 costs about 1.5× DDP's communication.
7. TP needs all-reduces inside every layer, on the critical path, so it needs NVLink bandwidth. PP only sends activations point-to-point between stages, which is much lighter.
8. Time when some stages sit idle as the pipeline fills and drains, about (p−1)/(m+p−1). More micro-batches (m) keep all stages busy more of the time.
9. Every GPU must send different, routing-dependent amounts of tokens to every other GPU, the hardest traffic pattern. If routing overloads one expert, its GPU becomes a straggler that everyone waits for.
10. Collectives are synchronous: every step waits for the slowest rank and the slowest flow. ECMP hashes a few huge flows, so two can collide on one link and set the pace for the whole job.
</details>

You now know how the model was *built*. Final stretch: how it gets to users at scale.
