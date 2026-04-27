# Month 10 — Distributed Training (Days 271–300)

**Goal of this month:** Understand how a 70B–600B model is trained across thousands of GPUs without falling over. By Day 300 you'll be able to explain DDP, FSDP, ZeRO stages, tensor parallelism, pipeline parallelism, and **3D parallelism** — the recipe behind every frontier model.

---

## Week 1 — Why Distribute?

### Day 271 — The Two Reasons We Distribute Training
1. **Speed** — N GPUs train ~N× faster (in theory).
2. **Capacity** — the model + gradients + optimizer state don't fit on one GPU.
Often both apply. Frontier training uses 10,000+ GPUs for both reasons combined.

### Day 272 — The Memory Anatomy of Training
Per parameter (with AdamW in mixed precision):
- **2 bytes** weights (BF16)
- **2 bytes** gradients (BF16)
- **4 bytes** master weights (FP32)
- **4 bytes** Adam first moment (FP32)
- **4 bytes** Adam second moment (FP32)
= **16 bytes per parameter** for state alone. Plus activations (depends on batch × seq_len × layers — often dominant).
A 7B model needs ~112 GB just for state. Doesn't fit on a 80 GB H100.

### Day 273 — Collective Communication 101
Distributed training is built on a few primitives implemented by NCCL:
- **broadcast** — one GPU's tensor → all GPUs
- **reduce** — all GPUs' tensors → summed on one GPU
- **all-reduce** — all GPUs' tensors → summed, copy on every GPU
- **all-gather** — concat tensors from every GPU → copy on every GPU
- **reduce-scatter** — sum, then split slices to different GPUs
The art of distributed training is choosing which primitive happens *when*.

### Day 274 — Latency vs Bandwidth in Collectives
NCCL all-reduce on 8 GPUs over NVLink takes microseconds for small tensors, milliseconds for big ones. Across nodes via InfiniBand, it's 10–100× slower. **Communication overhead is the gravity that distributed training fights.**

---

## Week 2 — Data Parallelism (DDP)

### Day 275 — Distributed Data Parallel (DDP) — The Easy Mode
Copy the entire model + gradients + optimizer state to each GPU. Each GPU processes a different batch slice. After the backward pass, **all-reduce** the gradients so every GPU has the same average. Take an optimizer step. Repeat.

### Day 276 — DDP Pros and Cons
**Pros:** simple, scales linearly with GPU count up to communication limits.
**Cons:** the *entire* model + optimizer state lives on every GPU. A 70B model = 1.1 TB of state per GPU = impossible. DDP alone tops out around 10B params.

### Day 277 — Gradient Accumulation
Use **micro-batches** to simulate a larger global batch without more memory: process 4 small batches, accumulate gradients, do one optimizer step. Standard trick to combine with any parallelism strategy.

### Day 278 — Gradient Checkpointing (Activation Recomputation)
Activations can dominate training memory. Trade compute for memory: don't save all activations during forward; instead, **recompute them** during backward. ~30% slowdown, often **5–10× memory reduction**. Universal in modern training.

---

## Week 3 — ZeRO and FSDP

### Day 279 — The Insight Behind ZeRO
DDP wastes memory: every GPU holds the *same* optimizer state, gradients, and weights. What if we **shard** them across GPUs and only gather what's needed when needed? That's **ZeRO** (Zero Redundancy Optimizer), from Microsoft DeepSpeed (2019).

### Day 280 — ZeRO Stage 1: Shard Optimizer States
Each GPU only holds 1/N of the optimizer states. Gradients still all-reduced as in DDP. **~4× memory reduction** for AdamW. Almost no extra communication. Free win.

### Day 281 — ZeRO Stage 2: + Shard Gradients
Each GPU only ever holds gradients for its slice. Implemented via **reduce-scatter** (instead of all-reduce). **~8× memory reduction**. Slight communication change.

### Day 282 — ZeRO Stage 3: + Shard Weights
Each GPU holds only its slice of the *weights too*. During forward/backward, the layer's full weights are temporarily **all-gathered** just before use, then freed. **Linear memory scaling with N GPUs** — finally enables 100B+ models with data parallelism alone. More communication overhead, partly hidden by overlap.

### Day 283 — FSDP: PyTorch's Native ZeRO-3
**Fully Sharded Data Parallel** (Meta, 2021) is essentially ZeRO-3 baked into PyTorch. By 2024 it's the standard PyTorch path for large model training. DeepSpeed remains popular but FSDP is built-in and easier to integrate with `torch.compile`, modern checkpointing, etc.

### Day 284 — Activation Sharding (FSDP+TP, ZeRO-Infinity)
Even with ZeRO-3, activations can blow up at long context. Further tricks shard activations across GPUs (sequence parallelism) or offload optimizer states to CPU RAM / NVMe (ZeRO-Infinity). Diminishing returns but enables truly absurd scales.

---

## Week 4 — Model Parallelism: TP and PP

### Day 285 — When Data Parallelism Isn't Enough
For *very* large models, even ZeRO-3 plus activation tricks isn't enough — or the per-GPU compute becomes too small to be efficient. You need to split the model itself. Two ways to slice:

### Day 286 — Tensor Parallelism (TP): Split Within a Layer
Split each weight matrix across GPUs (e.g., split a 4096×4096 matmul into 4 GPUs each doing 4096×1024). Each forward pass requires an **all-reduce** *inside every layer*. **Very communication-heavy** — only viable across GPUs with NVLink-class bandwidth (within a single node, typically). Also called **Megatron-style** parallelism (NVIDIA paper).

### Day 287 — Pipeline Parallelism (PP): Split Between Layers
Layers 1–20 on GPU 0, 21–40 on GPU 1, etc. Tokens flow through. Communication is just sending activations between adjacent stages — much less than TP. **Pipeline bubbles** (idle time at start/end of each batch) hurt efficiency; clever scheduling (1F1B, GPipe, Megatron-LM interleaved) reduces but doesn't eliminate them. Works across slower interconnects.

### Day 288 — Expert Parallelism (EP) for MoE
For Mixture-of-Experts models, route each token to the GPUs holding its chosen experts. Requires an **all-to-all** collective (every GPU sending a slice of tokens to every other). Load balancing is critical — bad routing kills throughput. Used by Mixtral, DeepSeek, GPT-4 (rumored).

### Day 289 — Sequence Parallelism (SP)
Shard the **sequence dimension** across GPUs for layers that are otherwise data-replicated (LayerNorm, dropout). Saves activation memory at long context. Standard in modern Megatron and FSDP.

### Day 290 — Context Parallelism / Ring Attention
For *very* long contexts (1M+ tokens), shard the attention computation itself across GPUs in a ring pattern, passing K/V chunks around. Enables training on sequences too long to fit on any single GPU. Used for Gemini 1M-token context, Llama 4, etc.

---

## Week 5 — 3D Parallelism and DeepSpeed

### Day 291 — Combining All Three: 3D Parallelism
Frontier training uses **DP × TP × PP** simultaneously. For example, a 70B run on 1024 H100s might use:
- TP=8 (within-node, NVLink)
- PP=8 (across nodes, InfiniBand)
- DP=16 (replicated training across pods)
Total: 8 × 8 × 16 = 1024 GPUs. Each axis is tuned to a different bandwidth tier of the network.

### Day 292 — DeepSpeed
Microsoft's training library. Pioneered ZeRO. Provides ZeRO-1/2/3, pipeline parallelism, MoE training, ZeRO-Infinity, ZeRO-Offload, and more. Still widely used (especially in HuggingFace/Accelerate stacks). FSDP overlaps in functionality and is preferred when "PyTorch-native" matters.

### Day 293 — Megatron-LM
NVIDIA's training framework. Source of tensor parallelism, sequence parallelism, distributed optimizer, and many of the tricks now in vLLM and TensorRT-LLM. Often combined with DeepSpeed (Megatron-DeepSpeed) for big training runs.

### Day 294 — NeMo, Axolotl, Colossal-AI, LLaMA-Factory
Higher-level frameworks that wrap PyTorch+FSDP/DeepSpeed/Megatron with sensible defaults. **Axolotl** and **LLaMA-Factory** dominate the open-source fine-tuning world; **NeMo** is NVIDIA's enterprise stack; **Colossal-AI** is a strong open competitor.

### Day 295 — How a Frontier Training Run Actually Looks
- 10,000–100,000 GPUs
- Months of compute
- Continuous monitoring; failures every few hours
- Checkpoint every ~30 min (TBs per checkpoint)
- Loss curves on a dashboard, watched 24/7
- Pre-planned data shuffling, learning rate schedule, batch size warmup
- Spike detection → rollback → skip bad data → resume
This is closer to running a particle accelerator than typical software work.

### Day 296 — The Engineering of Reliability
Network errors, GPU memory errors, NaN losses, NCCL hangs, bad nodes. Tools: **PyTorch Distributed Elastic** (auto-restart), **ECMP-aware schedulers**, **silent data corruption (SDC) detectors**. Meta's Llama 3 paper documented 419 hardware interruptions in a 54-day run.

### Day 297 — Distillation as a Training Strategy
Once you have a giant trained model, you can train smaller models to mimic it (Day 148). Distillation is now a major part of the pipeline — Llama-3.2 small models, Qwen-2.5 small models, gpt-4o-mini and many others are fundamentally distilled.

### Day 298 — RLHF Training Infrastructure
RLHF/DPO/RL on reasoning has its own distributed training pain. You may need to keep 4 models live (policy, reference, reward, value), run inference and training in the same loop, and synchronize across thousands of GPUs. Frameworks: **TRL**, **OpenRLHF**, **veRL**, **NeMo-Aligner**.

### Day 299 — Continuous / Online Training
Some labs continuously update their models on new data, A/B test variants, and ship updates daily. The training infrastructure starts to look like a CI/CD pipeline. Expect this to become more common as inference-derived data improves base models.

### Day 300 — Recap & "Explain it Back"
**Quiz yourself:**
1. What memory does training need that inference doesn't?
2. Walk through ZeRO stages 1, 2, 3.
3. What's the difference between tensor parallelism and pipeline parallelism, and where does each excel?
4. What is 3D parallelism?
5. Why does training a frontier model require an entire SRE/HPC team?

You now know how the model was *built*. Final stretch: how it gets to users at scale.
