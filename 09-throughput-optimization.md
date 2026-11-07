# Module 09 — Throughput Optimization

**Mastery: 67% → 75%** · Prerequisites: Module 08

**Goal of this module:** Understand the specific tricks that make modern inference 10–100× faster than a naïve PyTorch loop. By the end you'll be able to explain *why* decode is slow (memory bandwidth, not math), size a KV cache and say how many users fit on a GPU, explain PagedAttention, FlashAttention, speculative decoding, and continuous batching, and read the flags in `vllm serve --help` or an Ollama/llama.cpp config with confidence.

**How to read this module:** **Bold** marks a definition. Chapter 1 is the one idea everything else rests on: **decode is limited by how fast you can read memory**. Every later trick either reads less memory, reads it fewer times, or does more useful work per read. If prefill/decode (§3.29) or the KV cache (§2.37) feel fuzzy, reread those first.

---

## Chapter 1 — The Core Problem: Decode Is Memory-Bound

### §9.1 — Where the Time Goes
Recall the two phases of generation (§3.29, §8.3):
- **Prefill:** the whole prompt goes through the model in one parallel pass. Lots of math per byte of weights read, so the GPU's compute units stay busy.
- **Decode:** one new token per sequence per step. Each step still has to push the input through *every* layer, which means reading *every* weight from GPU memory, just to produce one token.

For a chat answer of 500 tokens, decode is 500 back-to-back passes. That's where almost all the wall-clock time goes, and it's what this module mostly attacks.

### §9.2 — The Bandwidth Ceiling (Worked Numbers)
A GPU has two speed limits: how fast it can **compute** (FLOP/s, §4.15) and how fast it can **read memory** (bytes/s, §6.12). Which one you hit depends on **arithmetic intensity**: FLOPs done per byte read.

Decode at batch size 1, Llama 3 8B in BF16 on an H100:
- Every weight (2 bytes) is read once and used for about 2 FLOPs (one multiply, one add), so arithmetic intensity is about **1 FLOP per byte**.
- The H100 can do ~989 trillion FLOP/s but read only ~3.35 TB/s. Its break-even point is `989 ÷ 3.35 ≈ 295` FLOPs per byte.
- At 1 FLOP/byte we're ~300× below that, so the compute units sit idle waiting for memory.

So the speed limit is simply: **tokens/sec ≤ memory bandwidth ÷ bytes of weights**.

```
3.35 TB/s ÷ 16 GB of weights ≈ 210 tokens/sec  (best case, one user)
```

That single line explains a lot:
- **Quantization speeds up decode** (Module 07): 4-bit weights are ~4× fewer bytes to read, so roughly 4× the ceiling.
- **Consumer GPUs are slower at decode** mainly because of bandwidth (an RTX 4090 has ~1 TB/s), not raw FLOPs.
- **MoE models decode fast for their size** (§2.36): each token only reads the *active* experts.

(This is the "roofline model" from §6.13 in action.)

### §9.3 — Batching: The Free Lunch
Here's the trick: the weights you read on each decode step can be **shared by every sequence in the batch**. Read the 16 GB once, and use it for 64 users' next tokens instead of one. Arithmetic intensity goes up ~64×, but the step barely gets slower.

From Lab 1 (idealized numbers, Llama 3 8B, H100, each sequence ~2,000 tokens deep):

| Batch size | Step time | Total tokens/s | Tokens/s per user |
|-----------|-----------|----------------|-------------------|
| 1 | 4.9 ms | 205 | 205 |
| 8 | 5.4 ms | 1,472 | 184 |
| 32 | 7.4 ms | 4,349 | 136 |
| 128 | 15.1 ms | 8,505 | 66 |

Total throughput rises ~40×, while each user's speed drops more gently. Why does the step get slower at all? Each sequence also has to read its **own KV cache** (§9.5), and that cost grows with batch size and context length. That's why so many tricks below shrink the KV cache.

**Network-engineer analogy:** a link carrying one tiny packet per frame wastes the frame overhead; aggregating many packets into each transmission (like frame aggregation in Wi-Fi) gets far more goodput out of the same medium.

### §9.4 — The Two Goals: Throughput vs. Latency
- **Throughput:** total tokens per second across all users. It determines **cost per token**.
- **Latency:** how fast *one* user gets their answer, measured as time to first token and time per output token (§9.35).

They pull against each other. Bigger batches raise throughput but slow each user (the table above). Every optimization in this module helps one, the other, or both:

| Goal | Main levers |
|------|-------------|
| More throughput | Bigger batches (§9.26), smaller KV cache (§9.8–§9.9), PagedAttention (§9.10), prefix caching (§9.11) |
| Lower latency | Speculative decoding (§9.20), CUDA graphs (§9.32), chunked prefill (§9.27), tensor parallelism (§9.29) |
| Both | Quantization (Module 07), FlashAttention (§9.15), kernel fusion (§9.33) |

---

## Chapter 2 — The KV Cache in Depth

### §9.5 — Why the KV Cache Exists
At decode step `t`, attention needs the Key and Value vectors (§2.6) of *all* previous tokens. Without a cache, you'd recompute K and V for tokens 1…t−1 on every step. Over a 1,000-token response that's about `1 + 2 + … + 1,000 ≈ 500,000` token-recomputations per layer, instead of 1,000.

The **KV cache** stores each token's K and V the first time they're computed and reuses them for the rest of the request. Prefill fills the cache for the whole prompt; each decode step adds one token's worth. It trades memory for compute, and that memory turns out to be the hard constraint.

### §9.6 — KV Cache Size: The Real Memory Hog
Per token, per layer: `2 (K and V) × KV heads × head size × bytes per number`. Multiply by layers, tokens, and concurrent requests.

| Model | Layers | KV heads × head size | Per token (BF16) | One 32K-token request |
|-------|--------|---------------------|------------------|-----------------------|
| Llama 3 8B | 32 | 8 × 128 | 128 KB | ~4.3 GB |
| Llama 3 70B | 80 | 8 × 128 | 320 KB | ~10.7 GB |

A batch of 32 such 70B requests needs ~340 GB of KV cache, **more than twice** the 140 GB of BF16 weights. At long context and high concurrency, the KV cache, not the model, is what fills your GPUs.

### §9.7 — The KV Cache Sets Your Concurrency
How many users fit on one GPU? Roughly:

```
users ≈ (VRAM − weights − overhead) ÷ KV cache per request
```

Llama 3 8B in BF16 on an 80 GB H100: `80 − 16 (weights) − ~4 (overhead) ≈ 60 GB` free. At 1.07 GB per 8K-token request, that's about **55 users at full 8K context**. At 32K context, about 14. This is exactly what vLLM's `--gpu-memory-utilization` and `--max-model-len` flags trade off (§8.23). It's also why a model that "fits" on a 24 GB card may only serve one or two long chats.

### §9.8 — MHA, MQA, GQA, and MLA
The biggest lever on KV size is *how many K/V heads* the model has (§2.37):
- **MHA (multi-head attention):** every query head has its own K and V. The original design and the biggest cache.
- **MQA (multi-query attention):** all query heads share a *single* K/V head. The cache shrinks by a factor of the head count (e.g., 32×), with some quality risk.
- **GQA (grouped-query attention):** groups of query heads share K/V. Llama 3 70B uses 8 KV heads for 64 query heads, an 8× smaller cache than MHA; Llama 3 8B uses 8 for 32 (4×). Quality loss is minimal, so almost every modern model uses it.
- **MLA (multi-head latent attention, DeepSeek-V2/V3):** stores a small compressed vector per token and expands it into K and V when needed. DeepSeek reported a KV cache over 90% smaller than its earlier dense models.

These are *architecture* choices, baked in at training time. You can't switch a model from MHA to GQA at serving time.

### §9.9 — KV Cache Quantization
Store K and V in 8 bits (FP8 or INT8) instead of BF16: half the memory, so roughly **twice the concurrent users or twice the context**, and fewer bytes to read per decode step. Quality cost is usually small at 8 bits; 4-bit KV is riskier, especially for long-context tasks (§7.23).
- vLLM: `--kv-cache-dtype fp8`
- llama.cpp: `--cache-type-k q8_0 --cache-type-v q8_0` (needs flash attention on)
- Ollama: `OLLAMA_KV_CACHE_TYPE=q8_0` (with `OLLAMA_FLASH_ATTENTION=1`)

### §9.10 — PagedAttention: Virtual Memory for the KV Cache
A request's KV cache grows one token at a time to a length nobody knows in advance. Older engines reserved space for the *maximum* length up front, as one contiguous chunk. The vLLM paper (2023) measured that this wasted **60–80%** of KV memory: on space reserved but never used, and on gaps between chunks (**fragmentation**).

**PagedAttention** (§8.18) splits the cache into small fixed-size **blocks** (16 tokens by default in vLLM), handed out only as a request actually grows. A per-request **block table** maps "token positions 0–15, 16–31, …" to wherever those blocks physically sit. It's exactly an operating system's virtual-memory page table, and the waste drops to a few percent (just the unfilled part of each request's last block).

**Network-engineer analogy:** router packet buffers. Instead of reserving a max-MTU buffer for every packet, many platforms chain small fixed-size buffer **particles** together, so a 64-byte packet doesn't pin 9 KB of memory.

In Lab 3, the same 100 requests waste 67% of reserved memory with max-length reservation versus 0.5% with 16-token blocks, so **about 3× as many requests fit**.

### §9.11 — Prefix Caching: Never Compute the Same Prompt Twice
Because blocks are independent, two requests that start with the *same* tokens (a long system prompt, the same document, the earlier turns of a conversation) can **point at the same physical blocks**. The engine identifies each full block by a hash of its tokens (and everything before it), and on a new request reuses every block that matches. Only the new part needs prefill, and if a shared block must change, it's copied first (**copy-on-write**).

Effect: in a chat with a 2,000-token system prompt, request #100 skips that prefill entirely, so **time to first token drops sharply**, and the shared blocks are stored once. vLLM enables this by default in its V1 engine (§8.20). API providers sell the same idea as discounted "cached input tokens" (§11.21). Practical tip: put the stable parts of a prompt first and the changing parts last, so the prefix actually matches.

### §9.12 — RadixAttention (SGLang)
SGLang (§8.27) organizes all cached prefixes as a **radix tree**, a prefix tree like the one routers use for longest-prefix-match lookups. A new request walks the tree to find its longest cached prefix. That handles tree-shaped workloads naturally: many branches from one system prompt, few-shot templates, agents that fork several attempts from the same context, and multi-turn chats. Cold branches are evicted least-recently-used first.

### §9.13 — When the Cache Is Full: Preemption and Offload
KV memory is finite. When new tokens need blocks and none are free, the engine must make room:
- **Preemption:** pause a running request and free its blocks. Later, either **recompute** its KV (re-run prefill) or **swap** it back from CPU memory. Users see this as a sudden latency spike.
- **Offload tiers:** newer setups (vLLM with LMCache, NVIDIA Dynamo, and others) keep evicted prefix blocks in CPU RAM or even SSD, so a returning conversation can reload its cache instead of recomputing it.

If your server logs show frequent preemption, you're out of KV memory. Options: lower `--max-num-seqs`, shorten the max context, quantize the KV cache, or add GPUs.

---

## Chapter 3 — FlashAttention and Friends

### §9.14 — Why Standard Attention Was Slow
GPU memory is a hierarchy, much like a router's TCAM-versus-DRAM split:

| Memory | Size (A100 example) | Bandwidth |
|--------|--------------------|-----------|
| On-chip SRAM (per-SM shared memory) | ~20 MB total | ~19 TB/s |
| HBM (the "VRAM", §6.11) | 40–80 GB | ~1.5–2 TB/s |

Naïve attention computes the full `N × N` score matrix (§3.27), *writes it to HBM*, reads it back for the softmax, writes the result, and reads it again to multiply by V. For long sequences that matrix is huge (32K tokens → ~1 billion scores per head), and the GPU spends most of its time shuttling it between SRAM and HBM. The math is fine; the memory traffic is the problem.

### §9.15 — FlashAttention (2022)
Tri Dao's **FlashAttention** computes exactly the same result while never writing the `N × N` matrix to HBM:
1. **Tile:** split Q, K, and V into blocks small enough to fit in SRAM.
2. **Online softmax:** process K/V blocks one at a time, keeping a running maximum and running sum for each row. When a new block reveals a bigger maximum, rescale what you've accumulated so far. (Like keeping an EWMA of interface rate without storing every packet.)
3. **Fuse** everything (scores, softmax, weighted sum of V) into one GPU kernel (§9.33), so intermediate values never leave the chip.

Results: typically **2–4× faster** attention, and the extra memory grows *linearly* with sequence length instead of quadratically. It's exact, not an approximation. It was a key enabler of today's long context windows (together with RoPE scaling, §3.24). In llama.cpp it's `-fa`; in Ollama, `OLLAMA_FLASH_ATTENTION=1`; vLLM uses it automatically.

### §9.16 — FlashAttention-2 and FlashAttention-3
- **FlashAttention-2 (2023):** better division of work across the GPU's cores, roughly 2× faster than v1.
- **FlashAttention-3 (2024):** written for the H100's newer hardware features (asynchronous data movement, FP8). It reached about **75% of the H100's theoretical peak** in BF16, which is very high for attention.

Each new GPU generation tends to get its own tuned version. Production engines all use some FlashAttention-style kernel.

### §9.17 — Attention Kernel Libraries
You'll see these names in engine configs and logs. They're different implementations of the same idea (tile, fuse, stay on-chip):
- **FlashAttention** (the reference implementations)
- **FlashInfer** (used by SGLang and vLLM, tuned for serving with paged KV caches)
- **PyTorch SDPA** (`scaled_dot_product_attention`, which picks a fast kernel automatically)
- **xFormers** (Meta), **cuDNN** attention (NVIDIA), and TensorRT-LLM's built-in kernels

Rarely something you need to change. Mostly useful for reading logs and benchmarks.

### §9.18 — Sliding-Window (Local) Attention
Instead of attending to *all* previous tokens, each layer attends only to the last `W` tokens (Mistral 7B used W = 4,096). Compute and KV memory per layer stop growing past W. The cost: a local layer can't directly see distant tokens, so modern models **interleave** local and full ("global") layers. Gemma 2 alternates them, Gemma 3 uses five local layers per global one, and OpenAI's open-weight gpt-oss alternates as well. The global layers keep long-range recall; the local ones keep memory down.

### §9.19 — Beyond Attention: Linear Attention and SSM Hybrids
Many designs try to break the O(n²) wall (§3.27) entirely:
- **Linear attention** variants keep a fixed-size running summary instead of attending to every past token.
- **State-space models (SSMs)** such as **Mamba** process the sequence like an RNN (§1.45) with a fixed-size state, so memory doesn't grow with context, but they are designed to train in parallel like transformers.

Pure versions tend to be weaker at precise recall ("what exactly did line 37 say?"). So the practical trend (2024–2025) is **hybrids**: mostly linear/SSM layers with a few full-attention layers mixed in. Examples include AI21's Jamba, NVIDIA's Nemotron-H, MiniMax's models, and Qwen3-Next. Standard transformers still dominate, but expect more hybrids, especially for very long context.

---

## Chapter 4 — Speculative Decoding

### §9.20 — The Core Idea
Decode leaves the GPU's compute mostly idle (§9.2). Checking 5 tokens costs almost the same as generating 1, because the weights are read once either way. So:
1. A small, fast **draft model** guesses the next `k` tokens (say 4), one at a time. That's cheap.
2. The big **target model** checks all `k` guesses in **one** forward pass, in parallel, just like prefill.
3. Keep the guesses up to the first one the target disagrees with, then take the target's own token at that spot. Repeat.

If 3 of 4 guesses are accepted, you got 4 tokens (3 drafts + 1 from the target) for the cost of one big-model pass plus a few small ones.

**Network-engineer analogy:** a sliding window with go-back-N. The sender transmits several segments without waiting, the receiver acknowledges up to the first bad one, and transmission resumes from there.

### §9.21 — Why It's Lossless
With a clever accept/reject rule, the output follows *exactly* the probabilities the target model alone would give, not an approximation:
- Accept a drafted token `x` with probability `min(1, p_target(x) / p_draft(x))`.
- If rejected, sample a replacement from the probability mass the draft *under-covered*: `max(0, p_target − p_draft)`, rescaled to sum to 1.

Lab 2 checks this with 200,000 samples. The target says mat/floor/roof = 0.600/0.300/0.100; speculative decoding with a *wrong* draft (0.4/0.5/0.1) produces 0.599/0.301/0.100. (With greedy decoding it's even simpler: accept while the draft's token equals the target's top token.) You lose nothing but some wasted work on rejected drafts.

### §9.22 — How Much Faster? (The Acceptance Math)
If each drafted token is accepted with probability `a` and you draft `k` tokens, the expected tokens per target pass is `(1 − a^(k+1)) ÷ (1 − a)`. From Lab 2, with the draft costing 5% of a target pass:

| Acceptance rate | k = 2 | k = 4 | k = 8 |
|-----------------|-------|-------|-------|
| 0.5 | 1.59× | 1.61× | 1.43× |
| 0.8 | 2.22× | 2.80× | 3.09× |
| 0.9 | 2.46× | 3.41× | 4.38× |

Lessons: **acceptance rate matters more than anything**, and drafting more tokens only pays off when acceptance is high. Real-world gains are typically **1.5–3×** after overheads. Acceptance is highest for predictable text (code, structured output, quoting the prompt) and lower for creative writing. The draft model must use the **same tokenizer** as the target (for example, a Qwen3 0.6B draft for a Qwen3 32B target).

### §9.23 — Draft-Free Variants: Medusa, EAGLE, MTP, N-gram
A separate draft model is extra memory and hassle. Alternatives:
- **Medusa:** a few extra "heads" on the big model that each predict 2, 3, 4… tokens ahead.
- **EAGLE (v1–v3):** a tiny trained module that drafts from the big model's own internal vectors (§2.13) rather than from text. Among the strongest methods as of 2025.
- **MTP (multi-token prediction):** models like DeepSeek-V3 are trained with extra heads that predict further ahead, which can be reused as a built-in drafter.
- **N-gram / prompt lookup:** no model at all. Guess that the next tokens copy a sequence already in the prompt. Very effective for code edits, RAG answers that quote sources, and summarization.

### §9.24 — Self-Speculative Decoding
Use the *same* model as both drafter and verifier: draft by running only the first few layers and exiting early (methods like **LayerSkip**), then verify with the full network. No second model in memory, but it usually requires a model trained to support early exit.

### §9.25 — When Speculative Decoding Hurts
Speculation spends spare compute to cut latency. Under heavy load, a big batch has *already* used up the spare compute (§9.3), so verifying drafts that get rejected is pure waste, and throughput can drop. It also costs extra memory for the draft and its KV cache.

| | Pros | Cons |
|---|---|---|
| Speculative decoding | 1.5–3× faster per user, identical output | Little or negative gain at high batch sizes; extra memory; needs a matching draft |

Engines increasingly turn speculation down or off automatically as load rises. Rule of thumb: great for a single user or a low-traffic local server; benchmark carefully before using it on a busy one.

---

## Chapter 5 — Batching, Scheduling, and Parallelism

### §9.26 — Continuous Batching (Deep Dive)
**Static batching** (the old way) collects N requests, runs them together until *all* finish, and only then starts the next batch. Short answers finish early and their slots sit idle while the longest one drags on.

**Continuous batching** (also called iteration-level or in-flight batching, introduced by the 2022 **Orca** paper and made famous by vLLM, §8.19) re-forms the batch **every decode step**. Finished requests leave immediately, and waiting requests join as soon as there's KV memory for them.

**Network-engineer analogy:** static batching is TDM: a slot is reserved for each flow whether it has data or not. Continuous batching is statistical multiplexing: capacity goes to whoever has packets right now. Same link, far more goodput.

Each step, the **scheduler** decides who runs, based on free KV blocks, fairness, priority, and how much prefill to mix with decode. Two vLLM knobs you'll see: `--max-num-seqs` (most sequences in a batch) and `--max-num-batched-tokens` (most tokens processed per step).

### §9.27 — Chunked Prefill
A 32K-token prompt takes a long time to prefill, and if it runs in one go, every other user's next token waits behind it. That's a visible stall in their stream. **Chunked prefill** splits big prompts into pieces (say, a few thousand tokens) and runs one piece per scheduling step alongside the ongoing decodes. Latency for everyone stays smooth, at a small cost in total throughput. It's on by default in vLLM's V1 engine; the chunk size is set by `--max-num-batched-tokens`.

### §9.28 — Prefill/Decode Disaggregation
Prefill (compute-heavy, bursty) and decode (bandwidth-heavy, steady) want different things from hardware. **Disaggregation** runs them on *separate* GPU pools: prefill servers process prompts, then ship the KV cache over a fast network to decode servers that stream the tokens. Each pool can be sized and tuned for its own job, and big prompts stop interfering with streaming. The cost is complexity and fast interconnect. Used at large scale (see §8.29, §11.18; examples include NVIDIA Dynamo, llm-d, and Moonshot's Mooncake). Not something a single-server setup needs.

### §9.29 — Tensor Parallelism Inside an Engine
Split each weight matrix across GPUs (§10.19): each GPU multiplies its slice, then an **all-reduce** (§10.4) combines the partial results. That happens twice per layer, every token, so it needs NVLink-class bandwidth (§6.17). Over PCIe or Ethernet it can become the bottleneck. Typical: TP = 2, 4, or 8 inside one server (`--tensor-parallel-size`).

Pros: fits bigger models *and* lowers latency, since each GPU reads only its share of the weights per token. Cons: communication overhead, and it needs fast links.

### §9.30 — Pipeline Parallelism for Inference
Split the model **by layers**: layers 1–40 on GPU 0, 41–80 on GPU 1. Tokens flow through like an assembly line. It needs far less interconnect bandwidth than tensor parallelism (only one handoff per GPU boundary), so it works across servers on slower networks. But a single request still passes through every stage in order, so latency doesn't improve, and stages sit idle unless enough requests keep the pipeline full (**pipeline bubbles**). llama.cpp's multi-GPU "layer split" and its CPU/GPU offload work this way.

### §9.31 — Expert Parallelism (for MoE)
In an MoE model (§2.36), each token uses only a few experts. **Expert parallelism** places different experts on different GPUs and sends each token (an all-to-all exchange, §10.21) to wherever its chosen experts live. It's how very large MoEs like DeepSeek-V3 are served across many GPUs. The hard part is **load balancing**: if the router sends too many tokens to one popular expert, that GPU becomes the hotspot while others idle. Engines replicate busy experts to compensate.

---

## Chapter 6 — Making the Kernels Fast

### §9.32 — CUDA Graphs
A **kernel** is one function launched on the GPU. A decode step for a 32-layer model launches hundreds of small kernels, and each launch costs a few microseconds of CPU and driver overhead. With a tiny batch, the GPU can spend a big share of the step just waiting for the next launch. **CUDA Graphs** record the whole sequence of launches once and replay it with almost no overhead. It's a big win for decode, used by vLLM and TensorRT-LLM. (vLLM's `--enforce-eager` turns it off, which is handy for debugging and saves a little memory, but it's slower.)

### §9.33 — Kernel Fusion
Combine several operations into one kernel, so intermediate results stay on-chip instead of making a round trip through HBM (§9.14). Examples: fused normalization plus matrix multiply, computing Q, K, and V in one combined projection, and the fused gated FFN (§2.15). FlashAttention (§9.15) is the most famous fused kernel. Engines write these by hand, in CUDA, or in **Triton** (the GPU programming language, §6.26).

### §9.34 — Compilation: TensorRT-LLM, torch.compile, MAX
Instead of running one operation at a time ("eager" mode), a **compiler** looks at the whole model graph ahead of time and plans memory, fuses kernels, and picks the fastest algorithm for each matrix shape. **TensorRT-LLM** (§8.25) goes furthest down this path. **torch.compile** brings much of the benefit to ordinary PyTorch with one line of code. Modular's **MAX** is another compiler-based engine. The trade-off is longer startup ("warming up") and less flexibility.

---

## Chapter 7 — Measuring and Putting It Together

### §9.35 — Measuring Performance: TTFT, TPOT, ITL, Goodput
- **TTFT (time to first token):** from request sent to first token received. Mostly prefill plus queueing time.
- **TPOT (time per output token):** average time between tokens after the first. The "typing speed."
- **ITL (inter-token latency):** the actual gap between each pair of tokens. Look at its *spread*, since stalls show up here (§9.27).
- **E2E latency:** `TTFT + TPOT × (output tokens − 1)`.
- **Throughput:** total tokens per second across the server.
- **Goodput:** requests per second that actually *meet your latency targets*. A server at huge throughput where half the users wait 10 seconds has poor goodput.

Typical chat targets (**SLOs**, §11.8): TTFT under ~1 second, TPOT under ~50 ms (20+ tokens/s, comfortably faster than people read). Always look at the slow tail (p95/p99, §11.12), not just averages. Reasoning models generate thousands of hidden tokens, so total response time matters more than TPOT for them.

### §9.36 — Putting It All Together
A modern serving stack combines nearly everything in this module: quantized weights, a GQA model with an FP8 KV cache, PagedAttention with prefix caching, FlashAttention kernels, CUDA graphs, and continuous batching with chunked prefill. Tensor parallelism is added for big models, and speculative decoding for latency when load allows. The vLLM paper reported up to ~24× the throughput of plain Hugging Face Transformers, and 2–4× over the best serving systems of its time. Exact gains depend heavily on workload.

Which knob to reach for:

| Symptom | Likely fix |
|---------|-----------|
| Single-user generation too slow | Quantize (Module 07), speculative decoding (§9.20), faster-bandwidth GPU |
| Can't fit enough concurrent users | FP8 KV cache (§9.9), shorter max context (§9.7), more GPUs / tensor parallelism |
| Slow first token with long shared prompts | Prefix caching (§9.11), stable-prefix prompt layout |
| Other users' streams stall when a big prompt arrives | Chunked prefill (§9.27), then disaggregation (§9.28) |
| Frequent preemption in logs | Less concurrency or context, or more KV memory (§9.13) |
| GPU underused at small batch | CUDA graphs (§9.32), bigger batches if latency allows |

### 🛠️ Build It — Hands-On Lab
**Lab 1: KV cache and batching calculator (§9.2–§9.7).** Pure Python, nothing to install.

```python
# Lab 1: KV cache size, how many users fit, and why batching is a free lunch.
GB = 1e9

# Llama 3 8B in BF16 on one H100 (80 GB, 3.35 TB/s memory bandwidth, ~989 TFLOP/s BF16)
layers, kv_heads, head_dim, bytes_per = 32, 8, 128, 2
weights_bytes = 8.03e9 * 2
vram, bandwidth, peak_flops = 80 * GB, 3.35e12, 989e12

kv_per_token = 2 * layers * kv_heads * head_dim * bytes_per      # K and V
print(f"KV cache per token: {kv_per_token / 1024:.0f} KB")

context = 8192
kv_per_request = kv_per_token * context
free = vram - weights_bytes - 4 * GB          # leave ~4 GB for activations/overhead
print(f"KV per {context}-token request: {kv_per_request / GB:.2f} GB")
print(f"Requests that fit at full {context} context: {int(free // kv_per_request)}")

# Each decode step reads every weight once (shared by the whole batch)
# plus each sequence's own KV cache. Compute: ~2 FLOPs per weight per sequence.
print("\nbatch  step(ms)  tokens/s total  tokens/s per user")
ctx_now = 2048                                 # average tokens already in each sequence
for batch in (1, 8, 32, 64, 128):
    bytes_read = weights_bytes + batch * kv_per_token * ctx_now
    t_memory = bytes_read / bandwidth
    t_compute = 2 * 8.03e9 * batch / peak_flops
    step = max(t_memory, t_compute)
    print(f"{batch:>5}  {step * 1000:8.1f}  {batch / step:14,.0f}  {1 / step:17,.0f}")
```

You should see 128 KB per token, 1.07 GB per 8K request, 55 requests fitting, and total throughput climbing from ~205 to ~8,500 tokens/s as the batch grows. These are idealized upper bounds; real engines reach a good fraction of them. Things to try: set `bytes_per = 1` (an FP8 KV cache). Change the weights to 4-bit (`8.03e9 * 0.5`). Plug in your own GPU's bandwidth (an RTX 4090 is about `1.0e12`, with 24 GB of VRAM).

**Lab 2: speculative decoding (§9.21–§9.22).**

```python
# Lab 2: speculative decoding. Part A proves the accept/reject rule is lossless.
# Part B estimates the speedup.
import random

random.seed(7)

# Next-token probabilities from the big target model and the small draft model.
target = {"mat": 0.6, "floor": 0.3, "roof": 0.1}
draft  = {"mat": 0.4, "floor": 0.5, "roof": 0.1}   # the draft is a bit wrong

def pick(dist):
    return random.choices(list(dist), weights=list(dist.values()))[0]

def speculative_step():
    x = pick(draft)                                  # draft proposes a token
    if random.random() < min(1.0, target[x] / draft[x]):
        return x                                     # accept it
    # reject: sample from the leftover probability the draft under-covered
    leftover = {t: max(0.0, target[t] - draft[t]) for t in target}
    return pick(leftover)

n = 200_000
counts = {t: 0 for t in target}
for _ in range(n):
    counts[speculative_step()] += 1
print("token   target  speculative output")
for t in target:
    print(f"{t:<6}  {target[t]:.3f}   {counts[t] / n:.3f}")

# Part B: expected tokens per expensive target pass, with k drafted tokens
# and acceptance rate a: (1 - a^(k+1)) / (1 - a). Draft cost c = 5% of a target pass.
print("\naccept  k  tokens/target-pass  speedup (draft cost 5%)")
for a in (0.5, 0.7, 0.8, 0.9):
    for k in (2, 4, 8):
        tokens = (1 - a ** (k + 1)) / (1 - a)
        speedup = tokens / (k * 0.05 + 1)
        print(f"  {a:.1f}   {k}  {tokens:18.2f}  {speedup:6.2f}x")
```

Part A's output should match the target's probabilities to within ~0.001, even though the draft is wrong. Things to try: make the draft *very* wrong (`{"mat": 0.05, "floor": 0.05, "roof": 0.9}`). The output still matches, but count how often drafts are rejected.

**Lab 3: why paging wins (§9.10).**

```python
# Lab 3: why PagedAttention wins. Compare reserving max length up front
# versus handing out 16-token blocks on demand.
import random

random.seed(3)
MAX_LEN, BLOCK = 4096, 16
lengths = [random.randint(200, 2500) for _ in range(100)]   # real final lengths

used = sum(lengths)
reserved_contiguous = MAX_LEN * len(lengths)
reserved_paged = sum(-(-n // BLOCK) * BLOCK for n in lengths)  # round up to whole blocks

for name, reserved in (("contiguous (max_len)", reserved_contiguous),
                       ("paged (16-token blocks)", reserved_paged)):
    waste = 100 * (reserved - used) / reserved
    print(f"{name:<24} reserved {reserved:>7,} token slots, wasted {waste:4.1f}%")

print(f"\nSame memory, how many requests fit? "
      f"contiguous: 100, paged: {reserved_contiguous * 100 // reserved_paged}")
```

Expect about 67% waste for contiguous reservation versus 0.5% for paging, so ~3× more requests fit. Try `BLOCK = 256`: waste rises, which is why blocks are kept small.

**Optional, on real hardware:**
- **Ollama / llama.cpp:** compare tokens/s (`ollama run <model> --verbose` prints the eval rate) with and without `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`, and check how much VRAM `nvidia-smi` shows at a long context. Raise `OLLAMA_NUM_PARALLEL` and watch per-request speed fall while total speed rises (§9.3).
- **vLLM** (from Module 08): enable a draft model with `--speculative-config` (the flag's format has changed across vLLM versions, so check `vllm serve --help`), using a small same-family draft such as `Qwen/Qwen3-0.6B` for a larger Qwen3 target. Measure TPOT at 1 user and at 32 users. Try `--kv-cache-dtype fp8` and note the "KV cache capacity" line in the startup log. Send 50 requests that share a 2,000-token system prompt and compare TTFT for the first and the last. Use vLLM's built-in benchmark tool to plot TTFT and TPOT.

**Deliverable:** a table of baseline versus each optimization, on a workload that resembles your real use.

### §9.37 — Checkpoint — Explain It Back
1. Why is decode limited by memory bandwidth rather than compute, and what's the rough tokens/s ceiling formula?
2. Why does batching raise total throughput almost for free, and what eventually slows it down?
3. How big is the KV cache per token for Llama 3 8B, and how does that limit concurrent users?
4. What's the difference between MHA, MQA, GQA, and MLA?
5. What problem does PagedAttention solve, and what's the networking or OS analogy?
6. How does prefix caching speed up time to first token?
7. What does FlashAttention avoid doing, and does it change the result?
8. Walk through one cycle of speculative decoding. Why is it lossless, and when does it hurt?
9. Why is continuous batching better than static batching, and what does chunked prefill fix?
10. Define TTFT, TPOT, and goodput. Which matters most for chat?

<details>
<summary>Answer key (try first!)</summary>

1. Each decode step reads every weight to produce one token per sequence, doing only ~1 FLOP per byte. Ceiling ≈ memory bandwidth ÷ weight bytes (e.g., 3.35 TB/s ÷ 16 GB ≈ 210 tokens/s).
2. The weights read each step are shared by every sequence in the batch. Eventually each sequence's own KV-cache reads (and then compute) dominate, and each user's speed drops.
3. 128 KB per token (BF16), so ~1 GB per 8K-token request. Free VRAM after the weights ÷ KV per request ≈ how many users fit (~55 at 8K on an 80 GB H100).
4. MHA: each query head has its own K/V. MQA: all heads share one. GQA: groups share, the common compromise. MLA: store a compressed vector and expand it into K/V. Each shrinks the cache more than the last.
5. Reserving maximum-length contiguous space wastes 60–80% of KV memory. Fixed-size blocks handed out on demand, tracked by a block table, are like OS virtual-memory pages or router buffer particles.
6. Requests sharing the same starting tokens reuse the already-computed KV blocks, so that part of the prompt skips prefill.
7. It never writes the N×N score matrix to slow HBM; it tiles, uses an online softmax, and fuses everything in on-chip SRAM. The result is mathematically identical.
8. A draft proposes k tokens; the target checks them all in one pass; keep tokens up to the first rejection, then take the target's token. The accept/reject rule exactly reproduces the target's distribution. It hurts at high load, where there's no spare compute to spend on drafts.
9. Static batching waits for the longest request; continuous batching refills freed slots every step (like statistical multiplexing). Chunked prefill splits big prompts so they don't stall other users' streams.
10. TTFT: time to the first token. TPOT: average time per later token. Goodput: requests per second that meet the latency targets. For chat, TTFT and TPOT both matter, and goodput is the real measure of a healthy server.
</details>

You now understand the optimization toolbox. Next: how those big models got *trained* across thousands of GPUs.
