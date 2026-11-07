# Module 08 — Inference Engines

**Mastery: 50% → 57%** · Prerequisites: Modules 06–07

**Goal of this module:** Understand the software that actually *runs* a trained model, from `ollama run` on your laptop to `vllm serve` on an 8×H100 server. By the end you'll be able to estimate how fast a model *can* run on given hardware, explain PagedAttention and continuous batching, and recommend the right engine for a deployment.

**How to read this module:** **Bold** marks a definition. It builds directly on prefill vs. decode (§3.29), the KV cache and GQA (§2.37), sampling (§3.31–§3.33), and quantization formats (Module 07). Chapter 1 is the theory every engine shares; Chapters 2–4 tour the engines.

---

## Chapter 1 — What an Inference Engine Does

### §8.1 — Training Frameworks Aren't Inference Engines
PyTorch is fantastic for training but a poor fit for serving: lots of Python overhead per step, no strategy for juggling many users at once, few kernels for quantized weights, and no built-in streaming. An **inference engine** is software rewritten for one job: turn prompts into tokens as fast and cheaply as possible. (A **kernel** is a small, highly optimized program that runs on the GPU, such as "multiply these two matrices" or "do attention" (§6.23).)

### §8.2 — The Life of a Request
What happens between hitting Enter and seeing words appear:

```
1. HTTP request arrives (JSON: messages, temperature, max_tokens)
2. Apply the chat template (§5.3), tokenize (§3.1)
3. Scheduler puts the request in a queue, waits for free GPU memory
4. PREFILL: run the whole prompt through the model in one pass,
   filling the KV cache (§8.6)
5. DECODE loop, one token per step:
      forward pass → logits → sampling (§3.31) → new token
      → detokenize → stream it to the client (§8.7)
6. Stop on EOS / stop sequence / max_tokens (§3.35), free the KV cache
```

Every engine in this module does these same steps. They differ in how cleverly they handle steps 3–5 when hundreds of requests arrive at once.

### §8.3 — The Two Phases: Prefill vs. Decode
From §3.29, every request has two very different workloads:
- **Prefill** processes the entire prompt in one big parallel pass. It's **compute-bound** (limited by how many math operations per second the GPU can do) and fast per token. It sets the **time to first token (TTFT)**.
- **Decode** generates one token per forward pass. It's **memory-bandwidth-bound** (limited by how fast weights can be read from memory, §6.12) and slow per token. It sets the "typing speed."

They want different things from the hardware, so engines optimize them separately, and the most advanced setups even run them on different GPUs (§8.29).

### §8.4 — Why Decode Is Slow: The Bandwidth Math
For every single decoded token, the GPU must read **every weight in the model** from memory once. The math per weight is trivial; the *reading* is the bottleneck. So at batch size 1:

```
max tokens/sec ≈ memory bandwidth (GB/s) ÷ model size (GB)
```

| Hardware (bandwidth) | 8B @ BF16 (16 GB) | 8B @ Q4_K_M (~5 GB) | 70B @ Q4_K_M (~42 GB) |
|----------------------|------|------|------|
| Dual-channel DDR5 PC (~80 GB/s) | 5 | 16 | 2 |
| Apple M-series Max (~400 GB/s) | 25 | 80 | 10 |
| RTX 4090 (~1,008 GB/s) | 63 | 202 | 24* |
| H100 SXM (~3,350 GB/s) | 209 | 670 | 80 |

\*Only if it fit, which it doesn't: a 4090 has 24 GB.

These are **upper bounds**. Real speeds are maybe 50–80% of them. But the table explains a lot:
- **Quantization speeds up decode** almost in proportion to the size reduction, because there are fewer bytes to read (§7.29).
- **A Mac with a big unified memory pool** runs large models acceptably, because its bandwidth is high for a non-GPU machine.
- **CPU-only inference is slow**, mainly because system RAM bandwidth is low, not because CPUs are bad at math.

You'll compute this table yourself in the lab.

### §8.5 — Batching: The Free Lunch
If reading the weights is the bottleneck, here's the trick: read them *once* and use them for many requests at the same time. With 32 requests in a batch, one pass over the weights produces 32 tokens, one per request, at nearly the same cost as producing one.

- **Throughput** (total tokens per second across all users) rises almost linearly with batch size, until the GPU finally runs out of compute or memory.
- **Per-user speed** drops a little as the batch grows.

This is why a serving GPU can handle dozens of users at once, and why the central problem of a production engine is *keeping the batch full* (§8.19) while *fitting everyone's KV cache in memory* (§8.18). Network analogy: one packet header lookup is expensive, so routers process packets in bursts to amortize the fixed per-lookup cost.

### §8.6 — The KV Cache and How Big It Gets
During decode, rather than recomputing the keys and values of the entire growing conversation for every new token, the engine **stores the K and V vectors of every past token** (§2.37) and computes only the new token's Q, K, V. That turns each step from "reprocess everything" into "process one token and look up the rest." Every modern engine does this. Module 09 goes deeper (§9.5–§9.9).

The cost is memory. Per token, the cache holds one K and one V vector per layer per KV head:

```
bytes per token = 2 (K and V) × layers × KV heads × head dim × bytes per value
Llama 3 8B:      2 × 32 × 8 × 128 × 2 bytes ≈ 131 KB per token
```

| Context | KV cache (one request, BF16) |
|---------|------------------------------|
| 8,192 tokens | ~1.1 GB |
| 32,768 tokens | ~4.3 GB |
| 131,072 tokens | ~17 GB, more than the 16 GB of weights |

Multiply by the number of concurrent users and the KV cache, not the weights, becomes the main limit on how many people one GPU can serve. That's why GQA (§2.37), KV cache quantization (§9.9), and PagedAttention (§8.18) matter so much.

### §8.7 — Streaming Output
Tokens come out one at a time, so engines **stream** them to the client as they're produced, usually with **Server-Sent Events (SSE)**: one HTTP response that stays open and delivers small chunks:

```
data: {"choices":[{"delta":{"content":"EX"}}]}
data: {"choices":[{"delta":{"content":"START"}}]}
data: {"choices":[{"delta":{"content":" is"}}]}
...
data: [DONE]
```

The total time is the same, but *perceived* latency drops from "wait 8 seconds for the whole answer" to "first word in a few hundred milliseconds."

### §8.8 — The Numbers You'll See on Every Benchmark
- **TTFT (time to first token):** how long until the first token appears. Mostly prefill, plus queueing.
- **TPOT / ITL (time per output token / inter-token latency):** the gap between tokens during decode. 20 ms per token ≈ 50 tokens/s per user.
- **Throughput:** total output tokens per second across all requests. This is what sets cost per token.
- **Concurrency:** how many requests are in flight at once.

There's a built-in tension: bigger batches raise throughput but also raise each user's TPOT. Module 09 (§9.25, §9.35) and Module 11 (§11.6) go deeper.

### §8.9 — The Engine Landscape
| Engine | Made by | Sweet spot |
|--------|---------|-----------|
| **llama.cpp** | Open source (Georgi Gerganov + community) | Local, CPU+GPU, GGUF files |
| **Ollama** | Ollama Inc. | Easiest local experience, built on llama.cpp's libraries |
| **LM Studio** | LM Studio | Desktop GUI, uses llama.cpp and MLX |
| **MLX / mlx-lm** | Apple | Fast on Apple Silicon Macs |
| **vLLM** | Open source (started at UC Berkeley) | The default open-source production server |
| **SGLang** | Open source (started at LMSYS) | High-throughput serving, prefix-heavy and structured workloads |
| **TensorRT-LLM** | NVIDIA | Maximum performance on NVIDIA GPUs |
| **TGI** | Hugging Face | Production server in the HF ecosystem |
| **ExLlamaV2/V3** | Open source | Fast single-user inference on consumer NVIDIA GPUs |
| **MLC-LLM / WebLLM** | Open source | Browsers (WebGPU), phones, edge devices |

---

## Chapter 2 — Local Inference: llama.cpp and Ollama

### §8.10 — llama.cpp: The Project That Started It All
In March 2023, shortly after Llama's weights leaked (§2.29), Georgi Gerganov ported the model to plain C/C++ so it could run on a MacBook CPU. Within months it supported GPU acceleration, almost every open model architecture, and a wide range of quantization formats. Its tensor library, **GGML**, and its file format, **GGUF** (§7.17), now power Ollama, LM Studio, KoboldCpp, Jan, and many more.

### §8.11 — How llama.cpp Works
- Loads a single **GGUF** file containing the quantized weights, tokenizer, chat template, and metadata.
- Runs its own kernels for CPU (AVX, NEON), CUDA, Metal (Mac), Vulkan, ROCm (AMD), and more. No PyTorch, no Python required.
- **Partial offload:** puts as many layers as fit in GPU VRAM (`-ngl` / `--n-gpu-layers`) and runs the rest from system RAM on the CPU. This lets you run models bigger than your VRAM, but every token must pass through the slow CPU part, so speed falls toward the CPU's bandwidth limit (§8.4). It's like a link aggregation where one member is a slow link: the whole flow slows down.
- Ships `llama-server`, an HTTP server with an OpenAI-compatible API (§8.32). It has several parallel slots, continuous batching, prompt caching, grammar-constrained output, and LoRA adapter support.

### §8.12 — Ollama: The Friendly Wrapper
Ollama (2023) wraps that machinery in a Docker-like experience:

```
ollama pull qwen3:8b
ollama run qwen3:8b
```

It handles downloads, a local model library, **Modelfiles** (like Dockerfiles: base model + template + parameters + optional LoRA `ADAPTER`), automatic GPU/CPU placement, loading and unloading models on demand, and a REST API on `localhost:11434`. That includes an OpenAI-compatible `/v1` endpoint. It's built on llama.cpp's GGML libraries; since 2025 Ollama has also run some model types (notably multimodal ones) on its own engine code on top of GGML. **This is how most people run LLMs locally.**

### §8.13 — How Ollama Picks Quantization
`ollama pull qwen3:8b` downloads the tag's default file, usually **Q4_K_M** (§7.18), the size/quality sweet spot for most people. You can ask for others by tag, such as `qwen3:8b-q8_0` or `qwen3:8b-fp16`. The available tags are listed on each model's page at `ollama.com/library`. Some newer models ship a different default (for example, models released natively in a 4-bit format), so check the page. Rule of thumb: the file size is roughly the VRAM the weights need, plus room for the KV cache (§8.6).

### §8.14 — Ollama Knobs That Actually Matter
| Setting | What it does | Why you care |
|---------|--------------|-------------|
| `num_ctx` (per request, in a Modelfile, or app setting) | Context length allocated | The default has historically been small (2,048, later 4,096). Long prompts get **silently truncated** unless you raise it, and a bigger value uses more memory (§8.6) |
| `OLLAMA_NUM_PARALLEL` | Requests served at once per model | Higher = more throughput, but each slot needs its own KV cache |
| `OLLAMA_KEEP_ALIVE` | How long a model stays loaded after use (default 5 minutes) | Avoids reload delays, but holds VRAM |
| `OLLAMA_FLASH_ATTENTION=1` | Uses FlashAttention kernels (§9.15) | Less memory at long context, often faster |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Quantizes the KV cache (§9.9) | Roughly halves KV memory, with a small quality cost |

`ollama ps` shows what's loaded and how it's split between GPU and CPU. "100% GPU" is what you want. Anything with a CPU share will be much slower (§8.11).

### §8.15 — Other Local Options
- **LM Studio:** a desktop GUI for browsing, downloading, and chatting. It uses llama.cpp, and MLX on Macs, and can run a local OpenAI-compatible server.
- **MLX / mlx-lm:** Apple's array framework. It's often the fastest option on Apple Silicon and uses its own MLX-format model files.
- **ExLlamaV2 / ExLlamaV3:** NVIDIA-only engines with their own EXL2/EXL3 formats (§7.20). Very fast single-user decode on consumer cards, a favorite of the local-LLM community.

### §8.16 — When llama.cpp / Ollama Is (and Isn't) the Right Choice
| Good fit | Poor fit |
|----------|----------|
| Local development on a laptop or workstation | Serving dozens or hundreds of concurrent users |
| Privacy: data never leaves the machine | Tight per-user speed targets under heavy load |
| One or a few users | Large multi-GPU tensor-parallel deployments |
| Macs (Metal support is excellent) | Squeezing maximum throughput out of datacenter GPUs |
| Running models bigger than VRAM via CPU offload | |
| Edge and embedded devices, mixed hardware | |

To be fair, llama.cpp's server *does* batch concurrent requests, and Ollama can serve several in parallel. What they lack, compared with vLLM or SGLang, is the memory management and scheduling that keep throughput high at large concurrency (§8.18–§8.20). For that, use vLLM, SGLang, or TensorRT-LLM.

---

## Chapter 3 — Production Inference: vLLM

### §8.17 — vLLM: The Open-Source Workhorse
Born at UC Berkeley's Sky Computing Lab in 2023, vLLM became the de facto open-source standard for production LLM serving, used by startups, clouds, and research projects alike. It supports nearly every popular open model on day one. Its founding idea was PagedAttention (§8.18). In 2025 it moved to a rewritten core ("V1") with a simpler, faster scheduler.

### §8.18 — PagedAttention
**The problem:** each request's KV cache grows token by token, and you don't know in advance how long the answer will be. Older engines reserved one *contiguous* chunk of GPU memory per request, sized for the maximum possible length. Most of it sat empty, and leftover gaps (**fragmentation**) couldn't be reused. The vLLM paper measured that existing systems used only about 20–40% of their KV cache memory for actual tokens.

**The fix**, borrowed from how operating systems manage RAM (**virtual memory paging**):
- Split the KV cache into small fixed-size **blocks** (16 tokens each by default).
- Give each request blocks *on demand* as it grows, from anywhere in GPU memory.
- Keep a per-request **block table** mapping "token positions 0–15 → block 812, 16–31 → block 27, ...".
- Requests that share the same prefix can point at the *same* blocks (§8.20).

**Network analogy:** router buffer memory carved into small fixed-size particles or cells that a packet chains together, instead of one max-MTU buffer per packet. Small packets stop wasting big buffers.

**Result:** waste dropped to a few percent, so far more requests fit in memory at once, and the batch (§8.5) gets bigger. The paper reported **2–4× higher throughput** than the best previous systems at the same latency.

### §8.19 — Continuous Batching
**Static batching** (the old way): collect a batch of requests, run them together until *all* finish, then start the next batch. If one request needs 1,000 tokens and the others need 10, those slots sit idle for 990 steps.

**Continuous batching** (also called **iteration-level scheduling**, introduced by the Orca paper in 2022 and made mainstream by vLLM): the scheduler re-decides the batch **every single step**. Finished requests leave immediately, and waiting requests join in the freed slots.

**Network analogy:** static batching is TDM, where each slot is reserved for its owner even when idle. Continuous batching is statistical multiplexing: capacity goes to whoever has traffic right now.

In the lab, a simple simulation of 12 requests on 4 slots finishes in **240 steps at 36% utilization** with static batching versus **100 steps at 86%** with continuous batching. Together with PagedAttention, it's the single biggest throughput win in modern serving (§9.26).

### §8.20 — Prefix Caching
Many requests start with the *same* tokens: a long system prompt, the same RAG documents, the same few-shot examples, or the earlier turns of a chat. Recomputing their KV cache every time is wasted prefill. **Automatic prefix caching** keeps the KV blocks for recently seen prefixes and reuses them, so only the new part of the prompt needs prefill. For long, repeated prefixes, it can cut TTFT many times over. SGLang's RadixAttention (§9.12) is a smarter version of the same idea. API providers sell it as "prompt caching" (§11.21).

### §8.21 — Tensor Parallelism in vLLM
For models too big for one GPU, `--tensor-parallel-size 4` splits each weight matrix across 4 GPUs (§10.19). Every layer then needs the GPUs to exchange partial results, so this needs a fast interconnect such as NVLink (§6.17). Over PCIe alone, the communication can eat much of the gain. It's standard for serving 70B+ models in BF16 or FP8. vLLM also supports pipeline parallelism (§9.30) and expert parallelism for MoE models (§9.31).

### §8.22 — Quantization in vLLM
vLLM runs BF16 by default and supports FP8, AWQ, GPTQ, and several other 4-bit and 8-bit formats (Module 07). **FP8 on H100-class GPUs** is often the production sweet spot: about half the memory of BF16 (so more room for KV cache, §8.6), near-BF16 quality, and the fastest tensor-core path (§7.6). GGUF support exists but is not vLLM's strength; for GGUF files, llama.cpp is the native home.

### §8.23 — Running vLLM
```
pip install vllm
vllm serve Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768
```

This serves an OpenAI-compatible API on port 8000 (§8.32). Flags worth knowing:
- `--gpu-memory-utilization`: the fraction of VRAM vLLM claims. Weights load first, and **everything left becomes KV cache blocks**, which sets how many requests fit at once.
- `--max-model-len`: the longest context allowed per request. Lower it if the startup log says there isn't enough KV cache memory.
- `--tensor-parallel-size`: the number of GPUs to split across (§8.21).
- `--enable-prefix-caching`: turns on §8.20 (on by default in recent versions).

### §8.24 — vLLM Architecture in 30 Seconds
- **API server:** speaks the OpenAI protocol and streams tokens (§8.7).
- **Scheduler:** every step, decides which requests run, mixing prefill chunks and decode steps (**chunked prefill**, §9.27), and handles waiting queues and pre-emption when memory runs out.
- **KV cache manager:** hands out and frees PagedAttention blocks (§8.18) and tracks cached prefixes (§8.20).
- **Workers:** one per GPU, running the model forward passes with optimized kernels and CUDA graphs (§9.32).

---

## Chapter 4 — TensorRT-LLM, TGI, SGLang, and the Rest

### §8.25 — TensorRT-LLM
NVIDIA's high-performance inference library. Its classic workflow **compiles** the model ahead of time into an optimized "engine" for a specific GPU type, precision, and batch shape, with fused kernels (§9.33). Newer versions add a PyTorch-based workflow that makes this much easier.

- **Pros:** often the fastest option on NVIDIA hardware, with first access to NVIDIA's newest formats (FP8, FP4 on Blackwell).
- **Cons:** NVIDIA-only, with a more complex build and deploy process and less flexibility. The speed gap over vLLM and SGLang varies by model and workload, and has narrowed over time, so benchmark your own case.

It's used inside NVIDIA's NIM containers and Triton deployments (§11.3).

### §8.26 — TGI (Hugging Face Text Generation Inference)
Hugging Face's production server: a Rust router in front of Python/CUDA model code. It added continuous batching, quantization, and multi-LoRA early, deploys easily with Docker, and powers HF's own inference products. It's convenient if you live in the Hugging Face ecosystem. **Check the project's current activity before choosing it**: by 2025 much of the community's momentum had moved to vLLM and SGLang, and Hugging Face itself supports those as backends.

### §8.27 — SGLang
Started at LMSYS (the team behind Chatbot Arena) in 2024. Two ideas:
- **RadixAttention** (§9.12): stores cached prefixes in a tree, so any shared prefix (system prompts, multi-turn chats, branching agent calls) is reused automatically.
- **A low-overhead scheduler** that keeps the GPU busy, plus fast constrained output (§8.31).

It's frequently at or near the top of throughput benchmarks, especially for prefix-heavy agent and RAG workloads, and it's used to serve some of the largest open MoE models, such as DeepSeek-V3/R1. Today vLLM and SGLang are the two leading open-source production engines. They borrow ideas from each other constantly.

### §8.28 — DeepSpeed-MII, MLC-LLM, and Friends
- **DeepSpeed-MII:** Microsoft's serving layer on top of DeepSpeed Inference. Less widely used than vLLM nowadays.
- **MLC-LLM / WebLLM:** compiles models to run in **the browser** (WebGPU), on iOS and Android, and on small devices. Great for "no server at all" demos.
- **ExLlamaV2/V3:** see §8.15.

### §8.29 — Disaggregated Serving (the Cutting Edge, 2024–2026)
Prefill is compute-bound and decode is memory-bound (§8.3), and running both on the same GPUs means they interfere: a big prefill stalls everyone's decode for a moment. **Disaggregated serving** runs prefill on one pool of GPUs and decode on another, then ships each request's KV cache across the network between them. Each pool can then use its ideal batch size, parallelism, and even GPU type.

The idea appeared in 2024 research (DistServe, Splitwise, and Moonshot AI's Mooncake for its Kimi service). It's now supported in vLLM, SGLang, and TensorRT-LLM, and orchestrated by frameworks like NVIDIA Dynamo and llm-d. DeepSeek described serving its models this way. The trade-off: moving gigabytes of KV cache needs a very fast network (InfiniBand or RDMA, §6.18), and it's only worth the complexity at large scale. More in §11.18.

### §8.30 — Speculative Decoding (Preview of Module 09)
Decode is slow because each token needs its own pass over the weights (§8.4). **Speculative decoding** guesses ahead: a small, fast **draft** model (or extra prediction heads) proposes several tokens, and the big **target** model checks all of them in *one* forward pass. That pass costs about the same as generating one token, because decode is memory-bound. Correct guesses are kept; at the first wrong one, the target's own token is used instead. The output is **exactly** what the big model would have produced, just faster, often 1.5–3× for predictable text like code. It's built into vLLM, SGLang, TensorRT-LLM, and llama.cpp. Details in §9.20–§9.24.

### §8.31 — Constrained Generation: Tool Calling and JSON Mode
Agents and apps need output that a program can parse, such as valid JSON matching a schema. **Constrained generation** guarantees it at sampling time (§3.32): at each step, the engine works out which tokens are *allowed* by the schema or grammar, and sets every other token's logit to `−∞` before sampling (the same trick as the causal mask, §2.10).

Example: the schema says the next field is `"status"`, which must be `"up"` or `"down"`. After `{"status": "`, every token except those starting `up` or `down` is masked out, so the model *cannot* produce `"maybe"`.

It's available as vLLM's structured outputs (via XGrammar, Outlines, and others), SGLang, llama.cpp grammars (GBNF) and JSON schemas, Ollama's `format` parameter, and TGI. Tool calling (§12.21) is built on the same machinery.

### §8.32 — The OpenAI-Compatible API Standard
Almost every engine and many hosted providers expose OpenAI's `/v1/chat/completions` request format:

```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "What is OSPF EXSTART?"}],
  "temperature": 0.7,
  "stream": true
}
```

So the same client code can talk to OpenAI, vLLM, SGLang, llama.cpp, LM Studio, or Ollama by changing only the **base URL** (and API key). Many other providers offer OpenAI-compatible endpoints alongside their native APIs. This commoditization of the interface was one of the most important developments of 2024–2025 (§11.2). Network analogy: it's like every vendor agreeing to speak BGP. What's under the hood can differ wildly, but they interoperate.

### §8.33 — How Closed Labs Probably Do It
OpenAI, Anthropic, and Google don't publish their serving stacks, so treat this as informed inference from papers, job postings, and public talks. They most likely use: heavily customized in-house engines built on the same ideas as this module (paged KV cache, continuous batching, prefix caching), custom kernels, low-precision weights (FP8/FP4), speculative decoding, disaggregated prefill/decode, large-scale tensor, pipeline, and expert parallelism, and custom networking and hardware (including Google's TPUs, §6.6). The principles match the open stacks; the scale, tuning, and hardware differ.

### §8.34 — Choosing an Engine
| Situation | Good default | Why |
|-----------|-------------|-----|
| One user, your laptop | Ollama or LM Studio | Easiest; runs GGUF on CPU and GPU |
| Mac (M-series) | Ollama, LM Studio, or MLX | Metal and MLX are well optimized |
| Model bigger than your VRAM | llama.cpp / Ollama | CPU offload (expect slower speed) |
| Small team, on-prem, a few users | llama.cpp server or Ollama, then vLLM as you grow | Simple first; scale when concurrency rises |
| Open-source production on NVIDIA/AMD GPUs | vLLM or SGLang | Paged KV cache, continuous batching, broad model support |
| Prefix-heavy agents or RAG, structured output | SGLang (or vLLM) | RadixAttention, fast constrained decoding |
| Maximum NVIDIA performance, stable model set | TensorRT-LLM | Compiled, fused kernels |
| Browser or mobile | MLC-LLM / WebLLM | WebGPU, on-device |

### 🛠️ Build It — Hands-On Lab
**Lab 1: how fast can it possibly go? (§8.4, §8.6).** Pure Python, nothing to install.

```python
# How fast can decode possibly go, and how big is the KV cache? (§8.4, §8.6)

def max_tokens_per_sec(model_gb, bandwidth_gb_s):
    # Batch of 1: every new token must read all the weights once.
    return bandwidth_gb_s / model_gb

def kv_cache_gb(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    # 2 = one K vector and one V vector, per layer, per KV head, per token.
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / 1e9

hardware = {"Dual-channel DDR5 PC": 80, "Apple M-series Max": 400,
            "RTX 4090": 1008, "H100 SXM": 3350}
models = {"8B @ BF16": 16.0, "8B @ Q4_K_M": 5.0, "70B @ Q4_K_M": 42.0}

print("Upper bound on single-user decode speed (tokens/s):")
print(f"{'':22}" + "".join(f"{m:>14}" for m in models))
for hw, bw in hardware.items():
    print(f"{hw:22}" + "".join(f"{max_tokens_per_sec(gb, bw):>14.0f}" for gb in models.values()))

print("\nLlama 3 8B KV cache (32 layers, 8 KV heads, head dim 128, BF16):")
print(f"  per token: {kv_cache_gb(32, 8, 128, 1) * 1e6:.0f} KB")
for tokens in (8_192, 32_768, 131_072):
    print(f"  {tokens:>7,} tokens -> {kv_cache_gb(32, 8, 128, tokens):5.1f} GB")
```

You should get the table from §8.4 and about 131 KB per token of KV cache (~17 GB at 128K tokens). Things to try: add your own GPU's memory bandwidth (it's on the spec sheet) and the size of a model file you actually run. Then compare with the real tokens/s that `ollama run <model> --verbose` reports as "eval rate." How close to the upper bound do you get?

**Lab 2: static vs. continuous batching (§8.19).** Pure Python.

```python
# Static vs continuous batching on one GPU with room for 4 requests at a time (§8.19).
# Each number is how many tokens that request needs to generate.
requests = [100, 10, 10, 10, 80, 20, 20, 20, 60, 5, 5, 5]
SLOTS = 4

def static_batching(reqs):
    steps = busy = 0
    for i in range(0, len(reqs), SLOTS):
        batch = reqs[i:i + SLOTS]
        steps += max(batch)          # the whole batch waits for its longest member
        busy += sum(batch)           # slot-steps doing useful work
    return steps, busy

def continuous_batching(reqs):
    queue, running, steps, busy = list(reqs), [], 0, 0
    while queue or running:
        while queue and len(running) < SLOTS:   # refill free slots every step
            running.append(queue.pop(0))
        steps += 1
        busy += len(running)
        running = [r - 1 for r in running if r > 1]
    return steps, busy

for name, fn in (("static", static_batching), ("continuous", continuous_batching)):
    steps, busy = fn(requests)
    print(f"{name:>10}: {steps:3} steps, slot utilization {100 * busy / (steps * SLOTS):.0f}%")
```

Expected: static takes 240 steps at 36% utilization; continuous takes 100 steps at 86%. Things to try: make all requests the same length (say all 20). Does continuous batching still win? (No. It wins because real output lengths vary wildly.) Then change `SLOTS` to 8.

**Lab 3 (optional, with real engines):**
1. **Local:** with Ollama running, try the OpenAI-compatible endpoint:
   ```
   curl http://localhost:11434/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "qwen3:8b", "messages": [{"role": "user", "content": "Explain OSPF EXSTART in one sentence."}]}'
   ```
   Run `ollama ps` to check the GPU/CPU split, and `ollama run qwen3:8b --verbose` to see prompt eval (prefill) versus eval (decode) speeds.
2. **Cloud:** rent a single GPU with 24 GB or more for an hour, `pip install vllm`, and run `vllm serve Qwen/Qwen3-8B`. Same API, now on port 8000.
3. **Same client, two backends:** with `pip install openai`, create `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`, then switch `base_url` to the vLLM server. That's the standard from §8.32.
4. **Benchmark:** run `vllm bench serve` (or a simple script firing concurrent requests) at concurrency 1, 8, and 32 against both. Plot total throughput versus concurrency, and watch which one keeps scaling.

**Deliverable:** your estimated-vs-measured tokens/s table plus a throughput-vs-concurrency chart.

### §8.35 — Checkpoint — Explain It Back
1. Walk through the life of a request from HTTP call to streamed tokens.
2. What's the difference between prefill and decode, and what limits each?
3. Why is single-user decode speed roughly bandwidth ÷ model size? What does that predict for quantized models?
4. Why does batching raise throughput almost for free?
5. How big is Llama 3 8B's KV cache per token, and why does the KV cache limit concurrency?
6. What problem does PagedAttention solve, and how?
7. Why does continuous batching beat static batching?
8. What does prefix caching save, and when does it help most?
9. When would you choose Ollama/llama.cpp over vLLM or SGLang, and vice versa?
10. How does constrained generation guarantee valid JSON?

<details>
<summary>Answer key (try first!)</summary>

1. Request arrives → chat template + tokenize → scheduler queues it → prefill fills the KV cache → decode loop (forward pass, sample, detokenize, stream) → stop on EOS/stop sequence/max tokens → free the cache.
2. Prefill processes the whole prompt in parallel and is limited by compute. Decode makes one token per pass and is limited by memory bandwidth.
3. Each token requires reading every weight once, and the math is cheap compared with the reading. Quantized models have fewer bytes to read, so they decode proportionally faster.
4. The weights are read once per step no matter how many requests share that step, so each extra request adds little cost.
5. About 131 KB per token (2 × 32 layers × 8 KV heads × 128 × 2 bytes), ~17 GB at 128K. Every concurrent request needs its own cache, so memory runs out before compute does.
6. Contiguous worst-case reservations wasted most KV memory. PagedAttention gives out small fixed-size blocks on demand, tracked by a block table, and can share blocks across requests.
7. Static batching makes everyone wait for the longest request, leaving slots idle. Continuous batching swaps requests in and out every step, keeping the GPU full.
8. It reuses the KV cache for repeated prompt beginnings (system prompts, RAG context, chat history), skipping that prefill. It helps most with long, frequently repeated prefixes.
9. Ollama/llama.cpp: local, private, few users, Macs, models bigger than VRAM. vLLM/SGLang: many concurrent users on datacenter GPUs, where throughput and memory management matter.
10. At each step, the engine masks out every token the schema or grammar wouldn't allow (sets its logit to −∞), so only valid continuations can be sampled.
</details>

You can now answer "How does Ollama work?" *and* "How does a big provider serve millions of users?" in real depth. Next module: the optimization tricks inside every fast engine.
