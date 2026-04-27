# Month 08 — Inference Engines (Days 211–240)

**Goal of this month:** Understand the software that actually *runs* a trained model — from `ollama run` on your laptop to `vllm serve` on a 8×H100 box. By Day 240 you'll be able to recommend the right engine for any deployment scenario.

---

## Week 1 — Why Inference Needs Its Own Software

### Day 211 — Training Frameworks Aren't Inference Engines
PyTorch is fantastic for training but inefficient for serving — too much Python overhead, no batching strategy, no quantization-aware kernels, no streaming output. Specialized **inference engines** rewrote everything for one job: serve tokens as fast and cheaply as possible.

### Day 212 — The Two Phases of Inference: Prefill vs Decode
Every request has two distinct workloads:
- **Prefill** — process the entire prompt in one big parallel matmul. Compute-bound. Fast per token.
- **Decode** — generate one token at a time, autoregressively. Memory-bandwidth-bound. Slow per token.
Different optimizations apply to each. Modern engines treat them separately (see "disaggregated serving," Day 234).

### Day 213 — The KV Cache (full deep dive in Month 9)
During decode, instead of recomputing attention over the entire growing sequence, the engine **caches the K and V vectors of every past token**. Each new token only computes its own Q, K, V and attends against the cache. Without KV caching, generation would be O(n²) per token instead of O(n). Every modern engine does this.

### Day 214 — Streaming Output
Tokens are generated one at a time, so it's natural to **stream** them to the client as they're produced (Server-Sent Events / WebSocket). Perceived latency drops dramatically — users see the first word in 100ms instead of waiting 5s for the full response.

### Day 215 — The Engine Landscape (the players)
- **llama.cpp / Ollama / LM Studio** — local, CPU+GPU, GGUF.
- **vLLM** — high-throughput Python/CUDA server. Most popular open source.
- **TensorRT-LLM** — NVIDIA's max-performance C++/CUDA engine.
- **TGI** (Text Generation Inference, Hugging Face) — production-ready Rust/Python.
- **SGLang** — fast structured-generation server, growing rapidly.
- **MLC-LLM** — cross-platform (web, mobile) via TVM.
- **ExLlamaV2** — fast single-GPU consumer inference.
- **MLX** — Apple's framework, fast on M-series Macs.

---

## Week 2 — Local Inference: llama.cpp and Ollama

### Day 216 — llama.cpp: The Project That Started It All
Georgi Gerganov, 2023. Ported Meta's Llama to plain C++ so it could run on a MacBook CPU. Within months it supported GPU acceleration, every model architecture, every quantization scheme. **Today it's the engine inside Ollama, LM Studio, KoboldCpp, Jan, and dozens more.**

### Day 217 — How llama.cpp Works
- Loads a **GGUF** file (weights + tokenizer + chat template + metadata).
- Runs custom CPU/CUDA/Metal/Vulkan/ROCm kernels — no PyTorch.
- Splits layers between GPU VRAM and CPU RAM (`n_gpu_layers`) — partial offload lets you run models larger than your VRAM.
- Exposes an HTTP server (`llama-server`) with an OpenAI-compatible API.

### Day 218 — Ollama: The Friendly Wrapper
Ollama (2023) wraps llama.cpp behind a Docker-like UX:
```
ollama pull qwen3:8b
ollama run qwen3:8b
```
It manages model downloads, model versioning (Modelfiles, like Dockerfiles), automatic GPU/CPU placement, and a local REST API on `localhost:11434`. **This is how non-experts run LLMs locally.**

### Day 219 — How Ollama Picks Quantization
By default `ollama pull qwen3:8b` grabs the **Q4_K_M** GGUF — the sweet spot most users want. You can request `qwen3:8b-instruct-q8_0`, `:fp16`, etc. Tags map directly to GGUF files on `ollama.com/library`.

### Day 220 — When Ollama/llama.cpp Is the Right Choice
- Local development on laptops or workstations
- Privacy-sensitive workloads (data never leaves the machine)
- Single-user / low-concurrency
- Mac users (Metal acceleration is excellent)
- Edge/embedded deployments

### Day 221 — When Ollama/llama.cpp Is **Not** the Right Choice
- Serving many concurrent users (no continuous batching → throughput collapses)
- Production at scale
- Multi-GPU tensor-parallel inference (limited support)
- Fine-tuned LoRA hot-swapping
For these → vLLM, TGI, TensorRT-LLM.

---

## Week 3 — Production Inference: vLLM

### Day 222 — vLLM: The Open-Source Workhorse
Born at UC Berkeley, 2023. Now the de-facto open-source standard for production LLM serving. Used by everyone from indie startups to LMSYS Arena. Core innovation: **PagedAttention** (Day 223).

### Day 223 — PagedAttention
The KV cache is huge and grows per-request. Naïve allocation wastes massive amounts of VRAM (over-allocate for max possible length, fragmentation). vLLM borrowed virtual memory's **paging** idea: split the KV cache into fixed-size blocks (e.g., 16 tokens each), allocate them on demand, share blocks across requests with the same prefix. **Result: 2–4× more throughput** than the previous best engines just from better memory management.

### Day 224 — Continuous Batching
Old "static" batching: wait for all requests in a batch to finish before starting new ones. If one request needs 1000 tokens and another needs 10, the GPU sits idle. **Continuous batching** (vLLM) adds and removes requests from the running batch every step. The GPU stays at near-100% utilization. **Single biggest throughput win in modern serving.**

### Day 225 — Prefix Caching
If two requests share a prefix (system prompt, RAG context, few-shot examples), there's no reason to recompute the KV cache for that prefix — keep it cached and reuse. vLLM's automatic prefix caching can reduce TTFT (time-to-first-token) by 5–10× for repeated system prompts.

### Day 226 — Tensor Parallelism in vLLM
For models too big for one GPU, vLLM shards each weight matrix across N GPUs (`--tensor-parallel-size 4`). The matmul becomes a synchronized cross-GPU operation. Requires fast interconnect (NVLink). Standard for serving 70B+ models.

### Day 227 — Quantization in vLLM
vLLM supports BF16 (default), FP8 (great on H100), AWQ, GPTQ, and various 4-bit kernels. **FP8 on H100** is often the production sweet spot — half the VRAM, near-BF16 quality, fastest tensor-core path.

### Day 228 — Running vLLM
```
pip install vllm
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768
```
Exposes an OpenAI-compatible REST API on port 8000. Drop-in replacement for `openai.com/v1`. This pattern (an OpenAI-compatible server) is the de-facto API standard in 2026.

### Day 229 — vLLM Architecture in 30 Seconds
- **Scheduler** decides which requests run each step (prefill vs decode, prefix caching, fairness).
- **Worker** executes the model on the GPU(s).
- **KV manager** handles PagedAttention blocks.
- **API server** speaks OpenAI protocol, streams tokens.

---

## Week 4 — TensorRT-LLM, TGI, SGLang, Others

### Day 230 — TensorRT-LLM
NVIDIA's flagship inference engine. **Compiles** the model into optimized CUDA kernels ahead of time (per GPU SKU, per quantization, per batch shape). Result: typically 1.5–2× faster than vLLM on the same hardware. Cost: more complex build process, less flexibility, NVIDIA-only. Used by NVIDIA NIM, Triton, and many enterprise stacks.

### Day 231 — TGI (Hugging Face Text Generation Inference)
Hugging Face's production server. Rust frontend + Python/CUDA backend. Production-ready since 2023, easy deploy via Docker, supports continuous batching, FP8, quantization, multi-LoRA. Strong choice if you're already in the HF ecosystem.

### Day 232 — SGLang
Newer (2024) entrant focused on **structured generation** — JSON-constrained output, multi-turn programs, RAG pipelines. Often beats vLLM on throughput in benchmarks. Innovative scheduling (RadixAttention for prefix sharing, zero-overhead batching). Worth watching.

### Day 233 — DeepSpeed-MII, MLC-LLM, ExLlamaV2
- **DeepSpeed-MII** — Microsoft's serving on top of DeepSpeed Inference. Less popular than vLLM nowadays.
- **MLC-LLM** — runs models in **the browser** via WebGPU, on iOS, Android, Raspberry Pi. Cross-platform.
- **ExLlamaV2** — fast single-GPU inference for consumer hardware; loved by the local LLM community.

### Day 234 — Disaggregated Serving (the cutting edge, 2024–2025)
Insight: prefill is compute-bound, decode is memory-bound. They have *different* optimal batch sizes, *different* optimal hardware. Run prefill on one set of GPUs and decode on another, ship the KV cache between them. Used by DeepSeek, vLLM v1, and TensorRT-LLM. Big throughput wins for long-context workloads.

### Day 235 — Choosing an Engine: The Decision Tree
1. **One user, your laptop** → Ollama / LM Studio
2. **Privacy + on-prem, low concurrency** → llama.cpp server
3. **Mac M-series** → Ollama or MLX
4. **Open-source production** → vLLM (default choice)
5. **Maximum NVIDIA performance** → TensorRT-LLM
6. **HF ecosystem comfort** → TGI
7. **Heavy structured output / agents** → SGLang
8. **Browser / mobile** → MLC-LLM

### Day 236 — How OpenAI/Anthropic Probably Do It
Closed-source, but based on papers and leaks: heavily customized stacks descended from internal forks of vLLM/TensorRT-LLM ideas, with proprietary kernels, FP8/FP4 quantization, disaggregated prefill/decode, massive tensor + pipeline parallelism, custom networking. The principles are the same as the open stacks — just at much larger scale and with secret sauce.

### Day 237 — The OpenAI API Standard
Because every engine speaks OpenAI's `/v1/chat/completions` protocol, you can swap providers (OpenAI → Anthropic → vLLM → Ollama) by changing a base URL. This commoditization is one of the most important developments of 2024–2025.

### Day 238 — Speculative Decoding (preview of Month 9)
A small "draft" model proposes 4–8 tokens at a time; the big "target" model verifies them in a single forward pass. If accepted, you got those tokens almost for free. Built into vLLM, TensorRT-LLM, llama.cpp. **2–3× decode speedup**, lossless.

### Day 239 — Tool Calling / JSON Mode
Modern engines support **constrained generation** — forcing the model's output to match a JSON schema, regex, or grammar. Done via masking the next-token logits at sampling time. Critical for agentic workloads (Month 12). Available in vLLM (`outlines`, `xgrammar`), SGLang, llama.cpp (`grammars`), TGI.

### Day 240 — Recap & "Explain it Back"
**Quiz yourself:**
1. What's the difference between prefill and decode, and why does it matter?
2. What problem does PagedAttention solve?
3. Why does continuous batching beat static batching?
4. When would you choose Ollama over vLLM, and vice versa?
5. How does Ollama actually run a model — what's underneath the friendly CLI?

You can now answer "How does Ollama work?" *and* "How does OpenAI serve at scale?" in real depth. Next month: the optimization tricks behind every fast engine.
