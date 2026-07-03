# The A-to-Z of AI — A 0% → 100% Mastery Guide

A complete, self-paced, low-math curriculum to help you **understand and explain** how modern AI and Large Language Models (LLMs) actually work — from the 1950s symbolic era through to today's GPU-accelerated inference servers like vLLM, Ollama, and OpenAI's stack.

> **Goal:** When someone asks you "How does Ollama / Qwen / OpenAI work?" — you can explain it confidently, end-to-end, without hand-waving.

This is not a timeline. There are no "months" or "days." It's a **mastery ladder**: 12 modules, each a foundation-plus-deep-dive on one subject, ordered so every module stands on the ones before it. Think of it as a guru bootcamp — everything Karpathy would put on the whiteboard, from the perceptron to PagedAttention, in one place. Go as fast or slow as you want; a section a sitting finishes it in roughly a semester, a section an hour finishes it in two intense weeks.

---

## How to Use This Guide

- **One numbered section (§) at a time.** Each is a self-contained 5–15 minute read.
- **Order matters, pace doesn't.** Module 08 (inference engines) only makes sense once you understand Module 06 (hardware) and Module 07 (quantization). Within that constraint, binge or sip freely.
- Every module ends with a **Checkpoint — "Explain it Back"**: questions you should be able to answer out loud, in plain English, to a non-technical coworker. Don't move on until you can.
- Several modules include a **🛠️ Build It hands-on lab** — do them; building beats reading.
- Math is kept minimal. Where a concept *requires* a formula, it's stated in one line and then explained in words.
- Looking up a specific term? Use the **[A-to-Z Index](./GLOSSARY.md)**.

---

## The Mastery Ladder

| Module | Mastery | Subject | Deep Dive |
|--------|---------|---------|-----------|
| [01](./01-foundations-of-ai.md) | 0% → 8% | Foundations of AI | Symbolic AI → Machine Learning → Deep Learning |
| [02](./02-the-transformer-era.md) | 8% → 17% | The Transformer Era | Attention, Transformers, BERT vs GPT |
| [03](./03-tokens-and-embeddings.md) | 17% → 25% | Tokens & Embeddings | Tokenizers, vocab, context windows, sampling |
| [04](./04-training-llms.md) | 25% → 33% | Training LLMs | Pretraining, datasets, loss, scaling laws |
| [05](./05-finetuning-and-alignment.md) | 33% → 42% | Fine-Tuning & Alignment | SFT, RLHF, DPO, LoRA, QLoRA |
| [06](./06-hardware-and-compute.md) | 42% → 50% | Hardware & Compute | CPU vs GPU vs TPU, VRAM, FLOPS, CUDA |
| [07](./07-quantization.md) | 50% → 58% | Quantization | FP16/BF16/INT8/INT4, GGUF, GPTQ, AWQ |
| [08](./08-inference-engines.md) | 58% → 67% | Inference Engines | llama.cpp, Ollama, vLLM, TensorRT-LLM, TGI, SGLang |
| [09](./09-throughput-optimization.md) | 67% → 75% | Throughput Optimization | KV cache, PagedAttention, FlashAttention, speculative decoding |
| [10](./10-distributed-training.md) | 75% → 83% | Distributed Training | DDP, FSDP, DeepSpeed/ZeRO, tensor & pipeline parallelism |
| [11](./11-serving-and-autoscaling.md) | 83% → 92% | Serving & Autoscaling | Triton, Ray Serve, latency vs throughput, prompt caching |
| [12](./12-ecosystem-and-future.md) | 92% → 100% | Ecosystem & Future | OpenAI, Ollama, Qwen/Llama/Mistral/DeepSeek, RAG, agents, MCP |

**Cross-reference notation:** `§8.13` means Module 08, section 13. `Module 09` means the whole module.

---

## The 5 Questions This Guide Will Let You Answer

1. **"How does an LLM actually generate text?"** — tokenization → embedding → transformer layers → next-token probability → sampling → repeat. (Modules 02, 03, 09)
2. **"Why do you need a GPU?"** — matrix math + memory bandwidth + VRAM holds the weights and the KV cache. (Modules 06, 07, 09)
3. **"How does Ollama run a model on my laptop?"** — quantized GGUF weights + llama.cpp inference engine + CPU/GPU offload. (Modules 07, 08)
4. **"How does OpenAI serve millions of users?"** — vLLM-style continuous batching + PagedAttention + tensor parallelism across many GPUs + autoscaling fleets. (Modules 08, 09, 10, 11)
5. **"What's the difference between training and inference?"** — Training is a months-long, thousands-of-GPU job that *makes* the weights. Inference is using those frozen weights to answer one prompt. (Modules 04, 10 vs Modules 08, 09)

---

## Required Background

None. If you can read JSON and have used ChatGPT once, you're qualified.

## Recommended (Optional) Hands-On Tools

- **[Ollama](https://ollama.com)** — run a small Qwen or Llama model on your own machine starting around Module 08.
- **[Hugging Face](https://huggingface.co)** — browse model cards starting Module 04.
- **A cloud GPU** (RunPod, Lambda, Vast.ai) — try `vllm serve` once you reach Module 08.

## Supplemental Resources

- [A-to-Z Index / Glossary](./GLOSSARY.md) — every key term, mapped to the section that teaches it.
- [Complete Roadmap to Become an Agentic AI Engineer.pdf](./Complete%20Roadmap%20to%20Become%20an%20Agentic%20AI%20Engineer.pdf) — downloaded from the shared Google Drive link on 2026-04-30.

## Audio (TTS)

The [`tts/`](./tts) folder contains an audio edition (one MP3 per module plus an overview). The current MP3s were generated from the previous month-based edition of this text — same content, older framing — and will be regenerated from these modules in a future pass. See [`tts/README.md`](./tts/README.md).

---

*Last updated: July 2026*
