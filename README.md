# AI / LLM Training Guide — 365 Days

A year-long, high-level, low-math training curriculum to help you **understand and explain** how modern AI and Large Language Models (LLMs) actually work — from the 1950s symbolic era through to today's GPU-accelerated inference servers like vLLM, Ollama, and OpenAI's stack.

> **Goal:** When someone asks you "How does Ollama / Qwen / OpenAI work?" — you can explain it confidently, end-to-end, without hand-waving.

---

## How to Use This Guide

- **One topic per day, ~5–15 minutes of reading.**
- Each month builds on the last. Don't skip ahead — Month 8 (inference engines) only makes sense once you understand Month 6 (hardware) and Month 7 (quantization).
- At the end of every month there's a **"Explain it Back"** checklist — questions you should be able to answer out loud, in plain English, to a non-technical coworker.
- Math is kept minimal. Where a concept *requires* a formula, it's stated in one line and then explained in words.

---

## Curriculum Map

| Month | Days | Theme | What You'll Learn |
|-------|------|-------|-------------------|
| [01](./month-01-foundations-of-ai.md) | 1–30   | Foundations of AI | Symbolic AI → Machine Learning → Deep Learning |
| [02](./month-02-the-transformer-era.md) | 31–60  | The Transformer Era | Attention, Transformers, BERT vs GPT |
| [03](./month-03-tokens-and-embeddings.md) | 61–90  | Tokens & Embeddings | Tokenizers, vocab, context windows, positional encoding |
| [04](./month-04-training-llms.md) | 91–120 | Training LLMs | Pretraining, datasets, loss, scaling laws |
| [05](./month-05-finetuning-and-alignment.md) | 121–150 | Fine-tuning & Alignment | SFT, RLHF, DPO, LoRA, QLoRA |
| [06](./month-06-hardware-and-compute.md) | 151–180 | Hardware & Compute | CPU vs GPU vs TPU, VRAM, FLOPS, CUDA |
| [07](./month-07-quantization.md) | 181–210 | Quantization | FP16/BF16/INT8/INT4, GGUF, GPTQ, AWQ |
| [08](./month-08-inference-engines.md) | 211–240 | Inference Engines | llama.cpp, Ollama, vLLM, TensorRT-LLM, TGI, SGLang |
| [09](./month-09-throughput-optimization.md) | 241–270 | Throughput Optimization | KV cache, PagedAttention, FlashAttention, speculative decoding, batching |
| [10](./month-10-distributed-training.md) | 271–300 | Distributed Training | DDP, FSDP, DeepSpeed/ZeRO, tensor & pipeline parallelism |
| [11](./month-11-serving-and-autoscaling.md) | 301–330 | Serving & Autoscaling | Triton, Ray Serve, BentoML, latency vs throughput, prompt caching |
| [12](./month-12-ecosystem-and-future.md) | 331–365 | Ecosystem & Future | OpenAI, Ollama, Qwen/Llama/Mistral/DeepSeek, RAG, agents, MCP |

---

## The 5 Questions This Guide Will Let You Answer

1. **"How does an LLM actually generate text?"** — tokenization → embedding → transformer layers → next-token probability → sampling → repeat. (Months 2, 3, 9)
2. **"Why do you need a GPU?"** — matrix math + memory bandwidth + VRAM holds the weights and the KV cache. (Months 6, 7, 9)
3. **"How does Ollama run a model on my laptop?"** — quantized GGUF weights + llama.cpp inference engine + CPU/GPU offload. (Months 7, 8)
4. **"How does OpenAI serve millions of users?"** — vLLM-style continuous batching + PagedAttention + tensor parallelism across many GPUs + autoscaling fleets. (Months 8, 9, 10, 11)
5. **"What's the difference between training and inference?"** — Training is a months-long, thousands-of-GPU job that *makes* the weights. Inference is using those frozen weights to answer one prompt. (Months 4, 10 vs Months 8, 9)

---

## Required Background

None. If you can read JSON and have used ChatGPT once, you're qualified.

## Recommended (Optional) Hands-On Tools

- **[Ollama](https://ollama.com)** — run a small Qwen or Llama model on your own machine starting around Month 8.
- **[Hugging Face](https://huggingface.co)** — browse model cards starting Month 4.
- **A cloud GPU** (RunPod, Lambda, Vast.ai) — try `vllm serve` once you reach Month 8.

---

*Last updated: April 2026*
