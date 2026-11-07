# The A-to-Z of AI — A 0% → 100% Mastery Guide

A complete, self-paced, low-math curriculum that takes you from **"what even is AI?"** to **building real LLM applications and agents**. It starts with the 1950s symbolic era, explains how modern Large Language Models (LLMs) actually work and how GPU servers like vLLM, Ollama, and OpenAI's stack run them, and finishes on the practical AI-engineering path: RAG, structured outputs, evaluation, agents, and shipping to production.

> **Goal:** Be able to explain how Ollama / Qwen / ChatGPT work, end to end, without hand-waving, *and* be able to build reliable systems around them.

This is not a timeline. There are no "months" or "days." It's a **mastery ladder**: 14 modules, each a foundation-plus-deep-dive on one subject, ordered so every module stands on the ones before it. Go as fast or slow as you want. One section per sitting finishes it in roughly a semester; one section per hour, in a few intense weeks.

---

## How to Use This Guide

- **One numbered section (§) at a time.** Each is a self-contained 5–15 minute read.
- **Order matters, pace doesn't.** Module 08 (inference engines) only makes sense once you understand Module 06 (hardware) and Module 07 (quantization). Within that constraint, binge or sip freely.
- **Terms are defined before they're used.** A word in **bold** is being defined right there; a forward reference (→ §M.N) points to the section that teaches it.
- Every module ends with a **Checkpoint — "Explain It Back"**: questions to answer out loud, in plain English, with a hidden answer key. Don't move on until you can.
- Every module has a **🛠️ Build It** lab. Most labs are pure Python with nothing to install; optional steps use real tools (Ollama, tiktoken, Unsloth, vLLM). Building beats reading.
- Math is kept minimal. Where a concept *requires* a formula, it's stated in one line, explained in words, and worked with real numbers.
- Looking up a term? Use the **[A-to-Z Index](./GLOSSARY.md)**.

---

## The Mastery Ladder

| Module | Mastery | Subject | Deep Dive |
|--------|---------|---------|-----------|
| **Part 1 — How AI and LLMs Work** | | | |
| [01](./01-foundations-of-ai.md) | 0% → 7% | Foundations of AI | Symbolic AI → ML → vectors → neural networks → deep learning |
| [02](./02-the-transformer-era.md) | 7% → 14% | The Transformer Era | Attention, the transformer block, GPT vs BERT, MoE, model families |
| [03](./03-tokens-and-embeddings.md) | 14% → 21% | Tokens & Embeddings | BPE tokenizers, embeddings & vector search, context windows, sampling |
| [04](./04-training-llms.md) | 21% → 29% | Training LLMs | Pretraining objective, data pipelines, compute & cost, scaling laws |
| [05](./05-finetuning-and-alignment.md) | 29% → 36% | Fine-Tuning & Alignment | SFT, RLHF/DPO/GRPO, LoRA/QLoRA, fine-tune vs RAG |
| **Part 2 — How Models Are Run** | | | |
| [06](./06-hardware-and-compute.md) | 36% → 43% | Hardware & Compute | GPU anatomy, VRAM & bandwidth math, roofline, interconnects, CUDA |
| [07](./07-quantization.md) | 43% → 50% | Quantization | Number formats, scales & outliers, GGUF/k-quants/i-quants, GPTQ, AWQ |
| [08](./08-inference-engines.md) | 50% → 57% | Inference Engines | Life of a request, llama.cpp/Ollama, vLLM, SGLang, TensorRT-LLM, choosing one |
| [09](./09-throughput-optimization.md) | 57% → 64% | Throughput Optimization | Bandwidth-bound decode, KV cache, FlashAttention, speculative decoding, batching |
| [10](./10-distributed-training.md) | 64% → 71% | Distributed Training | Collectives & fabrics, DDP, ZeRO/FSDP, tensor/pipeline/expert parallelism |
| [11](./11-serving-and-autoscaling.md) | 71% → 79% | Serving & Autoscaling | SLOs & percentiles, queueing, load balancing, autoscaling, cost |
| **Part 3 — Building With AI** | | | |
| [12](./12-ecosystem-and-future.md) | 79% → 86% | Ecosystem & Future | Choosing models, RAG, tools & agents, MCP, trends, "explain it" capstone |
| [13](./13-building-llm-applications.md) | 86% → 93% | Building LLM Applications | LLM APIs from code, prompts as interfaces, structured outputs, testing & evals, shipping |
| [14](./14-agentic-systems-and-your-path.md) | 93% → 100% | Agentic Systems & Your Path | Agent architecture, memory, tools, multi-agent, frameworks, project ladder, skills map |

**Cross-reference notation:** `§8.18` means Module 08, section 18 (PagedAttention). `Module 09` means the whole module.

---

## Pick Your Path

All paths start with **Part 1 (Modules 01–05)**. That's the foundation everything else assumes.

| Path | Route | For |
|------|-------|-----|
| **Full mastery** | 01 → 14 in order | Understanding the whole stack, from the perceptron to agents in production |
| **AI engineer (fast track)** | 01–05 → 08 → 11 (Chapters 2 and 4) → 12 → 13 → 14, then fill in 06, 07, 09, 10 | Building LLM apps and agents. Most AI-engineering work is "systems around LLMs," not training models |
| **Infrastructure / platform** | 01–05 → 06 → 07 → 08 → 09 → 10 → 11, then 12–14 | Running, scaling, and paying for models: GPUs, inference engines, clusters |

---

## The 6 Questions This Guide Will Let You Answer

1. **"How does an LLM actually generate text?"** Tokenization → embedding → transformer layers → next-token probability → sampling → repeat. (Modules 02, 03)
2. **"Why do you need a GPU?"** Matrix math, plus memory bandwidth, plus VRAM to hold the weights and the KV cache. (Modules 06, 07, 09)
3. **"How does Ollama run a model on my laptop?"** Quantized GGUF weights + the llama.cpp inference engine + CPU/GPU offload. (Modules 07, 08)
4. **"How does OpenAI serve millions of users?"** Continuous batching + PagedAttention + tensor parallelism across many GPUs + autoscaling fleets. (Modules 08, 09, 10, 11)
5. **"What's the difference between training and inference?"** Training is a months-long, thousands-of-GPU job that *makes* the weights. Inference uses those frozen weights to answer one prompt. (Modules 04, 10 vs. Modules 08, 09)
6. **"How would you build a reliable AI assistant or agent?"** RAG for knowledge, structured outputs and tools for actions, evals to prove it works, tracing and guardrails to run it safely. (Modules 12, 13, 14)

---

## Required Background

None. If you can read JSON and have used ChatGPT once, you're qualified. Basic Python helps for the labs, and every lab explains what each line does.

## Recommended (Optional) Hands-On Tools

- **[Ollama](https://ollama.com)**: run a small Qwen or Llama model on your own machine. Useful from Module 05's lab onward.
- **[Hugging Face](https://huggingface.co)**: browse model cards starting in Module 04.
- **A cloud GPU** (RunPod, Lambda, Vast.ai): try `vllm serve` once you reach Module 08.

## Supplemental Resources

- [A-to-Z Index / Glossary](./GLOSSARY.md): every key term, mapped to the section that teaches it.
- [Complete Roadmap to Become an Agentic AI Engineer.pdf](./Complete%20Roadmap%20to%20Become%20an%20Agentic%20AI%20Engineer.pdf): Lamhot Siagian's interview Q&A by topic (January 2026), downloaded 2026-04-30. Use it for interview practice after Modules 12–14.

**Credits:** Part 3's AI-engineering path was shaped by "The AI Engineer Learning Path" (an X post by @ParasVerma7454, April 2026), which builds on Alexey Grigorev's [AI Engineering Field Guide](https://github.com/alexeygrigorev/ai-engineering-field-guide), and by the roadmap PDF above. The course text itself is original.

## Audio (TTS)

The [`tts/`](./tts) folder contains an audio edition (one MP3 per module plus an overview). The current MP3s were generated from an earlier, month-based edition of this text and are out of date. They will be regenerated from these modules in a future pass. See [`tts/README.md`](./tts/README.md).

---

*Last updated: November 2026*
