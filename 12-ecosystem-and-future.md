# Module 12 — Ecosystem & Future

**Mastery: 79% → 86%** · Prerequisites: everything before it

**Goal of this module:** Tie everything together. By the end you'll be able to explain, confidently and in plain English, how Ollama, ChatGPT, Qwen, Claude, RAG systems, and AI agents actually work end to end. You'll also be able to pick a model and an architecture for a real project, and judge new announcements on your own.

**How to read this module:** **Bold** marks a definition. Module 02 (§2.28–§2.34) told each lab's origin story; Chapter 1 here covers where each one stands *now* and what makes it different. Vendor facts move fast. Anything marked "as of this writing" (late 2026) should be re-checked before you rely on it.

---

## Chapter 1 — The Players Today

### §12.1 — What Actually Differs Between Models
From §2.35 you know the leading models share almost the same skeleton: a decoder-only transformer with RoPE, RMSNorm, SwiGLU, GQA, and often MoE. So when one model beats another, the difference is almost never "a secret architecture." It's usually some mix of:

| Lever | What it means | Covered in |
|-------|---------------|-----------|
| **Data** | What went into pretraining and how well it was cleaned | §4.7–§4.11 |
| **Scale** | Parameters, tokens, compute | §4.14–§4.22 |
| **Post-training** | SFT, preference tuning, reasoning RL | Module 05 |
| **Serving** | Speed, price, context length, reliability | Modules 08–11 |
| **Distribution** | Who already has it in front of them (apps, cloud, phones) | This chapter |
| **License** | Whether you can download, modify, and sell with it | §5.28, §12.11 |

Keep this table in mind as you read the lab profiles below.

### §12.2 — OpenAI
Most details are not public. What's below is from OpenAI's own announcements plus reasonable inference.
- **Models:** the GPT-5 family (2025) and its successors, which merged the "o-series" reasoning models (§2.27) into one product line that decides how long to think. The top models are widely believed to be MoE. OpenAI also released **gpt-oss** (August 2025), Apache-2.0 open-weight models (MoE, ~20B and ~120B total parameters), its first open weights since GPT-2.
- **Training:** very large NVIDIA clusters, historically through Microsoft Azure and increasingly through its own large datacenter projects. Heavy synthetic data, RLHF, and reasoning RL (§5.7, §5.14).
- **Inference:** an internal stack built on the same ideas you learned in Modules 08–09 (low-precision weights, paged KV cache, continuous batching, prefill/decode split, tensor/pipeline/expert parallelism). The specifics are secret.
- **Products:** ChatGPT (hundreds of millions of weekly users), the API (whose format became the industry standard, §12.14), agents that browse and use a computer, and Codex for coding.
- **Edge:** the largest consumer user base and broad multimodal capability.

### §12.3 — Anthropic (Claude)
- **Models:** the Claude family, closed-weight. Strong at coding, long documents, careful instruction-following, and agentic tool use.
- **Distinguishing work:** Constitutional AI (§5.12), the **Model Context Protocol** (§12.23), Computer Use (2024: a model operating a desktop through screenshots and clicks), **Claude Code** (an agentic coding tool in the terminal), and an Agent SDK for building agents.
- **Context:** 200K tokens standard, with 1M-token options on some models (as of this writing).
- **Distribution:** its own apps and API, plus the major clouds (AWS Bedrock, Google Vertex AI). Strongest with enterprises and developers; its consumer audience is smaller than ChatGPT's.

### §12.4 — Google DeepMind (Gemini)
- **Models:** the Gemini family, natively multimodal (text, image, audio, video), with very long context (1M tokens and up). Google also releases open-weight **Gemma** models (§12.10).
- **Hardware:** trained and served mostly on Google's own **TPUs** (§6.6), not NVIDIA GPUs, a fully separate hardware stack from almost everyone else.
- **Distribution:** built into Search, Android, Workspace, and Chrome. Probably the biggest distribution advantage of any lab.

### §12.5 — xAI (Grok)
- Built **Colossus** in Memphis, reported as 100,000 H100s brought online in a few months in 2024, since expanded. Its stated goal is far larger.
- A strategy of spending heavily to catch up fast. Grok is available in X (Twitter), as its own app, and through an API.
- Has released older models as open weights (Grok-1 in 2024, Grok-2 in 2025).

### §12.6 — Meta (Llama)
- **Lineage:** Llama 2 → 3 → 3.1 (with a dense 405B, still one of the largest dense open models) → 3.2 (small and vision models) → 3.3 → **Llama 4** (2025, MoE: Scout ~109B total / 17B active, and Maverick ~400B total / 17B active).
- Llama 4's reception was mixed, and Chinese open models (Qwen, DeepSeek) overtook Llama as the community default for many uses. Meta reorganized its AI efforts in 2025. Watch whether it keeps releasing open weights at the top end.
- **Legacy:** the Llama architecture and tooling became the template the whole open ecosystem was built on.
- **License:** the Llama Community License. It's free for most uses but has restrictions (very large companies need a separate license) and is not OSI open source (§5.28).

---

## Chapter 2 — The Open-Weight Ecosystem

### §12.7 — Qwen (Alibaba)
- **Range:** every size from tiny to huge. Qwen3 (April 2025) spans 0.6B to 235B (the MoE `Qwen3-235B-A22B`), with switchable thinking (§2.27). The **2507** refreshes split it into separate Instruct and Thinking models.
- **Coding:** **Qwen3-Coder-480B-A35B** (July 2025) was the flagship open coding model when released.
- **Newer releases (as of this writing):** the Qwen3.5 family (February 2026; native multimodal, 256K+ context, up to a 397B MoE) and **Qwen3.6-27B / Qwen3.6-Plus** (April 2026), reported to beat much larger MoE models on agentic coding, with up to 1M context.
- **Strengths:** coding, math, multilingual (especially Chinese), and a huge range of sizes. Most models are Apache 2.0.
- **What you're running when you `ollama pull qwen3`.** It's also among the most popular bases for community fine-tunes (§5.30).

### §12.8 — DeepSeek
- **DeepSeek-V3** (December 2024): 671B MoE, ~37B active, trained largely in FP8 (§4.18).
- **DeepSeek-R1** (January 2025): the first open reasoning model competitive with o1, MIT-licensed. It showed the GRPO recipe publicly (§5.14). **R1-0528** (May 2025) improved it.
- **V3.1 / V3.2** (2025): hybrid thinking modes and an experimental sparse-attention design for cheaper long context (§9.19).
- **DeepSeek-V4** (preview April 2026, as of this writing): reported Flash (284B) and Pro (1.6T) variants with up to 1M context.
- **Signature:** efficiency engineering (MLA attention §2.37, fine-grained MoE, FP8, careful systems work) and publishing detailed technical reports.

### §12.9 — Mistral
- Paris-based, with a mix of open-weight (Apache 2.0) and commercial models: Mistral 7B, Mixtral 8x7B and 8x22B (open MoE pioneers, §2.30), the Mistral Small and Medium lines, Codestral for code, and **Magistral** reasoning models (2025).
- Strong on European languages, with a European data-sovereignty pitch. Chat product: **Le Chat**.

### §12.10 — Other Notable Open Players
- **Microsoft Phi:** small models trained on heavily synthetic, textbook-style data (§4.9).
- **Google Gemma:** small to mid-size open models from Gemini research, often trained with logit distillation (§5.30).
- **Moonshot (Kimi K2), Zhipu (GLM), MiniMax:** Chinese labs that released large, strong open-weight MoE models in 2025, especially good at agentic coding.
- **Cohere Command:** RAG- and enterprise-focused. Weights are released under a *non-commercial* license, so read it before you build on it.
- **AI2 OLMo, LLM360:** truly open, with data, code, and checkpoints all published (§5.28).
- **NVIDIA Nemotron:** open models and datasets, a lot of synthetic data, and post-training research.

### §12.11 — Choosing a Model: Open vs. Closed
Start with the task, then use this table:

| If you need... | Lean toward | Why |
|----------------|-------------|-----|
| The best quality on hard reasoning or agentic tasks | Closed frontier API | Usually still a step ahead (§12.34) |
| Data that must never leave your network | Open weights, self-hosted | Nothing is sent to a vendor |
| Lowest cost at high volume on a narrow task | Small open model, maybe fine-tuned | §5.21, §4.22 |
| Fastest time-to-prototype | Closed API | No hardware to set up |
| Offline or edge use | Small open model (quantized) | Module 07 |
| Freedom from vendor changes | Open weights, pinned version | You control upgrades |

Then check the **license** (Apache/MIT, Llama Community, non-commercial?), **context length**, **tool-calling support**, and whether it fits your VRAM after quantization (§6.30). A common pattern: prototype on a closed API, then move high-volume steps to an open model once you know exactly what you need.

---

## Chapter 3 — The Tools You'll Actually Use

### §12.12 — Hugging Face: The Hub
The GitHub of machine learning: well over a million models, hundreds of thousands of datasets, free hosting, **Spaces** for demos, and the `transformers` library. Learn to read a model repo name. It packs most of this course into one line:

```
Qwen3-30B-A3B-Instruct-2507-GGUF  →  file: ...Q4_K_M.gguf
│     │    │   │        │    │                  │
│     │    │   │        │    │                  └ 4-bit k-quant, medium (§7.18)
│     │    │   │        │    └ llama.cpp file format (§7.17)
│     │    │   │        └ release date: July 2025
│     │    │   └ post-trained to chat (§4.30)
│     │    └ 3B active parameters per token (§2.36)
│     └ 30B total parameters
└ model family
```

Always read the **model card**: license, intended use, chat template, recommended sampling settings (§3.33), and benchmark claims (with skepticism, §4.13).

### §12.13 — Local Inference Tools Recap
- **Ollama:** command line plus an API. The easiest way to run models locally (§8.12).
- **LM Studio:** a desktop app, great for browsing and trying models.
- **Jan, GPT4All:** open-source ChatGPT-style desktop apps.
- **Open WebUI:** a self-hosted ChatGPT-style web interface in front of Ollama or any OpenAI-compatible API.
- **llama.cpp:** the engine under most of the above (§8.10).
- **MLX:** Apple's framework, fast on Apple Silicon Macs.
- **vLLM / SGLang:** when you outgrow Ollama and need real multi-user serving (§8.17, §8.27).

### §12.14 — The OpenAI-Compatible API: The Lingua Franca
Almost every engine and vendor accepts the same request shape (§8.32, §11.2), so code written for one usually works with another by changing the URL:

```json
POST /v1/chat/completions
{
  "model": "qwen3:8b",
  "messages": [
    {"role": "system", "content": "You are a concise network assistant."},
    {"role": "user",   "content": "What does OSPF EXSTART mean?"}
  ],
  "temperature": 0.6,
  "max_tokens": 300,
  "stream": true
}
```

You'll recognize every field: messages become a chat template (§5.3), temperature and max tokens control generation (§3.31, §3.35), and streaming sends tokens as they're decoded (§8.7). Ollama serves this at `http://localhost:11434/v1`. One thing to remember: **the API is stateless**. The model remembers nothing between calls, so your app resends the whole conversation every time, which is why long chats get slower and pricier (and why prompt caching exists, §11.21).

---

## Chapter 4 — RAG: Giving Models Your Knowledge

### §12.15 — Retrieval-Augmented Generation (RAG): The Pattern
A model only knows its training data, up to its cutoff (§4.6). It has never seen your runbooks, tickets, or configs. **RAG** fixes this without retraining:
1. **Retrieve:** search your documents for the pieces most relevant to the question.
2. **Augment:** paste those pieces into the prompt.
3. **Generate:** ask the model to answer *using that text*, ideally citing it.

```
"Why is OSPF stuck in EXSTART?"
   → search the runbook index → top 3 chunks (one says "MTU mismatch")
   → prompt = instructions + 3 chunks + question
   → LLM: "Most likely an MTU mismatch between the interfaces [runbook-ospf.md]..."
```

It's an open-book exam instead of a memory test. Answers stay current (update a document and it's live), you can show sources, and hallucination drops because the facts are right there in the context. The term comes from a 2020 Meta paper.

### §12.16 — RAG Components
RAG has two phases. **Indexing** happens ahead of time:
1. **Loader:** parse PDFs, web pages, wikis, code, tickets.
2. **Chunker:** split into pieces (§12.17).
3. **Embedder:** turn each chunk into a vector (§3.18). Use the *same* embedding model for documents and queries.
4. **Vector database:** store vectors plus metadata (§3.20).

**Query time** happens on every question:

5. **Retriever:** embed the question, then run hybrid search (vectors + keywords, §3.20) for the top ~20–50 chunks.
6. **Reranker:** a cross-encoder rescores those and keeps the best ~3–8 (§3.18). Slower, but much more precise.
7. **Generator:** the LLM answers from the chosen chunks.

### §12.17 — Chunking: The Unglamorous Part That Matters Most
How you split documents often matters more than which model you use.
- **Size:** common choices are ~200–1,000 tokens. Too small and a chunk loses its context ("set it to 1500," set *what*?). Too big and a chunk mixes topics, which blurs its embedding and wastes prompt space.
- **Overlap:** repeat ~10–20% of text between neighboring chunks so an idea cut at a boundary survives in one of them.
- **Follow the structure:** split on headings, sections, or functions instead of every N characters. A config stanza or a runbook step should stay in one piece.
- **Add context:** prepend the document title and section heading to each chunk ("Runbook: OSPF > Troubleshooting > EXSTART"). Anthropic's "contextual retrieval" (2024) goes further and has an LLM write a short line explaining where each chunk fits.

### §12.18 — Why Naïve RAG Often Disappoints
"Garbage in, garbage out." Typical failures and fixes, roughly in order of payoff:
- **Missed matches:** pure vector search misses exact terms (error codes, hostnames), and pure keyword search misses synonyms. Use **hybrid search** (§3.20). You'll see both failures in the lab.
- **Right chunk ranked too low:** add a **reranker** (§12.16). It's often the single biggest quality boost.
- **Bad chunks:** fix chunking (§12.17).
- **Vague questions:** **query rewriting**. Have an LLM rephrase the question, split it into sub-questions, or write a hypothetical answer and search with that (**HyDE**).
- **Multi-hop questions** ("which branch VPNs use certs that expire this quarter?"): let an agent search several times (**agentic retrieval**, §12.22) or use a knowledge graph (**GraphRAG**).
- **The model ignores the context:** tell it to answer only from the provided text, to cite sources, and to say "not found" when the answer isn't there.

### §12.19 — Evaluating RAG
Measure the two halves separately (build a test set of 50–100 real questions first, §5.24):
- **Retrieval quality:**
  - **Recall@k:** how often is the correct chunk somewhere in the top k?
  - **MRR (mean reciprocal rank):** average of 1/rank of the first correct chunk.

  Worked example with three questions, where the correct chunk was at rank 1, rank 3, and not found:

  | Metric | Calculation | Result |
  |--------|-------------|--------|
  | Recall@3 | 2 of 3 questions found | **0.67** |
  | MRR | (1/1 + 1/3 + 0) ÷ 3 | **0.44** |

- **Answer quality:**
  - **Faithfulness / groundedness:** is every claim supported by the retrieved text? (An LLM judge can check this, §5.24.)
  - **Correctness:** compare against a reference answer.

If retrieval is poor, no model can save the answer. Fix retrieval first.

### §12.20 — Long Context vs. RAG
"Why not just paste the whole knowledge base into a 1M-token context window?" Sometimes you should: for one long contract or one codebase, long context is simple and works well. For a large, growing knowledge base, RAG usually wins:

| | Stuff everything in context | RAG |
|---|---|---|
| **Cost per question** (illustrative, at $3 per million input tokens) | 1M tokens ≈ **$3** | ~5K tokens ≈ **$0.015** (200× less) |
| **Latency** | Long prefill (§3.29), often many seconds | Fast |
| **Quality** | Can suffer from "lost in the middle" (§3.28) | Depends on retrieval quality |
| **Scale limit** | The context window | Effectively unlimited |
| **Setup** | None | An indexing pipeline to build and maintain |

Prompt caching (§11.21) narrows the gap when the same big context is reused across many questions. Many real systems combine the two: retrieve generously, then use a long-context model to read it.

---

## Chapter 5 — Tools, Agents, and MCP

### §12.21 — Tool Use / Function Calling
You give the model a list of **tools**, each with a name, a description, and a **JSON schema** for its arguments:

```json
{"name": "get_interface",
 "description": "Return speed, traffic, and error counters for one interface",
 "parameters": {"type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]}}
```

When the model decides it needs one, it doesn't answer. It outputs a structured **tool call**: `{"tool": "get_interface", "args": {"name": "ge-0/0/1"}}`.

The crucial point: **the model never runs anything itself.** *Your code* reads the tool call, runs the real function (with whatever permissions *you* give it), and sends the result back as a `tool` message. Then the model continues. Models are trained for this during post-training, and engines can force the output into valid JSON (§8.31). This is how chatbots get live data, do exact math (§3.11), run code, and search the web.

### §12.22 — Agents
An **agent** is an LLM with tools, running in a loop until the task is done:

```
goal → model thinks → picks a tool → your code runs it → result goes back → model thinks → ... → final answer
```

Examples: coding agents (Claude Code, OpenAI Codex, Cursor, Cline, Aider, OpenHands, Devin), browser and computer-use agents, and research agents that search, read, and summarize.

**Why agents are hard: errors compound.** If each step succeeds 95% of the time, a 20-step task succeeds only `0.95²⁰ ≈ 36%` of the time. At 99% per step, it's 82%. That's why agents have gotten dramatically better as models improved: small per-step gains multiply. It's also why good agent design includes:
- A **step limit** and a **budget** (tokens, time, money).
- Clear tool descriptions and helpful error messages, so the model can recover.
- **Checkpoints and human approval** before anything risky or irreversible.
- **Traces:** log every step so you can debug what it did (§11.24).

### §12.23 — MCP (Model Context Protocol)
Before MCP, every AI app needed custom integration code for every tool: M apps × N tools = M×N integrations. **MCP** (Anthropic, November 2024; now widely adopted, including by OpenAI, Google, and Microsoft) is an open standard so that **any MCP client** (Claude Desktop, Claude Code, Cursor, IDEs, other agent apps) can use **any MCP server** (a filesystem, a database, GitHub, Slack, your ticket system). That makes it M + N.

- A server offers **tools** (actions the model can call), **resources** (data it can read), and **prompts** (reusable templates).
- Messages are **JSON-RPC 2.0**, carried over **stdio** (a local process) or **HTTP** (a remote server).

**Network-engineer analogy:** MCP is to AI tools what NETCONF/YANG (or SNMP with standard MIBs) is to network devices. Instead of screen-scraping every vendor's CLI differently, any manager speaks one protocol to any device that implements it. A related standard, Google's **A2A** (agent-to-agent), covers agents talking to *other agents*.

### §12.24 — Agent Security: Least Privilege
Agents turn "the model said something wrong" into "the model *did* something wrong." The core threat is **prompt injection** (§5.26): instructions hidden in data the agent reads (a web page, an email, a ticket, a code comment).

A useful rule of thumb (Simon Willison's **"lethal trifecta"**): be very careful when an agent has *all three* of
1. access to **private data**,
2. exposure to **untrusted content**, and
3. a way to **send data out** (email, web requests, posting).

With all three, an injected instruction can steal your data. Defenses you'd recognize from network security:
- **Least privilege:** read-only credentials, scoped tokens, and only the tools the task needs.
- **Segmentation:** run agents in sandboxes or containers with no route to things they don't need.
- **Human approval** for destructive or outward-facing actions.
- **Logging and audit** of every tool call.

Treat an agent like an untrusted contractor with a badge: useful, but only let it into the rooms the job requires.

### §12.25 — Frameworks: LangChain, LlamaIndex, DSPy, LangGraph, and Friends
- **LangChain:** the earliest, with the most integrations, sometimes over-abstracted.
- **LlamaIndex:** strongest for RAG and data ingestion.
- **LangGraph:** agents as explicit graphs of steps with saved state.
- **DSPy:** treats prompts as programs and optimizes them automatically against your eval set.
- **Pydantic AI:** typed, lightweight, idiomatic Python.
- **Vendor SDKs** (OpenAI Agents SDK, Claude Agent SDK), plus **CrewAI** for multi-agent setups.

| Approach | Pros | Cons |
|----------|------|------|
| A framework | Fast start, many integrations | Abstractions hide the prompt; churns fast; harder to debug |
| Plain code + an SDK | You see exactly what's sent; easy to debug | You write the plumbing yourself |

The lab below is a whole agent loop in about 30 lines. Understand that first, then adopt a framework if it clearly saves you work.

---

## Chapter 6 — The Future

### §12.26 — Reasoning Models: The 2024–2026 Inflection
OpenAI o1 and o3, DeepSeek-R1, Qwen's thinking models, Gemini's thinking modes, Claude's extended thinking: all trained with RL to think before answering (§2.27, §5.14). There are huge gains on math, code, and science, at a cost: often **10–100× more output tokens** per answer, so slower and pricier responses (§3.13). Most products now let the model, or you, choose how much to think. This reshaped serving too: decode-heavy workloads (§3.29) and SLO targets (§11.8).

### §12.27 — Multimodal Convergence
Text, images, audio, and video increasingly go through the *same* transformer, with each input type turned into tokens and vectors (GPT-4o, Gemini, Llama 4, Qwen-VL and Omni). Voice conversations now respond in a few hundred milliseconds, close to human turn-taking, and video understanding is becoming standard.

### §12.28 — On-Device LLMs
Phones and laptops now run ~1B–8B models locally: Apple's on-device foundation model, Gemini Nano on Android, and Copilot+ PCs with NPUs (§6.7). Quantization (Module 07) and distillation (§5.30) are what make this possible. The benefits: privacy, low latency, offline use, and no per-token cost. The usual pattern is a hybrid: small tasks on device, hard ones sent to the cloud.

### §12.29 — World Models and Embodied AI
Models that predict video or simulate interactive 3D environments (OpenAI's Sora, DeepMind's Genie, Meta's V-JEPA) aim to learn how the physical world behaves. **Vision-language-action** models feed that into robots (Gemini Robotics, Physical Intelligence's π0). This is earlier-stage than chatbots, but it's where "AI that acts in the real world" is heading.

### §12.30 — AI for Science
**AlphaFold** predicts protein structures (its creators shared the 2024 Nobel Prize in Chemistry). Math systems reached olympiad level: AlphaProof/AlphaGeometry earned IMO silver in 2024, and general LLMs from Google and OpenAI reached gold-medal level in 2025. AI weather models (like GraphCast) rival traditional forecasts, and foundation models are emerging for chemistry, materials, and biology. It's less hyped than chatbots and possibly more important long-term.

### §12.31 — Continual Learning and Memory
Today's models are frozen after training. They learn nothing from talking with you (§12.14). Current workarounds live *outside* the weights: RAG over your history, "memory" features that save facts and re-inject them into prompts, and periodic fine-tunes. True **continual learning** (safely updating weights from real use without forgetting, §5.25, or being poisoned) is still mostly research (§10.34).

### §12.32 — Hardware Trends
- **NVIDIA:** Blackwell (B200/GB200, then Blackwell Ultra in 2025) → **Vera Rubin** (2026) → Rubin Ultra → **Feynman** (around 2028, per NVIDIA's public roadmap). See §6.8.
- **Google TPUs:** v6 (Trillium), then v7 (Ironwood, 2025), built for inference.
- **AMD:** the MI300/MI350 series, with MI400 in 2026, steadily closing the software gap.
- **Specialists:** Groq, Cerebras (wafer-scale), SambaNova, and others aim for very fast inference.
- **Custom chips:** AWS Trainium and Inferentia, Microsoft Maia, Meta MTIA. Every hyperscaler wants less dependence on NVIDIA.
- **Research frontier:** photonics and in-memory computing.

### §12.33 — Scaling Laws Are Bending, Not Breaking
Pretraining gains per dollar have slowed. Bigger runs still help, but less dramatically, and high-quality data is running short (the data wall, §4.12). The new scaling axes are **test-time compute** (reasoning, §12.26), **post-training RL** (§5.14), and **synthetic data** (§5.29). The Bitter Lesson (§2.38, §4.23) still holds; *what* gets scaled has shifted.

### §12.34 — The Open vs. Closed Race
By 2025–2026 the gap between the best open-weight models (Qwen, DeepSeek, Kimi, GLM, gpt-oss, Llama, Mistral) and the closed frontier (GPT, Claude, Gemini) became small for most everyday tasks, often estimated at months rather than years. OpenAI's gpt-oss release acknowledged the shift. Closed labs still tend to lead on the hardest reasoning and agentic tasks, frontier multimodal, and distribution. Whether that lead lasts is one of the big open questions of the late 2020s.

### §12.35 — Geopolitics of AI
US export controls (since 2022, tightened several times) restrict sales of top NVIDIA chips to China. NVIDIA made cut-down versions such as the H800 and H20, and the rules kept shifting. China has responded with domestic chips (Huawei Ascend), domestic manufacturing efforts, and efficiency-focused software (DeepSeek, §12.8). Chips, power, and talent are now treated as strategic national resources, and this will keep shaping who can build what (§4.33).

### §12.36 — Energy and Sustainability
Datacenters used about **4%** of US electricity in 2023, and a 2024 Lawrence Berkeley National Laboratory report projected roughly **7–12% by 2028**, driven largely by AI. Big AI companies have signed nuclear deals: Microsoft (restarting a Three Mile Island reactor), Amazon (Talen's Susquehanna plant), and Google (Kairos small modular reactors). Power availability, not just chips, now limits where clusters can be built. Efficiency work (quantization, MoE, distillation, better kernels) matters environmentally as well as financially.

### §12.37 — How to Stay Current (and Judge Hype)
**Sources:**
- **People:** Andrej Karpathy, Simon Willison, Sebastian Raschka, Jeremy Howard, Tri Dao, swyx, Nathan Lambert.
- **Podcasts:** Latent Space, Dwarkesh, No Priors, MLST.
- **Newsletters:** Import AI (Jack Clark), The Batch (Andrew Ng), AI News (smol.ai), Interconnects.
- **Papers:** Hugging Face Daily Papers, arXiv.
- **Just play:** Ollama plus a small Qwen on your own machine. Building beats reading.

**A five-question hype filter** for any new announcement:
1. **What's actually new:** data, scale, post-training, or architecture? (§12.1)
2. **Which benchmarks, and could they be contaminated?** (§4.13) Are there independent results?
3. **Total or active parameters?** What does it need to run? (§2.36, §6.30)
4. **What's the license?** (§12.11)
5. **Does it beat what you use today on *your* own test set?** (§5.24) That's the only benchmark that really matters.

---

### 🛠️ Build It — Hands-On Lab
**Lab 1: a toy RAG retriever (§12.15–§12.18).** Pure Python. It uses simple word counts as "embeddings" so you can see exactly why real systems need better ones.

```python
import math
import re
from collections import Counter

# Our "knowledge base": short runbook snippets an LLM was never trained on.
docs = [
    "OSPF neighbors stuck in EXSTART usually means an MTU mismatch between the two interfaces.",
    "BGP sessions that flap every 180 seconds point to a hold timer or keepalive problem.",
    "To reset a stuck OSPF neighbor, clear the OSPF process on one side during a maintenance window.",
    "Interface CRC errors usually come from a bad cable, a dirty fiber, or a duplex mismatch.",
    "The core switches back up their configs to the NAS every night at 02:00.",
    "VPN tunnels to the branch offices use IKEv2 with certificates that expire every year.",
]

STOP = {"the", "a", "an", "to", "in", "of", "on", "or", "and", "is", "that", "with",
        "from", "every", "one", "two", "my", "what", "why", "how", "do", "i", "are", "be"}

def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

def vectorize(text):
    return Counter(words(text))             # bag-of-words "embedding": word -> count

def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    length = lambda v: math.sqrt(sum(c * c for c in v.values()))
    return dot / (length(a) * length(b)) if a and b else 0.0

index = [(doc, vectorize(doc)) for doc in docs]   # indexing step: done once, ahead of time

def retrieve(question, k=2):
    q = vectorize(question)
    scored = sorted(((cosine(q, vec), doc) for doc, vec in index), reverse=True)
    return scored[:k]

for question in ["Why is my OSPF neighbor stuck in EXSTART?",
                 "My routing adjacency will not come up"]:
    print(f"\nQ: {question}")
    hits = retrieve(question)
    for score, doc in hits:
        print(f"   {score:.2f}  {doc[:60]}...")
    context = "\n".join(f"- {doc}" for _, doc in hits)
    prompt = f"Answer using only these notes:\n{context}\n\nQuestion: {question}"
    print(f"   prompt sent to the LLM: {len(prompt)} characters")
```

What you'll see, and why it matters:
- **Question 1** finds both OSPF notes, but ranks the "reset" note (0.55) *above* the note with the actual answer (MTU mismatch, 0.47). The question says "neighbor" and the answer note says "neighbors," and word counts treat those as different words. That's why rerankers (§12.16) and proper embeddings exist.
- **Question 2** fails completely. "Routing adjacency will not come up" shares no meaningful words with the OSPF notes, so the top hits are junk that matched "up" (as in "back *up*") and "come." Real embedding models (§3.18) place "adjacency" near "neighbor," which is exactly what fixes this. Meanwhile keyword search is what you'd want for exact strings like `EXSTART`: hence hybrid search.

Things to try: in `words()`, change `w for w in` to `w.rstrip("s") for w in` (a crude way to treat "neighbors" and "neighbor" as the same word). Question 1's MTU note jumps to first place (0.63), but question 2 still fails, because no amount of word-trimming turns "adjacency" into "neighbor." Then add your own runbook lines.

**Lab 2: an agent loop with a fake model (§12.21–§12.22).** Pure Python. The "model" is hard-coded rules, so you can see the loop itself.

```python
import json

# --- Tools: ordinary Python functions the "model" may ask us to run -----------
INTERFACES = {
    "ge-0/0/1": {"speed_mbps": 1000, "in_mbps": 930, "errors": 0},
    "ge-0/0/2": {"speed_mbps": 1000, "in_mbps": 120, "errors": 4521},
}

def get_interface(name):
    return INTERFACES.get(name, {"error": f"no such interface {name}"})

def percent(part, whole):
    return round(100 * part / whole, 1)

TOOLS = {"get_interface": get_interface, "percent": percent}

# --- A fake "LLM": hard-coded rules standing in for a real model -------------
# A real model would read the whole conversation and decide; ours just checks
# what it has learned so far. It replies either with a tool call or an answer.
def fake_model(messages):
    seen = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if not seen:
        return {"tool": "get_interface", "args": {"name": "ge-0/0/1"}}
    if len(seen) == 1:
        iface = seen[0]
        return {"tool": "percent", "args": {"part": iface["in_mbps"], "whole": iface["speed_mbps"]}}
    return {"answer": f"ge-0/0/1 is running at {seen[1]}% of line rate, so it is close to congested."}

# --- The agent loop: think -> act -> observe -> repeat -----------------------
messages = [{"role": "user", "content": "Is ge-0/0/1 congested?"}]
for step in range(1, 6):                      # hard cap so a confused model can't loop forever
    reply = fake_model(messages)
    if "answer" in reply:
        print(f"step {step}: FINAL ANSWER -> {reply['answer']}")
        break
    result = TOOLS[reply["tool"]](**reply["args"])   # WE run the tool, not the model
    print(f"step {step}: model asks {reply['tool']}({reply['args']}) -> {result}")
    messages.append({"role": "assistant", "content": json.dumps(reply)})
    messages.append({"role": "tool", "content": json.dumps(result)})
else:
    print("gave up: step limit reached")
```

It takes three steps: look up the interface, compute 93.0%, then answer. Things to try: make `fake_model` always return a tool call and watch the step limit save you. Change it to ask about `ge-0/0/9` and see how the error result flows back.

**Optional, with real tools:**
- **Real embeddings:** `ollama pull nomic-embed-text`, then POST `{"model": "nomic-embed-text", "input": "..."}` to `http://localhost:11434/api/embed`. Swap the result into Lab 1's `vectorize` (using the vector-list cosine from §1.23). Question 2 should now find the OSPF notes.
- **Real tool calling:** send Lab 2's tool schema (§12.21) to a tool-capable model (for example `qwen3`) through Ollama's `/api/chat` `tools` field, and replace `fake_model` with the real response.
- **An MCP server:** with the official Python SDK (`pip install mcp`), expose `get_interface` as an MCP tool in a few dozen lines and connect it to Claude Desktop or Claude Code.

### 🛠️ Capstone Projects (Previewed Here, Built in Modules 13–14)
Pick one or more. These show up on real AI-engineer job descriptions. Module 13 covers the engineering you need (APIs from code, structured outputs, evals, shipping), and Module 14 walks the full project ladder step by step:
1. **Document Q&A bot.** Chunk a folder of PDFs or runbooks (§12.17), embed them (`bge` or `nomic-embed-text`), store them in `pgvector` or Qdrant, and build an API endpoint that retrieves the top 20, reranks to 5, and answers with citations.
2. **Web research agent.** A tool-calling loop: search → fetch → summarize → decide the next step. Cap the iterations and token budget, and trace every step.
3. **Multi-agent workflow.** A planner breaks down the task, workers execute, and a critic reviews. Build it with LangGraph, or by hand.
4. **MCP server.** Expose your internal docs, ticket system, or read-only network inventory to Claude Desktop, Claude Code, or Cursor (mind §12.24).
5. **Eval harness.** For any of the above: a golden set of 50 cases, retrieval metrics (recall@k, MRR, §12.19), and LLM-as-judge scoring (§5.24), run automatically on every prompt change.

**Deliverable:** at least one in a public GitHub repo, with its eval results in the README. *This* is what gets you hired.

---

## Chapter 7 — Capstone: Explain It Back

The payoff of everything you've learned about *how AI works*. Say each of these out loud, without notes. Every sentence points to the section that taught it. If any part feels shaky, go back there.

### §12.38 — Practice Explanation: "What Even Is an LLM, in 30 Seconds?"
> "It's a huge neural network (§1.29) trained on one task: given some text, predict the next token (§1.1, §4.1), over trillions of tokens from the internet. Words become vectors (§1.42, §3.14), and stacked attention layers let every token pull in meaning from the ones before it (§2.4–§2.12). Billions of learned weights (§1.30) end up storing a compressed picture of language and the world. Then it's fine-tuned to follow instructions and be helpful (Module 05). It writes one token at a time (§3.29). It feels like reasoning, and on a growing range of tasks it effectively is, but it can also produce confident nonsense, because its job is *plausible* text, not *verified* text (§4.6)."

### §12.39 — Practice Explanation: "How Does Ollama Work?"
> "Ollama is a friendly wrapper around llama.cpp, a C++ inference engine (§8.10–§8.12). `ollama pull qwen3:8b` downloads a GGUF file (§7.17): the model's weights quantized from 16-bit down to about 4-bit (Module 07), plus the tokenizer and chat template (§3.1, §5.3). When you chat, llama.cpp loads as many layers as fit into GPU VRAM and runs the rest on the CPU (§6.10). It wraps your messages in the chat template, tokenizes them, and runs the prompt through every transformer layer in one parallel prefill pass (§3.29). Then it decodes one token at a time, sampling with temperature and top-p (§3.31–§3.32), storing each token's keys and values in the KV cache so nothing is recomputed (§9.5), and streaming each token to you, until the model emits its end-of-turn token (§3.35). It also serves an OpenAI-compatible API, so other apps can use it (§12.14)."

### §12.40 — Practice Explanation: "How Does ChatGPT / OpenAI Work?"
> "OpenAI pretrains very large transformer models, likely MoE (§2.36), on huge GPU clusters for months (§4.14–§4.16): next-token prediction on internet-scale text (Module 04). Then post-training: SFT on example conversations, preference tuning, and reasoning RL on checkable problems (§5.4–§5.14). To serve hundreds of millions of users, they run an optimized inference stack (Modules 08–09): low-precision weights, a paged KV cache, continuous batching, prefill and decode split across machines, and tensor, pipeline, and expert parallelism across many GPUs per copy of the model (§9.29–§9.31). Load balancers route requests across regions, autoscalers add and remove capacity, prompt caching cuts repeated work, and safety systems check inputs and outputs (Module 11). The API is stateless, so the app resends the conversation every turn (§12.14), and tools like search and code execution run in OpenAI's own code, not inside the model (§12.21)."

### §12.41 — Practice Explanation: "How Do RAG and Agents Work?"
> "A model only knows its training data up to its cutoff (§4.6). RAG gives it an open book: we split our documents into chunks, embed each chunk as a vector (§3.18), and store them in a vector database (§3.20). For each question, we find the most similar chunks with hybrid search, rerank them, paste the best few into the prompt, and ask the model to answer from them with citations (§12.15–§12.18). An agent goes further: the model gets tools described by JSON schemas, and it replies with tool calls that *our* code executes, feeding the results back in a loop until the task is done (§12.21–§12.22). MCP standardizes how those tools are plugged in (§12.23). Because errors compound over many steps and injected text can hijack an agent, we cap the steps, log everything, and give it the least privilege possible (§12.24)."

### §12.42 — Checkpoint — Explain It Back
1. If most top models share the same architecture, what actually makes one better than another?
2. Decode this name: `Qwen3-30B-A3B-Instruct-2507`, plus a `Q4_K_M.gguf` file.
3. What are the indexing steps and the query-time steps of RAG?
4. Why do chunk size and overlap matter?
5. Your RAG answers are bad. What do you check first, and what are two fixes?
6. Calculate recall@3 and MRR if the correct chunk was at ranks 2, 1, and not found.
7. When would you choose long context over RAG?
8. In tool calling, who actually runs the tool, and why does that matter?
9. Why does a 20-step agent fail so often, even with a good model?
10. What is the "lethal trifecta," and how do you defend against it?

<details>
<summary>Answer key (try first!)</summary>

1. Data, scale, post-training, serving, distribution, and license. Rarely the core architecture.
2. Qwen3 family, 30B total parameters, 3B active per token (MoE), chat-tuned, July 2025 release. The file is a GGUF for llama.cpp, quantized to a 4-bit k-quant (medium).
3. Indexing: load, chunk, embed, store. Query time: embed the question, hybrid-retrieve, rerank, generate with the chunks in the prompt.
4. Too small loses context, too large blurs topics and wastes the window. Overlap keeps ideas that span a boundary intact.
5. Check retrieval first (is the right chunk being found? measure recall@k). Fixes: hybrid search, a reranker, better chunking, query rewriting, instructing the model to cite and say "not found."
6. Recall@3 = 2/3 ≈ 0.67. MRR = (1/2 + 1/1 + 0) ÷ 3 = 0.5.
7. When the material is one bounded document or codebase that fits the window, especially if the context is reused and can be prompt-cached.
8. Your application code runs it, never the model. So you control permissions, validation, and logging, and the model can only do what your tools allow.
9. Errors compound: 95% per step over 20 steps is only about 36% overall.
10. Private data + untrusted content + a way to send data out. Defend with least privilege, sandboxing/segmentation, human approval for risky actions, and logging, and avoid giving one agent all three.
</details>

### §12.43 — Recap: You Understand the Stack
You now understand:
- The 70-year arc from symbolic AI to neural networks, vectors, and attention (Modules 01–02)
- How models see and produce text: tokens, embeddings, sampling (Module 03)
- How models are trained, fine-tuned, and aligned (Modules 04–05)
- The hardware and number formats they run on (Modules 06–07)
- The engines that serve them and the optimizations that make them fast (Modules 08–09)
- How frontier training scales across thousands of GPUs (Module 10)
- How real production services are built around all of this (Module 11)
- Who the players are, how RAG and agents work, and where the field is going (Module 12)

If someone asks you how AI works, at any depth from "explain it to my parents" to "explain it to our CTO," you've got it.

Understanding is half the job. The last two modules turn it into building: **Module 13** covers the engineering of LLM applications (calling models from code, structured outputs, testing and evals, shipping a service), and **Module 14** covers agents in practice and maps out your path as an AI engineer.
