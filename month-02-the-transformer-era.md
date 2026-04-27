# Month 02 — The Transformer Era (Days 31–60)

**Goal of this month:** Understand the architecture that powers *every* major LLM (GPT, Claude, Gemini, Llama, Qwen, Mistral, DeepSeek). By Day 60 you'll be able to draw a transformer block on a napkin and explain what each piece does.

---

## Week 1 — Attention Is All You Need

### Day 31 — The 2017 Paper That Changed Everything
"Attention Is All You Need" by Vaswani et al. (Google Brain, 2017) introduced the **Transformer**. It threw out RNNs entirely and showed that attention *alone*, applied in parallel, could outperform LSTMs on translation — while being far faster to train on GPUs. Every modern LLM descends from this paper.

### Day 32 — The Core Insight: Parallelism
RNNs process tokens one at a time (token 1 → token 2 → token 3...). That's a sequential bottleneck — you can't use a GPU's thousands of cores if the next step depends on the previous. Transformers process **the whole sequence at once** during training. That's why they scale.

### Day 33 — What Is "Attention"?
For each token, attention computes: *which other tokens in the sequence should I pay attention to, and how much?* It's a weighted sum where the weights are learned and depend on the content. When processing the word "it" in "The cat sat on the mat because it was tired," attention learns to weight "cat" highly.

### Day 34 — Queries, Keys, Values (Q, K, V)
The mechanical recipe: every token produces three vectors:
- **Query (Q):** "what am I looking for?"
- **Key (K):** "what do I contain?"
- **Value (V):** "what information will I share if you attend to me?"
Attention score = `Q · K` (dot product). Output = softmax(scores) × V. That's it. Everything else is plumbing.

### Day 35 — Multi-Head Attention
Instead of one attention computation, do 8, 16, 32, 64 of them in parallel — each "head" learns to attend to different things (syntax, coreference, topic, etc.). Concatenate the results. More heads = more types of relationships the model can track.

### Day 36 — Self-Attention vs Cross-Attention
- **Self-attention:** tokens attend to other tokens in the *same* sequence. Used inside encoders and decoders.
- **Cross-attention:** tokens in one sequence attend to tokens in another (e.g., decoder attending to encoder output in translation).
Decoder-only LLMs like GPT use *only* self-attention.

### Day 37 — Causal (Masked) Attention
For language *generation*, a token at position 5 cannot be allowed to peek at tokens 6, 7, 8 — that would be cheating. We mask future positions to `−∞` before the softmax. This is what makes GPT-style models "autoregressive" — they only see the past.

---

## Week 2 — The Full Transformer Block

### Day 38 — Anatomy of a Transformer Block
One block contains, in order:
1. **Layer Norm** (stabilization)
2. **Multi-head self-attention** (mixing across tokens)
3. **Residual connection** (add the input back — helps gradients flow)
4. **Layer Norm**
5. **Feed-Forward Network (FFN / MLP)** (two linear layers with a non-linearity, applied independently to each token)
6. **Residual connection**
A model is just N of these blocks stacked. GPT-3 has 96. Llama 3 70B has 80.

### Day 39 — The Feed-Forward Network (FFN)
Often overlooked, but the FFN is where most of a transformer's *parameters* live (typically 2/3 of them). It's a per-token "thinking" step: expand to 4× width, apply a non-linearity (GELU, SwiGLU), shrink back. Recent research suggests this is where factual knowledge is stored.

### Day 40 — Residual Connections (Skip Connections)
Each sub-layer's output is `output = sublayer(x) + x`. This trick (from ResNet, 2015) lets gradients flow backward through dozens of layers without vanishing. Without residuals, deep networks don't train.

### Day 41 — Layer Normalization
Re-centers and re-scales activations within each layer so they don't drift to extreme values. Training instability is the #1 enemy of deep networks; LayerNorm (and its cousin RMSNorm, used in Llama) is the seatbelt.

### Day 42 — Positional Encodings
Attention is **order-blind** by default — `"dog bites man"` and `"man bites dog"` would look identical. We inject position info by adding a position-dependent vector to each token's embedding. Original paper used sinusoidal encodings; modern models use **RoPE** (Rotary Position Embeddings) or **ALiBi** for better long-context behavior.

### Day 43 — Encoder, Decoder, Encoder-Decoder
The original 2017 transformer had both. Three families emerged:
- **Encoder-only** (BERT) — good for understanding/classification.
- **Decoder-only** (GPT, Llama, Qwen, Claude) — good for generation. **This won.**
- **Encoder-decoder** (T5, original translation models) — good for input→output transforms.

### Day 44 — Why Decoder-Only Won
Decoder-only models, trained on next-token prediction over the entire internet, turned out to be shockingly general. They can do classification, translation, summarization, coding, math — all by being prompted. Simplicity + scale beat the more "principled" encoder-decoder approach. This is the **bitter lesson** in action (Day 60).

---

## Week 3 — The GPT and BERT Families

### Day 45 — GPT-1 (2018)
117M parameters. OpenAI's first paper: pretrain a decoder transformer on books with next-token prediction, then fine-tune on specific tasks. Modest results, but the **recipe** was set.

### Day 46 — BERT (2018)
Google's encoder-only model, 340M params. Trained with **masked language modeling** (predict a hidden word from both sides). Crushed every NLP benchmark. Powered Google Search starting 2019. Still used today for embeddings and classification.

### Day 47 — GPT-2 (2019)
1.5B parameters. OpenAI initially refused to release it citing "misuse risk" — generating coherent multi-paragraph text was new and scary. First glimpse that **scale** alone produces qualitatively new abilities.

### Day 48 — GPT-3 (2020) — The "It's Real" Moment
175B parameters. The famous paper showed **few-shot learning** (in-context learning): give the model 3 examples in the prompt and it generalizes. No fine-tuning needed. This is when the field — and Silicon Valley — woke up.

### Day 49 — Emergent Abilities
As models grow past certain size thresholds, *new* capabilities appear that smaller models simply don't have: arithmetic, multi-step reasoning, code generation. This is controversial (some say it's a measurement artifact), but the pattern is real and reshaped the field's expectations.

### Day 50 — InstructGPT and ChatGPT (2022)
Raw GPT-3 was hard to use — you had to craft prompts cleverly. OpenAI applied **RLHF** (Month 5) to make it follow instructions. ChatGPT launched November 30, 2022, hit 100M users in 2 months, and made "LLM" a household term.

### Day 51 — GPT-4 (2023) and the Rise of Multimodality
Larger, smarter, accepted images as input. OpenAI stopped publishing architecture details. We're now in the **closed-frontier-model** era for the leading labs.

---

## Week 4 — The Open-Source Transformer Family

### Day 52 — Meta's LLaMA (Feb 2023)
Meta released a 7B–65B family for "research." Within a week, weights leaked publicly. The open-source LLM ecosystem exploded overnight. **Llama 2** (July 2023) was officially open-weight. **Llama 3** (2024), **Llama 4** (2025) followed.

### Day 53 — Mistral and Mixture-of-Experts
French startup **Mistral AI** released excellent small models (7B) and pioneered open **Mixture-of-Experts (MoE)** with Mixtral 8x7B — only ~2 of 8 experts run per token, so a 47B-parameter model has the *speed* of a 13B one.

### Day 54 — Alibaba's Qwen
Qwen (通义千问) is Alibaba's open model family. Qwen2, Qwen2.5, Qwen3 are *highly* competitive with closed models, strong at coding and multilingual (especially Chinese). When you `ollama pull qwen3`, you're downloading these weights.

### Day 55 — DeepSeek
Chinese lab that shocked the world in early 2025 with **DeepSeek-V3** and **DeepSeek-R1** — frontier-level reasoning at a fraction of the training cost, fully open-weight, MoE-based. Proved you don't need a $500M training run to compete.

### Day 56 — Anthropic's Claude
Closed-weight, API-only. Pioneered **Constitutional AI** (Month 5). Known for long context, strong reasoning, tool use, and the **MCP** protocol (Month 12) for connecting models to data.

### Day 57 — Google's Gemini
Google's flagship, the result of merging Google Brain and DeepMind. Natively multimodal (text+image+audio+video), very long context (1M+ tokens).

### Day 58 — The Architectural Convergence
Despite branding, the leading 2025 models all share roughly the same architecture: **decoder-only transformer + RoPE + RMSNorm + SwiGLU FFN + Grouped-Query Attention + (often) MoE**. The differences are in *data*, *scale*, and *post-training* — not the core network.

### Day 59 — Grouped-Query Attention (GQA)
A small but important optimization: instead of giving every attention head its own K and V vectors, share them across groups of heads. Drastically shrinks the **KV cache** (Month 9) without hurting quality. Used in Llama 2/3, Qwen, Mistral, basically everyone.

### Day 60 — Recap, "The Bitter Lesson," and Explain it Back
Rich Sutton's **Bitter Lesson** (2019): "The biggest lesson from 70 years of AI research is that general methods that leverage computation are ultimately the most effective." Translation: stop trying to be clever, just scale. The transformer + huge data + huge compute is the proof.

**Quiz yourself:**
1. What does Q · K compute, and what does softmax(Q·K) · V give you?
2. Why is causal attention needed for generation but not for classification?
3. Why did decoder-only beat encoder-decoder?
4. What's the difference between Llama, Qwen, and DeepSeek architecturally? (Mostly nothing!)
5. Where in a transformer does most of the parameter count live? (FFN.)
