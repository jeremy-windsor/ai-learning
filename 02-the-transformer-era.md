# Module 02 — The Transformer Era

**Mastery: 8% → 17%** · Prerequisites: Module 01

**Goal of this module:** Understand the architecture that powers *every* major LLM (GPT, Claude, Gemini, Llama, Qwen, Mistral, DeepSeek). By the end you'll be able to draw a transformer on a napkin, explain what each piece does, compute attention by hand on a toy example, and place today's model families on a timeline.

**How to read this module:** As in Module 01, a word in **bold** is being defined right there, and forward references point to the section that teaches them.

---

## Chapter 1 — Attention Is All You Need

### §2.1 — Bridge: What You Need From Module 01
Quick recap of the pieces this module builds on:
- A **token** is a word or piece of a word, the unit an LLM reads and writes (§1.1; Module 03 covers how text gets split up).
- Every token becomes an **embedding**: a vector of numbers where similar meanings sit close together (§1.42).
- The **dot product** measures how aligned two vectors are (§1.21). Attention is built on it.
- **RNNs/LSTMs** read one token at a time, which is slow and forgetful (§1.45–§1.47).
- **Attention** let a decoder look back at every input word and blend the relevant ones (§1.49).

One new word: a **prompt** is the text you give a model as input. The model's job is to continue it.

The problem left open in Module 01: word2vec gives "bank" one vector whether it's a river bank or a savings bank. We want **contextual embeddings**, where each token's vector is updated by the words around it. That's exactly what a transformer produces.

### §2.2 — The 2017 Paper That Changed Everything
"Attention Is All You Need" by Vaswani et al. (Google, 2017) introduced the **Transformer**. It threw out recurrence entirely and showed that attention *alone*, applied to all tokens in parallel, could beat LSTMs on translation while being far faster to train on GPUs. Every modern LLM descends from this paper.

### §2.3 — The Core Insight: Parallelism
RNNs process tokens one at a time (token 1 → token 2 → token 3...). Each step waits for the previous one, so a GPU's thousands of cores mostly sit idle. Transformers process **the whole sequence at once** during training: every token's attention can be computed simultaneously, as big matrix multiplies (§1.24). More parallel work means you can throw more GPUs at it, and that is why transformers scaled to billions of parameters while RNNs didn't.

### §2.4 — What Is "Attention"?
For each token, attention asks: *which other tokens in this sequence matter to me right now, and how much?* It then builds the token's new vector as a **weighted blend** of information from those tokens.

Example: in "The cat sat on the mat because **it** was tired," the token "it" should pull most of its information from "cat." After attention, the vector for "it" carries some cat-ness, so later layers know what "it" refers to.

Watch out for the word "weights," which now has two meanings:
- **Model weights** (§1.11) are the learned parameters, fixed after training.
- **Attention weights** are the percentages computed *fresh for every input* ("it" gives 70% to "cat," 5% to "mat"...). They're computed by the model weights, not stored.

### §2.5 — Softmax: Turning Scores Into Percentages
Attention first produces a raw **score** for each token (any number, positive or negative). **Softmax** turns a list of scores into percentages that are all positive and add up to 1:
1. Raise *e* (≈ 2.718) to the power of each score. This makes everything positive and exaggerates the gaps.
2. Divide each by the total.

```
scores   [2.0,  1.0,  0.1]
softmax  [0.66, 0.24, 0.10]   ← adds up to 1.0
```

The biggest score wins the largest share, but everyone gets something. Softmax shows up again at the very end of the model, where it turns scores for every possible next token into probabilities (§2.17, §3.30).

### §2.6 — Queries, Keys, Values (Q, K, V)
How does a token decide whom to attend to? Each token's embedding is multiplied by three learned matrices, `W_Q`, `W_K`, and `W_V`, to produce three vectors:
- **Query (Q):** "what am I looking for?"
- **Key (K):** "what do I contain?" (how others can find me)
- **Value (V):** "what information do I hand over if you attend to me?"

Then: **score = Q · K** (a dot product, so it's high when what I'm looking for matches what you contain), softmax the scores, and take the weighted sum of the Values.

**Network-engineer analogy:** a route lookup. The **query** is the destination address, the **keys** are the prefixes in the table, and the **values** are the next-hops. A normal router takes the single best match (longest prefix). Attention is a *soft* lookup: every entry matches a little, and you get a blend of all the next-hops weighted by how well each matched.

The three matrices are model weights learned during training. That's how the model learns *what* to look for.

### §2.7 — Attention, Step by Step
The full formula from the paper:

```
Attention(Q, K, V) = softmax( Q·K / √d ) × V
```

`d` is the length of the key vectors. Dividing by `√d` keeps scores from growing huge in high dimensions, which would make softmax put ~100% on a single token and stall learning.

A toy example with 3 tokens and 2-number vectors (you'll run this exact case in the lab). Focus on the token "sat," which asks "who did the sitting?":

| Token | Key | Value | Q·K for "sat" (Q = `[0, 2]`) | ÷ √2 | Softmax |
|-------|-----|-------|-----|------|---------|
| the | `[1, 0]` | `[1, 2]` | 0 | 0.00 | 0.14 |
| cat | `[0, 1]` | `[3, 4]` | 2 | 1.41 | **0.58** |
| sat | `[0.5, 0.5]` | `[5, 6]` | 1 | 0.71 | 0.28 |

New vector for "sat" = `0.14×[1,2] + 0.58×[3,4] + 0.28×[5,6] ≈ [3.3, 4.3]`. It's mostly "cat's" information. Every token does this at the same time, using matrix multiplies.

### §2.8 — Multi-Head Attention
One attention pass can only focus one way at a time. **Multi-head attention** runs several in parallel, each with its own `W_Q`, `W_K`, `W_V`. Each **head** can learn a different kind of relationship: one tracks "which noun does this pronoun mean," another "which verb goes with this subject," another just "the previous word."

The vector is split between heads. Llama 3 8B has 4,096-number vectors and 32 heads, so each head works on 128 numbers. The head outputs are concatenated back to 4,096 and mixed by one more learned matrix.

### §2.9 — Self-Attention vs. Cross-Attention
- **Self-attention:** tokens attend to other tokens in the *same* sequence.
- **Cross-attention:** tokens in one sequence attend to tokens in another, like the original translation setup where the French decoder looks at the English encoder output.

Decoder-only LLMs like GPT and Llama use *only* self-attention.

### §2.10 — Causal (Masked) Attention
When a model learns to *generate* text, the token at position 5 must not peek at tokens 6, 7, 8. Otherwise it would learn to copy the answer instead of predicting it. So before the softmax, every score for a future position is set to `−∞`. Softmax turns `−∞` into exactly 0%.

That makes the model **autoregressive**: each token only sees the past, and text is generated one token at a time, left to right (§3.29). In the lab, compare full and causal attention for the same toy input. "the" can only attend to itself, and "cat" can see "the" and "cat" but not "sat."

---

## Chapter 2 — The Full Transformer

### §2.11 — The Big Picture: Text In, Next Token Out
Before the details, here's the whole napkin drawing for a decoder-only LLM:

```
"The cat sat"
     │  tokenizer (Module 03)
     ▼
 3 token IDs (just numbers)
     │  embedding lookup (§3.14)
     ▼
 3 vectors × 4,096 numbers  (+ position info, §2.16)
     │
     ▼
 ┌──────────────────────────────┐
 │ Transformer block 1          │  attention: tokens share information
 │                              │  FFN: each token "thinks" on its own
 └──────────────────────────────┘
     ⋮   same block design repeated
         (32 blocks in Llama 3 8B, 80 in Llama 3 70B, 96 in GPT-3)
 ┌──────────────────────────────┐
 │ Transformer block N          │
 └──────────────────────────────┘
     │  final norm + output head (§2.17)
     ▼
 a score for every token in the vocabulary
     │  softmax → probabilities → pick one (§3.31)
     ▼
 " on"
```

Then " on" is appended to the input and the whole thing runs again for the next token. The vectors keep the same size (4,096 here) all the way through. Each block *refines* them, adding context and meaning.

### §2.12 — Anatomy of a Transformer Block
One block contains, in order (the modern "pre-norm" layout):
1. **Normalization** (§2.14)
2. **Multi-head self-attention** (§2.8): mixes information *across* tokens
3. **Residual add** (§2.13): add the block's input back in
4. **Normalization**
5. **Feed-forward network** (§2.15): processes each token *on its own*
6. **Residual add**

The rhythm to remember: **attention = tokens talk to each other; FFN = each token thinks alone.** A model is just N of these blocks stacked. (The 2017 paper put the normalization *after* each step. Moving it before turned out to train more stably, and almost everyone does that now.)

### §2.13 — Residual Connections and the Residual Stream
Each sub-layer's output is added to its input: `output = x + sublayer(x)`. The sub-layer only has to learn a *change* to the vector, not rebuild it from scratch. This trick, borrowed from ResNet (2015), lets gradients flow backward through dozens of layers without vanishing (§1.46). Without it, deep transformers don't train.

A useful mental picture is the **residual stream**: each token has one vector that flows straight up through the whole model, like a shared bus. Every attention and FFN sub-layer *reads* from the bus and *adds* its contribution back. By the top, the vector holds everything the layers have worked out about that token in context.

### §2.14 — Layer Normalization (LayerNorm and RMSNorm)
Re-scales each token's vector to a standard range so the numbers don't drift toward extreme values as they pass through many layers. Training instability is the #1 enemy of deep networks, and normalization is the seatbelt. **RMSNorm** is a simpler, cheaper version used by Llama, Qwen, and most modern models.

### §2.15 — The Feed-Forward Network (FFN)
Often overlooked, but the FFN (also called the **MLP**) holds most of a transformer's parameters, typically about two-thirds of the weights in the blocks. It's a small two-layer neural network (§1.29) applied to each token separately:
1. Expand the vector to a wider size (the 2017 paper used 4×: 512 → 2,048).
2. Apply a non-linear activation (§1.28): GELU, or **SwiGLU** in modern models. SwiGLU adds a third "gate" matrix, and Llama 3 8B expands 4,096 → 14,336.
3. Shrink back to the original size.

Research suggests much of a model's factual knowledge ("Paris is the capital of France") is stored in FFN weights. Attention routes information around; the FFN looks things up and transforms them. This is also the part that **Mixture-of-Experts** replaces (§2.36).

### §2.16 — Positional Encodings
Attention is **order-blind** by default. It compares every token with every other token, but nothing in the math says which came first. "Dog bites man" and "man bites dog" would look identical. So position information has to be injected:
- The original paper added a fixed wave pattern (**sinusoidal encoding**) to each token's embedding.
- Modern models use **RoPE** (Rotary Position Embedding), which rotates the Q and K vectors by an amount that depends on position. Attention scores then naturally reflect how far apart two tokens are. RoPE handles long context better.

Module 03 covers the options in detail (§3.21–§3.25).

### §2.17 — The Output Head: From Vector Back to a Word
After the last block, the model needs to turn the final vector of the *last* token into a prediction:
1. A final normalization.
2. Multiply by the **output matrix** (also called the **unembedding** or **LM head**), of size `vocab_size × 4,096`. That produces one raw score per vocabulary token: about 128,000 scores for Llama 3. These scores are called **logits**.
3. Softmax (§2.5) turns them into probabilities: " on" 62%, " down" 21%, " quietly" 3%...
4. A sampling rule picks one (§3.31).

Some models reuse the input embedding table as this output matrix ("tied embeddings," §3.17).

### §2.18 — Encoder, Decoder, Encoder-Decoder
The 2017 transformer had two halves. Three families grew out of it:
- **Encoder-only** (BERT): every token sees every other token in both directions. Good at *understanding*: classification, search, embeddings.
- **Decoder-only** (GPT, Llama, Qwen, and, as far as is publicly known, Claude and Gemini): causal attention, generates text left to right. **This won.**
- **Encoder-decoder** (T5, the original translation model): an encoder reads the input, and a decoder writes the output using cross-attention. Good at input → output transforms.

### §2.19 — Why Decoder-Only Won
Decoder-only models trained on next-token prediction over internet-scale text turned out to be shockingly general. They can classify, translate, summarize, write code, and do math, all just by being prompted. They're also simpler: one stack, one objective, and every token in the training text is a training example. Simplicity plus scale beat the more "principled" designs. That's the **Bitter Lesson** in action (§2.38).

---

## Chapter 3 — The GPT and BERT Families

### §2.20 — GPT-1 (2018)
117M parameters. OpenAI's first GPT ("Generative Pre-trained Transformer") paper: **pretrain** a decoder on thousands of books with next-token prediction, then **fine-tune** it on each specific task. The results were modest, but the **recipe** was set: general pretraining first, specialization second.

### §2.21 — BERT (2018)
Google's encoder-only model (110M and 340M parameters). Trained with **masked language modeling**: hide ~15% of the words and predict them using context from *both* sides. It crushed nearly every language benchmark and has powered parts of Google Search since 2019. BERT-style models are still widely used for embeddings and classification (§3.18).

### §2.22 — GPT-2 (2019)
1.5B parameters, about 10× GPT-1. OpenAI at first held back the full model, citing misuse risk. Coherent multi-paragraph text was new and unsettling. It released it in stages over 2019. GPT-2 was the first strong hint that **scale alone** produces new abilities.

### §2.23 — GPT-3 (2020): The "It's Real" Moment
175B parameters, about 100× GPT-2. The headline finding was **in-context learning** (also called **few-shot learning**): put three examples of a task in the prompt and the model picks up the pattern and continues it, with no fine-tuning and no weight changes. One general model could now be steered with text alone. This is when the field, and Silicon Valley, woke up. Module 04 covers the **scaling laws** that predicted this (§4.19).

### §2.24 — Emergent Abilities
As models grow, some abilities seem to appear suddenly: multi-step arithmetic, following complex instructions, writing working code. Smaller models just can't do them. This is debated. A 2023 paper argued the "suddenness" is partly an artifact of all-or-nothing scoring, and that skills improve smoothly under the hood. Either way, bigger models kept surprising people, and that reshaped the field's expectations.

### §2.25 — InstructGPT and ChatGPT (2022)
Raw GPT-3 was a **base model**: it continues text, but doesn't follow instructions. Ask it a question and it might write five more questions. OpenAI's **InstructGPT** (early 2022) fixed that with supervised fine-tuning plus **RLHF**: humans ranked answers, and the model was trained toward the preferred ones (§5.7). ChatGPT, built on GPT-3.5 with the same approach, launched **November 30, 2022**. It reportedly reached 100 million users in about two months and made "LLM" a household word. (Base vs. instruct models: §4.30, §5.1.)

### §2.26 — GPT-4 (2023) and Multimodality
Released March 2023. Much stronger at reasoning and coding, and it could accept **images** as well as text (**multimodal**). OpenAI published no architecture details, not even a parameter count. From here on, the leading labs mostly stopped publishing their designs: the **closed-frontier** era.

### §2.27 — Reasoning Models (2024–)
OpenAI's **o1** (September 2024) introduced a new axis of scaling. Before answering, the model writes a long internal **chain of thought**, working the problem step by step, checking itself, and backtracking. It's trained with reinforcement learning to make that thinking actually useful. Spending more compute *at answer time* is called **test-time compute**: harder problems get more thinking tokens.

DeepSeek-R1 (January 2025) showed the recipe in an open-weight model, and "thinking" modes quickly spread to Claude, Gemini, Qwen, and others. The architecture is the same transformer. What changed is the training (§5.14, §12.20).

---

## Chapter 4 — The Modern Model Landscape

### §2.28 — Open-Weight vs. Closed Models
- **Closed (API-only):** you send text to the company's servers and get text back. You never see the weights. Examples: GPT, Claude, Gemini.
- **Open-weight:** the parameter file (§1.30) is published, so you can download it and run it yourself, for example with `ollama pull`. Examples: Llama, Qwen, Mistral, DeepSeek, Gemma.

"Open-weight" isn't the same as "open source." You usually get the weights but not the training data or full training code, and licenses vary (§5.28).

| | Open-weight | Closed |
|---|---|---|
| **Pros** | Runs locally, private, no per-token fees, can be fine-tuned | Usually the most capable; no hardware to buy |
| **Cons** | You supply the GPU or VRAM; often a step behind the frontier | Data leaves your network; pricing and models can change under you |

Module 12 covers the current state of each lab (§12.1–§12.9). The sections below give the origin story of each.

### §2.29 — Meta's Llama
Meta released **LLaMA** (7B–65B) in February 2023 for researchers only. Within about a week the weights leaked publicly, and the open-model ecosystem exploded. **Llama 2** (July 2023) was officially open-weight with a commercial license. **Llama 3** (2024) and **Llama 4** (2025, Meta's first MoE models) followed. Llama's architecture became the template many others copied (§12.5).

### §2.30 — Mistral
A Paris startup founded in 2023. **Mistral 7B** (September 2023) beat much larger models and shipped under a permissive Apache 2.0 license. **Mixtral 8x7B** (December 2023) brought **Mixture-of-Experts** to open weights (§2.36).

### §2.31 — Alibaba's Qwen
Qwen (通义千问) is Alibaba's open model family. Qwen2.5 (2024) and Qwen3 (2025) are highly competitive, strong at coding, multilingual (especially Chinese), and released in sizes from under 1B to hundreds of billions of parameters, both dense and MoE. Qwen3 can switch between thinking and non-thinking modes (§2.27). Qwen models are among the most popular bases for community fine-tunes. When you `ollama pull qwen3`, you're downloading these weights (§12.6).

### §2.32 — DeepSeek
A Chinese lab that shocked the industry with **DeepSeek-V3** (December 2024: a 671B-parameter MoE that uses only 37B per token) and **DeepSeek-R1** (January 2025, the reasoning model from §2.27). Both were open-weight and near frontier quality. The widely quoted ~$5.6M training cost covers only the GPU time of the final training run, not research, failed experiments, or hardware, so the true cost was much higher. Even so, it was far cheaper than people assumed frontier models required, and it rattled markets (§12.7).

### §2.33 — Anthropic's Claude
Anthropic was founded in 2021 by former OpenAI researchers, and the first Claude was released in March 2023. Claude is closed-weight and API-only. Anthropic pioneered **Constitutional AI** (training against a written set of principles, §5.12) and created **MCP** (Model Context Protocol, November 2024), now a widely used standard for connecting models to tools and data (§12.18). The architecture isn't published.

### §2.34 — Google's Gemini
Google's flagship (December 2023), built after Google Brain and DeepMind merged. Trained from the start on text, images, audio, and video (**natively multimodal**), with very long context (Gemini 1.5 offered 1M tokens in 2024). Google also releases smaller open-weight models called **Gemma**.

---

## Chapter 5 — Modern Architecture Refinements

### §2.35 — The Architectural Convergence
Despite the branding, today's leading open models all share roughly the same design: **decoder-only transformer + RoPE + RMSNorm + SwiGLU FFN + Grouped-Query Attention + (often) Mixture-of-Experts**. The big differences are in *data*, *scale*, and *post-training* (Modules 04–05), not in the core network. If you understand this module, you understand the skeleton of almost every model you can download.

### §2.36 — Mixture-of-Experts (MoE)
In an MoE layer, the single FFN (§2.15) is replaced by many smaller FFNs called **experts**, plus a tiny **router**. For each token, the router scores the experts and sends the token to only the top few.

```
token vector → router → picks 2 of 8 experts → blend their outputs
```

That gives two different size numbers:
- **Total parameters:** every expert counts. All of them must sit in memory.
- **Active parameters:** only the experts a token actually uses. This sets the compute per token, and therefore speed.

| Model | Total | Active per token |
|-------|-------|------------------|
| Mixtral 8x7B | 46.7B | ~13B |
| DeepSeek-V3 | 671B | 37B |
| Qwen3-235B-A22B | 235B | 22B (the "A22B" in the name) |

**Pros:** much more knowledge for the same compute per token, so a big model runs at roughly the speed of a small one. **Cons:** you still need memory for *all* the experts, and balancing the router during training is tricky. That's why local engines often keep some experts in system RAM instead of VRAM, and why big MoE models get spread across GPUs by expert (§9.23).

### §2.37 — Grouped-Query Attention (GQA)
First, the **KV cache**: while generating, the model saves every previous token's Key and Value vectors so it doesn't recompute them for each new token. That cache grows with context length and quickly becomes a major memory cost (§9.1).

**GQA** shrinks it. Instead of every attention head having its own K and V, groups of query heads *share* one K/V pair. Llama 3 8B has 32 query heads but only 8 K/V heads, so the KV cache is 4× smaller with almost no quality loss.
- **MQA (Multi-Query Attention):** the extreme version, where all heads share a single K/V. Smallest cache, more quality risk.
- **MLA (Multi-head Latent Attention):** DeepSeek's alternative that compresses K/V into a small vector.

More detail: §9.3.

### §2.38 — The Bitter Lesson
Rich Sutton's 2019 essay *The Bitter Lesson*: "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin." Translation: hand-built cleverness keeps losing to simple methods plus more data and compute. The transformer, a simple, highly parallel design fed with internet-scale data on huge GPU clusters, is the clearest proof so far (§4.23).

The counterpoint is worth knowing too. Scale alone isn't the whole story. Data quality, post-training (RLHF, reasoning RL), and efficiency ideas like MoE and GQA are clever engineering, and DeepSeek showed that cleverness can substitute for a lot of money. The fair reading: *general* methods win, and clever work that helps those methods scale wins with them.

### 🛠️ Build It — Hands-On Lab
Compute attention by hand, with pure Python and no installs. This is exactly the example from §2.7.

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):
    biggest = max(scores)                      # subtract max for numeric safety
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

# Three tokens. In a real model these come from multiplying each token's
# embedding by learned matrices W_Q, W_K, W_V. Here we just make them up.
tokens = ["the", "cat", "sat"]
Q = [[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
K = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
V = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
d = len(Q[0])                                  # vector size per head

def attention(causal):
    for i, q in enumerate(Q):
        scores = [dot(q, k) / math.sqrt(d) for k in K]
        if causal:                              # no peeking at later tokens
            scores = [s if j <= i else float("-inf") for j, s in enumerate(scores)]
        weights = softmax(scores)
        out = [sum(w * v[c] for w, v in zip(weights, V)) for c in range(d)]
        pretty = ", ".join(f"{tokens[j]}={w:.2f}" for j, w in enumerate(weights))
        print(f"{tokens[i]:>4} attends: {pretty}  -> output {[round(x, 2) for x in out]}")

print("Full (bidirectional) attention:")
attention(causal=False)
print("\nCausal (GPT-style) attention:")
attention(causal=True)
```

What you should see: "sat" puts 58% of its attention on "cat." In causal mode, "the" can only attend to itself (100%), and "cat" can't see "sat." Things to try:
- Change "sat"'s query to `[1.0, 0.0]`. Who does it attend to now?
- Remove the `/ math.sqrt(d)` and multiply every query by 10. Watch softmax put nearly 100% on one token (why §2.7 scales the scores).
- Add a fourth token with its own Q, K, and V.

### §2.39 — Checkpoint — Explain It Back
1. Why did transformers train so much faster than RNNs?
2. What do Q, K, and V each represent, and what does `softmax(Q·K / √d) × V` produce?
3. What's the difference between model weights and attention weights?
4. Why is causal masking needed for generation?
5. Draw the napkin diagram: from text in to next token out.
6. What are the two halves of a transformer block, and what does each do?
7. Why did decoder-only beat encoder-decoder?
8. What's the difference between a base model and ChatGPT-style chat models?
9. In an MoE model, what's the difference between total and active parameters?
10. Architecturally, what's the difference between Llama, Qwen, and DeepSeek?

<details>
<summary>Answer key (try first!)</summary>

1. They process every token in parallel instead of one after another, so GPUs can be fully used.
2. Q is what a token is looking for, K is what each token offers, and V is the information it hands over. The formula gives each token a new vector: a blend of the other tokens' values, weighted by how well their keys matched its query.
3. Model weights are learned parameters fixed after training. Attention weights are percentages computed fresh for each input.
4. So the model can't see future tokens and must actually predict them. It also matches how generation works, one token at a time.
5. Text → tokens → embeddings (+ position) → N transformer blocks → final norm → output head → logits → softmax → pick a token → append and repeat.
6. Attention (tokens share information with each other) and the FFN (each token processed on its own), each wrapped in normalization and a residual add.
7. It's simpler, every token is a training signal, and next-token prediction at scale turned out to be general enough to do almost any task through prompting.
8. A base model just continues text. Chat models are further trained with supervised fine-tuning and RLHF to follow instructions and hold a conversation.
9. Total is every expert (all must be in memory). Active is only the experts each token uses (this sets speed).
10. Very little. All are decoder-only transformers with RoPE, RMSNorm, SwiGLU, and GQA, with DeepSeek and newer Qwen/Llama models adding MoE (and DeepSeek using MLA). The differences are mostly data, scale, and post-training.
</details>

Next: **Module 03 — Tokens & Embeddings**, a close look at what the model actually reads and how it picks each output token.
