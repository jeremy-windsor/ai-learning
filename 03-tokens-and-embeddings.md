# Module 03 — Tokens & Embeddings

**Mastery: 17% → 25%** · Prerequisites: Module 02

**Goal of this module:** Understand exactly what an LLM "sees" and how it picks each word it writes. It doesn't see characters or words. It sees **token IDs**, which it turns into **vectors**. By the end you'll know why "how many R's in strawberry?" trips models up, what a context window really is, why pricing is "per token," how embeddings power search, and what temperature and top-p actually do.

**How to read this module:** As before, **bold** marks a word being defined. This module zooms in on two ends of the napkin diagram in §2.11: the very bottom (text → tokens → vectors) and the very top (scores → the next token).

---

## Chapter 1 — What Is a Token?

### §3.1 — Computers Don't Read Words
Neural networks only work on numbers (§1.19). So before any text reaches a model, a **tokenizer** chops it into pieces called **tokens** and gives each piece an integer **token ID**. The full list of pieces the tokenizer knows is its **vocabulary**. The model only ever sees the IDs. On the way out, the tokenizer turns the model's chosen IDs back into text (**detokenizing**).

```
"Tokenization is fun!"  →  tokenizer  →  [3404, 2065, 374, 2523, 0]  →  model
```

The tokenizer is a separate, fixed piece of software that ships *with* each model. It's built once, before training, and never changes afterward. Every model family has its own.

### §3.2 — Why Not Just Use Characters?
You *could* give each character its own ID (`h`, `e`, `l`, `l`, `o`). The vocabulary would be tiny. But sequences get about 4× longer, and since attention cost grows with the *square* of the length (§3.27), that's roughly 16× more attention work. The model would also have to learn from scratch that "h-e-l-l-o" is one idea.

### §3.3 — Why Not Just Use Whole Words?
You *could* give each word one ID. But English alone has hundreds of thousands of words, plus every name, typo, new slang term ("doomscrolling"), and code identifier (`get_bgp_neighbors`). Any word not in the list is **out-of-vocabulary** and can't be represented. Dead end.

### §3.4 — Subword Tokenization: The Compromise
Modern tokenizers use **subwords**. Common words stay whole (`the`, ` neighbor`), and rare words get split into reusable pieces (`tokenization` → `token` + `ization`). You get a manageable vocabulary (about 32,000 to 200,000 entries) and nothing is ever unknown, because in the worst case a word just breaks into smaller pieces.

### §3.5 — Byte-Pair Encoding (BPE): How the Vocabulary Is Built
**BPE** was invented as a compression trick in 1994 and adapted for language models in 2015. Its training algorithm is surprisingly simple:
1. Start with every word split into single characters.
2. Count every pair of neighboring pieces across the whole training text.
3. Merge the most common pair into one new token and add it to the vocabulary.
4. Repeat steps 2–3 until the vocabulary reaches the size you want.

A real run on a tiny corpus (`low ×3, lower ×2, newest ×3, widest ×2`; `_` marks the end of a word). You'll run this yourself in the lab:

| Merge | Pair glued together | Times seen |
|------|------|------|
| 1 | `l` + `o` → `lo` | 5 |
| 2 | `lo` + `w` → `low` | 5 |
| 3 | `e` + `s` → `es` | 5 |
| 4 | `es` + `t` → `est` | 5 |
| 5 | `est` + `_` → `est_` | 5 |

After 8 merges, `newest` is just `new | est_`, while the rarer `widest` is `w | i | d | est_`. BPE discovered the suffix "-est" on its own, just from counting. Run it over trillions of characters with ~100,000 merges and you get a vocabulary where common words and word-parts are single tokens. **GPT-2 through GPT-4o, Llama 3, Qwen, Mistral, and DeepSeek all use BPE variants.**

To tokenize new text later, the tokenizer replays the same merges in the same order.

### §3.6 — Byte-Level BPE: Never Unknown
Since GPT-2, most tokenizers run BPE on **bytes** instead of characters. All text, in any language, plus emoji and even binary junk, is stored as bytes, and there are only 256 possible byte values. Start the vocabulary with all 256 and *any* input can be tokenized, even text the tokenizer has never seen. Common byte sequences get merged into bigger tokens just as in §3.5.

### §3.7 — WordPiece, Unigram, and SentencePiece
Other tokenizer designs you'll see on model cards:
- **WordPiece** (BERT): like BPE, but it picks merges by a slightly different score. Word-continuation pieces are marked with `##` (`token`, `##ization`).
- **Unigram** (T5, many multilingual models): works in reverse. It starts with a huge vocabulary and prunes the pieces that help least.
- **SentencePiece** (Google): a *library* that runs BPE or Unigram directly on raw text, treating spaces as an ordinary symbol (`▁`). That makes it work the same for languages without spaces, like Chinese and Japanese. Llama 1 and 2 used it. Llama 3 switched to a tiktoken-style BPE with a bigger vocabulary.

### §3.8 — Tiktoken and Vocabulary Sizes
OpenAI's tokenizer library is **tiktoken**. Its vocabularies have grown:

| Encoding | Used by | Vocabulary |
|----------|---------|-----------|
| `r50k_base` | GPT-3 | ~50,000 |
| `cl100k_base` | GPT-3.5, GPT-4 | ~100,000 |
| `o200k_base` | GPT-4o and later | ~200,000 |

Llama 3 uses ~128,000 tokens and Qwen ~152,000. A bigger vocabulary means fewer tokens per sentence, especially for non-English text and code. That's cheaper and faster, and it fits more into the context window. The cost: a bigger embedding table (§3.14) and output layer.

### §3.9 — Let's See It: A Real Tokenization
Real output from GPT-4's `cl100k_base` tokenizer:

```
"Tokenization is fun!"
→ ["Token", "ization", " is", " fun", "!"]
→ [3404,    2065,      374,   2523,   0]
```

Things to notice:
- **Leading spaces are part of the token.** `" is"` (with a space) is a different token from `"is"`.
- **Capitalization matters.** `"Token"` and `"token"` are different IDs.
- **Same text, different tokenizer, different IDs.** GPT-4o's `o200k_base` gives `[4421, 2860, 382, 2827, 0]` for the same sentence. IDs only mean something to the model they belong to.

### §3.10 — Why "How Many R's in Strawberry?" Fails
The model never sees letters. Here's what `cl100k_base` actually does:

| Text | Tokens |
|------|--------|
| `"strawberry"` | `str` · `aw` · `berry` (3 tokens) |
| `" strawberry"` (with a leading space, as in mid-sentence) | ` strawberry` (1 token) |

Mid-sentence, the whole word is *one* ID, a single number with no letters visible inside it. To count R's, the model must have *memorized* how that token is spelled, which is a separate fact it picks up only indirectly during training. Modern reasoning models usually get it right by spelling the word out letter by letter in their thinking (§2.27), which turns each letter into its own token.

### §3.11 — Numbers, Code, and Other Languages
Tokenization explains several famous LLM quirks:
- **Numbers get chopped oddly.** `12345678` becomes `123` · `456` · `78`, so digits don't line up by place value. That's one reason plain LLMs are shaky at arithmetic, and why tool use (a calculator or code) helps (§12.16).
- **Non-English text often costs more tokens.** In `cl100k_base`, the Chinese word 路由器 ("router") is three tokens, one per character. In older tokenizers, many languages took 2–4× more tokens than English for the same meaning, so they were slower and more expensive.
- **Code is full of odd pieces.** `def add(a, b): return a + b` is 10 tokens, including pieces like `(a` and `):`.
- **Glitch tokens.** In 2023 people found that GPT-3 behaved bizarrely on the token `" SolidGoldMagikarp"`. It was in the vocabulary (it was a common Reddit username in the tokenizer's training text) but almost never appeared in the model's training data, so its embedding was never properly learned.

### §3.12 — Special Tokens
Vocabularies also contain **special tokens** that never appear in normal text. They act as control signals:
- **BOS / EOS** (beginning / end of sequence): mark where a document starts and ends. When the model outputs EOS, generation stops (§3.35).
- **Chat tokens** like `<|im_start|>`, `<|im_end|>` (Qwen's ChatML) or `<|start_header_id|>`, `<|eot_id|>` (Llama 3): mark who is speaking. The **chat template** (§5.3) defines how a conversation gets wrapped in them.
- **Tool and thinking tokens:** many newer models have tokens that mark tool calls or `<think>` sections.

Get special tokens wrong (for example, the wrong chat template) and the model rambles, never stops, or answers as the wrong speaker.

### §3.13 — Tokens, Pricing, and Context Windows
Everything about using an LLM is measured in tokens:
- **Rules of thumb for English:** 1 token ≈ 4 characters ≈ ¾ of a word. 1,000 tokens ≈ 750 words ≈ 1.5 pages.
- **Pricing:** APIs charge per token, usually quoted per million. **Output tokens typically cost several times more than input tokens**, because each one needs its own forward pass (§3.29). Reasoning models also bill for their thinking tokens.
- **Context window:** the maximum number of tokens the model can handle at once, *prompt plus response combined*. Think of it as the model's working memory: anything outside the window, the model simply can't see. A 1M-token window holds roughly 750,000 words, about 1,500 pages.

---

## Chapter 2 — From Tokens to Vectors: Embeddings

### §3.14 — The Embedding Layer
The first layer of every LLM is a giant lookup table with one row per vocabulary token. For Llama 3 8B, it's `128,256 rows × 4,096 columns`. Token ID 2065 → grab row 2065 → a vector of 4,096 numbers. *That* vector is what the transformer blocks work on.

Two things worth knowing:
- **It's a lot of parameters:** `128,256 × 4,096 ≈ 525 million` numbers, about 6.5% of the whole 8B model.
- **It's just the one-hot idea from §1.41, done efficiently.** Multiplying a one-hot vector by a matrix just selects one row, so "lookup" and "one-hot × matrix" give the same answer. The table is learned by backprop (§1.34) along with every other weight.

### §3.15 — From Static to Contextual Embeddings
The vector from the lookup table is **static**: token "bank" always starts as the same vector, the word2vec limitation from §1.44. But that's only the starting point. As the vector rises through the residual stream (§2.13), every attention layer mixes in information from the surrounding tokens. By the middle layers, "bank" in "river bank" and "bank" in "bank account" have become very different vectors. Those **contextual embeddings** are where an LLM's understanding actually lives.

### §3.16 — Embedding Dimensions
The length of each token's vector is the model's **hidden size** (also called **model dimension** or `d_model`). Typical sizes: 768 (BERT-base), 4,096 (Llama 3 8B), 8,192 (Llama 3 70B), 12,288 (GPT-3 175B). Bigger means more room to store meaning, but more memory and slower math. The hidden size stays the same through every block (§2.11).

### §3.17 — Tied vs. Untied Embeddings
At the top, the output head (§2.17) needs its own `vocab_size × hidden_size` matrix to turn the final vector into one score per token. Some models **tie** the two, reusing the input embedding table as the output matrix. That saves hundreds of millions of parameters, which matters most for small models (Llama 3.2 1B and 3B and Gemma tie them). Bigger models like Llama 3 8B and 70B keep them separate (**untied**) for a little more quality.

### §3.18 — Sentence and Document Embeddings (A Different Beast)
When people say "embeddings" while talking about search or RAG, they usually mean something different: **one vector for a whole sentence, paragraph, or document**. These come from separate **embedding models**, such as OpenAI's `text-embedding-3-small`/`large`, `all-MiniLM-L6-v2`, `nomic-embed-text`, BGE, and Qwen3-Embedding.

How they work:
- Run the text through a transformer, then **pool** the per-token vectors into one, usually by averaging them.
- Train with **contrastive learning**: show the model pairs that belong together (a question and its answer, a title and its article) and pull their vectors together, while pushing unrelated texts apart.

Practical facts:
- **Sizes vary:** 384 numbers (MiniLM), 768 (nomic), 1,536 or 3,072 (OpenAI). Some models let you cut vectors shorter with only a small quality loss (**Matryoshka embeddings**).
- **Never mix models.** Vectors from two different embedding models live in unrelated spaces, so comparing them is meaningless. If you switch models, re-embed everything.
- This kind of model encodes both sides separately (a **bi-encoder**), which is fast. A **reranker** (cross-encoder) reads query and document *together* for a more accurate score, and is often used as a second pass on the top results (§12.13).

### §3.19 — Cosine Similarity in Practice
You learned the formula in §1.23. What changes in real systems:
- **Most embedding models output normalized vectors** (length 1). For those, cosine similarity equals the plain dot product, which is cheaper, so databases often use "inner product" as the metric.
- **Scores are relative, not absolute.** One model's "0.8" isn't another model's "0.8." Many models rarely produce scores below 0 for real text. Rank results by score, and if you need a cutoff, tune it on your own data rather than picking a magic number.
- **Similar isn't always relevant.** "How do I *enable* OSPF?" and "How do I *disable* OSPF?" are nearly identical as vectors. That's one reason real search systems add keyword matching and rerankers.

### §3.20 — Vector Databases and Approximate Search
Comparing a query against 10 million vectors one by one is too slow. A **vector database** (pgvector, Qdrant, Milvus, Weaviate, Pinecone, Chroma) stores the embeddings and finds the closest ones fast, using **approximate nearest-neighbor (ANN)** search.

The most common ANN method, **HNSW** (Hierarchical Navigable Small World), builds a layered graph that links each vector to its neighbors. A search starts on a sparse top layer, takes long hops toward the right region, then drops to denser layers for local steps. It's like hierarchical routing: get to the right area over the backbone first, then route within the area.

Trade-offs and features to know:
- **"Approximate"** means it can occasionally miss a true top match. You tune the speed-versus-recall trade-off.
- **Metadata filters:** "only search documents from the network team, newer than 2025."
- **Hybrid search:** combine vector similarity with classic keyword search (BM25), so exact terms like an error code or `EXSTART` still match.

This is the retrieval engine under every RAG system (§12.12–§12.14).

---

## Chapter 3 — Position, Context, and Sequence Length

### §3.21 — Why Position Has to Be Encoded
As §2.16 explained, attention compares every token with every other token but has no built-in sense of order. Without position information, "dog bites man" and "man bites dog" look the same. So position is injected somehow. Here are the main schemes, in historical order.

### §3.22 — Sinusoidal Positional Encoding (the Original Transformer)
Add a fixed pattern of sine and cosine waves at different frequencies to each token's embedding. Every position gets a unique pattern, like a binary counter where low digits flip fast and high digits flip slowly. Simple, nothing to learn, but it doesn't work well beyond the lengths seen in training.

### §3.23 — Learned Positional Embeddings
GPT-2 and BERT learned a separate vector for each position (position 1, position 2, ... up to 1,024 or 512). Problem: there's a hard wall. There is no vector for position 1,025, so the model can't go past its training length.

### §3.24 — RoPE (Rotary Position Embedding)
The modern winner, used by Llama, Qwen, Mistral, DeepSeek, and most others. Instead of adding anything to the embedding, RoPE **rotates** pieces of each Q and K vector (§2.6) by an angle that grows with position.

Picture each pair of numbers in the vector as a clock hand. Token 1's hand is turned a little, token 50's a lot. When attention takes the dot product of Q and K, the result depends on the *difference* between the two angles, which is how far apart the tokens are, not their absolute positions. **Relative** position is what language mostly cares about ("the word right before me").

RoPE can also be stretched after training. Tricks like **YaRN** and "RoPE scaling" let a model trained on 8,000 tokens be extended to 128,000 or more with a short extra round of training. That's how many models got their long context windows.

### §3.25 — ALiBi
An alternative: add no position information to the vectors at all, and instead subtract a penalty from attention scores that grows with distance, so far-away tokens get a little less attention. It generalizes well to longer inputs. It was used by BLOOM and MPT, but RoPE won.

### §3.26 — Context Window Length: 1K → 1M+
- GPT-2 (2019): 1,024 tokens
- GPT-3 (2020): 2,048
- GPT-4 (2023): 8K → 32K → 128K
- Claude 3 (2024): 200K
- Gemini 1.5 (2024): 1M+
- Llama 4 Scout (2025): 10M; Llama 4 Maverick (2025): 1M
- GPT-5 (2025): 400K (272K input + 128K output)
- Qwen3.6 / DeepSeek-V4 (2026): up to 1M

The growth came from RoPE extensions plus memory and speed tricks (Module 09: FlashAttention, the paged KV cache).

### §3.27 — Why Long Context Is Hard: Attention Is O(n²)
**Big-O notation** describes how cost grows as input grows. **O(n)** ("linear") means double the input, double the cost. **O(n²)** ("quadratic") means double the input, *four times* the cost.

Attention scores every token against every other token. With `n` tokens that's `n × n` scores:

| Context | Pairs to score |
|---------|----------------|
| 1,000 tokens | 1 million |
| 10,000 tokens | 100 million |
| 100,000 tokens | 10 billion |

So 100× more context means 10,000× more attention work. That's the fundamental wall, and most long-context engineering works around it. FlashAttention (§9.9) makes the work cheaper without changing the result. Sliding-window attention (§9.12) only looks at nearby tokens.

There's a second cost: the **KV cache** (§2.37) grows linearly with context, and at long lengths it can take more memory than the model weights (§9.2).

### §3.28 — Advertised vs. Effective Context
A 1M-token window doesn't mean the model *uses* 1M tokens well. Research found models recall facts at the start and end of a long prompt much better than facts buried in the middle (**"lost in the middle,"** 2023). The popular **needle-in-a-haystack** test (hide one fact in a huge document and ask for it) is easy compared to real tasks that need *combining* many facts spread through the text. Harder benchmarks like RULER show the usable length is often well below the advertised maximum. Practical advice: put the most important material at the start or end, and don't stuff the window just because you can (§12.15).

---

## Chapter 4 — Generation: How the Next Token Gets Picked

### §3.29 — Autoregressive Generation
The model writes one token at a time (§2.10):

```
prompt → forward pass → pick token 1 → append it
       → forward pass → pick token 2 → append it
       → ... until EOS, a stop sequence, or max tokens (§3.35)
```

This splits into two phases with very different performance:
- **Prefill:** the whole prompt is processed in one parallel pass. It's fast per token and limited by raw GPU compute. This is your wait for the first word, called **time to first token**.
- **Decode:** each new token needs its own forward pass through *all* the weights. It's slow per token and limited by memory bandwidth (§6.10). This is the "typing speed" you see.

The KV cache (§2.37, §9.1) is what keeps decode from recomputing the whole prompt every step. Module 08 builds on this split (§8.2).

### §3.30 — Logits and Softmax (Recap)
From §2.17: the output head gives one raw score (**logit**) per vocabulary token, about 128,000 numbers for Llama 3. Softmax (§2.5) turns them into probabilities that add up to 1. Everything below is about how you choose one token from that list.

### §3.31 — Temperature
**Temperature** divides every logit by a number `T` before the softmax:
- **T < 1** exaggerates the gaps: the top choice gets even more likely. Output is more focused and predictable.
- **T > 1** shrinks the gaps: unlikely tokens get a bigger share. Output is more varied and creative, and more likely to go off the rails.
- **T = 0** is treated as "always take the top token," called **greedy** decoding. (Libraries special-case it, since you can't divide by zero.)

Real output from the lab, after the prompt "The cat sat on the":

| Temperature | mat | floor | couch | roof | moon |
|-------------|-----|-------|-------|------|------|
| 0.5 | **0.84** | 0.11 | 0.04 | 0.00 | 0.00 |
| 1.0 | **0.61** | 0.22 | 0.14 | 0.03 | 0.00 |
| 1.5 | **0.49** | 0.25 | 0.18 | 0.07 | 0.02 |

Temperature doesn't change the *order*. It only changes how lopsided the bet is.

### §3.32 — Top-k, Top-p, and Min-p
These **truncate** the list (cut off the unlikely tail) before sampling, so a one-in-a-thousand bad token can never be picked:
- **Top-k:** keep only the `k` most likely tokens (say 40).
- **Top-p** (nucleus sampling): keep the smallest set of top tokens whose probabilities add up to at least `p`. With `p = 0.9` and the T=1.0 row above: mat (0.61) + floor (0.22) = 0.83, not enough; + couch (0.14) = 0.97 ≥ 0.9, so keep those three. It adapts automatically: it keeps few tokens when the model is confident and many when it's unsure.
- **Min-p:** keep tokens with at least `min_p ×` the top token's probability. With `min_p = 0.1` and a top choice at 0.61, anything under 0.061 is dropped.

After truncating, the remaining probabilities are rescaled to add up to 1, and one token is drawn at random by weight.

### §3.33 — Repetition Penalties and Sensible Defaults
- **Repetition / presence / frequency penalties** lower the logits of tokens that already appeared, to stop loops like "the the the" or the same sentence repeated. Too much and the model avoids words it legitimately needs.
- **Typical defaults:** temperature 0.6–0.8 with top-p 0.9–0.95 for chat, and lower temperature for code or factual extraction. Model cards often publish recommended settings. Qwen3, for example, suggests temperature 0.6, top-p 0.95, top-k 20 in thinking mode. Ollama applies the model's recommended settings from its Modelfile unless you override them.

| Setting | Pros | Cons |
|---------|------|------|
| Low temperature / greedy | Consistent, good for code and extraction | Repetitive; can get stuck in loops |
| High temperature | Creative, varied | More factual errors and nonsense |
| Top-p / min-p truncation | Cuts off garbage tokens | One more knob to tune |

### §3.34 — Why "Temperature 0" Isn't Truly Deterministic
Even with greedy decoding, you can get slightly different answers run to run. GPUs add numbers in parallel, and floating-point addition gives tiny differences depending on order. If two tokens are nearly tied, a difference in the 7th decimal place can flip the choice, and the rest of the response diverges from there. On a server, *which other requests share your batch* can change the math too (§9.20). For reproducibility you need a fixed seed, the same hardware, and ideally batch-invariant kernels.

### §3.35 — Stopping: EOS, Stop Sequences, and Max Tokens
Generation ends when one of these happens:
- The model outputs its **EOS / end-of-turn token** (§3.12). This is the normal ending, learned from the chat template.
- The output contains a **stop sequence** you set, such as `"\n\nUser:"`.
- It hits **max tokens**, your hard cap on response length.

If a model never stops or keeps talking as the "user," the chat template or EOS settings are usually wrong (§5.3).

### 🛠️ Build It — Hands-On Lab
**Lab 1: train a BPE tokenizer from scratch (§3.5).** Pure Python, nothing to install.

```python
from collections import Counter

# A tiny "training corpus". Each word is split into characters,
# and "_" marks the end of a word so merges can't cross word boundaries.
corpus = "low low low lower lower newest newest newest widest widest"
words = Counter(tuple(w) + ("_",) for w in corpus.split())

def most_common_pair(words):
    pairs = Counter()
    for word, count in words.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += count
    return pairs.most_common(1)[0]

def merge(words, pair):
    merged = {}
    for word, count in words.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])   # glue the pair into one token
                i += 2
            else:
                out.append(word[i])
                i += 1
        merged[tuple(out)] = count
    return merged

for step in range(1, 9):
    pair, count = most_common_pair(words)
    words = merge(words, pair)
    print(f"merge {step}: {pair[0]!r} + {pair[1]!r} (seen {count}x)")

print("\nFinal tokenization of each word:")
for word in words:
    print("  ", " | ".join(word))
```

Things to try: replace `corpus` with a paragraph of your own (a config file, a network runbook) and raise the number of merges to 30. Which pieces become tokens first?

**Lab 2: temperature and top-p (§3.31–§3.32).** Pure Python.

```python
import math
import random

random.seed(42)

# Pretend the model just read "The cat sat on the" and produced these logits.
logits = {" mat": 4.0, " floor": 3.0, " couch": 2.5, " roof": 1.0, " moon": -1.0}

def softmax(scores, temperature=1.0):
    scaled = {tok: s / temperature for tok, s in scores.items()}
    biggest = max(scaled.values())
    exps = {tok: math.exp(s - biggest) for tok, s in scaled.items()}
    total = sum(exps.values())
    return {tok: e / total for tok, e in exps.items()}

def top_p_filter(probs, p):
    kept, running = {}, 0.0
    for tok, pr in sorted(probs.items(), key=lambda kv: -kv[1]):
        kept[tok] = pr
        running += pr
        if running >= p:
            break
    total = sum(kept.values())
    return {tok: pr / total for tok, pr in kept.items()}   # renormalize

def sample(probs):
    return random.choices(list(probs), weights=list(probs.values()))[0]

for t in (0.5, 1.0, 1.5):
    probs = softmax(logits, t)
    row = "  ".join(f"{tok.strip()}={pr:.2f}" for tok, pr in probs.items())
    print(f"temperature {t}: {row}")

probs = top_p_filter(softmax(logits, 1.0), p=0.9)
print("\ntop-p 0.9 keeps:", {tok.strip(): round(pr, 2) for tok, pr in probs.items()})
print("10 samples:", [sample(probs).strip() for _ in range(10)])
```

Things to try: set temperature to `0.1` and `5.0`. Change `p` to `0.5`. Can " moon" ever be picked with top-p 0.9?

**Lab 3: a real tokenizer.** Run `pip install tiktoken` (ideally inside a virtual environment: `python3 -m venv venv && source venv/bin/activate`), then:

```python
import tiktoken

samples = [
    "Tokenization is fun!",
    "strawberry",
    " strawberry",
    "OSPF neighbor stuck in EXSTART",
    "路由器",                      # "router" in Chinese
    "12345678",
    "def add(a, b): return a + b",
]
for name in ("cl100k_base", "o200k_base"):   # GPT-4 and GPT-4o tokenizers
    enc = tiktoken.get_encoding(name)
    print(f"\n{name}  (vocab size {enc.n_vocab:,})")
    for text in samples:
        ids = enc.encode(text)
        pieces = [enc.decode([i]) for i in ids]
        print(f"  {len(ids):>2} tokens  {pieces}")
```

Add a few lines from a real router config. How many tokens is your longest config?

**Optional, bigger installs:**
- Compare Qwen's tokenizer: `pip install transformers`, then `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")`. (It's a small download and doesn't need an account. Meta's Llama repos require accepting a license on Hugging Face first.)
- Sentence embeddings: `pip install sentence-transformers` (this pulls in PyTorch, a large download), load `all-MiniLM-L6-v2`, embed five sentences about networking and five about cooking, and print the cosine similarity table. The topics should cluster.
- With Ollama running, send the same prompt with `"options": {"temperature": 0}`, then `0.7`, then `1.5` to `http://localhost:11434/api/generate`, and compare.

### §3.36 — Checkpoint — Explain It Back
1. What is a token, and why don't models use whole words or single characters?
2. How does BPE build its vocabulary?
3. Why does "how many R's in strawberry?" trip models up?
4. What is a context window, and why do output tokens cost more than input tokens?
5. What does the embedding layer do, and how is a token embedding different from a sentence embedding?
6. Why can't you compare vectors from two different embedding models?
7. How does RoPE encode position, and why is "relative" useful?
8. Why is attention O(n²), and what does that mean for long context?
9. What's the difference between prefill and decode?
10. What do temperature and top-p each do?

<details>
<summary>Answer key (try first!)</summary>

1. A token is a word or word-piece with an integer ID. Words leave too many words out of the vocabulary; characters make sequences long and expensive. Subwords balance the two.
2. Start from characters (or bytes), repeatedly merge the most frequent neighboring pair into a new token, and stop at the target vocabulary size.
3. Mid-sentence, " strawberry" is a single token, so the model never sees individual letters. It has to have memorized the spelling.
4. The maximum number of tokens (prompt plus response) the model can handle at once. Each output token needs its own forward pass, while the prompt is processed in one parallel pass.
5. It looks up one learned vector per token ID. A token embedding is one vector per token inside the LLM. A sentence embedding is one vector for a whole text, produced by a separate embedding model.
6. Each model learns its own unrelated "map" of meaning, so positions in one space mean nothing in the other.
7. It rotates parts of Q and K by an angle based on position, so their dot product depends on the distance between tokens. Language cares more about "how far apart" than "absolute position number."
8. Every token is scored against every other token, so n tokens means n² scores. 10× more context means about 100× more attention work.
9. Prefill processes the whole prompt in one parallel pass (limited by compute). Decode generates one token per pass (limited by memory bandwidth).
10. Temperature makes the probability list more lopsided (low) or more even (high). Top-p keeps only the most likely tokens that together reach probability p, cutting off the unlikely tail.
</details>

You now understand what the model *sees* and how it produces output. Next: how it learned to do any of this in the first place.
