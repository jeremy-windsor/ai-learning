# Month 03 — Tokens & Embeddings (Days 61–90)

**Goal of this month:** Understand exactly what an LLM "sees." It doesn't see characters or words — it sees **token IDs**, which it converts into **vectors**. By Day 90 you'll know why "strawberry has 3 R's" trips up models, what a context window actually is, and why pricing is "per token."

---

## Week 1 — What Is a Token?

### Day 61 — Computers Don't Read Words
Neural networks only operate on **numbers**. So before any text reaches a model, a **tokenizer** chops the text into pieces and assigns each piece a unique integer ID. The model only ever sees those integers.

### Day 62 — Why Not Just Use Characters?
You *could* tokenize at the character level (`['h','e','l','l','o']`). But sequences become very long, and the model has to learn that "h-e-l-l-o" means a greeting from scratch. Inefficient.

### Day 63 — Why Not Just Use Whole Words?
You *could* use one ID per word. But English alone has hundreds of thousands of words; rare/new words ("doomscrolling," your last name, code identifiers) become **out-of-vocabulary**. Dead-end.

### Day 64 — Subword Tokenization: The Compromise
Modern tokenizers split text into **subword pieces** — common words stay whole (`the`, `and`), rare words get broken into pieces (`tokenization` → `token` + `ization`). Best of both worlds: small vocabulary (~32k–200k), no unknown words.

### Day 65 — Byte-Pair Encoding (BPE)
Invented for compression in 1994, repurposed for NLP. The training algorithm: start with characters, repeatedly merge the most common adjacent pair into a new token. Run for N merges. The result is a vocabulary where common patterns are single tokens. **GPT-2/3/4, Llama, Qwen, Mistral all use BPE variants.**

### Day 66 — WordPiece and SentencePiece
- **WordPiece** (BERT) — similar to BPE, slightly different merge criterion.
- **SentencePiece** (Google, T5, Llama) — works on raw bytes, no pre-tokenization needed, language-agnostic. Handles Chinese, Japanese, code, emoji uniformly.

### Day 67 — Tiktoken and the OpenAI Tokenizers
OpenAI uses BPE via their **tiktoken** library. GPT-3 used `r50k_base` (~50k tokens). GPT-4 uses `cl100k_base` (~100k). GPT-4o uses `o200k_base` (~200k) — bigger vocab = fewer tokens per sentence = cheaper and faster, especially for non-English.

### Day 68 — Let's See It: A Tokenization Example
The sentence `"Tokenization is fun!"` might become:
`["Token", "ization", " is", " fun", "!"]` → `[5963, 2065, 318, 1257, 0]`
Notice: leading spaces are part of the token. Capitalization changes the token. This is why prompt formatting matters.

### Day 69 — Why "Strawberry Has Three R's" Fails
The model doesn't see `s-t-r-a-w-b-e-r-r-y`. It sees something like `["straw", "berry"]` — two tokens. Counting characters inside a token is a *separate* skill the model has to learn statistically. Newer models (with reasoning, tool use, or character-level fallback) get this right; older ones don't.

### Day 70 — Tokens, Pricing, and Context Windows
APIs charge **per token** (input + output). 1 token ≈ 4 English characters ≈ 0.75 words on average. A 1M-token context window holds roughly 750,000 words ≈ 1,500 pages of text.

---

## Week 2 — From Tokens to Vectors: Embeddings

### Day 71 — The Embedding Layer
The first layer of every LLM is a giant lookup table: **vocab_size × embedding_dim**. For Llama 3 8B that's `128,256 × 4,096`. Each token ID becomes a row of that table — a vector of ~4,000 numbers. *That* is what the rest of the network operates on.

### Day 72 — What Does an Embedding "Mean"?
Nothing, individually. Collectively, embeddings learned during training place semantically similar tokens near each other in high-dimensional space. The famous `king − man + woman ≈ queen` analogy works because of this geometric structure.

### Day 73 — Embedding Dimensions
Typical sizes: 768 (BERT-base), 4,096 (Llama 3 8B), 8,192 (Llama 3 70B), 12,288 (GPT-3 175B). Bigger = more capacity, more memory, slower. Embedding dim is also called the **hidden size** or **model dimension** (`d_model`).

### Day 74 — Tied vs Untied Embeddings
The *output* layer of an LLM also needs a `vocab_size × hidden_size` matrix to convert the final hidden state back to per-token logits. Many models **tie** the input and output embeddings (share weights) to save parameters. Llama unties them.

### Day 75 — Sentence/Document Embeddings (Different Beast)
When people say "embeddings" in a RAG context, they usually mean **dense sentence embeddings** from models like `text-embedding-3-large`, `bge`, `nomic-embed`. These are typically encoder models trained with **contrastive learning** so similar sentences are close in vector space. Used for search, clustering, RAG retrieval.

### Day 76 — Cosine Similarity
The standard way to measure "how close" two embeddings are. Range: −1 (opposite) to +1 (identical direction). Magnitude doesn't matter — only direction. This is the math behind every "find similar documents" feature.

### Day 77 — Vector Databases
Pinecone, Weaviate, Qdrant, Milvus, pgvector. They store millions of embeddings and do approximate nearest-neighbor search (ANN, e.g. HNSW algorithm) so you can find "the 10 most similar documents to this query" in milliseconds. Foundation of every RAG system.

---

## Week 3 — Position, Context, and Sequence Length

### Day 78 — Why Position Has To Be Encoded
Attention is permutation-invariant — without position info, "dog bites man" and "man bites dog" are identical. Position must be injected somehow.

### Day 79 — Sinusoidal Positional Encoding (Original Transformer)
Add a fixed pattern of sines/cosines at different frequencies to each token's embedding. Different positions = different patterns. Simple, works, doesn't extrapolate well beyond training length.

### Day 80 — Learned Positional Embeddings
GPT-2 used a separate learned vector per position. Problem: hard limit at training context length. Can't go beyond it.

### Day 81 — RoPE (Rotary Position Embedding)
The modern winner. Encodes position by **rotating** the Q and K vectors in 2D subspaces by angles proportional to position. Mathematically elegant, *relative* (token A and B's interaction depends on their distance, not absolute positions), and extrapolates well. Used by Llama, Qwen, Mistral, DeepSeek, and most others.

### Day 82 — ALiBi
Alternative scheme: add a position-dependent bias to attention scores. Strong long-context generalization. Used by some BLOOM and MPT models.

### Day 83 — Context Window Length: 2k → 1M+
- GPT-2 (2019): 1,024 tokens
- GPT-3 (2020): 2,048
- GPT-4 (2023): 8k → 32k → 128k
- Claude 3 (2024): 200k
- Gemini 1.5 (2024): 1M+
- Llama 4 (2025): 10M+
The growth is enabled by RoPE/ALiBi + memory tricks (Month 9: FlashAttention, paged KV cache).

### Day 84 — Why Long Context Is Hard: Attention Is O(n²)
Attention computes a score between every pair of tokens. Doubling the context = 4× compute and memory for attention. This is the fundamental scaling wall, and most "long-context" engineering is about working around it.

---

## Week 4 — Inference Mechanics: How Generation Actually Works

### Day 85 — Autoregressive Generation
The model predicts one token at a time. Process the prompt → predict token 1 → append → predict token 2 → append → ... until end-of-sequence or max length. Each step requires a full forward pass through every layer.

### Day 86 — Logits and the Softmax
The model's output for each position is a vector of `vocab_size` numbers called **logits** — one raw score per possible next token. Apply **softmax** to convert logits to probabilities that sum to 1.

### Day 87 — Sampling Strategies
Given the probability distribution over the next token, how do we pick?
- **Greedy:** always pick the highest probability. Boring, repetitive.
- **Temperature:** divide logits by T before softmax. T<1 = more focused; T>1 = more random.
- **Top-k:** only consider top k tokens.
- **Top-p (nucleus):** consider smallest set of tokens whose cumulative probability ≥ p (e.g., 0.9). Most common today.
- **Min-p, typical, mirostat:** newer variants.

### Day 88 — Why "Temperature 0" Is Not Truly Deterministic
Even with greedy decoding, GPU non-determinism (parallel reduction order, batching effects) can cause tiny differences run-to-run. For true reproducibility you also need to fix the seed and batch composition.

### Day 89 — Stop Sequences and Special Tokens
Models have special tokens like `<|im_start|>`, `<|im_end|>`, `<|eot_id|>` that delimit messages, system prompts, tool calls. The **chat template** (Day 122) defines how messages get serialized into tokens. Get this wrong and the model behaves badly.

### 🛠️ Build It — Month 3 Hands-On
Stop reading, start poking:
1. `pip install tiktoken` and tokenize the same sentence with `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o). Compare token counts. Try Chinese, code, emoji.
2. Try the same sentence in Llama 3's tokenizer (`pip install transformers`, `AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")`). Notice the differences.
3. Compute cosine similarity between embeddings of `"king"`, `"queen"`, `"man"`, `"woman"` using `sentence-transformers` (`all-MiniLM-L6-v2`). See the famous analogy emerge.
4. Write a 20-line script that calls an Ollama or OpenAI endpoint with `temperature=0.0`, `0.7`, `1.5` for the same prompt. Feel the difference.

**Deliverable:** a Jupyter notebook you can show a coworker that demonstrates "what an LLM sees."

### Day 90 — Recap & "Explain it Back"
**Quiz yourself:**
1. What is a token, and why don't models use words or characters directly?
2. What's the difference between a token embedding and a sentence embedding?
3. Why is attention O(n²), and what does that mean for long context?
4. Walk through what happens from raw text input to a single generated token.
5. What's the difference between top-k and top-p sampling?

You now understand what the model *sees* and how it produces output. Next month: how it learned in the first place.
