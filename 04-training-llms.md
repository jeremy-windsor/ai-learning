# Module 04 — Training LLMs

**Mastery: 21% → 29%** · Prerequisites: Modules 02–03

**Goal of this module:** Understand what it actually takes, in data, compute, money, and time, to *create* a foundation model. By the end you'll be able to explain what pretraining does (and why it causes hallucinations and knowledge cutoffs), estimate a training run's cost on the back of an envelope, explain scaling laws, and say why pretraining is reserved for a handful of labs.

**How to read this module:** It applies the training loop from §1.35 at enormous scale. If "loss," "gradient," or "batch" feel fuzzy, reread §1.32–§1.35 first.

---

## Chapter 1 — The Pretraining Objective

### §4.1 — Next-Token Prediction Is the Whole Game
**Pretraining** is the first, giant training stage that produces a model from random weights. Its *only* objective: **given a sequence of tokens, predict the next one** (§1.1). Nothing else is taught directly. Grammar, facts, translation, code, and reasoning all emerge because they help with that one task. To predict the next line of a Python file, it helps to understand Python. To predict the end of a mystery novel, it helps to track who did what.

It's **self-supervised** (§1.8): the text supplies its own answers. The sentence "OSPF uses link-state advertisements" contains several training examples at once:

| Model sees | Must predict |
|-----------|--------------|
| `OS` | `PF` |
| `OSPF` | ` uses` |
| `OSPF uses` | ` link` |
| `OSPF uses link` | `-state` |

### §4.2 — Cross-Entropy Loss for LLMs
You met cross-entropy in §1.32: look at the probability the model gave the *correct* next token, and take `−log` of it. Average that over every position in the batch, and that's the number backprop pushes down.

Numbers that help:
- **A brand-new model** guesses evenly across its whole vocabulary. For Llama 3's 128,256 tokens that's a loss of `ln(128,256) ≈ 11.8`.
- **A well-trained LLM** ends up around 2 or below on typical web text (the exact value depends on the data and tokenizer).
- Loss is measured in **nats** (natural log) or **bits** (log base 2). Same idea, different units.

### §4.3 — Teacher Forcing
During training, the model always gets the *real* previous tokens as input, never its own guesses. Even if it badly mispredicted token 5, position 6 still sees the true token 5. This is called **teacher forcing**. It keeps training stable and lets every position be trained at once (§4.4).

At inference there's no teacher: the model feeds on its own outputs (§3.29). That mismatch is one reason a small early mistake can snowball in a long generation.

### §4.4 — One Forward Pass, Thousands of Lessons
Thanks to the causal mask (§2.10), each position can only see the tokens before it. So a single 8,192-token training sequence holds 8,191 separate prediction tasks, all computed in *one* parallel forward pass. RNNs had to step through them one at a time (§1.45). This is the practical reason transformers train so efficiently.

### §4.5 — Perplexity
**Perplexity** = `e^loss`. It turns the loss into an intuitive number: *on average, the model was as unsure as if it were choosing between this many equally likely tokens.*

| Loss | Perplexity | Meaning |
|------|-----------|---------|
| 11.8 | ~128,000 | Random guessing across the whole vocabulary |
| 2.0 | ~7.4 | Like choosing among ~7 options each step |
| 1.5 | ~4.5 | Sharper still |

Lower is better. Perplexity is only comparable between models that use the same tokenizer and the same test text.

### §4.6 — What Pretraining Produces: Knowledge, Cutoffs, and Hallucination
To predict text well, the model has to squeeze an enormous amount of its training data into its weights: facts, styles, code patterns, relationships between ideas. Three consequences matter every day:
- **Knowledge cutoff.** The model knows nothing about anything after its training data was collected. That date is its **knowledge cutoff**. Ask about later events and it can only guess, unless you supply the information in the prompt (RAG, web search: §12.15).
- **It's not a database.** Facts aren't stored as rows you can look up. They're smeared across billions of weights as statistical patterns. Frequently repeated facts are reliable; rare ones (a niche RFC number, an obscure person) are fuzzy.
- **Hallucination.** The model's job is to produce *plausible next tokens*, not *true* ones. When it doesn't know, the most plausible continuation is often a confident, well-formatted, made-up answer: a fake citation, an invented CLI flag. Post-training (Module 05) reduces this and teaches models to say "I don't know," but it's built into how they learn. Treat factual details as claims to verify.

---

## Chapter 2 — The Data

### §4.7 — Internet-Scale Datasets
Modern pretraining uses **trillions** of tokens. (Roughly, 1 trillion tokens ≈ 4 TB of plain English text.) Common sources:
- **Common Crawl:** a non-profit crawl of the web since 2008, with petabytes of raw pages. Almost every lab starts here.
- **C4**, **RefinedWeb**, **FineWeb** (15T tokens, 2024), **DCLM**: cleaned and filtered versions of Common Crawl.
- **The Pile** (EleutherAI, 825 GiB), **RedPajama**, **Dolma**: open, mixed-source datasets built to recreate what labs use.
- **Code** from GitHub, **science** from arXiv and papers, plus **Wikipedia, books, StackExchange, Reddit, and math** for quality and variety.

Llama 3 trained on about **15 trillion tokens**. Frontier labs don't disclose their exact data.

### §4.8 — The Data Pipeline: From Web Page to Training Token
Raw web data is mostly junk: navigation menus, spam, duplicate pages, cookie banners. A typical cleaning pipeline:
1. **Extract text** from HTML (drop the menus, ads, and scripts).
2. **Identify the language** and route each page to the right bucket.
3. **Filter for quality:** heuristic rules (too short, too many symbols, repeated lines) plus a small classifier trained to score "is this educational, well-written text?"
4. **Deduplicate:** remove exact copies, then near-copies (with a technique called **MinHash**). The web repeats itself constantly, and duplicates waste compute and encourage memorization.
5. **Remove personal data and toxic content**, as far as possible.
6. **Decontaminate:** strip out anything matching known benchmark test sets (§4.13).

Often more than 90% of the raw crawl is thrown away. Much of a lab's real competitive advantage lives in this pipeline.

### §4.9 — Data Quality > Data Quantity (Eventually)
Early models used whatever text they could get. Then labs discovered that **filtering matters enormously**. Microsoft's **Phi** series ("Textbooks Are All You Need," 2023) showed that small models trained on carefully selected and textbook-style synthetic data (§5.29) can beat much larger models trained on raw web data. Hugging Face's FineWeb-Edu showed the same thing in the open: keeping only the most "educational" pages trained better models from fewer tokens.

### §4.10 — Tokenize Once, Then Pack
Before training starts, the whole dataset is tokenized (Module 03) into integer IDs and saved as compact binary files. Then documents are **packed**: glued end-to-end with EOS separators (§3.12) and cut into fixed-length chunks (say 8,192 tokens). No space is wasted on padding, and a masking trick usually stops one document from attending into the next.

### §4.11 — Data Mixing and Annealing
The model sees a *blend*. An *illustrative* mix might be 60% web, 15% code, 10% books, 10% science, 5% math and conversation. The mix strongly shapes the result: code-heavy training (DeepSeek-Coder, Qwen-Coder) produces strong coders. Exact recipes are among a lab's most guarded secrets.

Order matters too. Many labs finish pretraining with an **annealing** (or **mid-training**) phase: the learning rate drops (§4.27) while the mix shifts toward the highest-quality data (math, code, curated text, long documents). A small amount of excellent data at the end has an outsized effect.

### §4.12 — Epochs: Each Token Seen About Once
In classic ML you loop over a dataset for many **epochs** (§1.35). LLM pretraining is the opposite: most tokens are seen **once**, or a handful of times for the best data. There's so much text that the model rarely needs to see anything twice, and repeating data invites memorization (§1.16). Research suggests repeating data up to about 4 times is nearly as good as fresh data, with fast-shrinking returns after that. As labs run out of high-quality human text (the **data wall**), this matters more and more, and it's a big driver of synthetic data (§5.29).

### §4.13 — Contamination
If the training data contains a benchmark's *test questions*, the benchmark score is meaningless. The model memorized the answers, the LLM version of test-set leakage (§1.17). Labs try to detect and remove contamination, but because benchmarks are published on the web, it's a permanent arms race. It's why newer benchmarks keep some test questions private, and why you should trust your own evals (§5.24) over leaderboards.

---

## Chapter 3 — The Compute

### §4.14 — How Big Is "Big"?
| Model | Parameters | Training tokens | Compute |
|-------|-----------|-----------------|---------|
| GPT-3 (2020) | 175B | ~300B | ~3,640 petaFLOP-days; ~$4–5M estimated at the time |
| Llama 3 70B (2024) | 70B | ~15T | ~6.4M H100 GPU-hours (Meta's figure) |
| DeepSeek-V3 (Dec 2024) | 671B total (37B active, MoE) | ~14.8T | ~2.8M H800 GPU-hours for the final run |
| GPT-4 (2023) | Not disclosed (rumored ~1.8T, MoE) | Not disclosed | Rumored $60–100M |
| Frontier models (2025–26) | Not disclosed | Not disclosed | Reported clusters of 100,000+ GPUs; $500M+ runs |

### §4.15 — FLOPs and the 6ND Rule of Thumb
A **FLOP** is one floating-point operation (one multiply or one add). **FLOP/s** is how many a chip does per second.

Training a dense transformer takes about **`6 × N × D` FLOPs**, where N = parameters and D = training tokens. (Roughly 2 for the forward pass and 4 for the backward pass, per parameter, per token.)

Worked example, Llama 3 70B:
- `6 × 70×10⁹ × 15×10¹² ≈ 6.3 × 10²⁴ FLOPs`
- An H100 does about **1 petaFLOP/s** (10¹⁵) of dense BF16 math at peak. (Spec sheets often quote ~2 PFLOP/s, but that's "with sparsity," which training doesn't use.)
- At 100% efficiency: `6.3×10²⁴ ÷ 10¹⁵ ≈ 6.3 billion GPU-seconds ≈ 200 GPU-years`.
- Meta reported ~6.4M H100-hours, about **730 GPU-years**.

Real runs never hit peak. The fraction of peak actually achieved is called **MFU (model FLOPs utilization)**. Meta reported 38–43% MFU on its Llama 3 405B run. Communication between GPUs, memory stalls, failures, and restarts eat the rest. You'll compute these numbers yourself in the lab.

### §4.16 — GPU-Hours and the Money
Renting an H100 costs roughly $2–4 per hour. A frontier run uses 16,000–100,000+ GPUs for 2–6 months. Multiply it out and you get **$50M–$500M+** for the final run alone, before salaries, electricity, networking, data, and all the smaller experiments and failed attempts that come first. The headline number for any model's "training cost" usually excludes all of that (§2.32).

### §4.17 — Training Memory: The 16-Bytes-per-Parameter Rule
Running a model (inference) needs about **2 bytes per parameter** for BF16 weights. Training with the AdamW optimizer (§1.33) needs about **16 bytes per parameter**, before activations:

| What's stored | Bytes per parameter | Why |
|---------------|---------------------|-----|
| BF16 weights | 2 | Used in the forward and backward pass |
| BF16 gradients | 2 | One gradient per weight (§1.34) |
| FP32 master weights | 4 | High-precision copy for the updates (§4.18) |
| Adam state: momentum + variance | 4 + 4 | AdamW tracks two running averages per weight |
| **Total** | **16** | |

For a 70B model: `70B × 16 bytes ≈ 1.1 TB`, plus **activations** (the intermediate values saved for backprop, which grow with batch size and sequence length). An H100 has 80 GB, so you need *at least* 14 GPUs just to hold the training state, and in practice far more, joined by very fast links (NVLink, InfiniBand: §6.17–§6.18). That's why pretraining is a big-lab game, and why fine-tuning tricks like LoRA (§5.16) exist for everyone else.

### §4.18 — Mixed-Precision Training
Doing the math in 16-bit formats is roughly 2× faster and uses half the memory of 32-bit. But tiny weight updates can get rounded away at low precision. The standard fix is **mixed precision**: compute in **BF16** (§7.5) while keeping an FP32 "master copy" of the weights that receives the updates.

The older FP16 format also needed **loss scaling** (multiplying the loss up so small gradients don't round to zero). BF16 has a much wider range, so it usually doesn't. The frontier is moving to **FP8** for much of the math: DeepSeek-V3 was trained largely in FP8. (Quantization for *inference* is a separate, more aggressive game: Module 07.)

---

## Chapter 4 — Scaling Laws and Chinchilla

### §4.19 — The Scaling Hypothesis
OpenAI's **Kaplan et al. (2020)** found that loss falls along a smooth, predictable **power law** as you increase model size, data, and compute, across many orders of magnitude.

A power law means *every 10× increase buys a similar-sized improvement*. On a chart with log-scale axes, it's a straight line. That predictability changed the field: you can train small models cheaply, fit the line, and forecast how good a 100× bigger model will be *before spending the money*. It was the empirical case for "just make it bigger," and the story of GPT-2 → GPT-3 (§2.22–§2.23).

### §4.20 — Chinchilla (2022): The Course Correction
DeepMind's **Chinchilla** paper (Hoffmann et al.) showed that Kaplan's advice on *how to split* the compute was off. For a fixed budget, parameters and training tokens should grow **in roughly equal proportion**, at about **20 tokens per parameter**.

By that measure GPT-3 (175B parameters, 300B tokens, under 2 tokens per parameter) was badly undertrained. DeepMind proved it: **Chinchilla 70B**, trained on 1.4T tokens (20 per parameter) with the same compute as their 280B Gopher, beat both Gopher and GPT-3.

### §4.21 — The Post-Chinchilla Era: Train Small Models Long
Labs now train *well past* the Chinchilla ratio on purpose:

| Model | Tokens per parameter |
|-------|---------------------|
| Chinchilla-optimal | ~20 |
| Llama 3 70B (15T tokens) | ~214 |
| Llama 3 8B (15T tokens) | ~1,875 |

Why "waste" compute like this? See the next section.

### §4.22 — Compute-Optimal vs. Inference-Optimal
Chinchilla answers: "For a fixed *training* budget, what gives the best model?" But training happens once, and inference happens billions of times. A smaller model is cheaper and faster for *every single request* forever after.

So labs deliberately **over-train smaller models** on far more data. Each extra token of training still improves the model a little, and the saving at serving time dwarfs the extra training cost. That's why an 8B model in 2025 can beat a 70B model from 2023, and why small models you run in Ollama are so capable.

### §4.23 — The Bitter Lesson, Quantified (and Its New Axes)
Scaling laws are Sutton's Bitter Lesson (§2.38) with numbers attached. The recipe is industrialized: predict the loss from (parameters, tokens, FLOPs), plan the run, execute.

Two newer developments:
- **The data wall.** High-quality human text is finite, so the "D" in 6ND is getting harder to grow. Answers include synthetic data (§5.29), repeating the best data (§4.12), and multimodal data.
- **Test-time compute.** Reasoning models (§2.27) open a second scaling axis: instead of only a bigger model, spend more compute *thinking* at answer time. Accuracy on hard problems climbs with thinking length too (§12.33).

---

## Chapter 5 — Training Infrastructure (Preview of Module 10)

### §4.24 — Why Training Is Different From Inference
Training stores **weights + gradients + optimizer state + activations** (§4.17), roughly 8–20× the memory of inference. It also needs constant GPU-to-GPU communication to share gradients. A model you can *run* on 1 GPU may need 16 or more to *train*.

### §4.25 — Data Parallelism
The simplest way to scale out: put a full copy of the model on each of N GPUs, give each a different slice of the batch, then **average the gradients** across all GPUs (an "all-reduce," §10.4) so every copy makes the same update. N GPUs ≈ N× the throughput. It only works while the whole training state fits on one GPU (§10.7).

### §4.26 — Splitting the Model Itself
When the model is too big for one GPU, you split the model:
- **Pipeline parallelism:** different *layers* on different GPUs, like an assembly line (§10.20).
- **Tensor parallelism:** each big matrix multiply is split *across* GPUs within a layer (§10.19).
- **Sharded data parallelism (FSDP / ZeRO):** still data parallel, but each GPU stores only a *slice* of the weights, gradients, and optimizer state, and fetches the rest just in time (§10.11–§10.16).

Frontier runs combine data, tensor, and pipeline parallelism, which is called **3D parallelism** (§10.24) (often with sharding and, for MoE, expert parallelism on top).

### §4.27 — Learning-Rate Schedules: Warm Up, Then Cool Down
The learning rate (§1.33) isn't fixed during a run:
1. **Warmup:** start tiny and ramp up over the first few thousand steps. Large updates on random weights can wreck training immediately.
2. **Decay:** slowly lower it over the rest of the run (often along a cosine curve), so the model takes big strides early and fine, careful steps late.

A newer variant, **warmup-stable-decay (WSD)**, holds the rate flat and decays only at the end, which pairs naturally with the annealing phase in §4.11.

### §4.28 — The Loss Curve and What Goes Wrong
Engineers watch the loss curve constantly. It should fall fast at first, then keep falling more and more slowly. Trouble looks like:
- **Loss spikes:** a sudden jump, from a bad data batch, too high a learning rate, or numerical overflow. The fix is to roll back to a checkpoint, skip that data, or lower the learning rate.
- **Divergence:** the loss shoots up and never recovers. Restart from an earlier checkpoint with different settings.
- **Plateau:** improvement stalls earlier than the scaling laws predicted, a sign of a data or setup problem.

This is the day-to-day life of a pretraining team.

### §4.29 — Checkpointing and Restarts
At 10,000+ GPUs, hardware fails constantly. During Meta's 54-day Llama 3 405B run on 16,384 H100s, there were 419 unexpected interruptions, roughly **one every 3 hours**, mostly GPU and memory faults. So runs save **checkpoints** (a full snapshot of the training state) regularly and automatically restart from the last good one. Fast checkpointing and quick recovery are serious engineering problems at this scale (§10.31).

---

## Chapter 6 — Wrap-Up

### §4.30 — Base Models vs. Instruct Models
Pretraining produces a **base model**: a raw next-token predictor. It continues text but doesn't follow instructions. Ask it a question and it might write more questions. **Post-training** (Module 05) turns it into an **instruct** (or **chat**) model that answers helpfully. Labs often publish both, such as `Qwen3-8B-Base` and `Qwen3-8B`. Base models are mainly useful as starting points for your own fine-tuning.

### §4.31 — Why Open Weights Matter
When Meta, Alibaba, DeepSeek, or Mistral release weights, they hand the world tens of millions of dollars of compute for free. Anyone can fine-tune for a specialized task (Module 05), quantize to run on a laptop (Module 07), or study the model, without repeating pretraining. The open ecosystem stands on a handful of these foundation runs.

### §4.32 — The Cost of "Free" Models
You pay nothing for open weights, but someone paid for the training. Labs' motives include ecosystem influence (everyone builds on *their* architecture and tools), recruiting talent, undercutting competitors' paid APIs, and national or strategic goals. That competition is a big reason 2024–2026 moved so fast.

### §4.33 — The Compute Concentration
Only about a dozen organizations can credibly pretrain a frontier model in 2026: OpenAI, Anthropic, Google, Meta, xAI, Microsoft, Alibaba, DeepSeek, Mistral, and a few others. The barrier is the combination of capital (billions in GPUs), scarce chips, power and datacenter capacity, data pipelines, and experienced people. It's one of the most concentrated technologies in the world (§12.35).

### 🛠️ Build It — Hands-On Lab
**Lab 1: pretraining in miniature.** A "language model" that predicts the next *character* from just the one before it (a **bigram model**). Training is just counting. Pure Python.

```python
import math
import random
from collections import Counter, defaultdict

random.seed(1)

# Our entire "internet". Real pretraining uses ~15 trillion tokens; we use ~600 characters.
train_text = (
    "the router sends the packet to the next hop. the next hop checks the route "
    "table and sends the packet on. when the link goes down the router picks a new "
    "route. the protocol tells every router about the new path so the packets keep "
    "moving. a good network has more than one path to every place. "
    "the switch learns the address of each port and sends the frame to the right port. "
)
test_text = "if every link fails, a backup path through a different switch keeps traffic flowing. "

# Tokenizer: one token per character (Module 03 explains why real models use subwords).
vocab = sorted(set(train_text + test_text))

# "Training": count how often each character follows each other character.
counts = defaultdict(Counter)
for prev, nxt in zip(train_text, train_text[1:]):
    counts[prev][nxt] += 1

def prob(prev, nxt):
    # +1 smoothing so unseen pairs get a small probability instead of zero
    c = counts[prev]
    return (c[nxt] + 1) / (sum(c.values()) + len(vocab))

def avg_loss(text):
    losses = [-math.log(prob(p, n)) for p, n in zip(text, text[1:])]
    return sum(losses) / len(losses)

uniform = math.log(len(vocab))
print(f"vocab size: {len(vocab)} characters")
print(f"untrained (uniform guess) loss: {uniform:.2f}  perplexity: {math.exp(uniform):.1f}")
for name, text in (("train", train_text), ("test", test_text)):
    loss = avg_loss(text)
    print(f"{name:>5} loss: {loss:.2f}  perplexity: {math.exp(loss):.1f}")

# "Inference": generate text one token at a time (autoregressive sampling).
out = "t"
for _ in range(80):
    options = vocab
    weights = [prob(out[-1], ch) for ch in options]
    out += random.choices(options, weights=weights)[0]
print("\nsample:", out)
```

What you should see: training drops the loss from 3.26 (random guessing, perplexity 26) to about 2.3 (perplexity ~10). The test loss is worse, about 2.9, because the model has partly memorized its tiny training set (§1.16). The generated sample is mostly gibberish with a few real fragments ("of," "tho"). With only *one* character of context, that's the best it can do. Real LLMs see thousands of tokens of context through attention, and that's the whole difference. Things to try: paste in a few pages of your own text as `train_text`. Does the test loss improve?

**Lab 2: estimate a training run (§4.15–§4.16).**

```python
# Back-of-the-envelope pretraining cost, using the 6 x N x D rule (§4.15).
H100_PEAK = 989e12        # BF16 dense FLOP/s (H100 SXM spec sheet)
MFU = 0.40                # realistic fraction of peak actually achieved
PRICE_PER_GPU_HOUR = 2.50 # USD, rough 2025-26 cloud rental
GPUS = 16_000

models = {                # name: (parameters, training tokens)
    "GPT-3":       (175e9, 300e9),
    "Llama 3 8B":  (8e9,   15e12),
    "Llama 3 70B": (70e9,  15e12),
}
for name, (n, d) in models.items():
    flops = 6 * n * d
    gpu_hours = flops / (H100_PEAK * MFU) / 3600
    days = gpu_hours / GPUS / 24
    cost = gpu_hours * PRICE_PER_GPU_HOUR
    print(f"{name:<12} {flops:.1e} FLOPs  {gpu_hours / 1e6:5.2f}M GPU-hours  "
          f"{days:5.1f} days on {GPUS:,} GPUs  ~${cost / 1e6:,.0f}M  "
          f"({d / n:,.0f} tokens/param)")
```

You should get about 4.4M GPU-hours and ~$11M for Llama 3 70B. Meta's real figure was ~6.4M GPU-hours. The formula is a *lower bound*: it ignores failed runs, experiments, and real-world inefficiency. Also notice GPT-3 would cost about $1M on modern H100s, versus several million on 2020 hardware. Things to try: add a 405B model trained on 15T tokens. Lower MFU to 0.25.

### §4.34 — Checkpoint — Explain It Back
1. What is the only objective during pretraining, and why is it called self-supervised?
2. What does a loss of 11.8 versus 2.0 mean for a Llama 3 model?
3. Why does pretraining cause hallucinations and a knowledge cutoff?
4. Name four steps of a pretraining data pipeline.
5. What is the 6ND rule, and roughly how many GPU-hours did Llama 3 70B take?
6. Why does training need ~16 bytes per parameter when inference needs ~2?
7. What did Chinchilla show, and why do labs now train far past it?
8. What's the difference between data parallelism and splitting the model across GPUs?
9. Why does the learning rate warm up and then decay?
10. What's the difference between a base model and an instruct model?

<details>
<summary>Answer key (try first!)</summary>

1. Predict the next token. The "labels" are just the next tokens of the text itself, so no human labeling is needed.
2. 11.8 is `ln(128,256)`: random guessing over the vocabulary. 2.0 (perplexity ~7.4) means it's about as unsure as choosing among ~7 options.
3. The model learns to produce plausible text, not to look up verified facts, so when unsure it produces plausible fiction. It only knows what was in its training data, which ends at a collection date.
4. Any four of: HTML text extraction, language identification, quality filtering, deduplication, personal-data and toxicity removal, decontamination, tokenization and packing.
5. Training FLOPs ≈ 6 × parameters × tokens. About 6.4M H100-hours (Meta's figure).
6. Training also stores gradients, an FP32 master copy of the weights, and two Adam optimizer values per weight (2+2+4+4+4 bytes), plus activations.
7. For a fixed budget, grow parameters and tokens together at about 20 tokens per parameter. Labs over-train smaller models because they're far cheaper to serve billions of times.
8. Data parallelism copies the whole model to every GPU and splits the *data*. Pipeline and tensor parallelism split the *model* (by layer, or within each layer); FSDP/ZeRO shard the training state across GPUs.
9. Big steps on random weights at the start can wreck training, so it starts small. Lowering it late lets the model settle into a good spot.
10. A base model only continues text. An instruct model has been post-trained to follow instructions and chat.
</details>

You now know how a model is *born*. Next: how it's made *useful*.
