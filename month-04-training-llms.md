# Month 04 — Training LLMs (Days 91–120)

**Goal of this month:** Understand what it actually takes — in data, compute, money, and time — to *create* a foundation model. By Day 120 you'll be able to explain the difference between pretraining, fine-tuning, and inference, and why pretraining is reserved for a handful of labs on Earth.

---

## Week 1 — The Pretraining Objective

### Day 91 — Next-Token Prediction Is the Whole Game
Despite all the magic, the *only* thing a base LLM is trained to do is: **given a sequence of tokens, predict the next one**. Nothing else. Reasoning, translation, code, jokes — all emerge from getting really good at this one task on enough data.

### Day 92 — Cross-Entropy Loss
For each predicted position, compare the model's probability distribution over the vocab against the actual next token (a one-hot vector). Cross-entropy measures "how surprised was the model by the correct answer?" Lower = better. Average it across billions of training tokens.

### Day 93 — Teacher Forcing
During training, the model sees the *correct* previous tokens at every position, not its own predictions. This lets all positions be trained in parallel (one forward pass = N predictions). Inference is different — there it has to use its own outputs.

### Day 94 — One Forward Pass, N Loss Terms
A 4,096-token training sequence yields 4,095 prediction tasks (predict each token from the ones before it) — all computed in a single GPU pass. This is why transformers train so efficiently.

### Day 95 — Perplexity
A more interpretable version of cross-entropy: `perplexity = exp(loss)`. Roughly: "on average, the model was choosing between this many equally-likely next tokens." Lower perplexity = better language model. Reported in nearly every paper.

---

## Week 2 — The Data

### Day 96 — Internet-Scale Datasets
Modern pretraining uses **trillions** of tokens. Common sources:
- **Common Crawl** — petabytes of raw web pages
- **C4** (Colossal Cleaned Common Crawl) — Google's filtered web corpus
- **The Pile** (EleutherAI) — 800GB curated mix
- **RedPajama, RefinedWeb, FineWeb** — open recreations of training mixes
- **GitHub** for code, **arXiv** for science, **Wikipedia, books, Reddit, StackExchange** for quality

### Day 97 — Data Quality > Data Quantity (Eventually)
Early models just used whatever they could get. Then labs discovered that **filtering** matters enormously: dedup, remove low-quality pages, balance domains, remove toxic content. **Phi-3** showed a small model trained on heavily-curated data can beat much larger models on cruddy data.

### Day 98 — Tokenization Happens Once
Before training, the entire dataset is tokenized into integer IDs and packed into binary files. Packing means concatenating documents with separator tokens, then chunking into fixed-length sequences (e.g., 8,192 tokens each).

### Day 99 — Data Mixing
The model sees a *blend*: maybe 60% web, 15% code, 10% books, 10% science, 5% conversation. The mix dramatically affects the resulting model's strengths. Code-heavy training (DeepSeek-Coder, Qwen-Coder) produces strong coders. The recipe is one of every lab's most-guarded secrets.

### Day 100 — Contamination
If your training data contains the *test set* of a benchmark, your benchmark scores are meaningless. Labs spend major effort detecting and removing contamination — but it's an unsolvable arms race as benchmarks become public.

---

## Week 3 — The Compute

### Day 101 — How Big Is "Big"?
- **GPT-3** (2020): 175B params, ~300B tokens, ~3,640 PFLOP-days, ~$4M est.
- **Llama 3 70B** (2024): 70B params, **15T tokens**, tens of thousands of H100-days
- **GPT-4** (rumored): ~1.8T params (MoE), ~13T tokens, ~$60–100M
- **Frontier 2025 models**: 100k+ GPU clusters, $500M+ training runs

### Day 102 — FLOPs and the 6N Rule of Thumb
A useful approximation: training a dense transformer requires about `6 × N × D` FLOPs, where N = parameters and D = training tokens. For a 70B model on 15T tokens: ~6 × 10²⁴ FLOPs. An H100 does ~1 PFLOP/s on BF16, so ideally ~70 GPU-years. Real life: 10–30× longer due to communication, restarts, suboptimal utilization.

### Day 103 — GPU-Months and the Money
Renting an H100 costs ~$2–4/hour. A frontier training run uses 16,000–100,000 GPUs for 2–6 months. Multiply it out: $50M–$500M, before salaries, electricity, networking, and the failed attempts.

### Day 104 — Why You Can't Pretrain at Home
A 70B model in BF16 weights alone is 140 GB — far beyond any single consumer GPU. Add gradients (140 GB), optimizer state for AdamW (560 GB), and activations (hundreds of GB depending on context). You need dozens of A100/H100s with high-bandwidth interconnect (NVLink, InfiniBand). This is why pretraining is a big-lab game and fine-tuning is what mortals do.

### Day 105 — Mixed Precision Training
Train weights in BF16 (16-bit) for speed, but keep an FP32 master copy for stability. Combined with **loss scaling**, this 2× speedup is universal in modern training. (Quantization for *inference* is a separate, more aggressive game — Month 7.)

---

## Week 4 — Scaling Laws and Chinchilla

### Day 106 — The Scaling Hypothesis
OpenAI's **Kaplan et al. (2020)** scaling laws paper showed that loss decreases as a smooth power law in **model size**, **dataset size**, and **compute** — over many orders of magnitude. This was the empirical justification for "just make it bigger."

### Day 107 — Chinchilla (2022) — The Course Correction
DeepMind showed Kaplan was wrong about *how* to allocate compute. The optimal is to scale parameters and tokens **roughly equally** — about **20 tokens per parameter**. GPT-3 was massively under-trained. **Chinchilla 70B** beat GPT-3 175B by training on more data with fewer parameters.

### Day 108 — The Chinchilla Era
Post-2022, every serious lab follows Chinchilla scaling — or *exceeds* it for inference-deployed models. **Llama 3 70B** trained on 15T tokens (215 tokens/param) — way past Chinchilla optimal *for training*, but optimal when you also care about cheap *inference*.

### Day 109 — Compute-Optimal vs Inference-Optimal
For a fixed training budget, Chinchilla maximizes model quality. But once trained, that model is used billions of times. So labs deliberately "over-train" smaller models on more data — they're slightly worse than a Chinchilla-optimal larger model, but much *cheaper* to serve.

### Day 110 — The Bitter Lesson, Quantified
Scaling laws are the bitter lesson with numbers attached. The recipe is now industrialized: predict the loss you'll get from a given (params, tokens, FLOPs). Plan the run. Execute. Done.

---

## Week 5 — Training Infrastructure (preview of Month 10)

### Day 111 — Why Training Is Different from Inference
Training requires storing **weights + gradients + optimizer state + activations** for backprop. That's typically 10–20× the memory of inference. A model you can *run* on 1 GPU may need 16+ GPUs to *train*.

### Day 112 — Data Parallelism (DDP)
Easiest scaling: copy the entire model to N GPUs, give each a different batch slice, average the gradients. Works until the model no longer fits on one GPU. Month 10 covers this in depth.

### Day 113 — Model/Tensor/Pipeline Parallelism
When the model is too big for one GPU, split *the model itself* across GPUs. Each technique partitions differently — by layer (pipeline), by within-layer matrix dimension (tensor), or by parameter shard (FSDP/ZeRO). Frontier training combines all three (**3D parallelism**).

### Day 114 — Checkpointing and Restarts
Training runs for months. Hardware fails constantly at 10,000+ GPU scale. You checkpoint weights every few hours so you can restart the run from the last good state. **Meta's Llama 3 paper** reported a hardware failure roughly every 3 hours during training.

### Day 115 — The Loss Curve and What Goes Wrong
Training loss should drop smoothly. Spikes mean instability — bad data batch, learning rate too high, numerical issues. Fixing them requires rolling back to a checkpoint, skipping the bad data, or reducing the LR. This is the day-to-day life of pretraining engineers.

---

## Week 6 — Wrap-Up

### Day 116 — Base Models vs Instruct Models
Pretraining produces a **base model** — a raw next-token predictor. It doesn't follow instructions; ask it a question and it might generate more questions. **Instruction tuning** (Month 5) turns a base model into a helpful assistant. Both are released for Llama, Qwen, Mistral, etc.

### Day 117 — Why Open Weights Matter
When Meta/Alibaba/DeepSeek release base model weights, the world saves billions in compute. Anyone can fine-tune for a specialized task without re-doing the pretraining. The open-source ecosystem stands on the shoulders of these few foundation runs.

### Day 118 — The Cost of "Free" Models
Even though you don't pay for a base model, *training* it cost Meta or Alibaba tens of millions. Their motivation: ecosystem dominance, talent attraction, and breaking competitors' moats. This dynamic is what makes 2024–2026 such a fast-moving period.

### Day 119 — The Compute Concentration
Only ~10 organizations on Earth can credibly pretrain a frontier model in 2026: OpenAI, Anthropic, Google, Meta, xAI, Microsoft, Alibaba, DeepSeek, Mistral, and a handful of others. This is a more concentrated industry than oil, semiconductor fabs, or aerospace.

### Day 120 — Recap & "Explain it Back"
**Quiz yourself:**
1. What is the only objective during pretraining?
2. What does Chinchilla say about scaling?
3. Why can't you pretrain a 70B model at home?
4. What's the difference between a base model and an instruct model?
5. About how many tokens does a frontier model train on today?

You now know how a model is *born*. Next: how we make it *useful*.
