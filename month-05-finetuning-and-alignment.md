# Month 05 — Fine-Tuning & Alignment (Days 121–150)

**Goal of this month:** Understand how a raw base model becomes ChatGPT, Claude, or `qwen3:8b-instruct`. By Day 150 you'll know what SFT, RLHF, DPO, LoRA, and QLoRA mean — and which one *you* would use to customize a model for your team.

---

## Week 1 — From Base Model to Assistant

### Day 121 — Why a Base Model Isn't an Assistant
Prompt a base model with `"What is the capital of France?"` and you may get back `"What is the capital of Germany? What is the capital of Spain?"` — it just continues the pattern. Base models are autocomplete, not assistants.

### Day 122 — The Chat Template
Instruct/chat models are trained on conversations formatted with **special tokens** marking roles. Example (ChatML, used by Qwen and OpenAI):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```
Llama 3 uses `<|start_header_id|>user<|end_header_id|>` etc. Get the template wrong → garbage output. Hugging Face stores it in `tokenizer_config.json`.

### Day 123 — Supervised Fine-Tuning (SFT)
Take the pretrained base model and continue training it on a curated dataset of `(prompt, ideal response)` pairs. Same loss (next-token prediction) but only on the assistant's tokens. Dataset sizes: 10k–1M examples. This single step turns a base model into a usable assistant.

### Day 124 — Where SFT Data Comes From
- Human contractors writing high-quality responses
- Distillation from a stronger model (use GPT-4 to generate training data for your smaller model)
- Public datasets: Alpaca, Dolly, OpenAssistant, ShareGPT, UltraChat
- Domain-specific (medical, legal, code) datasets

### Day 125 — The Limits of SFT Alone
SFT teaches the model *how* to respond. But it can't easily teach **preferences** — "be more concise," "refuse harmful requests," "show your reasoning." For preferences you need a different signal: comparisons.

---

## Week 2 — RLHF and Its Successors

### Day 126 — Reinforcement Learning from Human Feedback (RLHF)
The technique that made ChatGPT possible. Three stages:
1. **SFT** (Day 123)
2. **Reward model:** humans rank multiple responses to the same prompt; train a separate model to predict the rank.
3. **RL fine-tuning** (PPO algorithm): the LLM generates responses, the reward model scores them, the LLM is updated to get higher scores — while a KL penalty prevents it from drifting too far from the SFT model.

### Day 127 — Why RLHF Is Hard
PPO is finicky — unstable, hyperparameter-sensitive, requires keeping 4 models in memory simultaneously (policy, reference, reward, value). Many teams spent months getting it to work. There had to be a simpler way.

### Day 128 — Direct Preference Optimization (DPO)
Stanford 2023: prove that you can skip the reward model and PPO entirely. Just take preference pairs `(prompt, preferred, rejected)` and train the LLM directly with a clever loss function. **Same quality, far simpler.** DPO replaced PPO at most labs by 2024.

### Day 129 — DPO Variants
**IPO, KTO, ORPO, SimPO** — newer variants tweaking DPO's loss for stability or removing the need for paired data. KTO only needs `(prompt, response, good/bad)` labels — much easier to collect.

### Day 130 — Constitutional AI (Anthropic)
Anthropic's twist: instead of (only) human preferences, use a **written constitution** ("be helpful, be harmless, be honest, refuse to help with weapons...") and have the model critique and revise its own outputs against those principles. Scales beyond what humans can label. Foundational to Claude.

### Day 131 — RLAIF (RL from AI Feedback)
Use a strong model (e.g., GPT-4) to generate the preference labels instead of humans. Cheaper, faster, almost as good. Most modern alignment uses some mix of human + AI feedback.

### Day 132 — Reasoning Models: RL on Chain-of-Thought
**OpenAI o1** (Sept 2024) and **DeepSeek-R1** (Jan 2025) showed that RL on **verifiable** tasks (math problems with known answers, code with passing tests) teaches the model to produce long internal reasoning before answering. The model essentially learns to "think harder when it helps." This is the basis of every "reasoning" model.

---

## Week 3 — Parameter-Efficient Fine-Tuning (PEFT)

### Day 133 — The Problem with Full Fine-Tuning
Full fine-tuning a 70B model requires 16+ GPUs and creates a 140 GB checkpoint *per fine-tune*. If you want 50 specialized variants for 50 customers, that's 7 TB of weights. Not practical.

### Day 134 — LoRA (Low-Rank Adaptation, 2021)
The breakthrough idea: freeze the original weights, and add small **low-rank update matrices** alongside them. Instead of updating a 4096×4096 matrix (16M params), update two 4096×r and r×4096 matrices where r might be 8 or 16 — a few hundred KB instead of 64 MB.

### Day 135 — Why LoRA Works So Well
The empirical finding: fine-tuning updates are intrinsically low-rank. You don't need to change the whole weight matrix to teach a model a new domain — a small low-rank delta is enough. LoRA captures 95%+ of full fine-tuning quality at <1% of the parameters.

### Day 136 — QLoRA (2023)
Combine LoRA with **4-bit quantization** of the base model. The frozen base model is loaded in 4 bits (saves 4× memory), and you train tiny LoRA adapters in BF16 on top. A 70B model that needed 16 GPUs for full fine-tuning now fine-tunes on a *single* 48 GB GPU. Game-changing for individuals and small teams.

### Day 137 — Adapter Hot-Swapping
Because LoRA adapters are tiny (10–500 MB each), one server can hold a base model and dynamically swap in different adapters per request. Tools like **LoRAX** and **vLLM's multi-LoRA** turn this into a multi-tenant production capability.

### Day 138 — Other PEFT Methods
- **Prefix tuning, prompt tuning** — learn soft "virtual tokens" prepended to every prompt.
- **IA³, BitFit** — even smaller parameter footprints.
LoRA/QLoRA dominate in practice because the ecosystem (PEFT library, vLLM, axolotl) supports them best.

---

## Week 4 — Doing It Yourself

### Day 139 — When You Should Fine-Tune (and When You Shouldn't)
**Don't fine-tune** for: adding facts (use RAG), small style tweaks (use system prompt + few-shot), tasks the base model already does well.
**Do fine-tune** for: enforcing a structured output format reliably, teaching a domain vocabulary/style at scale, improving small-model performance on a narrow task, distilling a big model's behavior into a small one.

### Day 140 — RAG vs Fine-Tuning vs Both
- **RAG** (Retrieval-Augmented Generation) injects fresh facts into the prompt. Use for changing knowledge.
- **Fine-tuning** changes the model's *behavior*. Use for teaching skills or style.
- They compose well — fine-tune for the format/style, retrieve for the facts.

### Day 141 — The Fine-Tuning Stack
- **Hugging Face transformers + PEFT + TRL** — the standard library combo.
- **Axolotl** — a YAML-config wrapper that handles 90% of cases for you.
- **Unsloth** — heavily optimized fine-tuning, 2× faster on consumer GPUs.
- **OpenAI / Together / Fireworks fine-tuning APIs** — managed services if you don't want infrastructure.

### Day 142 — Evaluation: The Hardest Part
After fine-tuning, *did you make it better?* Common approaches:
- **Held-out test set** with metrics (accuracy, BLEU, ROUGE, exact match)
- **LLM-as-judge** — use GPT-4/Claude to score outputs
- **Human eval** — slow but the gold standard
- **Public benchmarks** — MMLU, GSM8K, HumanEval, MT-Bench, IFEval
A model that scores higher on your benchmark but feels worse subjectively is a sign you're overfitting to the metric.

### Day 143 — Catastrophic Forgetting
Fine-tune too aggressively and the model forgets what it knew. Mitigations: lower learning rate, mix in some original-domain data, use LoRA (only changes a small fraction).

### Day 144 — Safety Tuning and Jailbreaks
Aligned models refuse harmful requests. Attackers find prompt patterns ("DAN," "developer mode," "grandmother's bedtime story") that bypass refusals. Labs continuously red-team and patch. There's no perfect solution — alignment is ongoing.

---

## Week 5 — Wrap-Up

### Day 145 — The Full Pipeline, End-to-End
1. **Pretraining** (Month 4) → base model
2. **SFT** → instruction-following model
3. **Preference tuning** (DPO/RLHF) → helpful, harmless, honest
4. **(Optional)** Reasoning RL → o1/R1-style thinking
5. **(Optional)** Domain fine-tuning (LoRA) → your specialized model
6. **Quantization** (Month 7) → deployable model
7. **Inference engine** (Month 8) → serving

### Day 146 — Why "Open-Weight" Doesn't Mean "Open-Source"
You get the **weights** but rarely the training data, training code, or RLHF data. So you can run, fine-tune, and study the model — but you can't reproduce it from scratch. Truly open projects (OLMo, LLM360, Pythia) publish everything; most "open" models don't.

### Day 147 — Synthetic Data
Top labs increasingly train (and fine-tune) on data *generated by other models*. **Microsoft's Phi**, **Nvidia's Nemotron**, and many recent Qwen/DeepSeek improvements lean on this. Concerns about "model collapse" from too much synthetic data exist but haven't materialized yet at the frontier.

### Day 148 — Distillation
Take a strong "teacher" model, generate millions of (prompt, response) pairs, train a smaller "student" model to imitate them. **Llama-3.2-1B**, **Qwen2.5-0.5B**, and most "small but smart" models are distilled from larger siblings. This is how a 3 GB model can feel surprisingly capable.

### Day 149 — The Future of Alignment
Active research: **process reward models** (reward each reasoning step, not just the answer), **scalable oversight** (humans supervising models smarter than themselves), **interpretability** (can we read what's in the weights?). Critical for safety as capabilities grow.

### 🛠️ Build It — Month 5 Hands-On
1. Pick a tiny base model (`Qwen2.5-0.5B` or `Llama-3.2-1B`) and run a **QLoRA** fine-tune on a small dataset (e.g., 1k rows of Alpaca) using **Unsloth** (`pip install unsloth`) — fits on a free Colab T4.
2. Compare base vs fine-tuned outputs on 10 held-out prompts. Use a stronger model (Claude/GPT-4o) as a **judge** to score them — that's your first eval harness.
3. Save the LoRA adapter (~50 MB), reload it, run inference. You've now done end-to-end PEFT.
4. Try **DPO** on a preference dataset (`trl` library, ~50 lines of code).

**Deliverable:** a fine-tuned `.safetensors` LoRA adapter + a side-by-side eval table.

### Day 150 — Recap & "Explain it Back"
**Quiz yourself:**
1. What three stages does RLHF have?
2. Why did DPO replace PPO?
3. What does LoRA actually freeze and what does it train?
4. When should you choose fine-tuning over RAG, and vice versa?
5. What's a chat template and why does getting it wrong break things?

Halfway there. You now understand how models are made and shaped. The second half of the year is about **how we run them**.
