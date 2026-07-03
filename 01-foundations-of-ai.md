# Module 01 — Foundations of AI

**Mastery: 0% → 8%** · Prerequisites: none

**Goal of this module:** Understand the 70-year arc from "AI is hand-written rules" to "AI is a giant statistical pattern-matcher trained on data." By the end you should be able to explain *why* the field shifted away from symbolic AI and toward neural networks.

---

## Chapter 1 — What Even Is AI?

### §1.1 — The word "Artificial Intelligence"
Coined at the **Dartmouth Workshop in 1956** by John McCarthy. The original definition was "making a machine behave in ways that would be called intelligent if a human did them." That definition is deliberately vague — and 70 years later it still is. AI is an *aspiration*, not a single technology.

### §1.2 — Two Schools of AI: Symbolic vs. Connectionist
From the very beginning there were two camps:
- **Symbolic AI ("Good Old-Fashioned AI" / GOFAI):** Encode human knowledge as rules and logic. "IF temperature > 100 AND patient has cough THEN suggest flu test."
- **Connectionist AI:** Build networks of simple units (loosely inspired by neurons) and let them *learn* patterns from examples instead of being told rules.
The symbolic camp dominated from ~1956 to ~1990. The connectionist camp dominates today. Modern LLMs are 100% connectionist.

### §1.3 — Expert Systems (the 1980s peak of symbolic AI)
Systems like **MYCIN** (medical diagnosis) and **XCON** (configuring DEC computers) were thousands of hand-written `IF-THEN` rules. They worked — narrowly. But they couldn't *learn*, couldn't handle ambiguity, and broke the moment reality didn't match the rule book.

### §1.4 — The First "AI Winter" (late 1970s) and Second (late 1980s)
Twice, hype outran reality. Funding dried up. Researchers learned a hard lesson: **handcrafted rules don't scale**. There are too many edge cases in the real world. This pain is what eventually pushed the field toward learning-from-data.

### §1.5 — Machine Learning enters
**Machine Learning (ML)** is a sub-field of AI where the program *infers* the rules from examples instead of being given them. You show it 10,000 photos labeled "cat" or "not cat" and it figures out the pattern itself. Term coined by Arthur Samuel (1959) but didn't go mainstream until the 1990s when computers got fast enough.

### §1.6 — Three Flavors of Machine Learning
- **Supervised learning:** labeled examples (photo → "cat"). Most common.
- **Unsupervised learning:** find structure in unlabeled data (clustering customers).
- **Reinforcement learning:** an agent takes actions, gets rewards, learns a policy (game-playing, robotics, RLHF for LLMs).
LLM training touches *all three*: **self-supervised pretraining** (next-token prediction — the "labels" come from the text itself, no human labeling needed), then supervised fine-tuning, then reinforcement learning from human feedback.

### §1.7 — Checkpoint
Could you explain to a friend, in 60 seconds: "Why did AI shift from writing rules to learning from data?" If yes, move on.

---

## Chapter 2 — Classical Machine Learning

### §1.8 — Features and Labels
Before deep learning, ML required **feature engineering** — humans deciding which numbers about the data mattered. For spam detection: number of exclamation marks, presence of the word "viagra," sender domain age, etc. The model learned weights for these *human-chosen* features.

### §1.9 — Linear Regression and Logistic Regression
The two simplest ML models. Linear regression predicts a number (house price). Logistic regression predicts a probability (spam vs not-spam). Both essentially draw a line/plane through the data. Still used everywhere today for tabular data.

### §1.10 — Decision Trees and Random Forests
A decision tree is literally a flowchart of `if/else` questions, but the splits are *learned* from data. A random forest is hundreds of trees voting. Powerful, interpretable, and dominant for tabular business data even in 2026.

### §1.11 — Support Vector Machines (SVMs)
The hot algorithm of the 2000s. Finds the "widest road" separating two classes. Mostly displaced by neural nets for images and text but still appears in scientific computing.

### §1.12 — The Bias / Variance Tradeoff
- **High bias (underfitting):** model is too simple, misses the pattern.
- **High variance (overfitting):** model memorizes the training data and fails on new data.
Every ML technique — and every LLM trick — is ultimately about navigating this tradeoff.

### §1.13 — Train / Validation / Test Splits
You always split your data three ways. Train teaches the model. Validation tunes settings. Test is the final, untouched honest exam. Violating this discipline ("test set leakage") is the #1 way ML teams fool themselves.

### §1.14 — Why Classical ML Plateaued for Images and Text
Hand-engineered features stop working when the input is raw pixels or raw words. There are too many subtle patterns (curve of an ear, sarcasm in a sentence) for a human to enumerate. Something had to learn the features themselves. Enter deep learning.

---

## Chapter 3 — Neural Networks: The Original Idea

### §1.15 — The Perceptron (1958)
Frank Rosenblatt's **perceptron** was the first artificial neuron — inputs × weights, summed, passed through a step function. It could learn simple patterns. The New York Times claimed it would soon "walk, talk, see, write." It couldn't. Marvin Minsky and Seymour Papert's 1969 book *Perceptrons* showed single-layer perceptrons couldn't even learn XOR. Funding collapsed.

### §1.16 — Multi-Layer Networks Solve XOR
Stack perceptrons in layers and you can represent *any* function (universal approximation theorem). The catch: nobody knew how to train multi-layer networks efficiently — until backpropagation.

### §1.17 — Backpropagation (1986)
Rumelhart, Hinton, and Williams popularized **backprop** — an algorithm that uses calculus's chain rule to figure out how much each weight contributed to the error, and nudges every weight slightly to reduce the error. This is *still* how every neural network on Earth, including today's frontier models, is trained.

### §1.18 — Activation Functions
Each neuron applies a non-linear function (sigmoid, tanh, ReLU) to its sum. Without non-linearity, stacking layers is mathematically equivalent to one layer. **ReLU** (`max(0, x)`) became dominant in the 2010s because it's fast and trains well.

### §1.19 — Gradient Descent
The optimization recipe: compute error, compute the gradient (direction of steepest increase), step the weights *opposite* that direction. Repeat millions of times. Variants like **SGD**, **Adam**, **AdamW** (the LLM standard) tweak this.

### §1.20 — Loss Functions
The number the model is trying to minimize. For classification: **cross-entropy loss**. For regression: **mean squared error**. LLMs use cross-entropy on the next-token prediction task. Remember this — it's the only objective an LLM has during pretraining.

### §1.21 — The Second AI Winter Ends Quietly
Through the 1990s and 2000s, neural nets worked but were *slow* and beaten by SVMs on most benchmarks. Researchers like Geoffrey Hinton, Yann LeCun, and Yoshua Bengio kept the flame alive. Two things eventually changed the game: **GPUs** and **big datasets**.

---

## Chapter 4 — The Deep Learning Revolution

### §1.22 — ImageNet (2009)
Fei-Fei Li's team labeled **14 million images** across 20,000 categories. For the first time, neural nets had enough data to actually learn. ImageNet became the annual benchmark that drove progress.

### §1.23 — AlexNet (2012) — The Big Bang
A deep convolutional neural network (**CNN**) by Krizhevsky, Sutskever, Hinton crushed ImageNet by 10+ percentage points using **two consumer GPUs**. This is the moment modern AI started. Every researcher pivoted to deep learning within a year.

### §1.24 — Convolutional Neural Networks (CNNs)
CNNs use small filters that slide over images, detecting edges, then textures, then object parts, then objects. They mirror how the visual cortex works. Dominant for vision from 2012 to ~2020 (when Vision Transformers started to compete).

### §1.25 — Recurrent Neural Networks (RNNs) and LSTMs
For *sequential* data (text, speech, time series), researchers used **RNNs** — networks with a "memory" that carries information from one step to the next. **LSTMs** and **GRUs** were better variants. They worked but were slow (had to process tokens one-by-one) and forgot things over long sequences.

### §1.26 — Word Embeddings: word2vec (2013) and GloVe (2014)
A breakthrough: represent each word as a ~300-dimensional vector of numbers, learned so that similar words are close together. Famously: `king − man + woman ≈ queen`. This is the seed idea of *every* modern LLM — words are points in a high-dimensional space.

### §1.27 — Sequence-to-Sequence Models (2014)
Google's **seq2seq** used an RNN encoder to read a sentence, compress it into a vector, and an RNN decoder to write a translation. It worked — barely. The bottleneck: that single compressed vector lost too much information.

### §1.28 — The Attention Mechanism (2014–2015)
Bahdanau et al. fixed the bottleneck: instead of compressing the whole input into one vector, let the decoder *attend* to (look back at) any input position when generating each output word. This is the single most important idea in modern AI. We'll spend all of Module 02 on it.

### §1.29 — Why Deep Learning Works At All (intuition)
Three ingredients had to converge: (1) **enough data** (internet-scale), (2) **enough compute** (GPUs), (3) **enough algorithmic tricks** (backprop, ReLU, dropout, batch norm, attention). None alone were sufficient. All three together unlocked the 2010s revolution.

### §1.30 — Checkpoint — Explain it Back
Quiz yourself:
1. What was wrong with symbolic AI?
2. What is backpropagation, in one sentence?
3. Why did 2012 (AlexNet) change everything?
4. What is a word embedding?
5. Why are RNNs slow?

If you can answer all five, you're ready for **Module 02 — The Transformer Era**, where everything we use today was invented.
