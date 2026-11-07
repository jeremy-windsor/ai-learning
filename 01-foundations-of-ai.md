# Module 01 — Foundations of AI

**Mastery: 0% → 7%** · Prerequisites: none

**Goal of this module:** Understand the 70-year arc from "AI is hand-written rules" to "AI is a giant statistical pattern-matcher trained on data." By the end you should be able to explain *why* the field moved away from symbolic AI toward neural networks, what a neural network actually *is* (neurons, weights, layers), how one learns, and why **vectors** became the data type everything in modern AI runs on.

**How to read this module:** A word in **bold** is being defined right there. If a term shows up before it's taught, the text points you to the section that teaches it (for example "→ §1.25").

---

## Chapter 1 — What Even Is AI?

### §1.1 — Where This Is All Going: What Is an LLM?
This guide ends with you explaining how ChatGPT, Claude, Qwen, and Ollama work. All of those are **Large Language Models (LLMs)**. Here's the one-paragraph version so the word doesn't sneak up on you later:

> An **LLM** is a computer program that has read an enormous amount of text and learned to do one thing extremely well: **predict the next chunk of text**. Give it "The capital of France is" and it predicts " Paris". Repeat that prediction over and over, one chunk at a time, and you get paragraphs, code, and answers.

Break the name apart:
- **Large:** it has billions of adjustable numbers inside it (→ §1.30).
- **Language:** it works on text (and, in newer models, images and audio too).
- **Model:** a mathematical function that takes input and produces output. Its behavior comes from numbers it *learned* from data, not rules a person wrote (→ §1.11).

The "chunks of text" are called **tokens**. A token is roughly a word or a piece of a word ("unbelievable" might be `un` + `believ` + `able`). Module 03 covers them in depth. For now, think "token ≈ word."

Everything in Modules 01 and 02 is the 70-year story of how we got to a machine that can do this.

### §1.2 — Starter Vocabulary
Ten words you'll see constantly. You don't need to master them yet; this is a map.

| Term | One-line meaning | Taught in |
|------|------------------|-----------|
| **AI** | Making machines do things we'd call intelligent | §1.3 |
| **Machine Learning (ML)** | Programs that learn rules from examples instead of being told them | §1.7 |
| **Model** | A function with adjustable numbers inside it | §1.11 |
| **Weight / Parameter** | One adjustable number inside a model | §1.11, §1.30 |
| **Training** | Showing a model examples and adjusting its weights until it gets them right | §1.12, §1.35 |
| **Inference** | Using a finished, trained model to answer new questions | §1.12 |
| **Vector** | A list of numbers that describes something | §1.19 |
| **Neural network** | A model built from layers of simple math "neurons" | §1.26–§1.29 |
| **Token** | A word or piece of a word, the unit an LLM reads and writes | §1.1, Module 03 |
| **Embedding** | A vector that captures the meaning of a word or thing | §1.42–§1.44 |

### §1.3 — The Word "Artificial Intelligence"
Coined by John McCarthy for the **Dartmouth Workshop in 1956**. The original pitch: "making a machine behave in ways that would be called intelligent if a human were so behaving." That definition is deliberately vague, and 70 years later it still is. AI is an *aspiration*, not a single technology.

### §1.4 — Two Schools of AI: Symbolic vs. Connectionist
From the very beginning there were two camps:
- **Symbolic AI ("Good Old-Fashioned AI" / GOFAI):** Encode human knowledge as rules and logic. "IF temperature > 100 AND patient has cough THEN suggest flu test."
- **Connectionist AI:** Build networks of simple units (loosely inspired by brain neurons) and let them *learn* patterns from examples instead of being told rules.

The symbolic camp dominated from ~1956 to ~1990. The connectionist camp dominates today. Every LLM is 100% connectionist.

**Network-engineer analogy:** Symbolic AI is a hand-written ACL or static route table. You decide every rule, and it does exactly what you wrote, no more. Connectionist AI is closer to a routing protocol that *converges* on an answer from what it observes, except the "protocol" here is learning from examples.

### §1.5 — Expert Systems (the 1970s–80s peak of symbolic AI)
Systems like **MYCIN** (medical diagnosis, 1970s) and **XCON** (configuring DEC computer orders, 1980s) were thousands of hand-written `IF-THEN` rules. They worked, narrowly. But they couldn't *learn*, couldn't handle ambiguity, and broke the moment reality didn't match the rule book.

### §1.6 — The Two "AI Winters" (mid-1970s and late 1980s)
Twice, hype outran reality and funding dried up. Researchers learned a hard lesson: **handcrafted rules don't scale**. The real world has too many edge cases. That pain is what pushed the field toward learning from data.

### §1.7 — Machine Learning Enters
**Machine Learning (ML)** is a sub-field of AI where the program *infers* the rules from examples instead of being given them. Show it 10,000 photos labeled "cat" or "not cat" and it figures out the pattern itself. Arthur Samuel coined the term in 1959 (his program learned to play checkers better than he did), but ML didn't go mainstream until the 1990s, when computers got fast enough.

### §1.8 — Three Flavors of Machine Learning
- **Supervised learning:** you give labeled examples (photo → "cat"). The most common kind.
- **Unsupervised learning:** you give unlabeled data and the program finds structure in it (grouping customers by behavior).
- **Reinforcement learning (RL):** an **agent** takes actions, gets rewards or penalties, and learns a strategy (game-playing, robotics).

LLM training uses all three ideas:
1. **Self-supervised pretraining:** the model reads text and tries to predict the next token. The "label" is simply the real next word in the text, so no human labeling is needed. (Module 04)
2. **Supervised fine-tuning:** humans write example conversations to teach it to act like an assistant. (§5.4)
3. **Reinforcement learning from human feedback (RLHF):** humans rate answers and the model is rewarded for the good ones. (§5.7)

### §1.9 — Checkpoint
Could you explain to a friend in 60 seconds why AI shifted from writing rules to learning from data? If yes, move on.

---

## Chapter 2 — Classical Machine Learning: "Learning" Means Tuning Numbers

### §1.10 — Features and Labels
Every ML problem starts with a table of examples. Each column describing the example is a **feature** (an input), and the answer you want is the **label** (the output).

For spam detection:

| Exclamation marks | Has "free money" | Sender domain age (days) | **Label: spam?** |
|---|---|---|---|
| 7 | yes | 3 | spam |
| 0 | no | 4,000 | not spam |

Before deep learning, humans had to choose the features. That's called **feature engineering**, and it was most of the work.

### §1.11 — What a "Model" Actually Is: A Formula With Knobs
This section is the foundation for everything else, so go slowly.

Say you want to predict a house's price. You might guess the price works like this:

```
price = (w1 × square_feet) + (w2 × bedrooms) + b
```

- `square_feet` and `bedrooms` are the **inputs** (features).
- `w1` and `w2` are **weights**. A weight says *how much an input counts*. A big `w1` means square footage matters a lot. A weight near zero means that input barely matters. A negative weight means more of that input pushes the answer *down*.
- `b` is the **bias**: a starting value added no matter what the inputs are (a base price).

Weights and biases together are called **parameters**. A **model** is just this kind of formula plus its parameters.

Suppose training found `w1 = 150`, `w2 = 10,000`, `b = 50,000`. Then a 2,000 sq ft, 3-bedroom house is predicted at `150×2000 + 10000×3 + 50000 = $380,000`.

**Network-engineer analogy:** EIGRP's composite metric is a weighted formula. The K-values decide how much bandwidth versus delay count toward the final metric. You *set* the K-values by hand. Machine learning is the idea of having the computer *discover* the right K-values automatically, by looking at thousands of examples where the right answer is already known.

### §1.12 — Training vs. Inference (First Look)
How does the computer find good weights? Roughly:
1. Start with random weights.
2. Make a prediction for an example where you know the true answer.
3. Measure how wrong it was (the **error**).
4. Nudge each weight a little in the direction that would have made the error smaller.
5. Repeat for thousands or millions of examples.

That loop is **training**. Once the weights stop improving, you freeze them. Using the frozen model to answer new questions is **inference**. The two are very different jobs: training a frontier LLM takes months on thousands of GPUs, while inference is one prompt in, one answer out. (Chapter 4 explains the loop in detail. Module 04 covers training at scale, and Modules 08–09 cover inference.)

### §1.13 — Linear Regression and Logistic Regression
The two simplest ML models, and both are exactly the "formula with knobs" from §1.11.
- **Linear regression** predicts a number (house price). It's the formula above: a weighted sum of inputs plus a bias. On a graph, it draws a straight line (or flat plane) through the data.
- **Logistic regression** predicts a probability (how likely is this email spam?). It computes the same weighted sum, then squashes the result into the 0–1 range with the **sigmoid** function (an S-shaped curve: big negative → near 0, big positive → near 1).

Remember logistic regression: a single artificial neuron (§1.26) is basically the same thing. Both models are still used everywhere today for table-style data.

### §1.14 — Decision Trees and Random Forests
A **decision tree** is a flowchart of `if/else` questions ("Is domain age < 30 days? → Are there > 3 exclamation marks?"), but the questions and split points are *learned* from data. A **random forest** is hundreds of trees voting. Powerful, easy to inspect, and still dominant for table-style business data in 2026.

### §1.15 — Support Vector Machines (SVMs)
The hot algorithm of the late 1990s and 2000s. It finds the "widest road" separating two classes of points. Mostly replaced by neural networks for images and text, but still used in scientific computing.

### §1.16 — The Bias / Variance Tradeoff
(Different "bias" from §1.11, confusingly. This one means "how wrong your assumptions are.")
- **High bias (underfitting):** the model is too simple and misses the pattern. Like fitting a straight line to data that curves.
- **High variance (overfitting):** the model memorizes the training examples, including their noise, and fails on new data. Like a student who memorized the practice exam answers but didn't learn the subject.

Every ML technique, and every LLM training trick, is ultimately about steering between these two.

### §1.17 — Train / Validation / Test Splits
You always split your data three ways:
- **Train** (~80%): the model learns from this.
- **Validation** (~10%): you use this to tune settings and to notice overfitting.
- **Test** (~10%): locked away until the very end, then used once as the honest final exam.

If test data leaks into training (**test set leakage**), your scores look great and mean nothing. It's the #1 way ML teams fool themselves.

### §1.18 — Why Classical ML Plateaued for Images and Text
Feature engineering stops working when the input is raw pixels or raw words. There are too many subtle patterns (the curve of an ear, sarcasm in a sentence) for a human to list as features. Something had to *learn the features themselves*. That something is the neural network. But first you need the language neural networks speak: vectors.

---

## Chapter 3 — Vectors: The Data Type of AI

Every input, every internal value, and every output of a modern AI model is a vector. This chapter is worth extra time. Vectors are also directly useful on their own today: semantic search, recommendations, and RAG (Module 12) are all "vector math on embeddings."

### §1.19 — What Is a Vector?
A **vector** is an ordered list of numbers. That's it.

```
[2.0, 3.5, -1.0]
```

Each position in the list is a **dimension**, and the order matters: position 1 always means the same thing.

You already use vectors. One row of SNMP interface counters is a vector:

```
[in_mbps, out_mbps, errors, drops]  →  [840.2, 120.5, 3, 0]
```

The spam-email row in §1.10 is a vector too: `[7, 1, 3]`. In ML, every example becomes a vector before the model sees it.

### §1.20 — Vectors as Points in Space
A 2-dimensional vector like `[3, 4]` can be drawn as a point on graph paper: 3 across, 4 up. (Or as an arrow from the origin to that point.) A 3-D vector is a point in a room.

AI vectors have hundreds or thousands of dimensions: 300 for classic word vectors, 4,096 inside Llama 3 8B, 12,288 inside GPT-3. Nobody can picture 4,096-D space, and you don't need to. The math works exactly the same as in 2-D. **The key intuition carries over: things that are similar end up as points that are close together.**

### §1.21 — The Dot Product (The Most Important Operation in AI)
The **dot product** of two vectors: multiply matching positions, then add everything up.

```
[2, 3, 1] · [4, 1, 5]  =  (2×4) + (3×1) + (1×5)  =  8 + 3 + 5  =  16
```

Two reasons it matters:
1. **It *is* the weighted sum from §1.11.** `price = w1×sqft + w2×bedrooms` is just `[w1, w2] · [sqft, bedrooms]`. Every neuron computes a dot product.
2. **It measures how aligned two vectors are.** Vectors pointing the same direction give a big positive dot product. Unrelated (perpendicular) vectors give about zero. Opposite vectors give a big negative one. Attention in Module 02 is built entirely on this.

GPUs exist in AI largely because they can do trillions of these multiply-and-add operations per second.

### §1.22 — Length and Distance
- **Length** (also called **magnitude** or **norm**) of a vector: `sqrt(x1² + x2² + ...)`. The length of `[3, 4]` is `sqrt(9 + 16) = 5`, just like the hypotenuse of a triangle.
- **Distance** between two vectors: subtract them, then take the length of the result. `[1, 1]` and `[4, 5]` are `sqrt(3² + 4²) = 5` apart.

### §1.23 — Cosine Similarity: "Do These Point the Same Way?"
**Cosine similarity** is the dot product divided by both lengths:

```
cosine(a, b) = (a · b) / (length(a) × length(b))
```

Dividing by the lengths removes size, so only *direction* is compared. The result always lands between −1 and +1:
- **+1:** same direction (very similar)
- **0:** unrelated
- **−1:** opposite

A toy example with made-up 3-D "word vectors" (you'll compute these yourself in the lab):

| Pair | Cosine |
|------|--------|
| cat `[0.9, 0.8, 0.1]` vs dog `[0.8, 0.9, 0.2]` | **0.99** (very similar) |
| cat vs truck `[0.1, 0.1, 0.9]` | 0.24 |
| dog vs truck | 0.31 |

This one formula powers "find similar documents," "customers who bought this also bought," duplicate detection, and the retrieval step of every RAG system (§3.19, §12.15). It works the same in 4,096 dimensions as in 3.

### §1.24 — Matrices: Many Vectors at Once
A **matrix** is a grid of numbers: rows and columns. You can think of it as a stack of vectors.

**Matrix × vector** means: take the dot product of the vector with *each row* of the matrix, and collect the answers into a new vector.

```
| 1  2 |   | 5 |     | 1×5 + 2×6 |     | 17 |
| 3  4 | × | 6 |  =  | 3×5 + 4×6 |  =  | 39 |
```

Hold on to this: **one layer of a neural network is a matrix × vector.** Each row of the matrix is one neuron's weights. Multiply, and you get every neuron's output at once.

A **tensor** is the general term for these arrays: a single number, a vector (1-D), a matrix (2-D), or a stack of matrices (3-D and up). It's why the most popular AI framework is called Tensor-*Flow*, and why PyTorch's core object is `torch.Tensor`.

### §1.25 — Why GPUs Matter for AI
A matrix multiply is millions of independent multiply-and-add operations. None of them has to wait for the others. A CPU has a few dozen powerful cores; a GPU has thousands of simple ones that can all do multiply-adds at once. That makes GPUs a perfect fit for AI math. (Module 06 goes deep.)

---

## Chapter 4 — Neural Networks, From the Ground Up

### §1.26 — The Artificial Neuron
An **artificial neuron** does three steps:
1. Multiply each input by its weight.
2. Add them up and add a bias (a dot product plus a bias, §1.21).
3. Pass the result through an **activation function** (§1.28), which decides how strongly the neuron "fires."

Worked example: a neuron that judges "is this link in trouble?" Inputs scaled 0–1:

| Input | Value | Weight | Input × weight |
|-------|-------|--------|----------------|
| Utilization | 0.9 | 2.0 | 1.80 |
| Error rate | 0.2 | 3.0 | 0.60 |
| Latency | 0.7 | 1.5 | 1.05 |
| **Bias** | | | **−3.00** |
| **Sum** | | | **0.45** |

Run 0.45 through a sigmoid and you get **0.61**: about 61% "trouble." The neuron weighs error rate most heavily (weight 3.0), and the negative bias means it takes several bad signals together to push it over 50%.

Nobody typed in those weights. In a real network they're **learned** from examples (§1.32–§1.35). That's the whole trick.

### §1.27 — The Perceptron (1958) and the XOR Problem
Frank Rosenblatt's **perceptron** was the first artificial neuron built to learn. It used a "step" activation: output 1 if the sum is above zero, else 0. The New York Times reported it would one day "walk, talk, see, write, reproduce itself and be conscious of its existence." It couldn't.

In 1969, Marvin Minsky and Seymour Papert's book *Perceptrons* showed that a single perceptron can't even learn **XOR** ("exclusive or": output 1 when exactly one input is 1):

| A | B | XOR |
|---|---|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Why it can't:** a single neuron can only draw *one straight line* to split its inputs into "yes" and "no." Plot the four XOR cases on graph paper and you'll see the two "1" cases sit on opposite corners. No single straight line separates them. Funding for neural networks collapsed for over a decade. (You'll watch a neuron fail at XOR yourself in the lab.)

### §1.28 — Activation Functions: Why Non-Linearity Matters
The activation function is applied after the weighted sum. Common ones:
- **Step:** 0 or 1. The original; too blunt to train well.
- **Sigmoid:** smooth S-curve from 0 to 1 (§1.13).
- **Tanh:** like sigmoid but from −1 to +1.
- **ReLU** (`max(0, x)`): negative → 0, positive → unchanged. Dead simple, fast, and trains well. Dominant since ~2012.
- **GELU / SwiGLU:** smoother cousins of ReLU used inside today's LLMs (§2.15).

**Why bother?** Without a non-linear activation, stacking layers is pointless: a chain of weighted sums collapses into one big weighted sum, which can only draw one straight line again. The bend in the activation lets layers combine into curves and complex shapes.

### §1.29 — Layers: From One Neuron to a Network
A **neural network** is neurons arranged in **layers**:

```
 Input layer        Hidden layer(s)        Output layer
 (your data)        (learned features)     (the answer)

   x1 ──┐         ┌── h1 ──┐
        ├────────►├── h2 ──├──────────► output
   x2 ──┘         └── h3 ──┘
```

- **Input layer:** just the raw numbers (the vector) you feed in.
- **Hidden layers:** neurons whose outputs feed the next layer. They learn their own intermediate features. Nobody tells them what to detect.
- **Output layer:** produces the final answer.
- In a **fully-connected** (or **dense**) layer, every neuron takes input from every neuron in the previous layer.
- **Deep learning** just means networks with many hidden layers.

**Layers solve XOR.** Use two hidden neurons: one computes "A OR B," the other "NOT (A AND B)." Feed both into an output neuron that computes AND. The result is 1 only when exactly one input is 1: XOR. Each neuron still draws one straight line, but combining two lines carves out the shape a single neuron couldn't.

The **universal approximation theorem** (1989) takes this to the limit: a network with just one hidden layer and a non-linear activation can approximate *any* reasonable function, given enough neurons. The catch for decades: nobody had a good way to *train* multi-layer networks. That came with backpropagation (§1.34).

### §1.30 — Parameters: What "8B" Means
Every connection has a weight, and every neuron has a bias. Count them all and you get the **parameter count**.

A tiny layer with 4 inputs and 3 neurons has `4 × 3 = 12` weights plus `3` biases = **15 parameters**.

When you see "Llama 3 **8B**," that means about **8 billion** of these numbers. Stored at 2 bytes each, that's about 16 GB just for the weights. That's why model size and GPU memory (VRAM) are so tightly linked (Modules 06 and 07). The file you download with `ollama pull` is mostly these numbers.

### §1.31 — The Forward Pass
Running data through the network, start to finish, is the **forward pass**:

```
input vector
  → layer 1: matrix × vector, add biases, apply activation
  → layer 2: matrix × vector, add biases, apply activation
  → ...
  → output vector
```

Inference (§1.12) is just forward passes. When an LLM writes one token, that's one forward pass through every layer.

### §1.32 — Loss Functions: Measuring "How Wrong"
To learn, the network needs one number that says how wrong it was. That's the **loss** (smaller is better).
- **Mean squared error (MSE)**, for predicting numbers: square the difference between prediction and truth. Predict $380k for a $400k house → error 20k → squared 400 million. Squaring makes big misses hurt a lot more than small ones.
- **Cross-entropy**, for picking a category: look at the probability the model gave the *correct* answer and take `−log` of it.

  | Model's probability for the right answer | Loss |
  |------|------|
  | 0.9 (confident and right) | 0.11 |
  | 0.5 (unsure) | 0.69 |
  | 0.1 (confident and wrong) | 2.30 |

  Confident wrong answers are punished hardest.

LLMs use cross-entropy on next-token prediction: "what probability did you give the token that actually came next?" Remember this: it's the *only* objective an LLM has during pretraining (§4.2).

### §1.33 — Gradient Descent: Walking Downhill in the Fog
Picture the loss as a hilly landscape, where your position is the current setting of every weight and the height is the loss. You want the lowest valley, but it's foggy and you can only feel the slope under your feet.

**Gradient descent:** feel which way is downhill (that slope is the **gradient**), take a small step that way, repeat.
- The step size is the **learning rate**. Too big and you leap over the valley; too small and training takes forever.
- **Stochastic gradient descent (SGD):** estimate the slope from a small random **batch** of examples instead of the whole dataset. It's noisier but much faster.
- **Adam** and **AdamW** are smarter step-takers that adapt the step size for each weight. AdamW is the LLM standard.

With billions of weights, the "landscape" has billions of directions, but the recipe is identical.

### §1.34 — Backpropagation: Assigning the Blame
Gradient descent needs to know, for *every single weight*, "if I nudge you up slightly, does the loss go up or down, and by how much?" **Backpropagation** computes that efficiently.

It starts at the output (where the error is known) and works *backward* layer by layer, using the chain rule from calculus to pass each layer's share of the blame to the layer before it. One backward pass gives you the direction for every weight at once.

Analogy: after an outage, a postmortem starts at the symptom and traces back hop by hop, assigning each device its share of the blame, so each one knows exactly what to fix.

The math was worked out in the 1970s, but Rumelhart, Hinton, and Williams' 1986 paper made it famous. It is *still* how every neural network on Earth is trained, including today's frontier LLMs.

### §1.35 — The Training Loop, Assembled
Now §1.12 in full detail:

```
repeat until the loss stops improving:
    grab a batch of examples                 # e.g. 32 emails
    forward pass  → predictions              # §1.31
    compute loss  → one "wrongness" number   # §1.32
    backpropagation → a gradient per weight  # §1.34
    update: weight -= learning_rate × gradient   # §1.33
```

Vocabulary:
- **Step:** one trip around the loop.
- **Batch:** the examples used in one step.
- **Epoch:** one full pass through the whole training set.

Pretraining an LLM is this exact loop, run for months across thousands of GPUs. The lab at the end of this module is this loop in about 20 lines of Python.

### §1.36 — Neural Networks in the Wilderness (1990s–2000s)
Through the 1990s and 2000s, neural nets worked but were slow and usually lost to SVMs and random forests. A small group, especially Geoffrey Hinton, Yann LeCun, and Yoshua Bengio, kept the flame alive (they shared the 2018 Turing Award for it). Two things eventually changed the game: **GPUs** and **big datasets**.

---

## Chapter 5 — The Deep Learning Revolution

### §1.37 — ImageNet (2009)
Fei-Fei Li's team labeled about **14 million images** across ~20,000 categories. The yearly ImageNet competition used a 1.2-million-image, 1,000-category subset. For the first time, neural nets had enough labeled data to really learn, and the competition became the scoreboard that drove progress.

### §1.38 — AlexNet (2012): The Big Bang
A deep neural network by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton won ImageNet 2012 with a top-5 error of 15.3% versus 26.2% for second place, trained on **two consumer gaming GPUs**. That gap was so large that nearly every vision researcher switched to deep learning within a year or two. This is the moment modern AI started.

### §1.39 — Convolutional Neural Networks (CNNs)
AlexNet was a **CNN**. Instead of connecting every pixel to every neuron, a CNN slides small **filters** (say, 3×3 pixels of learned weights) across the image. Early layers learn to detect edges, middle layers textures and shapes, later layers object parts and whole objects. Nobody programmed "detect an edge." The network learned that edges are useful. The design was loosely inspired by the visual cortex. CNNs dominated vision from 2012 until Vision Transformers began competing around 2020.

### §1.40 — The Deep Learning Toolkit
A few tricks made deep networks practical. You'll meet them again inside transformers:
- **ReLU** (§1.28): keeps gradients healthy in deep networks.
- **Dropout** (2012): during training, randomly switch off some neurons each step so the network can't rely on any one of them. Reduces overfitting.
- **Batch normalization** (2015): re-scale each layer's outputs to a stable range so training doesn't blow up. Transformers use a cousin called LayerNorm (§2.14).
- **Residual connections** (ResNet, 2015): add a layer's input back to its output so signals and gradients can skip through very deep stacks (§2.13).
- **GPUs + frameworks** (CUDA, later PyTorch): made all this fast and easy to write.

---

## Chapter 6 — Deep Learning Meets Language

### §1.41 — The Problem: Text Isn't Numbers
Neural networks need vectors, so how do you turn the word "cat" into numbers?

The naive approach is **one-hot encoding**: make a vector as long as your vocabulary (say 50,000 words), all zeros except a single 1 in the slot for that word.

```
cat  = [0, 0, 1, 0, 0, ... 0]    (50,000 numbers)
dog  = [0, 0, 0, 1, 0, ... 0]
```

Two problems. The vectors are huge and almost all zeros. Worse, the cosine similarity between *any* two different words is exactly 0, so "cat" is as different from "dog" as from "refrigerator." The representation knows nothing about meaning.

### §1.42 — Word Embeddings: word2vec (2013) and GloVe (2014)
The breakthrough: give every word a short, **dense** vector (around 300 numbers, all meaningful), and *learn* those numbers.

**word2vec** (Google, 2013) learned them with a simple game: given a word, predict the words around it (or the reverse). Words used in similar contexts ("cat" and "dog" both appear near "pet," "fed," "vet") get pushed toward similar vectors. The idea goes back to a 1957 line from the linguist J.R. Firth: *"You shall know a word by the company it keeps."* **GloVe** (Stanford, 2014) reached similar vectors by counting how often words appear together.

A learned vector like this is called an **embedding**: a vector that captures meaning, where similar things land close together.

### §1.43 — Meaning as Geometry
Once words are points in space, you can do math with meaning:
- **Similarity:** the cosine between "cat" and "dog" is high; between "cat" and "invoice," low.
- **Clusters:** countries group together, verbs group together, programming languages group together.
- **Directions mean things:** the famous `king − man + woman ≈ queen`. The step from "man" to "woman" is roughly the same direction as from "king" to "queen," so the "gender" relationship is captured as a direction.

Honest caveat: the analogy trick is a bit cherry-picked (it only works if you exclude the input words from the answers, and many analogies fail). The *similarity* and *clustering* properties are the robust, useful part.

Also note: the individual dimensions usually don't mean anything a human can name. There's no "dimension 17 = furriness." Meaning lives in the overall position, not in single numbers. (The toy vectors in §1.23 were simplified.)

### §1.44 — Why Vectors Still Matter Today
Embeddings didn't go away when LLMs arrived. They became *more* central:
- **Inside every LLM:** the first thing an LLM does is look up an embedding vector for each token (§3.14). Every layer after that reads and writes vectors.
- **Semantic search:** turn every document and every question into an embedding, then return the documents with the highest cosine similarity. It finds "router won't form an adjacency" when you searched "OSPF neighbor stuck in EXSTART," even with no shared keywords.
- **RAG (Retrieval-Augmented Generation):** semantic search finds relevant documents and hands them to an LLM to answer from (§12.15). Stored in **vector databases** like pgvector, Qdrant, or Milvus (§3.20).
- **Recommendations, deduplication, clustering, anomaly detection:** all "embed things, then compare vectors."

One big limitation of word2vec: each word gets *one* vector, so "bank" (river) and "bank" (money) share a vector. Modern models fix this with **contextual embeddings**: the vector for a word changes depending on the sentence around it. That's exactly what transformers do (Module 02).

### §1.45 — Recurrent Neural Networks (RNNs)
Text is a **sequence**: order matters, and length varies. A regular network takes a fixed-size input. A **Recurrent Neural Network (RNN)** handles sequences by reading one word at a time and carrying a running memory forward.

That memory is a vector called the **hidden state**. At each step:

```
new_hidden_state = f( current_word_embedding, previous_hidden_state )
```

"Unrolled" over a sentence:

```
 "The"  →  [RNN] → h1
 "cat"  →  [RNN] → h2   (h2 depends on "cat" AND h1)
 "sat"  →  [RNN] → h3   (h3 depends on "sat" AND h2)
```

The same weights are reused at every step. By the end, the hidden state is supposed to summarize the whole sentence. It's a bit like a router that only remembers a running summary of every packet so far instead of storing the packets.

Two big problems:
1. **Slow:** step 3 can't start until step 2 is done, so you can't spread the work across a GPU's thousands of cores.
2. **Forgetful:** information from early words fades over long sequences (§1.46).

### §1.46 — The Vanishing Gradient Problem
During training, backpropagation (§1.34) has to send blame backward through *every time step*. At each step the signal gets multiplied by a number, and those numbers are often less than 1. Multiply 0.5 by itself 20 times and you get 0.00000095. The learning signal from early words effectively vanishes, so a plain RNN can't learn that word 3 matters to word 40. (Numbers bigger than 1 cause the opposite problem: **exploding gradients**.)

### §1.47 — LSTMs and GRUs: RNNs With Gates
The **LSTM (Long Short-Term Memory)**, invented by Hochreiter and Schmidhuber in 1997, fixes forgetting by adding a second memory track and three **gates**.

- **Cell state:** a "conveyor belt" of memory running alongside the hidden state. Information can ride along it for many steps without being multiplied down at each one.
- **Gates:** small learned neural layers that output values between 0 (closed) and 1 (open), controlling the flow:
  - **Forget gate:** what to erase from the conveyor belt ("new sentence subject, forget the old one").
  - **Input gate:** what new information to write onto it.
  - **Output gate:** what part of the memory to reveal as this step's output.

Think of the gates as learned, per-item valves (or a QoS policy deciding what to keep, drop, or pass), and the network learns *when* to open each one.

The **GRU (Gated Recurrent Unit, 2014)** is a simpler version with two gates instead of three. It's often just as good and faster.

LSTMs powered Google Translate, Siri, and Alexa in the mid-2010s. But they were still sequential (slow) and still struggled with very long text. Module 02 is the story of what replaced them.

### §1.48 — Sequence-to-Sequence Models (2014)
**Seq2seq** (Sutskever, Vinyals, and Le at Google, 2014) tackled translation with two LSTMs:
- An **encoder** reads the English sentence and squeezes it into one final hidden-state vector.
- A **decoder** starts from that vector and writes the French sentence one word at a time.

It worked, barely. The problem: the entire meaning of a 50-word sentence had to fit into *one* fixed-size vector. Long sentences got mangled. That vector was a **bottleneck**.

### §1.49 — The Attention Mechanism (2014–2015)
Bahdanau, Cho, and Bengio removed the bottleneck. Instead of relying on one compressed vector, the decoder may *look back at every encoder hidden state* each time it writes a word, and decide how much each input word matters right now. Translating "chat" back to "cat," it focuses on "cat" in the source.

How does it decide what to focus on? It scores how relevant each input vector is with (essentially) a **dot product** (§1.21), turns the scores into percentages, and takes a weighted blend. That's attention: the single most important idea in modern AI. Module 02 is built on it.

---

## Chapter 7 — Putting It Together

### §1.50 — Why Deep Learning Works at All
Three ingredients had to come together:
1. **Enough data:** internet-scale text and images.
2. **Enough compute:** GPUs doing matrix math in parallel (§1.25).
3. **Enough algorithmic tricks:** backprop, ReLU, dropout, normalization, residuals, attention.

None was enough alone. Together they unlocked the 2010s revolution, and the transformer (Module 02) put all three into overdrive.

### 🛠️ Build It — Hands-On Lab
Pure Python, nothing to install. Save each block as a `.py` file and run it with `python3 file.py`.

**Lab 1: train a neuron (the training loop from §1.35).**

```python
import math
import random

random.seed(0)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def train(data, steps=5000, lr=0.5):
    # Two inputs -> two weights, plus one bias. That's 3 parameters.
    w1, w2, b = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
    for step in range(steps):
        for (x1, x2), target in data:
            # Forward pass: weighted sum, then activation
            out = sigmoid(w1 * x1 + w2 * x2 + b)
            # How wrong were we, and which way should each knob move?
            error = out - target
            w1 -= lr * error * x1
            w2 -= lr * error * x2
            b  -= lr * error
    return w1, w2, b

def show(name, data):
    w1, w2, b = train(data)
    print(f"{name}: w1={w1:.2f} w2={w2:.2f} b={b:.2f}")
    for (x1, x2), target in data:
        out = sigmoid(w1 * x1 + w2 * x2 + b)
        print(f"  {x1} {x2} -> {out:.2f}  (want {target})")

AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
show("AND", AND)
show("XOR", XOR)
```

What you should see: AND is learned perfectly (outputs near 0, 0, 0, 1). XOR fails: the outputs stay stuck between about 0.35 and 0.6 and never reach 0 or 1, because one neuron can only draw one straight line (§1.27). Things to try:
- Change `lr` to `0.01` and `5.0`. What happens to the AND result?
- Print the weights every 1,000 steps to watch them move.

(The `error * x1` update is the gradient for this setup: sigmoid plus cross-entropy loss. Backprop computes the same kind of thing automatically for networks with many layers.)

**Lab 2: vector similarity (§1.21–§1.23).**

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def length(a):
    return math.sqrt(dot(a, a))

def cosine(a, b):
    return dot(a, b) / (length(a) * length(b))

# Toy 3-dimensional "embeddings". Real ones have hundreds of dimensions.
words = {
    "cat":   [0.9, 0.8, 0.1],
    "dog":   [0.8, 0.9, 0.2],
    "truck": [0.1, 0.1, 0.9],
}
for a in words:
    for b in words:
        if a < b:
            print(f"{a:>5} vs {b:<5} cosine = {cosine(words[a], words[b]):.2f}")
```

Things to try: add a "car" vector close to "truck" and check that it scores high against truck and low against cat. Then double every number in "dog" and confirm the cosine doesn't change (only direction counts). In Module 03's lab you'll do the same with *real* embeddings.

### §1.51 — Checkpoint — Explain It Back
Answer out loud, in plain English:
1. What is an LLM, in one sentence?
2. What was wrong with symbolic AI?
3. What is a weight, and what does "a model with 8 billion parameters" mean?
4. What does a single artificial neuron compute?
5. Why can't one neuron learn XOR, and how do layers fix it?
6. Describe the training loop: forward pass, loss, backprop, update.
7. What is a vector, and what does cosine similarity tell you about two of them?
8. What is a word embedding, and why is one-hot encoding worse?
9. Why are RNNs slow, and what problem do LSTMs solve?
10. What bottleneck did attention remove?

<details>
<summary>Answer key (try first!)</summary>

1. A program that learned, from huge amounts of text, to predict the next token, and generates text by doing that repeatedly.
2. Hand-written rules don't scale: they can't learn, can't handle ambiguity, and break on edge cases.
3. A weight is an adjustable number saying how much one input counts. 8B parameters means about 8 billion such learned numbers (weights plus biases).
4. A weighted sum of its inputs (a dot product), plus a bias, passed through an activation function.
5. One neuron draws a single straight line, and XOR's "yes" cases sit on opposite corners. Hidden layers combine several lines (for example OR and NAND, then AND).
6. Run a batch through the network, measure how wrong it was with the loss, use backprop to get each weight's gradient, step each weight a little downhill, and repeat.
7. An ordered list of numbers. Cosine similarity says whether two vectors point the same way (+1), are unrelated (0), or point opposite ways (−1).
8. A learned dense vector where similar words land close together. One-hot vectors are huge and make every pair of words equally unrelated.
9. They process tokens one at a time, so they can't run in parallel. LSTMs use a cell state and gates so information (and gradients) survive across long sequences.
10. Seq2seq squeezed the whole input into one vector. Attention lets the decoder look back at every input position and blend the relevant ones.
</details>

If you can answer these, you're ready for **Module 02 — The Transformer Era**, where the architecture behind every modern LLM was invented.
