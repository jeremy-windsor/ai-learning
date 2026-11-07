# Module 11 — Serving & Autoscaling

**Mastery: 83% → 92%** · Prerequisites: Modules 08–09

**Goal of this module:** Take a working inference engine and turn it into a reliable, scalable, multi-tenant production service. By the end you'll be able to set latency targets and measure them correctly, explain why a GPU fleet at 90% busy feels broken, size a fleet with Little's Law, decide between self-hosting and an API on real numbers, and draw the architecture of a modern LLM app.

**How to read this module:** **Bold** marks a definition. Much of this module is classic distributed-systems and networking thinking (queues, load balancing, policers, SLOs) applied to GPUs. If you've run a production network, you already have most of the intuition. The new part is that every "packet" here is a request that holds a GPU slot for seconds, not microseconds.

---

## Chapter 1 — From Engine to Service

### §11.1 — Engine vs. Service
An **inference engine** (vLLM, SGLang, TGI, llama.cpp; Module 08) is one process that loads a model and generates tokens. A **service** is everything you wrap around it so real users can depend on it: load balancing, autoscaling, authentication, rate limiting, retries, monitoring, multi-region failover, model versioning, and safety filters. In a serious deployment, the engine is maybe 20% of the system.

Vocabulary for this module:
- **Replica:** one running copy of the model server. It might span several GPUs with tensor parallelism (§8.21).
- **Gateway:** the front door that every request passes through (auth, quotas, routing).
- **RPS / QPS:** requests (or queries) per second arriving at the service.
- **Concurrency:** how many requests are being worked on at the same moment.

### §11.2 — The OpenAI-Compatible API Layer
Almost every serving stack exposes the same HTTP endpoints that OpenAI popularized: `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` (§8.32). vLLM, SGLang, TGI, LiteLLM, and Ollama (at `http://localhost:11434/v1`) all speak it. Clients like the OpenAI Python SDK, LangChain, and LlamaIndex can switch providers by changing one base URL.

Two details matter for serving:
- **Streaming:** with `"stream": true`, tokens come back as they're generated, over **Server-Sent Events (SSE)**, a long-lived HTTP response that sends one small chunk per token (§8.7). Users see text after the time to first token instead of waiting for the whole answer. Your load balancers and proxies must allow long-lived connections and must not buffer the response.
- **Newer API shapes exist** (OpenAI's Responses API, Anthropic's Messages API), but `/v1/chat/completions` is still the common language. Build *your* internal API to it, so apps can swap models without code changes.

### §11.3 — NVIDIA Triton Inference Server (the Server, Not the Language)
A multi-framework, multi-model server. It was the standard way to serve models before LLM-specific engines like vLLM existed. Today it's still common for **non-LLM** models (vision, embeddings, recommendation), and as a front end for **TensorRT-LLM** (§8.25). It provides request batching, model ensembles (chaining models), and dynamic model loading. (Not to be confused with the Triton *GPU programming language*, §6.26.)

### §11.4 — Serving Platforms: Ray Serve, BentoML, KServe, Serverless
General-purpose layers that sit above the engine:
- **Ray Serve:** Python-native and scales across a Ray cluster. Good for **compound AI systems** (§11.29) where a retriever, a reranker, and an LLM form one service.
- **BentoML:** packaging and serving with a strong developer experience.
- **KServe:** Kubernetes-native model serving, the cloud-native standard. Newer Kubernetes projects (the Gateway API Inference Extension, `llm-d`) add LLM-aware routing on top (§11.14).
- **Serverless GPU** (Modal, Replicate, RunPod Serverless, and others): you write a function, and they handle scaling, including scaling to zero. Managed model APIs (AWS Bedrock, Azure, Google Vertex) go one step further: you never see a GPU.

### §11.5 — Choosing Your Serving Architecture
| Situation | Typical choice | Pros | Cons |
|-----------|---------------|------|------|
| One team, one model, steady traffic | `vllm serve` replicas behind a load balancer | Simple, cheap, easy to debug | You build autoscaling, auth, and monitoring yourself |
| Many models, many teams | KServe or Ray Serve on a shared GPU cluster | Shared platform, consistent operations | Real platform-engineering effort |
| Bursty or occasional traffic | Serverless GPU or a managed API | No idle GPUs; scales to zero | Cold starts (§11.15); higher per-token price at volume |
| Huge scale, latency-critical | Custom stack (TensorRT-LLM / SGLang / vLLM with disaggregation, §11.18) | Best performance per dollar | Needs a dedicated team |
| Small scale, any of the above | Just use an API | Zero ops | Data leaves your network; vendor lock-in |

---

## Chapter 2 — Measuring It: Latency, Throughput, and SLOs

### §11.6 — Latency vs. Throughput
From §9.35: **TTFT** (time to first token) is the prefill wait, and **TPOT** (time per output token) is the decode speed (§3.29).
- **Latency:** how long *one user* waits. For a full answer: `TTFT + TPOT × (output tokens − 1)`.
- **Throughput:** how many tokens per second the *whole system* produces across all users.

Worked example: TTFT 300 ms, TPOT 30 ms, a 500-token answer → `0.3 + 0.030 × 499 ≈ 15.3 seconds` total. With streaming, the user *starts reading* after 0.3 seconds, which is why streaming matters so much.

The two trade off. Bigger batches (§8.19) mean more total tokens per second per GPU (higher throughput), but each user's tokens come out a little slower (worse TPOT). Production tuning is choosing the right point on that curve for your product.

### §11.7 — Percentiles: Why Averages Lie
Latency is a *distribution*, not one number. **Percentiles** describe it:
- **p50 (median):** half of requests are faster than this. The "typical" experience.
- **p95 / p99:** 95% or 99% of requests are faster. The bad-day experience.

An average hides the slow requests. If 99 requests take 1 second and one takes 60 seconds, the average is about 1.6 s, which looks fine. But the p99 user waited a minute. At scale, *someone* is always the p99 user, and in a chat session with 20 turns, most users hit a p95-level delay at least once. Always set targets and alerts on p95/p99, not averages. (Same lesson as monitoring link jitter: the average is fine while voice calls break.)

### §11.8 — SLIs, SLOs, SLAs, and Error Budgets
Terms from site reliability engineering (SRE):
- **SLI** (service level *indicator*): the thing you measure. Example: "TTFT for chat requests."
- **SLO** (service level *objective*): your internal target for it. Example: "p99 TTFT under 2 s over 30 days."
- **SLA** (service level *agreement*): a promise to a customer, with consequences such as refunds. SLAs should be looser than SLOs.
- **Error budget:** the failure an SLO allows. 99.9% availability allows about 43 minutes of downtime per 30 days. While budget remains you can ship risky changes; when it's spent, you slow down and fix reliability.

An example SLO set for a chat product:
- p50 TTFT < 500 ms, p99 < 2 s
- p50 TPOT < 30 ms, p99 < 80 ms
- 99.9% of requests succeed

Reasoning models (§2.27) need different SLOs. They may "think" for 30 seconds before the first visible token, so target total time-to-answer and show progress, rather than TTFT.

### §11.9 — Queueing Basics: Utilization and Little's Law
Two ideas from queueing theory explain most serving surprises.

**1. Waiting time explodes as utilization approaches 100%.** In the simplest queue, average waiting time is proportional to `ρ / (1 − ρ)`, where ρ (rho) is **utilization**, the fraction of time the server is busy:

| Utilization | Relative wait |
|-------------|---------------|
| 50% | 1× |
| 80% | 4× |
| 90% | 9× |
| 95% | 19× |

Going from 80% to 95% busy doesn't add a little delay. It nearly *quintuples* it. Network engineers know this from interface buffers: a link at 95% isn't "5% free," it's a link with deep queues and ugly latency. GPU fleets behave the same way. You'll watch it happen in the lab.

**2. Little's Law:** `L = λ × W`.
- **L** = average number of requests in the system (in flight)
- **λ** (lambda) = arrival rate (requests per second)
- **W** = average time each request spends in the system

Example: 20 requests/s, each taking 12.5 s (TTFT plus a 400-token answer) → `20 × 12.5 = 250` requests in flight at once. If one replica handles about 32 concurrent requests within your SLO, you need at least 8 replicas *before* any headroom. LLM requests are long, so concurrency gets big fast. This is the core of capacity planning (§11.31).

### §11.10 — Load Testing: Find the Knee
Use **`vllm bench serve`**, NVIDIA **`genai-perf`**, **`locust`**, or **`k6`**. Rules that matter:
- **Use realistic prompt and output lengths.** Short chat, long-context RAG, and long code generation stress very different parts of the engine (prefill vs. decode, §3.29). A test with all-identical prompts also hits the prefix cache (§8.20) and looks unrealistically fast.
- **Sweep the load:** 1, 2, 4, 8, ... 256 concurrent users, recording p50 and p99 TTFT and TPOT at each step.
- **Find the knee:** the point where adding load stops raising throughput and starts mostly raising latency. Run production comfortably *below* it.
- **Prefer an open-loop test** (requests arrive on a schedule, like real users) over a closed loop (each virtual user waits for its last reply). Closed loops politely slow down when the server struggles, which hides the queueing explosion from §11.9.

### §11.11 — Latency Budgeting
Every hop gets its share of the SLO, just like budgeting delay across a WAN path. For a 1-second target on a 25-token reply:

| Hop | Budget |
|-----|--------|
| Client ↔ gateway network | 100 ms |
| Gateway: auth, quota, routing | 20 ms |
| Queueing at the replica | 50 ms |
| TTFT (prefill) | 200 ms |
| 24 more tokens × 25 ms TPOT | 600 ms |
| **Total** | **970 ms** |

Only 30 ms of headroom, which means p99 will blow through it. Options: shorter outputs, a faster or smaller model, more replicas (less queueing), or a looser SLO. Writing the budget down makes those trade-offs visible.

### §11.12 — Tail Latency Is the Real Enemy
In LLM serving, p99 latency is often 10× p50 or worse. Common causes:
- **A long prompt arrives** and its big prefill stalls everyone else's decode on that replica. *Fix:* chunked prefill (§9.27), or separate prefill and decode (§11.18).
- **Very long outputs** hold a slot for minutes. *Fix:* max-token limits per tier.
- **KV cache pressure** (§9.6): the engine runs out of cache memory and has to pause or evict requests (**preemption**). *Fix:* fewer concurrent sequences, KV-cache quantization (§9.9), more replicas.
- **Bad routing:** one replica is overloaded while others idle (§11.14).
- **Cold starts, a sick node, a cold cache.** *Fix:* warm pools, health checks, and taking bad nodes out of rotation.

---

## Chapter 3 — Scaling Out

### §11.13 — Horizontal vs. Vertical Scaling
- **Vertical:** use a bigger GPU. There's a ceiling: the largest GPU you can buy (B200-class in 2025–26).
- **Horizontal:** run more replicas of the same model behind a load balancer. This is how inference mostly scales.
- **Within a replica:** tensor or pipeline parallelism (§9.29–§9.30) splits *one* replica across several GPUs when the model doesn't fit on one. This is for fitting the model, not for adding capacity.

### §11.14 — Load Balancing for LLMs
LLM requests are wildly uneven: one is 50 tokens, the next is a 100,000-token RAG prompt. The policy matters a lot.
- **Round-robin:** ignores replica state. Two monster requests can land on the same replica while others idle.
- **Least-loaded** (least outstanding requests, or better, least outstanding *tokens*, or least KV-cache use): send work to whoever has the most room. In the lab, it cuts p99 by about a quarter at light load and roughly in half at 80–95% load, compared with round-robin.
- **Prefix-aware (cache-aware) routing:** requests sharing a long prefix (the same system prompt, the same document) go to the replica that already has that prefix in its KV cache (§8.20, §9.11), skipping most of the prefill. It's like session affinity, but chosen for cache hits. The tension: pure affinity piles load onto "hot" replicas, so good routers balance cache hits against load. Projects like the Kubernetes Gateway API Inference Extension, `llm-d`, the vLLM production stack, SGLang's router, and NVIDIA Dynamo implement versions of this.

Network-engineer analogy: round-robin is per-packet load sharing with no awareness of flow size. Least-loaded is closer to a smart link-utilization-aware balancer. Prefix-aware routing is like keeping a flow on the path where the state already lives.

### §11.15 — Autoscaling on GPUs
CPU web servers start in seconds. A GPU replica of a 70B model must download about 140 GB of weights (at 1–2 GB/s, that's one to two minutes), load them into VRAM, and warm up (compile kernels, capture CUDA graphs, §9.32). That's **minutes, not seconds**, assuming a GPU is even available. So:
- **Scale on leading indicators:** queue depth, waiting requests, KV-cache use, TTFT trending up. *Not* CPU use, which tells you nothing about a GPU server.
- **Keep a warm pool** of ready replicas, and scale ahead of known daily peaks.
- **Make cold starts cheaper:** cache weights on local NVMe on each node, pre-pull container images, and use fast weight-loading formats.
- **Scale to one, not zero,** for latency-sensitive apps. Scale-to-zero suits internal tools and batch jobs that can tolerate a slow first request.
- **Scale down slowly.** Each replica in flight has long requests. Drain them, stopping new traffic and letting current requests finish, before removing it.

### §11.16 — Spot / Preemptible GPUs
Spare cloud capacity, often 50–80% cheaper than on-demand, which the cloud can reclaim with short notice: about **2 minutes on AWS, about 30 seconds on Google Cloud and Azure**.
- **Good for:** batch jobs, offline evals, embedding big document sets, asynchronous workloads that can retry.
- **Risky for:** synchronous chat. Mitigations are graceful draining, retrying cut-off requests on another replica, and keeping an on-demand base layer with spot capacity only for overflow.

### §11.17 — Multi-Region, Multi-Cloud
Users are far from a single region, and GPU supply in any one region or cloud is unreliable. Serious services deploy across regions (lower latency, failover) and often across clouds and GPU providers to hedge supply. The costs are real: copying weights everywhere, data-residency rules (where prompts may be processed), and more moving parts to operate.

### §11.18 — Disaggregated Serving in Production
Revisiting §8.29. Prefill is compute-bound and decode is memory-bandwidth-bound (§3.29), and running both on the same GPU makes them interfere (§11.12). **Disaggregated serving** runs a **prefill pool** and a separate **decode pool**, and ships each request's KV cache from one to the other over fast networking (NVLink, InfiniBand, or RDMA). Each pool can be sized and tuned for its own job.

Research systems like Splitwise and DistServe (2024) showed big gains. Moonshot AI's Mooncake and DeepSeek have reported running it at large scale, and open tooling (NVIDIA Dynamo, `llm-d`, and the major engines) has been adding support. It pays off at high volume with mixed prompt lengths. For a handful of replicas, the extra complexity usually isn't worth it.

---

## Chapter 4 — Multi-Tenancy, Cost, and Reliability

### §11.19 — Several Models on One GPU
Small models often don't need a whole GPU. Ways to share:
- **MIG (Multi-Instance GPU),** on A100/H100-class cards: carve one physical GPU into as many as 7 isolated slices, each with its own memory and compute. Strong isolation, but the slice sizes are fixed.
- **Time-slicing:** processes take turns on the GPU. Easy, but there's no memory isolation, and they slow each other down.
- **MPS (Multi-Process Service):** lets several processes run on the GPU *at the same time*, with better use than time-slicing and weaker isolation than MIG.
- **Adapter swapping** (vLLM multi-LoRA, LoRAX; §5.19): one base model with many customer fine-tunes as small adapters, chosen per request. Often the most cost-effective option for per-customer fine-tunes.

### §11.20 — Rate Limiting and Quotas
Limit and bill in **tokens**, not just requests. One request can be 50 tokens or 200,000. Track input and output tokens separately, since output tokens cost much more to produce (§3.13). Typical limits apply per API key, per user, and per team: requests per minute, tokens per minute, and monthly spend.

The standard algorithm is the **token bucket**, which works exactly like a policer on a router interface. A bucket refills at a steady rate (like CIR) up to a maximum size (like the burst size). Each request spends from it, and a request that would overdraw it is rejected. Over the limit, return **HTTP 429** with a `Retry-After` header so well-behaved clients back off. You'll build one in the lab.

Fairness matters too. Without per-tenant limits, one customer's batch job can fill every replica's queue and ruin everyone else's p99. That's the "noisy neighbor" problem.

### §11.21 — Prompt Caching for Cost
OpenAI, Anthropic, and Google all discount **cached input tokens**: typically 50–90% off when the start of your prompt matches a recent request. Some providers charge a small premium to *write* the cache, or require you to mark cacheable sections. The mechanism underneath is the prefix cache in their engines (§8.20, §9.11).

**Design rule: put the long, stable parts first** (system prompt, tool definitions, reference documents) and the parts that change (the user's question) last. A timestamp or user ID at the *top* of the system prompt breaks every cache hit. Self-hosted engines reward exactly the same habit.

### §11.22 — Cost Modeling: API vs. Self-Hosting
- **API cost** = requests/month × tokens/request × price per token (input and output priced separately).
- **Self-hosted cost** = GPU $/hour × replicas × hours, *whether or not they're busy*, plus engineers, on-call, monitoring, and spare capacity.

The deciding variable is **utilization**. In the lab's example, a 12-GPU fleet at $2.50 per GPU-hour costs about $21,900 a month. Its cost per million output tokens:

| Average utilization | Cost per million output tokens |
|---------------------|-------------------------------|
| 5% | ~$13.00 |
| 35% | ~$1.86 |
| 80% | ~$0.81 |

Idle GPUs are where self-hosting loses. Compare those numbers to the API price for a model of similar quality; small open models served by API providers are often cheap. Self-hosting tends to win with steady high volume, strict data-privacy needs, a custom fine-tune, or very low latency requirements. At small or spiky scale it rarely pays off, once people's time is counted.

### §11.23 — Observability: What to Measure
Per request:
- TTFT, TPOT, total latency, queue time
- Input, cached, and output tokens; cost
- Model and version, replica, tenant
- Outcome: success, timeout, out-of-memory, safety-filter block, client disconnect

Per replica and fleet:
- Running and waiting requests, batch size, KV-cache use and preemptions (vLLM exports these as Prometheus metrics)
- GPU use, memory, temperature, errors (DCGM)
- Prefix-cache hit rate

And *quality*: run LLM-as-judge evals (§5.24) on a sample of real traffic, because a model can be fast, healthy, and wrong.

### §11.24 — Tracing and Debugging
LLM-specific observability tools such as **Langfuse, Helicone, Arize Phoenix, and OpenLLMetry** (built on OpenTelemetry) capture the full trace of each request: prompt, retrieved documents, tool calls, model responses, timing, and cost at every step. These traces are essential for debugging agent loops and for catching **prompt regressions** (a prompt or model change that quietly made answers worse). Prompts can contain sensitive data, so set retention rules and redact.

### §11.25 — Guardrails: Content Filtering and Safety in Production
Usually two layers around the model:
1. **Input checks:** block clearly abusive requests and flag likely prompt injection (§5.26) before the main model sees them.
2. **Output checks:** scan responses for personal data, secrets, harmful content, or signs that an injection worked.

Tools include Llama Guard (a small open model trained to classify safety), NeMo Guardrails, AWS Bedrock Guardrails, and Azure AI Content Safety. Run them on small, cheap models, because they add latency to every request; budget for it (§11.11). Guardrails reduce risk but don't eliminate it, so also limit what tools and data the model can reach.

### §11.26 — Retries, Timeouts, Fallbacks, and Idempotency
LLM calls fail: overloaded replicas, 429s, timeouts, crashed engines.
- **Retry with exponential backoff and jitter** (wait 1 s, 2 s, 4 s, plus randomness), or a retry storm can turn a brief overload into an outage.
- **Timeouts at every layer,** plus a cap on output tokens. For agents, cap the number of steps and the total token budget.
- **Circuit breakers and fallbacks:** if the primary model or provider is failing, stop hammering it and route to a backup (another region, another provider, or a smaller model) with an honest quality trade-off.
- **Idempotency keys:** a unique ID per request, so a retried request that *did* succeed isn't run (and billed, or acted on) twice. This matters most when an agent's tool call changes something.
- **Streaming caveat:** if a stream dies halfway through, you can't silently retry; the user has already seen half an answer. Decide how your UI handles it.

### §11.27 — Versioning and Rollouts
Treat models like software releases:
- **Pin exact versions** (a dated API model name, or a specific weights hash), never a floating "latest."
- **Canary rollouts:** 1% → 10% → 50% → 100% of traffic, comparing latency, errors, and eval scores at each step, with instant rollback.
- **Regression evals** on a fixed test set (§5.24) before every swap.
- **Plan for deprecation.** API providers retire old models on a schedule, so a forced migration is guaranteed eventually.

*Every* model change is a behavior change for users, even a "minor" one: different refusals, formatting, and tone.

---

## Chapter 5 — Patterns and the Real World

### §11.28 — The Modern LLM App Architecture
```
Client
  → API gateway (TLS, auth, quotas / rate limits)      §11.20
  → Cache check (exact or semantic)                    §11.30
  → Router (pick a model / tier)                       §11.29
  → Load balancer (least-loaded or prefix-aware)       §11.14
  → Inference replicas (vLLM / SGLang / TRT-LLM)       Module 08
       ↘ Tools (search, DB, code execution)            §12.21
       ↘ RAG (vector DB, retriever, reranker)          §3.20, §12.16
       ↘ Guardrails (input / output checks)            §11.25
  → Observability everywhere (metrics, traces, evals, cost)  §11.23–§11.24
```

### §11.29 — Compound Systems and Model Routing
A modern "ChatGPT" request is rarely one model call. It might go: a classifier → a router → maybe a tool or retrieval step → maybe a reasoning model → a safety check → the response. Berkeley researchers (2024) called these **compound AI systems**, and it's how most production LLM apps actually work.

**Model routing** is a big cost lever. Send easy requests to a small, cheap, fast model and escalate only hard ones to a large or reasoning model. If 80% of traffic can be handled by a model one-tenth the price, total cost drops by about 70%. The risk: a bad router quietly sends hard questions to a weak model, so evaluate the router itself. Your serving platform must support these multi-step pipelines (Ray Serve, LangGraph, or your own orchestration) and trace them end-to-end (§11.24).

### §11.30 — Caching at Every Layer
From the outside in:
1. **Exact-match response cache:** the identical request got an answer recently, so return it. Safe, but hit rates are usually low for chat.
2. **Semantic cache:** if a new query's embedding is very close (§3.19) to a past query, return the stored answer. Higher hit rate, but risky: "How do I enable OSPF on this interface?" and "How do I *disable* OSPF on this interface?" are near-identical vectors. Use high thresholds, limit it to FAQ-style traffic, and never use it for personalized or permission-sensitive answers.
3. **Prompt/prefix cache inside the engine** (§8.20, §11.21): automatic, always worth designing for.
4. **Embedding cache for RAG:** don't re-embed unchanged documents or repeated queries.

Each layer that hits saves everything below it.

### §11.31 — Capacity Planning
Inputs: peak requests per second, prompt and output length distributions, the model and GPU, your SLOs, and per-replica capacity *measured* by load testing (§11.10). Then:
1. **Concurrency** via Little's Law (§11.9): in flight = peak RPS × time per request.
2. **Replicas** = in flight ÷ requests per replica within SLO, × headroom (often 1.5–2×, to cover spikes, a failed node, and deploys).
3. **Cost** = replicas × GPUs per replica × $/GPU-hour × hours (§11.22).

Worked example (Lab 3): 20 RPS at peak, 400-token answers at 30 ms TPOT, TTFT 0.5 s → 12.5 s per request → 250 in flight. With 32 concurrent requests per replica and 1.5× headroom → **12 replicas**, about $21,900/month at $2.50/GPU-hour. Redo it whenever traffic or the model changes. Usage of LLM features tends to grow fast.

### 🛠️ Build It — Hands-On Lab
**Lab 1: watch a queue explode (§11.9, §11.14).** Pure Python, nothing to install. It simulates random request arrivals into 4 replicas and compares round-robin with least-loaded routing.

```python
import random

# Simplified model: 4 replicas, each handles one request at a time (FIFO).
# Real engines batch many requests (§8.19), but the queueing lesson is the same.
REPLICAS = 4
MEAN_SERVICE = 2.0          # seconds per request, on average
N_REQUESTS = 20_000

def percentile(values, p):
    values = sorted(values)
    return values[int(p / 100 * (len(values) - 1))]

def simulate(utilization, policy, seed=7):
    rng = random.Random(seed)
    arrival_rate = utilization * REPLICAS / MEAN_SERVICE   # requests per second
    free_at = [0.0] * REPLICAS      # when each replica finishes its current work
    t, latencies = 0.0, []
    for i in range(N_REQUESTS):
        t += rng.expovariate(arrival_rate)                  # random (Poisson) arrivals
        service = rng.expovariate(1 / MEAN_SERVICE)         # some requests are long
        if policy == "round-robin":
            r = i % REPLICAS
        else:                                               # "least-loaded"
            r = min(range(REPLICAS), key=lambda k: free_at[k])
        start = max(t, free_at[r])                          # wait if the replica is busy
        free_at[r] = start + service
        latencies.append(free_at[r] - t)                    # queueing + service time
    return percentile(latencies, 50), percentile(latencies, 99)

print("util   policy         p50     p99")
for util in (0.5, 0.8, 0.9, 0.95):
    for policy in ("round-robin", "least-loaded"):
        p50, p99 = simulate(util, policy)
        print(f"{util:>4.0%}   {policy:<13} {p50:5.1f}s  {p99:6.1f}s")
```

What you should see: with round-robin, p99 goes from about 12 s at 50% utilization to about 83 s at 95%, while the *work per request* never changed. Least-loaded routing cuts p99 by about a quarter at 50% load and roughly in half from 80% up (about 44 s at 95%). Things to try: set `REPLICAS = 16` at the same utilization (bigger pools queue less), or make service times more uneven.

**Lab 2: a token-bucket rate limiter (§11.20).** Pure Python.

```python
# Token-bucket rate limiter: the same idea as a policer on a router interface.
# Here the "bucket" holds LLM tokens a customer may spend, refilled at a steady rate.

class TokenBucket:
    def __init__(self, rate_per_sec, burst):
        self.rate = rate_per_sec    # refill speed (like CIR)
        self.capacity = burst       # bucket size (like burst size, Bc)
        self.tokens = burst
        self.last = 0.0

    def allow(self, now, cost):
        # Refill for the time that has passed, but never above capacity.
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if cost <= self.tokens:
            self.tokens -= cost
            return True
        return False                # caller should return HTTP 429 + Retry-After

# One API key: 1,000 LLM tokens/sec sustained, bursts up to 5,000.
bucket = TokenBucket(rate_per_sec=1_000, burst=5_000)

# (time in seconds, tokens this request will use)
requests = [(0.0, 2_000), (0.1, 2_000), (0.2, 2_000), (1.0, 1_500), (4.0, 4_000)]
for now, cost in requests:
    ok = bucket.allow(now, cost)
    print(f"t={now:>3}s  request for {cost:>5,} tokens -> "
          f"{'allowed' if ok else 'REJECTED (429)'}   bucket now {bucket.tokens:,.0f}")
```

What you should see: the first two requests ride the burst, the third is rejected, and the 4,000-token request at t=4 is also rejected (the bucket has only refilled to 3,500). Things to try: how would you tell the client how long to wait? Compute `Retry-After` as `(cost − tokens) / rate`.

**Lab 3: capacity and cost (§11.9, §11.22, §11.31).** Pure Python.

```python
# Capacity and cost planning with Little's Law (§11.9).
import math

# --- Workload (your assumptions) ---
peak_rps = 20               # requests per second at the busiest hour
avg_output_tokens = 400
tpot = 0.030                # seconds per output token at your SLO (§11.6)
ttft = 0.5                  # seconds to first token
seqs_per_replica = 32       # concurrent requests one replica handles within SLO (from load tests)
headroom = 1.5              # spare capacity for spikes and failures

# --- Little's Law: requests in flight = arrival rate x time each spends in the system ---
time_in_system = ttft + tpot * avg_output_tokens
in_flight = peak_rps * time_in_system
replicas = math.ceil(in_flight * headroom / seqs_per_replica)
print(f"each request takes ~{time_in_system:.1f}s -> ~{in_flight:.0f} requests in flight at peak")
print(f"replicas needed with {headroom}x headroom: {replicas}")

# --- Self-host cost vs an API, per million output tokens ---
gpu_hour = 2.50             # USD per GPU-hour (1 GPU per replica here)
monthly_hours = 730
avg_utilization = 0.35      # traffic isn't at peak all day
tokens_per_sec_per_replica = seqs_per_replica / tpot   # ~ aggregate decode speed
monthly_tokens = replicas * tokens_per_sec_per_replica * 3600 * monthly_hours * avg_utilization
monthly_cost = replicas * gpu_hour * monthly_hours
print(f"self-host: ${monthly_cost:,.0f}/month for ~{monthly_tokens / 1e9:.1f}B output tokens "
      f"= ${monthly_cost / (monthly_tokens / 1e6):.2f} per million tokens")
for util in (0.05, 0.35, 0.80):
    t = replicas * tokens_per_sec_per_replica * 3600 * monthly_hours * util
    print(f"  at {util:.0%} average utilization: ${monthly_cost / (t / 1e6):.2f} per million tokens")
```

You should get 250 requests in flight, 12 replicas, and about $1.86 per million output tokens at 35% utilization ($13.02 at 5%). These inputs are *assumptions*. Replace `seqs_per_replica` and `tpot` with numbers from your own load test, and compare the result to a real API price list.

**Lab 4 (optional, real tools): a small serving stack.**
1. Start Ollama or vLLM, and confirm the OpenAI-compatible endpoint works: `curl http://localhost:11434/v1/models` (Ollama) or `curl http://localhost:8000/v1/models` (vLLM).
2. Put a small **FastAPI** gateway in front of it: API-key auth, the token bucket from Lab 2 per key (in memory, or in Redis to share across gateway copies), and a log line per request with TTFT, token counts, and latency.
3. `docker compose up` the gateway, Redis, and the engine together.
4. Add tracing with Langfuse or Phoenix (§11.24).
5. Load-test with `vllm bench serve`, `k6`, or `locust` at 1, 10, and 100 concurrent users with realistic prompt lengths. Plot p50/p99 TTFT and TPOT, find the knee (§11.10), and write SLOs you could actually meet.

**Deliverable:** a compose stack a teammate can start with one command, plus a one-page performance report (the knee, p99s, and replicas needed for 10× your test load).

### §11.32 — Checkpoint — Explain It Back
1. What's the difference between an inference engine and an inference service?
2. Why should SLOs use p99 instead of averages?
3. What is an error budget?
4. Why does latency explode as utilization nears 100%?
5. Use Little's Law: 10 requests/s, 20 s each. How many are in flight?
6. Why is GPU autoscaling harder than CPU autoscaling, and what should you scale on?
7. What does prefix-aware routing gain, and what's its risk?
8. How does a token bucket rate limiter work, and what should the server return when it rejects?
9. Why does utilization decide whether self-hosting beats an API?
10. What's a compound AI system, and how does model routing cut cost?

<details>
<summary>Answer key (try first!)</summary>

1. The engine is one process that runs the model. The service wraps engines with load balancing, autoscaling, auth, quotas, monitoring, failover, versioning, and guardrails.
2. Averages hide the slow requests that users actually notice. At scale, many users hit the tail every session.
3. The amount of failure the SLO allows (99.9% ≈ 43 minutes per 30 days). Spend it on change; when it's gone, prioritize reliability.
4. Waiting time grows like ρ/(1−ρ): any burst finds the servers busy and the queue has no time to drain. 80% → 95% busy is roughly a 5× longer wait.
5. L = λ × W = 10 × 20 = 200 requests in flight.
6. A new replica must download and load huge weights and warm up, which takes minutes, and GPUs may not be available. Scale on queue depth, waiting requests, KV-cache use, or TTFT, not CPU, and keep warm capacity.
7. Requests sharing a prefix hit an existing KV cache, skipping most of the prefill (faster TTFT, less compute). The risk is overloading "hot" replicas, so it must be balanced against load.
8. A bucket refills at a steady rate up to a burst size; each request spends tokens and is rejected if it would overdraw. Return HTTP 429 with a Retry-After header.
9. You pay for GPUs every hour whether they're busy or not, so cost per token is set by how busy they are. At low utilization, the per-token cost can be several times an API's price.
10. An app built from several steps and models (classifier, router, tools, RAG, LLM, safety check). Routing sends easy requests to a cheap small model and only hard ones to the expensive model.
</details>

You can now design real systems. Last module: the ecosystem and what's coming.
