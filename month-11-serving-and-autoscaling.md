# Month 11 — Serving & Autoscaling (Days 301–330)

**Goal of this month:** Take a working inference engine and turn it into a reliable, scalable, multi-tenant production service. By Day 330 you'll understand how to architect a system that serves millions of LLM requests without melting down or blowing the budget.

---

## Week 1 — From Engine to Service

### Day 301 — Engine vs Service
An inference engine (vLLM, TGI) is a single process. A **service** is what wraps it: load balancing, autoscaling, observability, auth, rate limiting, retries, multi-region failover, model versioning, A/B testing. The engine is maybe 20% of the production system.

### Day 302 — The OpenAI-Compatible API Layer
Almost every serving stack exposes `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`. This standardization means clients (LangChain, LlamaIndex, your app) can swap providers trivially. Build *your* internal API to this contract too.

### Day 303 — NVIDIA Triton Inference Server (the server, not the language)
Multi-framework, multi-model server. Was the dominant solution before vLLM/TGI became LLM-specific. Still used heavily for **non-LLM** AI (vision, embeddings, recommendation) and as a backend for **TensorRT-LLM**. Provides batching, model ensembling, dynamic loading.

### Day 304 — BentoML, Ray Serve, KServe, Modal
General-purpose model serving platforms:
- **Ray Serve** — Python-native, scales horizontally on a Ray cluster. Good for compound AI systems (RAG + LLM + reranker as one service).
- **BentoML** — packaging + serving framework with strong DX.
- **KServe** — Kubernetes-native model serving, the cloud-native standard.
- **Modal / Replicate / Banana** — serverless GPU platforms; you write a function, they handle scaling.

### Day 305 — Choosing Your Serving Architecture
- **Single team, single model** → just `vllm serve` behind a load balancer.
- **Multiple models, multiple teams** → KServe or Ray Serve cluster.
- **Bursty/sporadic traffic** → serverless (Modal, Replicate, Bedrock).
- **Latency-critical at huge scale** → custom stack on TensorRT-LLM with NVIDIA NIM or in-house.

---

## Week 2 — Latency, Throughput, and SLOs

### Day 306 — The Two Numbers Everyone Cares About
- **Latency** — how long does *one user* wait? (TTFT + TPOT × tokens.)
- **Throughput** — how many tokens/sec can the *system* generate across all users?
These trade off. Bigger batches → higher throughput, *worse* latency. Production tuning is choosing the right point on this curve for your workload.

### Day 307 — Define Your SLOs Up Front
Example chat product:
- p50 TTFT < 500ms, p99 < 2s
- p50 TPOT < 30ms, p99 < 80ms
- 99.9% availability
Reasoning workloads (long thinking phases) need different SLOs — total response time within X seconds, regardless of token rate.

### Day 308 — Load Testing
Use tools like **`vllm bench`, `genai-perf`, `locust`, `k6`**. Test with realistic prompt distributions (chat-length vs RAG-length vs code-length skew throughput dramatically). Measure across concurrency levels — find your engine's "knee."

### Day 309 — Latency Budgeting
For a 1-second SLO: 100ms network + 200ms TTFT + 30ms × 25 tokens = 950ms. No headroom. Each subsystem (load balancer, gateway, engine, network) needs its own latency budget.

### Day 310 — Tail Latency Is the Real Enemy
P99 latency is often 10× P50 in LLM serving. Causes: long prompts (prefill spike), long outputs (one user holding a slot), KV cache pressure (eviction), bad node, cold cache, model loading, GC pauses. Mitigations: chunked prefill, smarter scheduling, request limits, prewarming.

---

## Week 3 — Scaling Out

### Day 311 — Horizontal vs Vertical Scaling
- **Vertical**: bigger GPU. There's a ceiling (B200 is the max, for now).
- **Horizontal**: more replicas of the same model server behind a load balancer. Mostly how you scale inference.
- **Within-replica** scaling: tensor/pipeline parallelism *inside* one replica for huge models that don't fit on one GPU.

### Day 312 — Load Balancing Considerations
Naïve round-robin works but ignores per-replica state. Better: **least-pending-tokens** or **prefix-aware** routing — send requests sharing a prefix to the same replica to maximize prefix-cache hits. Some products (GKE Inference Gateway, Anyscale Endpoints, NVIDIA NIM) do this automatically.

### Day 313 — Autoscaling on GPUs
Cold-start a 70B model = pull 140 GB of weights + warm caches → minutes, not seconds. So:
- Maintain a **warm pool** of replicas
- Scale on a leading indicator (queue depth, KV cache util) not lagging (CPU)
- **Pre-pull model images** to all nodes
- Consider **scale-to-1** instead of scale-to-zero for latency-sensitive apps

### Day 314 — Spot/Preemptible GPUs
50–80% cheaper than on-demand. Risk: 30-second termination notice. Mitigations: graceful drain, checkpointed KV cache, request hand-off to other replicas. Worth it for batch / async workloads, risky for latency-sensitive synchronous traffic.

### Day 315 — Multi-Region, Multi-Cloud
Latency to users matters → deploy in multiple regions. GPU availability is unreliable → multi-cloud (AWS + GCP + Azure + Lambda + CoreWeave) hedges supply risk. The added ops complexity is real but increasingly necessary for serious scale.

### Day 316 — Disaggregated Serving in Production (Day 234 revisited)
At very high scale, run a **prefill cluster** (compute-bound, big batch) and a **decode cluster** (memory-bound, smaller batch), shipping KV caches between them over fast networking. Pioneered by DeepSeek, Mooncake, Splitwise. Big wins for mixed prompt-length workloads.

---

## Week 4 — Multi-Tenancy, Cost, Observability

### Day 317 — Multi-Model on One GPU
Several smaller models can share a GPU via:
- **Time-slicing** (MIG on A100/H100, MPS) — partition the GPU into slices.
- **Adapter swapping** (LoRAX, vLLM multi-LoRA) — one base model, many fine-tunes loaded as adapters and routed per request. Hugely cost-effective for SaaS with per-customer fine-tunes.

### Day 318 — Token Accounting and Quotas
Bill / rate-limit by tokens. Track input vs output tokens (output is much more expensive — each token requires a full forward pass). Per-user, per-team, per-API-key quotas. Backpressure when limits hit.

### Day 319 — Prompt Caching for Cost
OpenAI, Anthropic, Google all offer prompt caching at the API layer — 50–90% discount on cached tokens. Underlying mechanism: prefix sharing in their inference engines (Day 246). Architect prompts so the long, stable parts come *first* (system prompt, tool defs, RAG context).

### Day 320 — Cost Modeling
- Per-token pricing × expected QPS × tokens/request = monthly cost
- Compare to self-hosted: GPU $/hr × replica count × hours
- Break-even is usually around millions of tokens/day for popular open-weight models
- Don't forget: ops cost, on-call, redundancy. Self-hosting at small scale rarely pencils out.

### Day 321 — Observability: What to Log
- TTFT, TPOT, total latency per request
- Tokens in / tokens out
- KV cache utilization, GPU util, GPU memory
- Queue depth, batch size distribution
- Error rates by type (OOM, timeout, content filter)
- Per-user/per-prompt cost and latency
- Output quality (LLM-as-judge eval pipelines on a sample)

### Day 322 — Tracing and Debugging
**Langfuse, Helicone, Arize, Phoenix, OpenLLMetry** — observability tools tailored to LLM apps. Capture full prompt/response/tool-call traces. Critical for debugging agent failures and prompt regressions.

### Day 323 — Content Filtering and Safety in Production
Two layers usually:
1. **Input filter** — block obviously harmful prompts before they hit the model.
2. **Output filter** — scan generated tokens for PII, harmful content, prompt injection results.
Tools: Llama Guard, Azure AI Content Safety, AWS Bedrock Guardrails, NeMo Guardrails. Run them on a smaller/cheaper model.

### Day 324 — Retries, Timeouts, Idempotency
LLM calls fail (overloaded GPU, NaN, timeout). Retry with exponential backoff. Timeout *generation* (cap max output tokens). For agent loops, cap iteration count or token budget. Idempotency keys for non-streaming endpoints.

### Day 325 — Versioning and Rollouts
Treat models like software — semantic versioning, canary deploys (1%/10%/50%/100%), automated regression suites, instant rollback. *Every* model swap is a behavior change for users; treat accordingly.

---

## Week 5 — Patterns and the Real World

### Day 326 — The Modern LLM App Architecture
```
Client → API Gateway → Auth/Quota → Cache → Router →
   ↓
Inference replicas (vLLM/TRT-LLM behind LB)
   ↘ Tools (search, DB, code exec)
   ↘ RAG (vector DB, retriever, reranker)
   ↘ Guardrails (input/output filters)
   ↘ Observability (traces, evals, costs)
```

### Day 327 — Compound Systems Are the Norm
A "ChatGPT" today is rarely one model call. It's: classifier → router → maybe tools → maybe RAG → maybe a reasoning model → final formatting model → safety filter. **Compound AI systems** (Berkeley term) are how production LLM apps actually work. Your serving infra needs to support them — Ray Serve, LangGraph, custom orchestration.

### Day 328 — Caching at Every Layer
1. CDN for static prompt templates / responses to FAQ questions
2. Semantic cache: if a new query is ≈ a past query, return the cached answer
3. KV/prefix cache inside the engine
4. Embedding cache for RAG
Each layer reduces cost and latency further down.

### Day 329 — Capacity Planning
Inputs: expected QPS, prompt length distribution, output length distribution, model choice. Outputs: # of GPUs, # of replicas, headroom for spikes (typically 2×), cost per month. Re-do this every quarter — usage grows fast.

### 🛠️ Build It — Month 11 Hands-On
Build a small but real serving stack:
1. **FastAPI gateway** in front of vLLM/Ollama. Add API-key auth, per-key token quotas (Redis counters), request logging.
2. **Dockerize** it. `docker-compose up` should bring up FastAPI + Redis + Ollama.
3. Wire **Langfuse** or **Phoenix** for tracing — capture every prompt/response/latency/cost.
4. Add an **input guardrail** (regex or Llama Guard) and an **output guardrail** (PII scanner).
5. Load-test with `k6` or `locust` at 1, 10, 100 concurrent users. Find your knee. Set SLOs accordingly.
6. Bonus: deploy to a single Kubernetes node (`kind` or k3s) with KServe or a simple Deployment + Service.

**Deliverable:** a running compose stack + a perf report + a `README` that any teammate can `docker-compose up`.

### Day 330 — Recap & "Explain it Back"
**Quiz yourself:**
1. Difference between an inference engine and an inference service?
2. Why is GPU autoscaling harder than CPU autoscaling?
3. What's a compound AI system, and why does it matter for serving?
4. Where in your stack should prompt caching live?
5. Why do output tokens cost more than input tokens?

You can now design real systems. Last month: the ecosystem and what's coming.
