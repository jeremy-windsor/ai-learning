# Parsed Links for Input

Generated from repository input files for reference.

## Source Repository

- https://github.com/jeremy-windsor/ai-learning

## Links Found

| Source file | URL | Notes |
|---|---|---|
| `README.md` | https://ollama.com | Tool/platform reference |
| `README.md` | https://huggingface.co | Tool/platform reference |
| `links-for-input.md` | https://x.com/ParasVerma7454/status/2048366074169835559?s=20 | X post transcribed below |

---

## X Post Transcript

**Author:** @ParasVerma7454 (Luffytaro)  
**Post:** https://x.com/ParasVerma7454/status/2048366074169835559  
**Timestamp:** Sun Apr 26 11:38:23 +0000 2026  
**Engagement at capture:** 1088 likes · 156 reposts · 15 replies

### The AI Engineer Learning Path

A practical roadmap on what to learn, in what order, based on real-world demand across ~2000 job descriptions.

If you’re trying to break into AI engineering (not just ML theory), this is the path that actually maps to how systems are built in production.

---

### The Core: 20% Skills That Drive 80% of the Work

This is the part most people get wrong.

AI engineering is not about training models from scratch. It’s about building systems around LLMs.

#### 1. LLM Fundamentals

Start here. Everything builds on this.

- How LLMs work (at a high level)
- What they’re good at vs where they fail
- Working with APIs like OpenAI and Anthropic
- Structured outputs (JSON, schemas, tool responses)
- Prompt engineering across different tasks

**Goal:** Move from “chatting with models” to controlling them predictably.

#### 2. RAG (Retrieval-Augmented Generation)

This is the backbone of most real-world AI systems.

- Injecting custom data into LLMs
- Vector search + semantic retrieval
- Tools like Elasticsearch, Qdrant
- Chunking strategies (this matters more than people think)
- Handling real data: PDFs, web pages, transcripts

**Practice projects:**

- FAQ assistant
- Document Q&A system
- Internal knowledge search

#### 3. AI Agents

This is where things get interesting — and messy.

- Tool calling (LLMs that can act, not just respond)
- Agent loop: think → act → observe → repeat
- Frameworks: LangChain, PydanticAI, OpenAI Agents SDK, Google ADK
- Model Context Protocol (MCP)
- Multi-agent systems (routing, coordination, pipelines)

**Practice projects:**

- Web research agent
- Data extraction pipeline
- Multi-agent workflow

#### 4. Testing AI Systems

Underrated, but critical.

- Testing tool usage and outputs
- Evaluating consistency
- Using LLMs as judges (yes, meta — but useful)

**Goal:** Make AI systems reliable, not just impressive.

#### 5. Monitoring & Observability

If you can’t see what your system is doing, you can’t fix it.

- Tracing agent workflows
- Logging interactions
- Cost tracking
- Feedback loops
- Dashboards (Grafana, OpenTelemetry)

**Real-world impact:** This is what separates demos from production systems.

#### 6. Evaluation

Most engineers skip this — and it shows.

- Offline eval datasets
- Measuring retrieval quality
- Synthetic data generation
- Prompt optimization based on results

**Goal:** Move from “it feels good” → it’s measurable.

#### 7. Production Systems

This is where AI engineers become valuable.

- Turning notebooks into real services
- Deployment (Streamlit for quick prototypes)
- Cloud platforms: AWS / GCP / Azure
- Guardrails and safety layers
- Parallel processing for scale

---

### The Supporting Skills (What Job Descriptions Actually Ask For)

These aren’t optional — they show up everywhere.

#### Python & Engineering Basics

- Python (used in ~80%+ roles)
- Testing, CI/CD, code quality
- Git workflows

#### Web Development (For Real Products)

- FastAPI (backend standard for AI apps)
- React / Next.js (frontend layer)
- APIs: REST / GraphQL

#### Cloud & Infrastructure

- At least one: AWS, GCP, or Azure
- Docker (non-negotiable)
- Kubernetes (for scale)
- Terraform (infra as code)

#### Databases

- PostgreSQL (default choice)
- Vector DBs: Pinecone, Weaviate, Qdrant, pgvector
- Redis (caching, sessions)

#### ML Fundamentals (Just Enough)

You don’t need to be a researcher — but you need context.

- PyTorch basics
- Embeddings (very important)
- Fine-tuning (when APIs aren’t enough)
- Model evaluation basics

#### Data Engineering

- ETL pipelines
- Airflow, Spark, Kafka
- Tools like Databricks, Snowflake

#### Languages Beyond Python

- TypeScript (huge for full-stack AI)
- SQL (mandatory for real data work)
- Java / Go (for backend-heavy roles)

---

### The Typical AI Engineering Stack

A modern AI system usually looks like this:

- Frontend: React / Next.js
- Backend: FastAPI
- AI orchestration: LangChain, LangGraph, PydanticAI
- LLMs: OpenAI, Anthropic, Groq, local models
- Vector DB: Pinecone / Weaviate / Qdrant
- Infra: Docker + Kubernetes + Cloud
- Monitoring: OpenTelemetry, Grafana
- Evaluation: LLM judges, tools like Evidently

---

### Skill Priority (If You’re Short on Time)

#### Must-Have

- Python
- Prompt engineering
- RAG systems
- One cloud platform
- Docker

#### High-Value

- LangChain or PydanticAI
- FastAPI
- TypeScript
- CI/CD
- Kubernetes
- PyTorch basics

#### Differentiators (What Gets You Hired Faster)

- Agent frameworks (LangGraph, CrewAI)
- Fine-tuning models
- Evaluation systems
- Vector databases
- Multi-agent architectures

---

### Final Take

AI engineering is not about chasing hype tools.

It’s about building reliable systems where LLMs are just one component.

If you focus on:

- RAG
- Agents
- Evaluation
- Production systems

You’ll already be ahead of most candidates.

---

### Credit

This roadmap is heavily inspired by the original work from Alexey Grigorev:

- https://github.com/alexeygrigorev/ai-engineering-field-guide
