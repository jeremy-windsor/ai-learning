# Module 14 — Agentic Systems & Your AI Engineering Path

**Mastery: 93% → 100%** · Prerequisites: Module 13

**Goal of this module:** Go from "I know what an agent is" (§12.22) to "I can design, build, and operate one that doesn't fall over." By the end you'll know when *not* to build an agent, how to structure one as a controlled state machine, how agent memory and tools should be designed, when multiple agents are worth the cost, and how to choose a framework. The last chapter turns the whole course into a practical path: a ladder of networking projects, the skills employers ask for, and how to practice for interviews.

**How to read this module:** As before, **bold** marks a definition. This module assumes Module 12's Chapters 4–5 (RAG, tools, agents, MCP, agent security) and Module 13 (calling LLMs from code, structured outputs, testing, evaluation, shipping). It doesn't re-teach those. It builds the agent-specific layer on top.

---

## Chapter 1 — From Chatbot to Agent

### §14.1 — Where We Are
Look back at how far you've come. Modules 01–05 explained what a model *is*. Modules 06–11 explained how it *runs*. Module 12 surveyed the ecosystem and introduced RAG, tools, and agents. Module 13 showed how to call models from code and ship a reliable LLM feature.

This module finishes the path most AI engineering job descriptions describe: **LLM fundamentals → RAG → agents → testing, evaluation, and monitoring → production**. Agents come last for a reason. They combine everything before them, and they magnify every weakness. A shaky retrieval step or an untested prompt becomes a shaky *action*.

### §14.2 — Workflows vs. Agents
Two very different ways to build with LLMs:
- A **workflow** is a fixed path that *your code* decides. The LLM fills in some steps: "extract fields from this syslog line, then classify severity, then draft a ticket." The order never changes.
- An **agent** lets *the model* decide the path: which tool to call next, when it has enough information, and when it's done (§12.22).

Most real systems sit somewhere in between:

| Pattern | Who decides the next step? | Example |
|---------|---------------------------|---------|
| Single LLM call | Nobody, there's one step | "Summarize this outage report" |
| Prompt chain | Your code, fixed order | Extract → classify → draft ticket |
| Router | The LLM picks one of a few fixed branches | "Is this a BGP, OSPF, or hardware question?" → specialist prompt |
| Agent | The LLM picks steps in a loop until done | "Find out why ge-0/0/1 is slow" |

**Network-engineer analogy:** a workflow is a static route: predictable, easy to audit. An agent is a dynamic routing protocol: it adapts to what it discovers, but you need guardrails (filters, max-prefix limits, dampening) or it will happily do something surprising.

### §14.3 — When *Not* to Use an Agent
Agents are the most powerful option and the least reliable one. Remember the compounding math from §12.22: 95% success per step over 20 steps is about 36% overall. A 4-step workflow with the same per-step quality succeeds `0.95⁴ ≈ 81%` of the time, and you can unit-test each step.

Use the simplest pattern that works:
- **The steps are known in advance?** Use a workflow.
- **There are only a few possible paths?** Use a router.
- **The path truly depends on what you discover along the way** (troubleshooting, research, coding)? Now an agent earns its cost.

| | Workflow | Agent |
|---|---|---|
| **Pros** | Predictable, cheap, testable step by step, easy to explain to auditors | Handles open-ended problems; adapts to surprises |
| **Cons** | Breaks on cases you didn't anticipate | Costlier, slower, harder to test, can go off the rails |

A good habit: build the workflow first. Promote a part of it to an agent only when you have examples it can't handle.

### §14.4 — The Agent Loop as a State Machine
The loop in §12.22 ("think → call a tool → observe → repeat") is easier to control if you draw it as a **state machine** (also called a **graph**): **nodes** are steps, and **edges** are the allowed transitions between them.

```
          ┌──────────────── repair edge (with feedback) ───────────────┐
          ▼                                                             │
 start → ROUTER ──"call a tool"──► EXECUTOR ──► back to ROUTER          │
          │                                                             │
          └──"I have an answer"──► VERIFIER ──fail──────────────────────┘
                                      │
                                      └──pass──► DONE
          (any node) ── budget exceeded ──► STOPPED
```

You already know this pattern. The OSPF neighbor state machine (Down → Init → 2-Way → ExStart → Exchange → Loading → Full) has explicit states, explicit events that move between them, and timeouts that kick you back. That's why it's debuggable: when a neighbor is stuck in ExStart, you know exactly where to look. An agent drawn as a graph gets the same benefit. A trace shows which node it was in, and every transition is one you allowed.

This idea is popular enough that a common pattern name, **ReAct** ("reason + act," 2022), describes the loop, and frameworks like LangGraph make the graph explicit (§14.28).

### §14.5 — Deterministic Shell, Probabilistic Core
The design principle for every reliable agent: **the LLM proposes, your code disposes.**
- **The probabilistic core:** the model chooses the next step and writes text. It will sometimes be wrong (§4.6, §5.8).
- **The deterministic shell:** your code validates every tool call against a schema, checks permissions, enforces budgets, runs verifiers, logs everything, and decides when to stop.

Anything that *must* be true (never delete a config without approval, never exceed $2 per task, always cite evidence) belongs in the shell, as code. Never rely on a sentence in the prompt for it. Prompts are suggestions to a statistical model; code is policy.

---

## Chapter 2 — Agent Architecture

### §14.6 — State: The Agent's Working Record
**State** is everything the agent knows about the current task, stored as structured data your code owns:

```
{"target": "ge-0/0/1", "step": 3, "max_steps": 6, "status": "running",
 "observations": [{"tool": "get_utilization", "result": {"util_pct": 93}}],
 "feedback": ["missing checks: ['get_errors']"], "answer": null}
```

Good state design:
- **Typed and minimal.** Only what later steps need. A huge state is a huge prompt.
- **Separate from the prompt.** You *render* a prompt from state each step. That way you can replay, inspect, and test the state without the model.
- **Serializable** (JSON), so you can checkpoint and resume it (§14.12) and attach it to traces (§11.24).

### §14.7 — Routers
A **router** decides which branch to take. Two kinds:
- **Rule-based:** keywords, regex, or metadata ("if the ticket mentions `BGP`, use the BGP specialist"). Free, instant, and predictable, but brittle.
- **LLM-based:** ask a model to pick from a *small, fixed list* of labels, with a structured output so it can't invent a new one (Module 13). It handles fuzzy cases but costs a call.

A common combination is rules first, then an LLM for whatever the rules can't classify, plus an "unsure" label that goes to a human or a safe default. This is model routing (§11.29) applied to *tasks* instead of models.

### §14.8 — Planner, Executor, Verifier
Splitting responsibilities makes agents more reliable:
- **Planner:** turns the goal into steps ("check utilization, then error counters, then recent config changes").
- **Executor:** carries out one step at a time by calling tools.
- **Verifier** (or **critic**): checks the result *before* it's accepted.

The verifier is the most underrated piece. Put **deterministic checks first**: is the JSON valid, does the answer cite the evidence that was gathered, are all required checks done, does the proposed config pass a syntax check? Use an LLM-as-judge (§5.24) only as a second layer for things code can't check. If verification fails, don't throw the work away. Follow a **repair edge** back to the router with the failure reason as feedback, as in the diagram in §14.4.

### §14.9 — Stop Conditions and Budgets
An agent must always be able to stop. Every loop needs:
- A **success condition:** the verifier passed.
- **Budgets:** maximum steps, wall-clock time, tokens, and money.
- **Loop detection:** the same tool called with the same arguments repeatedly is a sign the agent is stuck.
- **Escalation:** when a budget runs out, return partial results and ask a human. Don't just fail silently.

Why budgets matter, with numbers. Each step re-sends the growing conversation, so input tokens grow every step. Suppose step 1 sends 3,000 tokens and each step adds 1,500 more (tool output plus reasoning):

```
total input over 20 steps = 20 × 3,000 + 1,500 × (0 + 1 + ... + 19)
                          = 60,000 + 285,000 = 345,000 tokens
```

At illustrative prices of $3 per million input tokens and $15 per million output tokens, with 500 output tokens per step, that's about `$1.04 + $0.15 ≈ $1.19` per task. Run it 10,000 times a month and it's ~$12,000. A runaway loop that hits 100 steps costs far more than 5× as much, because the context keeps growing. Prompt caching (§11.21) makes the repeated prefix much cheaper, and summarization (§14.15) keeps it from growing without limit.

### §14.10 — Failure Handling: Retries, Fallbacks, Repairs
Agents fail in different ways, and each needs a different response (the general patterns are in §11.26):

| Failure | Example | Response |
|---------|---------|----------|
| Transient | Tool API timeout, HTTP 503 | Retry with exponential backoff (a capped number of times) |
| Invalid tool call | Missing argument, wrong type | Return the validation error to the model so it can fix the call |
| Wrong answer | Verifier rejects it | Repair edge with feedback (§14.8) |
| Persistent | Device unreachable, permission denied | Fallback: a simpler tool, a cached answer, or ask the user |
| Stuck | Same call repeated, budget nearly gone | Stop and escalate (§14.9) |

Log every attempt. A trace showing "retried 3 times, then fell back" is the difference between a 5-minute fix and a mystery.

### §14.11 — Human-in-the-Loop Approvals
For anything **irreversible or outward-facing** (pushing a config, deleting data, sending email, spending money), the agent should *propose* and a human should *approve*. In graph terms, the agent pauses at an **approval node**, saves its state, shows the exact proposed action and why, and waits.

Network engineers already work this way: change tickets, peer review, and a maintenance window before a config push. An agent should produce the same artifacts: a diff, the evidence, a rollback plan, and a clear "approve / reject / edit" decision. Approval also handles prompt injection (§5.26, §12.24). Even if poisoned text convinces the model to do something bad, it can only *ask*.

### §14.12 — Long-Running Tasks: Checkpoint, Resume, Idempotency
Some agent tasks run for minutes or hours: a fleet-wide audit, a large research report, waiting on a human approval overnight. Processes crash, deploys restart containers, and approvals take time. So:
- **Checkpoint** the state (§14.6) after every step, to a file, a database, or a queue.
- **Resume** from the last checkpoint instead of starting over. The lab does exactly this: it "crashes" after step 2 and a new process picks up at step 3.
- Make tools **idempotent**: running the same call twice has the same effect as running it once. "Set interface description to X" is idempotent; "append a line to the config" is not. Without idempotency, a retry after a crash can apply a change twice. Give write operations an **idempotency key** (a unique ID per intended action) so the system can recognize and skip a repeat.

This is the same discipline as configuration management (Ansible's "desired state") and transactional commits (Junos `commit confirmed`).

---

## Chapter 3 — Memory

### §14.13 — The Kinds of Agent Memory
Models are frozen. They don't learn from your conversations (§12.31). Everything an agent "remembers" is text your code stores and puts back into the prompt. It helps to separate:
- **Short-term memory:** the current task's conversation and observations, living inside the context window (§3.13).
- **Long-term memory:** information kept *across* tasks, stored outside the model and retrieved when relevant. Three useful flavors:
  - **Semantic:** facts. "Site B's core switches are Arista; the NOC's on-call rotation is in PagerDuty."
  - **Episodic:** past experiences. "Last time ge-0/0/2 showed CRC errors, it was a dirty optic."
  - **Procedural:** how-tos. "To check an MPLS LSP on this platform, run these three commands."

### §14.14 — Short-Term Memory: Budgeting the Context Window
The context window is a hard limit (§3.13), and quality drops long before you hit it (§3.28). Treat it like a link you're traffic-engineering: allocate it deliberately.

An example budget for a 128K-token model:

| Use | Tokens |
|-----|--------|
| System prompt and rules | 2,000 |
| Tool definitions (20 tools × ~200 tokens each) | 4,000 |
| Retrieved documents (RAG, §12.15) | 16,000 |
| Long-term memory snippets | 2,000 |
| Summary of older turns | 2,000 |
| Recent turns, word for word | 10,000 |
| Reserved for the model's output | 8,000 |
| **Total** | **44,000** |

Note: tool definitions are sent on *every* call. An agent with 100 tools burns ~20,000 tokens per step just describing them, and gets worse at choosing among them. That's a strong reason to keep each agent's tool list short (§14.18) or to load tools on demand.

### §14.15 — Summarization and Compaction
When the conversation outgrows its budget, keep the last few turns word for word and **summarize** the older ones into a short note. This is often called **compaction**.

**Network-engineer analogy:** it's route summarization. Ten /24s become one /21. The table gets smaller, but you lose specificity, and a bad summary can black-hole traffic. The lab shows exactly that: after compaction, the summary still says `4120` but has lost that it meant *CRC errors* pointing to *a bad optic*. Mitigations:
- Ask the summarizer to keep **decisions, open questions, and facts with their meaning** ("ge-0/0/2: 4,120 CRC errors → suspected bad optic"), not loose fragments.
- Store important facts as **structured long-term memory** (§14.16) instead of trusting a prose summary.
- Keep the original transcript in storage so the agent can look details back up if needed.

### §14.16 — Long-Term Memory Stores
Long-term memory is mostly RAG pointed at the agent's own history:
- A **vector store** (§3.20) of past conversations, notes, and resolved incidents, searched by meaning (§3.18).
- A **structured store** (a SQL table or key-value store) for facts that need exact answers: device inventory, owners, maintenance windows. Don't make an embedding search guess at a serial number.
- A **memory write step:** after a task, the agent (or a separate summarizer) decides what's worth keeping, and stores it with a source, a date, and maybe an expiry.

Research systems like MemGPT (2023, now Letta) treat the context window like RAM and long-term storage like disk, paging information in and out. That's a nice mental model even if you build something simpler.

### §14.17 — What to Remember, Forget, and Protect
Memory creates new risks:
- **Staleness:** "Core-1 runs version X" was true in March. Store dates, prefer recent facts, and let some memories expire (a TTL).
- **Poisoning:** if an agent saves text from untrusted content into memory, an injected instruction can come back in *future* sessions (§5.26). Store facts, not instructions, and record where each one came from.
- **Privacy:** never store secrets, passwords, or tokens in memory. Be deliberate about personal data, and give users a way to see and delete what's remembered.
- **Contradictions:** when a new fact conflicts with an old one, replace it explicitly. Don't keep both and let the model guess.

---

## Chapter 4 — Tools in Depth

### §14.18 — Designing a Good Tool
To the model, a tool *is* its name, description, and schema (§12.21). It never sees your code. Design tools the way you'd write a good CLI for a junior engineer:
- **Clear names and descriptions** that say *when* to use the tool, not just what it does: "Use to read live counters for ONE interface. Does not change anything."
- **Strict schemas:** required fields, enums for fixed choices (`"direction": "in" | "out"`), and patterns for formats like interface names.
- **The right size.** One giant tool (`run_any_cli_command`) is powerful and dangerous. Twenty tiny tools overwhelm the model and the context budget (§14.14). Aim for task-shaped tools like `get_interface_counters` or `get_bgp_neighbor`.
- **Compact outputs.** Return the 10 fields the model needs, not 5,000 lines of `show tech`. Every token of tool output is paid for on every later step (§14.9).

### §14.19 — Errors the Model Can Act On
When a tool fails, the error message goes back to the model, which will try to recover. Write errors for that reader:
- ❌ `Error: 400`
- ✅ `Interface "ge-0/0/9" not found on core-1. Valid interfaces: ge-0/0/0 to ge-0/0/3. Check the name and try again.`

A good error says what went wrong, why, and what to do next. That one change often fixes more agent failures than any prompt tweak. Also decide which errors the model should see at all: an authentication failure should go to a human or monitoring, not to the model to "work around."

### §14.20 — Read vs. Write Tools, and Blast Radius
Classify every tool by what it can affect:
- **Read-only** (`show` commands, search, queries): safe to let the agent call freely, within rate limits (§11.20).
- **Write** (config changes, tickets, messages): require validation, a **dry-run** mode that shows the diff without applying it, and human approval (§14.11).
- **Destructive or high blast radius** (reload a device, bulk changes): usually shouldn't be agent tools at all.

Start every agent project **read-only**. A troubleshooting agent that only gathers evidence and writes a recommendation is already valuable, and it can't break anything.

### §14.21 — MCP Servers as Packaged Tools
MCP (§12.23) lets you package tools once and use them from any MCP client. A "network inventory" MCP server can serve Claude Desktop, your IDE, and your own agent. Practical points:
- **Trust:** an MCP server runs code and returns text that goes into the prompt. A third-party server is third-party code *plus* a prompt-injection channel. Review it like any dependency, pin versions, and prefer servers you can read.
- **Scope:** run separate servers with separate credentials for read and write capabilities, so a client can be given only the read one.
- **Local vs. remote:** stdio servers run as local processes with your user's permissions; HTTP servers need real authentication.

### §14.22 — Sandboxing and Least Privilege in Practice
§12.24 gave the principles. Applied to tools:
- **Credentials per tool**, scoped to the minimum: a read-only API token for counters, nothing else.
- **Sandboxes** for anything that executes code or shell commands: a container with no network route to production and no secrets mounted.
- **Allow-lists** over block-lists: the agent may call these 6 tools with these argument patterns. Everything else is denied.
- **Egress control:** if the agent doesn't need to reach the internet, don't give it a route there. This breaks the "send data out" leg of the lethal trifecta (§12.24).
- **Audit logs** of every call, with arguments and results, tied to the task ID.

It's an ACL and a jump host, applied to software.

---

## Chapter 5 — Multi-Agent Systems

### §14.23 — Why Split Into Multiple Agents?
A **multi-agent system** gives different parts of a task to separate agents, each with its own prompt, tools, and context. Good reasons:
- **Context isolation:** each agent's window holds only what its job needs, so it stays sharp (§3.28).
- **Specialization:** a "log analyst" with log tools and a "config reviewer" with config tools, each with a focused prompt.
- **Parallelism:** several researchers can search different sources at the same time.
- **Separation of duties:** the agent that proposes a change isn't the one that approves it.

### §14.24 — Multi-Agent Patterns
- **Supervisor (orchestrator–workers):** one agent breaks the task down, delegates to workers, and combines the results.
- **Pipeline:** agents in a fixed sequence, each handing its output to the next (really a workflow whose steps are agents).
- **Critic / debate:** one agent produces, another critiques, and they iterate. This is the verifier from §14.8 with more independence.
- **Handoff:** an agent passes the whole conversation to a specialist ("transferring you to the BGP agent"), like a call-center transfer.
- **Parallel fan-out:** send the same question to several agents or sources, then merge or vote.

### §14.25 — The Cost of Coordination
Multi-agent systems are easy to over-build. The costs:
- **Tokens:** every agent has its own prompt and context, and they pass results back and forth. Anthropic reported in 2025 that its multi-agent research system used roughly 15× the tokens of a normal chat. That was worth it for hard research questions and wasteful for simple ones.
- **Error compounding** (§12.22) now happens across agents *and* within each one. A worker's confident mistake becomes the supervisor's "fact."
- **Latency and debugging:** more hops, and a trace that spans several agents (§11.24).

| | Single agent with tools | Multi-agent |
|---|---|---|
| **Pros** | Cheaper, simpler, one trace to read | Context isolation, parallelism, specialization, separation of duties |
| **Cons** | Context gets crowded on big tasks | Many more tokens, coordination bugs, harder to test |

Rule of thumb: start with one agent. Split only when you can name the specific problem the split solves.

### §14.26 — Agents Talking to Agents
Inside one application, agents usually talk through your own code: a function call or a message on a queue. Across organizations or products, standards are emerging. Google's **A2A** (Agent2Agent, 2025, now under the Linux Foundation) defines how independent agents describe their capabilities and exchange tasks.

The distinction from MCP (§12.23): **MCP connects an agent to tools and data. A2A connects an agent to another agent** that has its own reasoning. In network terms, MCP is like NETCONF to a device, and A2A is closer to BGP peering between autonomous systems: each side keeps its own policy and decides what to accept.

---

## Chapter 6 — Frameworks in Practice

### §14.27 — Learn the Loop First
Every agent framework wraps the same few pieces: a model call, tool schemas, a loop, state, and maybe a graph and checkpoints. If you've done the labs in Module 12 and this module, you know what's inside. That matters, because:
- When something breaks, you'll know which layer to look at.
- Frameworks change fast. APIs from a year ago are often deprecated. The concepts don't change.
- Sometimes 100 lines of plain Python plus an SDK *is* the best framework (§12.25).

### §14.28 — The Framework Landscape
As of this writing (verify current status; this area moves monthly):

| Framework | Style | Strengths | Watch out for |
|-----------|-------|-----------|---------------|
| **LangGraph** | Explicit graphs of nodes and edges, with built-in checkpointing | Long-running workflows, human-in-the-loop, fine control | More concepts to learn; tied to the LangChain ecosystem |
| **PydanticAI** | Typed Python agents with validated structured outputs | Clean, Pythonic, model-agnostic, great with schemas | Younger, smaller ecosystem |
| **OpenAI Agents SDK** | Agents, handoffs, guardrails, built-in tracing | Simple multi-agent handoffs; works with other providers too | Best experience is with OpenAI's platform |
| **Claude Agent SDK** | The harness behind Claude Code, exposed as a library | Strong tool use, file and shell tools, subagents | Best experience is with Claude |
| **Google ADK** | Agent Development Kit: code-first, multi-agent | Google Cloud / Gemini integration, A2A support | Best experience is on Google Cloud |
| **CrewAI** | Role-based "crews" of agents | Fast to prototype multi-agent role play | Less control over the exact flow |
| **AutoGen** (Microsoft) | Agents that converse with each other | Research-friendly multi-agent conversations | Microsoft has been consolidating it into a newer agent framework, so check its status |

### §14.29 — Choosing a Framework and Avoiding Lock-In
Choose from requirements, not popularity:
- **Need explicit control, long runs, approvals, resumability?** A graph framework (LangGraph) or your own state machine.
- **Mostly structured extraction and simple tool use, in clean Python?** PydanticAI, or plain SDK calls.
- **Committed to one vendor's platform?** That vendor's SDK is the smoothest path.

To limit lock-in:
- Put the model provider behind your own small interface (or the OpenAI-compatible API, §12.14), so you can switch models, including to a local one in Ollama.
- Keep your **prompts, tool schemas, and evaluation sets** in your own files, not buried in framework objects. They're your real assets.
- Keep business logic in plain functions that the framework *calls*, not inside framework classes.

### §14.30 — Testing and Observing Agents
Module 13 covered testing and evaluating LLM features. Agents add one more layer: you have to judge the **trajectory** (the path the agent took), not just the final answer.
- **Unit-test the deterministic shell:** tools, validators, routers, and budget logic, with no model involved.
- **Replay tests:** run the agent against *recorded* tool results, so tests are fast and repeatable.
- **Trajectory checks:** did it call the required tools? Did it avoid forbidden ones? How many steps and how many tokens did it take? A correct answer reached in 40 steps is a regression if it used to take 6.
- **Task success rate** on a fixed set of scenarios, tracked over time like any other metric (§5.24).
- **Traces in production** (§11.24): every node, tool call, token count, and decision, so "the agent went weird" becomes a specific step you can inspect.

---

## Chapter 7 — Your Path

### §14.31 — The Project Ladder
Build these in order. Each one adds a new skill, and each produces something useful in a network job. "Done when" means you have an eval that proves it (§5.24, §12.19).

| # | Project | What it adds | Course sections | Done when |
|---|---------|--------------|-----------------|-----------|
| 1 | **Runbook FAQ bot**: answers questions from your team's runbooks | RAG basics, chunking, citations | §12.15–§12.20, §3.18–§3.20, Module 13 | It answers 20 test questions correctly *with the right source*, and says "not found" for 5 trick questions |
| 2 | **Config and RFC Q&A**: "which routers have BFD on their OSPF interfaces?" | Hybrid search for exact terms, structure-aware chunking, structured outputs | §3.20, §12.17, §8.31, Module 13 | It finds exact matches like `EXSTART` or an RFC section number every time |
| 3 | **Network research agent** (read-only): investigates "why is this link slow?" with counters, logs, and docs | Tool design, the agent loop, verifiers, budgets | §12.21–§12.24, Chapters 1–2 and 4 of this module | It solves 10 recorded scenarios within a step budget and cites evidence |
| 4 | **Log and config extraction pipeline**: syslog → structured incident records | A *workflow*, schema validation, batch processing, maybe a fine-tuned small model | §5.21, §8.31, Module 13, §14.3 | It hits ≥95% field accuracy on a labeled sample, at a known cost per 1,000 lines |
| 5 | **Multi-agent incident triage with approval**: correlates alerts, proposes a fix with a diff and rollback plan, and waits for a human | Multi-agent design, memory, approvals, checkpoint/resume, security | Chapters 2–5 of this module, §12.24 | No write happens without approval, it resumes after a restart, and its traces explain every decision |

Run projects 1–4 against a local model in Ollama first (cheap, private), then compare with a frontier API model (§12.11).

### §14.32 — The Skills Map
Job descriptions for AI engineering roles mostly ask for *software engineering around LLMs*, not model research. A prioritized map, with where this course covers each skill:

| Tier | Skill | Covered in course? |
|------|-------|--------------------|
| **Must-have** | Python (clean, typed, tested) | Labs throughout; Module 13 |
| | Calling LLM APIs, prompting, structured outputs | Module 13, §12.14, §8.31 |
| | RAG | §12.15–§12.20 |
| | Docker | Module 13 |
| | One cloud (AWS, GCP, or Azure) | Not in depth; learn one |
| **High-value** | FastAPI (Python web APIs) | Module 13 |
| | SQL and PostgreSQL (plus pgvector) | §3.20; learn SQL separately |
| | Evaluation and testing of LLM systems | §5.24, §12.19, §14.30, Module 13 |
| | Observability: tracing, logging, cost tracking | §11.23–§11.24, §14.30, Module 13 |
| | Git, CI/CD | Module 13 |
| | One agent framework | §14.28 |
| **Differentiators** | Production agents with approvals and memory | This module |
| | Fine-tuning a small model | §5.15–§5.23 |
| | Serving and infrastructure (vLLM, GPUs, Kubernetes) | Modules 06–11 |
| | Multi-agent systems | Chapter 5 of this module |
| | TypeScript / React (for full-stack roles) | Not covered |

Honest trade-offs:
- **Kubernetes:** high value for platform and serving roles (Module 11), lower for app-focused roles. Learn Docker well first.
- **TypeScript:** important if you'll build user interfaces. Skip it if you're aiming at back-end, platform, or network-automation work.
- **Deep ML math and PyTorch:** "just enough" is enough for most AI engineering jobs. Modules 01–05 cover the concepts. Go deeper only if you want to train or fine-tune models seriously.
- **Frameworks:** learn one well, plus the plain loop. Knowing five superficially impresses nobody.

### §14.33 — Your Advantage as a Network Engineer
You're not starting from zero. Much of what makes agents work in production is what you already do:
- **Change management** → approvals, dry-runs, rollback plans (§14.11, §14.20).
- **Least privilege and segmentation** → agent sandboxes and scoped credentials (§14.22).
- **State machines and protocol debugging** → agent graphs and traces (§14.4, §14.30).
- **Monitoring, SLOs, capacity planning** → LLM serving and observability (Module 11).
- **Automation** (Ansible, NETCONF, Python scripts) → tool design and idempotency (§14.12, §14.18).
- **Domain knowledge** → the thing generic AI engineers lack. A team that needs an AI agent to troubleshoot its network needs someone who knows what a correct answer looks like.

Natural directions: **AI for network operations** (troubleshooting assistants, config analysis, NOC copilots), **AI infrastructure and platform engineering** (GPU clusters, fabrics, serving: Modules 06–11), or general **AI application engineering**.

### §14.34 — Practicing for Interviews
The repo includes `Complete Roadmap to Become an Agentic AI Engineer.pdf`, an interview Q&A set organized by topic (Python, LLM fundamentals, frameworks, memory, tools, RAG, agents, deployment). Use it actively:
1. Read a question and **answer it out loud first**, in your own words, before reading the model answer.
2. Compare. Where you were vague, find the course section that covers it and reread it.
3. For each topic, have **one project story ready** from the ladder (§14.31): what you built, what broke, how you measured and fixed it. Interviewers care far more about real failures and fixes than about definitions.
4. Practice the capstone explanations in §12.38–§12.41 until you can give each in under two minutes.

### 🛠️ Build It — Hands-On Lab
Pure Python, nothing to install. Both labs use a **fake model** so you can see the control logic clearly. Swapping in a real model later changes one function.

**Lab 1: a graph agent with a verifier, a step budget, and checkpoint/resume (§14.4–§14.12).**

```python
import json

# The "network" our tools can see. In real life these would be API or SNMP calls.
INTERFACES = {
    "ge-0/0/1": {"util_pct": 93, "crc_errors": 0},
    "ge-0/0/2": {"util_pct": 20, "crc_errors": 4120},
}

def get_utilization(name):
    return {"util_pct": INTERFACES[name]["util_pct"]}

def get_errors(name):
    return {"crc_errors": INTERFACES[name]["crc_errors"]}

TOOLS = {"get_utilization": get_utilization, "get_errors": get_errors}   # the allow-list

def fake_model(state):
    """Stands in for an LLM. Like a real one, it sometimes jumps to conclusions."""
    seen = {obs["tool"] for obs in state["observations"]}
    target = state["target"]
    if "get_utilization" not in seen:
        return {"action": "tool", "tool": "get_utilization", "args": {"name": target}}
    if "get_errors" not in seen and not state["feedback"]:
        return {"action": "answer", "text": f"{target} is congested."}    # premature!
    if "get_errors" not in seen:
        return {"action": "tool", "tool": "get_errors", "args": {"name": target}}
    util = next(o["result"]["util_pct"] for o in state["observations"] if o["tool"] == "get_utilization")
    crc = next(o["result"]["crc_errors"] for o in state["observations"] if o["tool"] == "get_errors")
    cause = "congestion" if util > 80 and crc == 0 else "a physical-layer fault"
    return {"action": "answer",
            "text": f"{target}: utilization {util}%, CRC errors {crc}. Likely cause: {cause}."}

def verifier(state, text):
    """Deterministic check: the answer must cite every piece of evidence gathered,
    and both checks must have been run."""
    tools_run = {o["tool"] for o in state["observations"]}
    if tools_run != set(TOOLS):
        return False, f"missing checks: {sorted(set(TOOLS) - tools_run)}"
    for obs in state["observations"]:
        for value in obs["result"].values():
            if str(value) not in text:
                return False, f"answer does not cite {obs['tool']} result {value}"
    return True, "ok"

def save(state, path):
    with open(path, "w") as f:
        json.dump(state, f)

def run(state, path, crash_after=None):
    while state["status"] == "running":
        if state["step"] >= state["max_steps"]:                  # stop condition: budget
            state["status"] = "stopped: step budget used up"
            break
        decision = fake_model(state)                             # router: what next?
        state["step"] += 1
        if decision["action"] == "tool":
            result = TOOLS[decision["tool"]](**decision["args"])  # executor
            state["observations"].append({"tool": decision["tool"], "result": result})
            print(f"step {state['step']}: tool {decision['tool']} -> {result}")
        else:
            ok, reason = verifier(state, decision["text"])       # verifier
            if ok:
                state["status"], state["answer"] = "done", decision["text"]
                print(f"step {state['step']}: answer ACCEPTED")
            else:
                state["feedback"].append(reason)                 # repair edge: back to router
                print(f"step {state['step']}: answer REJECTED ({reason})")
        save(state, path)                                        # checkpoint after every step
        if crash_after == state["step"]:
            print("--- process crashed! ---")
            return

path = "agent_checkpoint.json"
state = {"target": "ge-0/0/1", "step": 0, "max_steps": 6, "status": "running",
         "observations": [], "feedback": [], "answer": None}
run(state, path, crash_after=2)

with open(path) as f:                                            # a new process starts...
    resumed = json.load(f)
print(f"--- resumed from checkpoint at step {resumed['step']} ---")
run(resumed, path)
print("final:", resumed["status"], "|", resumed["answer"])
```

What you should see:

```
step 1: tool get_utilization -> {'util_pct': 93}
step 2: answer REJECTED (missing checks: ['get_errors'])
--- process crashed! ---
--- resumed from checkpoint at step 2 ---
step 3: tool get_errors -> {'crc_errors': 0}
step 4: answer ACCEPTED
final: done | ge-0/0/1: utilization 93%, CRC errors 0. Likely cause: congestion.
```

The model jumped to a conclusion at step 2, the deterministic verifier caught it, and the repair edge sent it back to gather the missing evidence. The crash cost nothing, because the state was checkpointed. (The lab writes `agent_checkpoint.json` in the current folder.) Things to try:
- Change the target to `ge-0/0/2`. What cause does it report?
- Set `max_steps` to 3. What happens, and what *should* a production system do next (§14.9)?
- Make the fake model always answer prematurely. Watch the budget stop the loop.

**Lab 2: short-term memory with a token budget and compaction (§14.14–§14.15).**

```python
import re

BUDGET = 60          # max "tokens" of history we send each turn (we count words as tokens)
KEEP_RECENT = 2      # always keep the last 2 turns word-for-word

def tokens(text):
    return len(text.split())

def fake_summarize(turns):
    """Stands in for an LLM summarizer: keeps only the 'facts' (interface names and numbers)."""
    facts = []
    for turn in turns:
        for item in re.findall(r"\d{1,2}:\d{2}|[a-z]+-\d+/\d+/\d+|\d+%?", turn):
            if item not in facts:
                facts.append(item)
    return "SUMMARY of earlier turns, key facts: " + ", ".join(facts)

history = []
conversation = [
    "user: ge-0/0/1 on the core router is dropping packets since 09:00",
    "assistant: I checked ge-0/0/1. Utilization is 93% and output drops are 1200 per minute",
    "user: what about the uplink ge-0/0/2, is it any better than the first one",
    "assistant: ge-0/0/2 is at 20% with 4120 CRC errors, which points to a bad optic or cable",
    "user: ok so should I move traffic off the congested link first or fix the optic first",
    "assistant: move traffic first since that is a quick change, then replace the optic on ge-0/0/2",
]
for n, turn in enumerate(conversation, start=1):
    history.append(turn)
    total = sum(tokens(t) for t in history)
    if total > BUDGET:
        old, recent = history[:-KEEP_RECENT], history[-KEEP_RECENT:]
        history = [fake_summarize(old)] + recent
        new_total = sum(tokens(t) for t in history)
        print(f"turn {n}: {total} tokens > {BUDGET} -> summarized "
              f"{len(old)} turns, now {new_total} tokens")
    else:
        print(f"turn {n}: {total} tokens (budget {BUDGET})")

print("\nWhat the model sees on the next turn:")
for line in history:
    print("  ", line)
```

What you should see: the history grows to 56 tokens, then compaction kicks in at turns 5 and 6 and keeps it under 60. The final summary reads `ge-0/0/1, 09:00, 93%, 1200, ge-0/0/2, 20%, 4120`. The numbers survived, but the *meaning* didn't: nothing says 4120 was CRC errors or that it pointed to a bad optic. That's the route-summarization problem from §14.15. Things to try: improve `fake_summarize` so it keeps "4120 CRC errors" together, then think about how you'd *test* a real LLM summarizer for the same failure.

**Optional, real tools:**
- Rebuild Lab 1 with **LangGraph** (`pip install langgraph`): make router, executor, and verifier nodes, add a checkpointer, and point the model at a local Ollama model through its OpenAI-compatible endpoint (§12.14). Compare how much of your hand-written code the framework replaced.
- Or rebuild it with **PydanticAI**: define the two tools as typed functions and make the final answer a Pydantic model with `util_pct`, `crc_errors`, and `cause` fields, so the verifier becomes schema validation plus one evidence check.
- Add an approval node (§14.11) before a fake `set_interface_description` write tool, and make it idempotent (§14.12).

### §14.35 — Checkpoint — Explain It Back
1. What's the difference between a workflow and an agent, and when should you choose the workflow?
2. Why does drawing an agent as a state machine make it easier to control and debug?
3. What does "the LLM proposes, your code disposes" mean in practice?
4. What does a verifier check, and what's a repair edge?
5. Why do agent costs grow faster than the number of steps?
6. What's an idempotent tool, and why does checkpoint/resume need it?
7. What are short-term and long-term memory, and what can go wrong with summarization?
8. What makes a good tool description and a good tool error message?
9. When is a multi-agent system worth its cost?
10. Name three things you'd keep outside any agent framework to avoid lock-in.

<details>
<summary>Answer key (try first!)</summary>

1. In a workflow, your code fixes the order of steps; in an agent, the model chooses the next step in a loop. Choose the workflow whenever the steps are known in advance: it's cheaper, more predictable, and testable step by step.
2. Every state and allowed transition is explicit, so you can enforce limits on each edge, and a trace shows exactly where it went wrong, like a neighbor stuck in ExStart.
3. The model only suggests the next action or answer. Code validates it, checks permissions and budgets, runs verifiers, and decides whether it actually happens.
4. Deterministic rules first (valid schema, required checks done, evidence cited), then optionally an LLM judge. A repair edge sends a failed result back to the router with the failure reason as feedback instead of giving up.
5. Each step re-sends the whole growing context, so total input tokens grow roughly with the square of the step count. 20 steps in the example was ~345,000 input tokens.
6. Running it twice has the same effect as running it once. After a crash you may re-run the step that was in progress, and only idempotent tools (or idempotency keys) make that safe.
7. Short-term memory is the current task inside the context window. Long-term memory is stored outside the model and retrieved later. Summaries can keep the facts but lose their meaning, like route summarization losing specificity.
8. A description that says when to use the tool (and whether it changes anything), with a strict schema. An error that says what went wrong, why, and what to try next.
9. When the task really benefits from context isolation, specialization, parallel work, or separation of duties, and the extra tokens (reported around 15× for research) are justified.
10. Prompts, tool schemas, and evaluation sets, plus business logic in plain functions and the model provider behind your own interface.
</details>

### §14.36 — Final Recap: The Whole Journey
You started at "what is AI?" and finished designing production agents. The whole course, one sentence per module:

1. **Foundations (01):** AI moved from hand-written rules to learning weights from data, using vectors and neural networks trained by backprop.
2. **Transformers (02):** attention lets every token look at every other in parallel, and stacked transformer blocks power every modern LLM.
3. **Tokens & embeddings (03):** text becomes token IDs, IDs become vectors, and the next token is picked from a probability list.
4. **Training (04):** next-token prediction over trillions of tokens, governed by scaling laws and enormous compute.
5. **Fine-tuning & alignment (05):** SFT, preference tuning, and RL turn a base model into an assistant, and LoRA lets you customize one.
6. **Hardware (06):** inference is mostly limited by memory bandwidth, and VRAM decides what fits.
7. **Quantization (07):** fewer bits per weight make models smaller and faster, at a measurable quality cost.
8. **Inference engines (08):** llama.cpp, Ollama, vLLM, and friends turn weights into a fast, batched service.
9. **Throughput (09):** the KV cache, batching, FlashAttention, and speculative decoding squeeze out speed.
10. **Distributed training (10):** collectives over fast fabrics let thousands of GPUs train one model.
11. **Serving (11):** SLOs, queueing, load balancing, autoscaling, and cost turn an engine into a reliable product.
12. **Ecosystem (12):** the labs, the open models, RAG, tools, agents, MCP, and where it's all heading.
13. **Building AI applications (13):** calling models from code, structured outputs, testing, evaluation, and shipping.
14. **Agentic systems (14):** controlled loops, memory, tools, and multi-agent designs, plus your path from here.

Where to go next: pick project 1 from the ladder (§14.31) and build it this week, then keep climbing. Use §12.37 to stay current without chasing every headline. The best way to keep what you've learned is to build with it and to explain it to someone else.
