# Agent Analytics + NAT Integration — Plan (no code yet)

Planning doc for two related features in the **Agent Lab**:
- **Part A — Native agent analytics** (in-browser run report + admin/student dashboards)
- **Part B — Real NVIDIA NeMo Agent Toolkit (NAT) integration** (Python sidecar on Render)

Grounded in the current stack: Next.js 15 / React 19 / TS, TiDB via Drizzle, Render (Docker) hosting.
Reuses what already exists: `usage` metering, `audit_log`, `rate_limits`, `/api/health`, the admin
dashboard pattern (`/admin/usage` + `UsageDashboard.tsx`), the ReAct loop in `AgentLab.tsx`, tools in
`src/lib/agentTools.ts`, and `/api/chat` token metering.

---

## Part A — Native agent analytics (recommended first; no new infra)

### A1. Data model — new `agent_runs` table (`src/lib/db/schema.ts`)
| column | purpose |
|---|---|
| id, userId, ts | identity |
| agentName, agentType (`react`/`workflow`), provider, model | what ran |
| task (truncated) | the goal |
| iterations, maxIterations, hitMax | loop behaviour |
| toolCalls (json: `[{tool, count, errors}]`), toolCallCount | tool usage |
| promptTokens, completionTokens, totalTokens, estimated | cost basis |
| costUsd (computed) | $ per run |
| latencyMs, steps (json: `[{i, phase, ms, tokens, tool?}]`) | profiling / timeline |
| outcome (`success`/`max_iters`/`error`/`parse_fail`/`stopped`), errorMsg | result |

Migration via the existing flow: `npm run db:generate` → commit → `apply-to-existing-db.sql` delta.

### A2. Instrumentation (client — `AgentLab.tsx`)
Wrap the existing run loop in a collector:
- stamp start/end + per-iteration timing,
- tally tool calls (from `parseReAct` Action),
- detect outcome (reached Final Answer vs max-iters vs parse/tool/provider error),
- accumulate tokens (agent calls are non-streaming → `/api/chat` can return `usage`; else estimate from text length as we already do).

On run end: `POST /api/agent/runs` with the summary.

### A3. Endpoints
- `POST /api/agent/runs` — auth + `rateLimitDb` + payload caps; writes one `agent_runs` row (userId from session).
- `GET /api/agent/runs?scope=me` — a student's own run history.

### A4. Layer 1 — in-lab "Run report" (client, teaching)
Panel after each run: outcome badge, total ms, iterations, tokens, **est. cost**, tool-usage bars, and a
**step timeline** (Thought/Action/Observation with per-step ms + tokens). Highlight the slowest / most
expensive step — this is the NAT-flavored "bottleneck" idea, done natively. Cost from a new
`src/lib/pricing.ts` per-model price map (tokens → $).

### A5. Layer 2 — dashboards (persistent)
- **Admin** `/admin/agents`: total runs, success rate, avg iterations, avg tokens/cost per run, most-used
  tools, failure-reason breakdown, cost by model/user, runs over time. Reuse `UsageDashboard` layout.
- **Student**: "My agent runs" (Agent Lab tab or My Projects) — personal history + trends.

### A6. Files touched / effort
`schema.ts` (+table), migration, `/api/agent/runs`, `pricing.ts`, `AgentLab.tsx` (instrument + report
panel), `/admin/agents` page + `AgentAnalytics` component, nav link, unit tests (pricing, aggregation).
**Effort:** ~the same as the usage-dashboard work. No new infrastructure. Keeps the glass-box model.

---

## Part B — Real NAT integration (Python sidecar on Render)

### B1. Architecture
```
Browser ──> Next.js (Render web service) ──internal HTTP──> nat-service (Render private Docker service)
                    │ decrypts provider key                         │ runs agent via nvidia-nat
                    │ quota + rate limit                            │ returns answer + profiler report
```
NAT runs the agent **server-side in Python**; the browser never talks to it directly.

### B2. The Python service (`nat-service/`)
- `python:3.12-slim` + `nvidia-nat` + FastAPI + uvicorn.
- `POST /run` body: `{agentType, model, provider:{baseUrl, apiKey}, systemPrompt, tools:[...], task, knowledge?:{docs}}`.
- Builds a NAT workflow config from the body, registers the tool functions, runs it, returns
  `{answer, trace, profiler:{perStep:[{name, ms, tokens}], totalMs, totalTokens, costUsd, bottleneck}}`.
- Auth: verify a shared-secret header. Provider key arrives **per-request from the Next server only** —
  never stored in Python, never sent from the browser.

### B3. Tool parity — the real work
NAT tools are **Python functions**; the lab's tools are JS (`agentTools.ts`). Each supported tool must be
reimplemented/registered in Python:
- `calculator`, `datetime` — trivial.
- `http` — reimplement with the **same SSRF guard** (`isBlockedHost`/DNS check).
- `knowledge` (RAG) — hardest: the index is built **in the browser** from uploaded docs, so the Next
  server must ship the doc text to `nat-service`, which rebuilds the index in Python (TF-IDF or NAT's
  retriever). Launch with a **declared supported-tool set** (start: calculator + http + knowledge).

### B4. Next.js side
- `POST /api/agent/nat-run`: auth, quota + rate limit, decrypt provider key **server-side**, forward to
  `NAT_SERVICE_URL/run` with the shared secret, return the profiler report. Key never reaches the client.
- `AgentLab.tsx`: a **"Run via NAT (real profiler)"** toggle; render NAT's report (reuse the Layer-1 UI +
  NAT-specific fields). Labelled "runs server-side via NVIDIA NeMo Agent Toolkit."
- Meter NAT's returned tokens into the `usage` table too, so cost accounting stays unified.

### B5. Deployment (Render + Docker)
- `nat-service/Dockerfile` (uvicorn).
- `render.yaml` Blueprint = two services: the Node web service + a **private** Docker service. They talk
  over Render's private network; the Python service is not publicly exposed.
- Env: `NAT_SERVICE_URL`, `NAT_SHARED_SECRET` (provider key passed per-request, not stored).
- Cost: one small Render instance for the Python service. LLM cost unchanged (reuses your provider).
- Note: on Render's free tier the Python service sleeps → cold-start latency on first NAT run.

### B6. Risks / trade-offs
- **Loses the glass box** for NAT runs (executes server-side) — frame it as an *advanced "production tool"*
  mode, not the default runtime.
- **Two tool implementations** (JS + Python) to keep in sync — maintenance cost; mitigate with a small
  supported set.
- NAT config translation + RAG-context shipping are the bulk of the effort, not NAT itself.

### B7. Phasing
1. **Spike (~1 day):** calculator-only agent, hardcoded, prove the profiler round-trip Render→Render.
2. Add `http` + `knowledge`, config translation, token metering.
3. UI polish + supported-tool matrix.

---

## Recommended sequence
1. **Part A** (native analytics) — immediate value, no infra, keeps the browser model.
2. **Part B spike** — prove real NAT on Render with one tool.
3. **Part B full** — expand tool parity + UI.

## Open decisions (need your call before building)
1. **NAT-supported tools at launch** — calculator only / +http / +knowledge / all?
2. **Who sees NAT mode** — students, or admin/demo only?
3. **Cost display** — maintain a per-model price map for $ figures, or show tokens only?
4. **License/cost** — NAT is Apache-2.0 (free); LLM + the extra Render service are the only real costs.
