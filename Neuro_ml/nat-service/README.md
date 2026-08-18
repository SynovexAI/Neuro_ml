# NAT agent service

A small Python sidecar that runs agents through the **NVIDIA NeMo Agent Toolkit**
(NAT, Apache-2.0) and returns the answer plus timing. The Next.js app calls it
over Render's private network; the provider API key is passed **per-request** and
never stored here.

## What it does
- `GET /health` → `{ "ok": true }`
- `POST /run` → builds a NAT `react_agent` workflow, runs it, returns the answer, latency, and a per-step profiler.
- Built-in tools: `calculator`, `current_datetime` (NAT core).
- **MCP tools**: `mcp_servers` in the request become `mcp_client` function groups (HTTP headers / stdio env resolved by the Next proxy).
- **RAG ingest**: `knowledge.docs` are chunked + retrieved (TF-IDF) and the top passages injected as context (`context_used`).
- **Per-step profiler**: subscribes to NAT intermediate steps → `profiler.steps` (name, type, ms, tokens); best-effort, degrades to `total_ms`.

## Request / response
```jsonc
// POST /run   header: X-NAT-Secret: <NAT_SHARED_SECRET>
{
  "task": "What is 18% of 2450?",
  "model": "llama-3.3-70b",
  "base_url": "https://api.groq.com/openai/v1",   // your provider (OpenAI-compatible)
  "api_key": "<passed per-request, never stored>",
  "temperature": 0.0,
  "system_prompt": "You are a careful reasoning agent.",
  "tools": ["calculator", "current_datetime"]
}
// ->
{ "answer": "...", "latency_ms": 1600, "model": "llama-3.3-70b",
  "tool_names": ["calculator","current_datetime"], "unsupported_tools": [], "profiler": { "total_ms": 1600 } }
```

## Run locally
NAT needs Python 3.11–3.13 (not 3.14). With `uv`:
```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements.txt
```
```bash
export NAT_SHARED_SECRET=dev-secret
uvicorn app.main:app --reload --port 8000
```
Then `POST http://localhost:8000/run` with the body above.

## Deploy (Render)
`render.yaml` defines a **private** Docker service (`type: pserv`) — internal-only.
Set `NAT_SHARED_SECRET` on it, and on the Next.js service set `NAT_SERVICE_URL`
(e.g. `http://nat-service:8000`) + the same `NAT_SHARED_SECRET`.

## Notes / not-yet
- Detailed per-step profiler data (latency/tokens per Thought/Action) needs
  intermediate-step subscription — phase 2. `profiler.total_ms` is populated now.
- The provider key is written to a temp config file for the duration of a run,
  then deleted. Fine for a trusted internal service; hardening (in-memory config)
  is a follow-up.
