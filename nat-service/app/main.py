"""FastAPI sidecar that runs agents through the NVIDIA NeMo Agent Toolkit.

The Next.js app calls POST /run over Render's private network with a shared
secret. This service never stores the provider key — it arrives per-request.
"""
import logging
import os

from fastapi import FastAPI, Header, HTTPException

from . import rag
from .config_builder import build_config
from .nat_runner import run_workflow
from .schemas import RunRequest, RunResponse

log = logging.getLogger("nat-service")
app = FastAPI(title="NAT agent service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, x_nat_secret: str | None = Header(default=None)) -> RunResponse:
    secret = os.environ.get("NAT_SHARED_SECRET")
    if secret and x_nat_secret != secret:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    config, tool_names, unsupported = build_config(req)
    message = req.task if not req.system_prompt else f"{req.system_prompt}\n\nTask: {req.task}"

    # RAG ingest: ground the run in supplied documents (retrieval + context inject).
    context_used = False
    if req.knowledge and req.knowledge.docs:
        message, context_used = rag.augment(message, req.task, req.knowledge.docs)

    try:
        result = await run_workflow(config, message)
    except Exception as e:  # noqa: BLE001 — surface a clean error to the caller
        log.exception("NAT run failed")
        raise HTTPException(status_code=502, detail=f"NAT run failed: {e}") from e

    return RunResponse(
        answer=result["answer"],
        latency_ms=result["latency_ms"],
        model=req.model,
        tool_names=tool_names,
        unsupported_tools=unsupported,
        context_used=context_used,
        profiler=result.get("profiler", {"total_ms": result["latency_ms"]}),
    )
