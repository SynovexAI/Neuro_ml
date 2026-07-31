"""Best-effort per-step profiler for a NAT run.

Subscribes to NAT's intermediate-step stream, pairs START/END events by id, and
reports each step's name, type, duration and token usage. Everything is guarded
so a NAT API change degrades to a total-latency-only report rather than failing
the run.
"""
from typing import Any


def subscribe(collector: list[Any]) -> None:
    """Subscribe the collector to the current run's intermediate steps."""
    try:
        from nat.builder.context import Context
        ctx = Context.get()
        ctx.intermediate_step_manager.subscribe(
            on_next=lambda item: collector.append(item),
            on_error=lambda _e: None,
            on_complete=lambda: None,
        )
    except Exception:
        pass  # profiling is optional; the run continues either way


def _payload(item: Any) -> Any:
    return getattr(item, "payload", item)


def _tokens(payload: Any) -> int:
    for attr in ("usage_info", "token_usage"):
        u = getattr(payload, attr, None)
        if u is not None:
            tot = getattr(u, "total_tokens", None)
            if tot:
                return int(tot)
            pt = getattr(u, "prompt_tokens", 0) or 0
            ct = getattr(u, "completion_tokens", 0) or 0
            if pt or ct:
                return int(pt) + int(ct)
    return 0


def build(collector: list[Any], total_ms: int) -> dict[str, Any]:
    try:
        opens: dict[Any, tuple[str, str, float]] = {}
        steps: list[dict[str, Any]] = []
        for item in collector:
            p = _payload(item)
            uid = getattr(p, "UUID", None) or getattr(p, "uuid", None)
            state = str(getattr(p, "event_state", "")).upper()
            etype = str(getattr(p, "event_type", ""))
            name = getattr(p, "name", None) or etype or "step"
            ts = getattr(p, "event_timestamp", None)
            if "START" in state:
                opens[uid] = (name, etype, ts)
            elif "END" in state and uid in opens:
                nm, et, t0 = opens.pop(uid)
                ms = round((ts - t0) * 1000) if (ts and t0) else None
                steps.append({"name": nm, "type": et.split(".")[-1].lower(), "ms": ms, "tokens": _tokens(p)})
        out: dict[str, Any] = {"total_ms": total_ms}
        if steps:
            out["steps"] = steps
        return out
    except Exception:
        return {"total_ms": total_ms}
