"""Run a generated NAT config and return the answer, timing, and profiler.

Uses NAT's documented programmatic entry point:
    async with load_workflow(config_path) as session_manager:
        async with session_manager.run(message) as runner:
            answer = await runner.result(to_type=str)
"""
import os
import tempfile
import time
from typing import Any

import yaml

from nat.runtime.loader import load_workflow

from . import profiler
from . import tools  # noqa: F401 — importing registers our custom calculator group


async def run_workflow(config: dict[str, Any], input_message: str) -> dict[str, Any]:
    start = time.perf_counter()
    collector: list[Any] = []
    # NAT loads config from a file. The provider key lives in this temp file only
    # for the duration of the run, then it's deleted.
    fd, path = tempfile.mkstemp(suffix=".yml", prefix="nat_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        async with load_workflow(path) as session_manager:
            async with session_manager.run(input_message) as runner:
                profiler.subscribe(collector)  # best-effort per-step capture
                answer = await runner.result(to_type=str)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    total_ms = round((time.perf_counter() - start) * 1000)
    return {"answer": answer, "latency_ms": total_ms, "profiler": profiler.build(collector, total_ms)}
