"""Translate a RunRequest into a NAT workflow config (the YAML `nat` consumes).

Schema mirrors NVIDIA's examples: top-level `function_groups` / `functions` /
`llms` / `workflow`, an OpenAI-compatible LLM, a `react_agent` workflow, and
`mcp_client` function groups for admin-connected MCP servers.
"""
import re
from typing import Any

from .schemas import RunRequest

# built-in tool id -> (config section, config entry).
SUPPORTED_TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "calculator": ("function_groups", {"_type": "calculator"}),
    "current_datetime": ("functions", {"_type": "current_datetime"}),
}


def _safe_name(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_]", "_", (name or "mcp").lower()).strip("_") or "mcp"
    n, i = base, 1
    while n in taken:
        i += 1
        n = f"{base}_{i}"
    taken.add(n)
    return n


def build_config(req: RunRequest) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return (config_dict, wired_tool_names, unsupported_tool_ids)."""
    llm: dict[str, Any] = {"_type": "openai", "model_name": req.model, "temperature": req.temperature}
    if req.base_url:
        llm["base_url"] = req.base_url
    if req.api_key:
        llm["api_key"] = req.api_key

    function_groups: dict[str, Any] = {}
    functions: dict[str, Any] = {}
    tool_names: list[str] = []
    unsupported: list[str] = []

    for t in req.tools:
        entry = SUPPORTED_TOOLS.get(t)
        if entry is None:
            unsupported.append(t)
            continue
        section, cfg = entry
        (function_groups if section == "function_groups" else functions)[t] = cfg
        tool_names.append(t)

    # MCP servers -> mcp_client function groups.
    taken = set(tool_names)
    for s in req.mcp_servers:
        server: dict[str, Any] = {"transport": s.transport}
        if s.transport == "stdio":
            if s.command:
                server["command"] = s.command
            if s.args:
                server["args"] = s.args
            if s.env:
                server["env"] = s.env
        else:
            if s.url:
                server["url"] = s.url
            if s.headers:
                server["headers"] = s.headers
        name = _safe_name(s.name, taken)
        function_groups[name] = {"_type": "mcp_client", "server": server}
        tool_names.append(name)

    config: dict[str, Any] = {
        "llms": {"app_llm": llm},
        "workflow": {"_type": "react_agent", "tool_names": tool_names, "llm_name": "app_llm", "verbose": True},
    }
    if function_groups:
        config["function_groups"] = function_groups
    if functions:
        config["functions"] = functions
    return config, tool_names, unsupported
