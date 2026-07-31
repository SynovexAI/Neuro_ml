"""Translate a RunRequest into a NAT workflow config (the YAML `nat` consumes).

Schema mirrors NVIDIA's examples: top-level `function_groups` / `functions` /
`llms` / `workflow`, an OpenAI-compatible LLM, and a `react_agent` workflow.
"""
from typing import Any

from .schemas import RunRequest

# tool id -> (config section, config entry). Spike scope: the two tools NAT
# ships in core. http / knowledge(RAG) / MCP are wired in later phases.
SUPPORTED_TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "calculator": ("function_groups", {"_type": "calculator"}),
    "current_datetime": ("functions", {"_type": "current_datetime"}),
}


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

    config: dict[str, Any] = {
        "llms": {"app_llm": llm},
        "workflow": {
            "_type": "react_agent",
            "tool_names": tool_names,
            "llm_name": "app_llm",
            "verbose": True,
        },
    }
    if function_groups:
        config["function_groups"] = function_groups
    if functions:
        config["functions"] = functions
    return config, tool_names, unsupported
