"""Request/response contract between the Next.js app and this NAT service."""
from pydantic import BaseModel, Field


class McpServerCfg(BaseModel):
    """A resolved MCP server config (the Next proxy already decrypted secrets and
    built the headers/env, so this service just maps it into the NAT config)."""
    name: str
    transport: str = "streamable-http"   # streamable-http | sse | stdio
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    headers: dict[str, str] | None = None


class KnowledgeCfg(BaseModel):
    docs: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    task: str = Field(description="The user task / question for the agent.")
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    system_prompt: str | None = None
    agent_type: str = "react_agent"                         # react_agent | tool_calling_agent
    tools: list[str] = Field(default_factory=list)          # built-in tool ids
    mcp_servers: list[McpServerCfg] = Field(default_factory=list)
    knowledge: KnowledgeCfg | None = None


class RunResponse(BaseModel):
    answer: str
    latency_ms: int
    model: str
    tool_names: list[str]
    unsupported_tools: list[str] = Field(default_factory=list)
    context_used: bool = False        # true if RAG context was injected
    # { total_ms, steps: [{ name, type, ms, tokens }] } — steps best-effort.
    profiler: dict = Field(default_factory=dict)
