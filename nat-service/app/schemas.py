"""Request/response contract between the Next.js app and this NAT service."""
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(description="The user task / question for the agent.")
    model: str = Field(description="Model name, e.g. 'llama-3.3-70b'.")
    base_url: str | None = Field(default=None, description="OpenAI-compatible base URL of the provider.")
    api_key: str | None = Field(default=None, description="Provider API key (passed per-request, never stored).")
    temperature: float = 0.0
    system_prompt: str | None = None
    # Tool ids the Next app requested. Only supported ones are wired; the rest
    # come back in `unsupported_tools`.
    tools: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    answer: str
    latency_ms: int
    model: str
    tool_names: list[str]
    unsupported_tools: list[str] = Field(default_factory=list)
    # Profiler summary. `total_ms` is always present; per-step data is added in a
    # later phase (intermediate-step subscription).
    profiler: dict = Field(default_factory=dict)
