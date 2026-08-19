# NAT requires Python 3.11-3.13 (not 3.14). Pin 3.12.
FROM python:3.12-slim

WORKDIR /app

# System deps: build tools for NAT wheels, plus Node.js + uv so MCP stdio servers
# can spawn inside this runtime (npx for @modelcontextprotocol/* ; uvx for the
# Python servers like mcp-server-fetch / postgres-mcp).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# uv / uvx (for the Python-based MCP servers)
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm the free MCP servers so first use doesn't download at cold start.
# Best-effort (|| true) so a registry hiccup or renamed community package never
# breaks the image build — uvx/npx will fetch on demand if a warm is missing.
RUN npm install -g @modelcontextprotocol/server-memory @modelcontextprotocol/server-sequential-thinking || true
RUN for p in mcp-server-fetch mcp-server-time duckduckgo-mcp-server wikipedia-mcp arxiv-mcp-server postgres-mcp; do uv tool install "$p" || true; done
# make uv-installed tools available on PATH for uvx
ENV PATH="/root/.local/bin:${PATH}"

COPY app ./app

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
