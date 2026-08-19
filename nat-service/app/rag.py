"""Lightweight in-memory retrieval (TF-IDF over word overlap) used to ground a
NAT run in admin/student-supplied documents — no external embedder needed.

Docs are chunked, scored against the task, and the top chunks are injected as
context ahead of the task. This is the 'RAG ingest' step done in-process.
"""
import math
import re


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _chunks(text: str, size: int = 60, overlap: int = 12) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return out


def retrieve(docs: list[str], query: str, k: int = 3) -> list[str]:
    chunks: list[str] = []
    for d in docs:
        chunks.extend(_chunks(d))
    if not chunks:
        return []
    n = len(chunks)
    tokenized = [_tokens(c) for c in chunks]
    df: dict[str, int] = {}
    for toks in tokenized:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    q = _tokens(query)
    scores = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in set(q):
            if t in tf:
                s += (tf[t] / max(1, len(toks))) * math.log(n / (df.get(t, 1)) + 1)
        scores.append(s)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:k] if scores[i] > 0]


def augment(message: str, task: str, docs: list[str]) -> tuple[str, bool]:
    """Return (possibly-augmented message, whether context was injected)."""
    top = retrieve(docs, task, 3)
    if not top:
        return message, False
    context = "\n---\n".join(top)
    return f"Use the following context to answer.\n\nContext:\n{context}\n\n{message}", True
