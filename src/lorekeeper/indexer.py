"""Thin wrapper around LightRAG: build/update the graph, retrieve context, and
generate grounded answers.

Design note (grounding): LightRAG's built-in answer prompt pushes the model to
"generate a comprehensive answer", which a small local model obeys by padding
with its own training knowledge. So we use LightRAG only for RETRIEVAL
(aquery_data) and generate the answer ourselves with a strict prompt we fully
control. If the retrieved notes don't contain the answer, we return a fixed
"not in your notes yet" and show no sources.

LightRAG import paths move between versions; if an import fails, `pip show
lightrag-hku` and adjust ONLY this file.
"""
from __future__ import annotations

import inspect
from typing import Iterable

from .adapters.base import Document
from .config import CONFIG

_rag = None  # lazily-initialised singleton

NOT_IN_NOTES = "That's not in your notes yet."

STRICT_ANSWER = (
    "You answer strictly from the user's personal notes, provided below as NOTES. "
    "Use ONLY the NOTES. Do not use outside or general knowledge, and do not add facts, "
    "list items, or examples that are not in the NOTES, even if you know them to be true. "
    "If the NOTES do not actually contain the answer, reply with EXACTLY this and nothing "
    f"else: {NOT_IN_NOTES} "
    "Otherwise write a direct, well-structured prose answer. Do not include citations, "
    "reference ids, or a sources list (sources are shown separately)."
)


async def _build_rag():
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts):
        safe = [t if (t and t.strip()) else " " for t in texts]
        vecs = await ollama_embed.func(safe, embed_model=CONFIG.embed_model, host=CONFIG.ollama_host)
        return np.asarray(vecs, dtype=np.float32)

    CONFIG.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(CONFIG.working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=CONFIG.llm_model,
        llm_model_kwargs={"host": CONFIG.ollama_host, "options": {"num_ctx": CONFIG.num_ctx}},
        embedding_func=EmbeddingFunc(
            embedding_dim=CONFIG.embed_dim,
            max_token_size=8192,
            func=_embed,
        ),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass
    return rag


async def get_rag():
    global _rag
    if _rag is None:
        _rag = await _build_rag()
    return _rag


def _ainsert_kwargs(rag, doc) -> dict:
    """Pass ids AND file_paths when supported. file_paths is what gives chunks a
    real source for citation — without it LightRAG stores 'unknown_source'."""
    try:
        params = inspect.signature(rag.ainsert).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if "ids" in params:
        kw["ids"] = [doc.doc_id]
    if "file_paths" in params:
        kw["file_paths"] = [doc.doc_id]
    return kw


async def index_documents(docs: Iterable[Document]) -> int:
    if not CONFIG.is_indexer:
        raise RuntimeError(
            "This machine is not the indexer (LORE_INDEXER=false). "
            "Indexing runs on the desktop/GPU box; the laptop queries the committed graph."
        )
    rag = await get_rag()
    n = 0
    for doc in docs:
        await rag.ainsert(doc.text, **_ainsert_kwargs(rag, doc))
        n += 1
    return n


async def complete(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    """Direct, non-RAG chat completion against the local Ollama model."""
    import ollama

    client = ollama.AsyncClient(host=CONFIG.ollama_host)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat(
        model=CONFIG.llm_model,
        messages=messages,
        options={"num_ctx": CONFIG.num_ctx, "temperature": temperature},
    )
    message = resp["message"] if isinstance(resp, dict) else resp.message
    content = message["content"] if isinstance(message, dict) else message.content
    return content or ""


def _ref_path(item) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("file_path", "reference", "source", "doc_id", "source_id", "id"):
            v = item.get(k)
            if isinstance(v, str) and v:
                return v
    return None


async def _retrieve(question: str, mode: str = "hybrid"):
    """Use LightRAG for graph-aware retrieval only. Returns (context_text, chunks)."""
    from lightrag import QueryParam

    rag = await get_rag()
    data = await rag.aquery_data(question, param=QueryParam(mode=mode))
    payload = data.get("data", data) if isinstance(data, dict) else {}
    chunks = payload.get("chunks") or [] if isinstance(payload, dict) else []
    parts = []
    for c in chunks:
        if isinstance(c, dict):
            t = (c.get("content") or "").strip()
            if t:
                parts.append(t)
    return "\n\n---\n\n".join(parts), chunks


def _sources_from_chunks(chunks) -> list[dict]:
    sources, seen = [], set()
    for c in chunks:
        fp = _ref_path(c)
        if not fp or fp == "unknown_source" or fp in seen:
            continue
        seen.add(fp)
        title = fp.rsplit("/", 1)[-1]
        if title.endswith(".md"):
            title = title[:-3]
        sources.append({"title": title, "path": fp})
        if len(sources) >= 8:
            break
    return sources


async def query_with_sources(question: str, mode: str = "hybrid", strict: bool = True):
    """Return (answer, sources). Retrieval via LightRAG; generation by us, strict."""
    try:
        context, chunks = await _retrieve(question, mode)
    except Exception:
        context, chunks = "", []

    if strict and not context.strip():
        return NOT_IN_NOTES, []

    if strict:
        answer = await complete(f"NOTES:\n{context[:8000]}\n\nQUESTION: {question}", system=STRICT_ANSWER)
    else:
        from lightrag import QueryParam

        rag = await get_rag()
        answer = await rag.aquery(question, param=QueryParam(mode=mode))

    sources = _sources_from_chunks(chunks)
    if answer.strip().lower().startswith("that's not in your notes"):
        sources = []
    return answer, sources


async def query(question: str, mode: str = "hybrid", strict: bool = True) -> str:
    answer, _ = await query_with_sources(question, mode=mode, strict=strict)
    return answer
