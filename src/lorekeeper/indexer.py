"""Thin wrapper around LightRAG: build/update the graph, run graph queries,
and a direct LLM completion for structured tasks (placement).

LightRAG import paths move between versions; if an import fails, `pip show
lightrag-hku` and adjust ONLY this file.

Embedding note: LightRAG's bundled `ollama_embed` is decorated for bge-m3
(1024-dim), so we call the unwrapped `.func` and declare our own 768 dim for
nomic-embed-text, and pad empty chunks Ollama would otherwise drop.
"""
from __future__ import annotations

import inspect
from typing import Iterable

from .adapters.base import Document
from .config import CONFIG

_rag = None  # lazily-initialised singleton


async def _build_rag():
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts):
        safe = [t if (t and t.strip()) else " " for t in texts]
        vecs = await ollama_embed.func(
            safe, embed_model=CONFIG.embed_model, host=CONFIG.ollama_host
        )
        return np.asarray(vecs, dtype=np.float32)

    CONFIG.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(CONFIG.working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=CONFIG.llm_model,
        llm_model_kwargs={
            "host": CONFIG.ollama_host,
            "options": {"num_ctx": CONFIG.num_ctx},
        },
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


def _supports_ids(rag) -> bool:
    try:
        return "ids" in inspect.signature(rag.ainsert).parameters
    except (TypeError, ValueError):
        return False


async def index_documents(docs: Iterable[Document]) -> int:
    if not CONFIG.is_indexer:
        raise RuntimeError(
            "This machine is not the indexer (LORE_INDEXER=false). "
            "Indexing runs on the desktop/GPU box; the laptop queries the committed graph."
        )
    rag = await get_rag()
    use_ids = _supports_ids(rag)
    n = 0
    for doc in docs:
        if use_ids:
            await rag.ainsert(doc.text, ids=[doc.doc_id])
        else:
            await rag.ainsert(doc.text)
        n += 1
    return n


STRICT_GROUNDING = (
    "Answer ONLY using the information in the provided Knowledge Base context. "
    "Do not use any outside or general knowledge, and do not add background the "
    "context does not contain. If the answer is not in the context, reply with "
    "exactly: That's not in your notes yet. Prefer citing the note titles you drew from."
)


async def query(question: str, mode: str = "hybrid", strict: bool = True) -> str:
    from lightrag import QueryParam

    rag = await get_rag()
    param = QueryParam(mode=mode, user_prompt=STRICT_GROUNDING if strict else None)
    return await rag.aquery(question, param=param)


async def complete(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    """Direct, non-RAG chat completion against the local Ollama model.

    Used for structured tasks (placement) where we supply the context
    ourselves rather than retrieving it from the graph.
    """
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


async def query_with_sources(question: str, mode: str = "hybrid", strict: bool = True):
    """Return (answer, sources) where sources is a list of {title, path} of the
    notes the retrieval drew from."""
    from lightrag import QueryParam

    rag = await get_rag()
    answer = await rag.aquery(
        question, param=QueryParam(mode=mode, user_prompt=STRICT_GROUNDING if strict else None)
    )
    sources: list[dict] = []
    try:
        data = await rag.aquery_data(question, param=QueryParam(mode=mode))
        items = []
        if isinstance(data, dict):
            items = data.get("references") or data.get("chunks") or []
        seen = set()
        for it in items:
            fp = _ref_path(it)
            if not fp or fp == "unknown_source" or fp in seen:
                continue
            seen.add(fp)
            title = fp.rsplit("/", 1)[-1]
            if title.endswith(".md"):
                title = title[:-3]
            sources.append({"title": title, "path": fp})
    except Exception:
        pass
    return answer, sources
