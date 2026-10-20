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


async def query(question: str, mode: str = "hybrid") -> str:
    from lightrag import QueryParam

    rag = await get_rag()
    return await rag.aquery(question, param=QueryParam(mode=mode))


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
