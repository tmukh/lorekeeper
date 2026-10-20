"""Thin wrapper around LightRAG: build/update the graph from any adapter,
and run queries.

LightRAG's import paths move between versions. If an import here fails, run
`pip show lightrag-hku`, check its README for your version, and adjust ONLY
this file — it is the single point of contact with the engine.

NOTE on embeddings (the one gotcha): LightRAG's bundled `ollama_embed` is
decorated with `@wrap_embedding_func_with_attrs(embedding_dim=1024)` because
its default model is bge-m3 (1024-dim). We use `nomic-embed-text` (768-dim),
so we call the UNWRAPPED `ollama_embed.func` and declare 768 on our own
EmbeddingFunc — otherwise LightRAG validates 768 vectors against 1024 and
every flush fails. We also pad empty/whitespace chunks, which Ollama would
otherwise silently drop (breaking the 1:1 text->vector count).
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
        # .func = the undecorated function (no 1024/bge-m3 wrapper).
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
            embedding_dim=CONFIG.embed_dim,  # 768 for nomic-embed-text
            max_token_size=8192,
            func=_embed,
        ),
    )
    await rag.initialize_storages()
    # Newer LightRAG also requires initialising the pipeline-status store:
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
