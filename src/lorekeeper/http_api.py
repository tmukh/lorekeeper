"""Plain REST facade for the Obsidian plugin.

  GET  /health                       -> {ok, model, vault, indexer}
  POST /query   {question, mode?}    -> {answer, sources:[{title,path}]}
  POST /suggest {draft_text, title?} -> {result, plan:{...}}
  POST /reindex                      -> {indexed}
"""
from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import indexer, placement
from .adapters.obsidian import ObsidianAdapter
from .config import CONFIG


async def _health(request):
    return JSONResponse({"ok": True, "model": CONFIG.llm_model, "vault": str(CONFIG.vault_path), "indexer": CONFIG.is_indexer})


async def _query(request):
    body = await request.json()
    question = (body or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "missing 'question'"}, status_code=400)
    answer, sources = await indexer.query_with_sources(question, (body.get("mode") or "hybrid"))
    return JSONResponse({"answer": answer, "sources": sources})


async def _suggest(request):
    body = await request.json()
    draft = (body or {}).get("draft_text", "").strip()
    if not draft:
        return JSONResponse({"error": "missing 'draft_text'"}, status_code=400)
    plan = await placement.suggest_data(draft, (body or {}).get("title", ""))
    return JSONResponse({"result": placement.format_plan(plan), "plan": plan})


async def _reindex(request):
    n = await indexer.index_documents(ObsidianAdapter(CONFIG.vault_path).documents())
    return JSONResponse({"indexed": n})


def build_app() -> Starlette:
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/query", _query, methods=["POST"]),
        Route("/suggest", _suggest, methods=["POST"]),
        Route("/reindex", _reindex, methods=["POST"]),
    ]
    middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]
    return Starlette(routes=routes, middleware=middleware)


def run_rest(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    uvicorn.run(build_app(), host=host, port=port, log_level="info")
