"""Plain REST facade for the Obsidian plugin.

  GET  /health                       -> {ok, model, vault, indexer}
  POST /query   {question, mode?}    -> {status, answer, sources:[{title,path}]}
                                        status: full | partial | topic_only | none
  POST /suggest {draft_text, title?} -> {result, plan:{...}}
  POST /reindex                      -> {indexed}
  POST /shutdown                     -> {ok}  (exits this process; used by the plugin's Restart)
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
    r = await indexer.answer(question, (body.get("mode") or "hybrid"), debug=bool(body.get("debug")))
    return JSONResponse(r)  # {status, answer, sources}


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


async def _shutdown(request):
    """Exit this server process. The plugin calls this on Restart/unload because on
    Windows killing the venv's python.exe launcher leaves the real interpreter
    running, still serving the OLD code on the port."""
    import asyncio
    import os

    asyncio.get_running_loop().call_later(0.3, os._exit, 0)
    return JSONResponse({"ok": True})


def build_app() -> Starlette:
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/query", _query, methods=["POST"]),
        Route("/suggest", _suggest, methods=["POST"]),
        Route("/reindex", _reindex, methods=["POST"]),
        Route("/shutdown", _shutdown, methods=["POST"]),
    ]
    middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]
    return Starlette(routes=routes, middleware=middleware)


def run_rest(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    uvicorn.run(build_app(), host=host, port=port, log_level="info")
