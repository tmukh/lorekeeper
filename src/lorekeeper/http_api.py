"""Plain REST facade for the Obsidian plugin.

The plugin is a browser-like client, so instead of making it speak MCP we
expose the same placement/query capabilities as simple JSON endpoints on
localhost. Reuses uvicorn+starlette (already pulled in by mcp[cli]).

Endpoints:
  GET  /health                      -> {ok, model, vault, indexer}
  POST /query   {question, mode?}   -> {answer}
  POST /suggest {draft_text, title?}-> {result}
"""
from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import indexer, placement
from .config import CONFIG


async def _health(request):
    return JSONResponse(
        {"ok": True, "model": CONFIG.llm_model, "vault": str(CONFIG.vault_path), "indexer": CONFIG.is_indexer}
    )


async def _query(request):
    body = await request.json()
    question = (body or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "missing 'question'"}, status_code=400)
    answer = await indexer.query(question, (body.get("mode") or "hybrid"))
    return JSONResponse({"answer": answer})


async def _suggest(request):
    body = await request.json()
    draft = (body or {}).get("draft_text", "").strip()
    if not draft:
        return JSONResponse({"error": "missing 'draft_text'"}, status_code=400)
    result = await placement.suggest(draft, (body or {}).get("title", ""))
    return JSONResponse({"result": result})


def build_app() -> Starlette:
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/query", _query, methods=["POST"]),
        Route("/suggest", _suggest, methods=["POST"]),
    ]
    # localhost-only, but permissive CORS so the Obsidian renderer is never blocked.
    middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]
    return Starlette(routes=routes, middleware=middleware)


def run_rest(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    uvicorn.run(build_app(), host=host, port=port, log_level="info")
