"""lorekeeper MCP server — exposes the knowledge graph as tools (mcp 2.x API).

Transports:
  - stdio (default): Claude Code / Claude Desktop.
  - streamable-http: the Obsidian plugin (localhost:8765).

Run:
  python -m lorekeeper.server          # stdio
  python -m lorekeeper.server --http   # http on 127.0.0.1:8765
"""
from __future__ import annotations

import argparse

from mcp.server import MCPServer

from . import indexer, placement
from .adapters.obsidian import ObsidianAdapter
from .config import CONFIG

mcp = MCPServer("lorekeeper")


@mcp.tool()
async def query_kb(question: str, mode: str = "hybrid") -> str:
    """Answer a question grounded in the knowledge graph.

    mode: 'local' (specific entities), 'global' (themes/summaries),
    'hybrid' (default, best all-round), or 'naive' (plain vector RAG).
    """
    return await indexer.query(question, mode=mode)


@mcp.tool()
async def related_notes(text: str, k: int = 10) -> str:
    """Find notes related to the given text/topic by graph + embedding proximity."""
    return await indexer.query(
        f"List up to {k} notes most related to the following, each with a one-line reason:\n\n{text}",
        mode="local",
    )


@mcp.tool()
async def suggest_placement(draft_text: str, title: str = "") -> str:
    """Decide where a draft note belongs: folder (existing or a new subfolder),
    kebab-case filename, matching template, and links grounded in real notes."""
    return await placement.suggest(draft_text, title)


@mcp.tool()
async def reindex() -> str:
    """Re-index the vault into the graph (incremental; indexer machine only)."""
    adapter = ObsidianAdapter(CONFIG.vault_path)
    n = await indexer.index_documents(adapter.documents())
    return f"Indexed/updated {n} notes from {CONFIG.vault_path}."


@mcp.tool()
def graph_stats() -> str:
    """Quick health view of the graph store (files present and their sizes)."""
    wd = CONFIG.working_dir
    if not wd.exists():
        return "No graph store yet. Run `python scripts/reindex.py` on the indexer machine first."
    lines = [f"Store: {wd}", f"Indexer: {CONFIG.is_indexer}", f"Model: {CONFIG.llm_model}"]
    for f in sorted(wd.glob("*")):
        if f.is_file():
            lines.append(f"  {f.name}: {f.stat().st_size / 1024:.1f} KiB")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="lorekeeper MCP server")
    ap.add_argument("--http", action="store_true", help="serve streamable-http on 127.0.0.1:8765")
    args = ap.parse_args()
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
