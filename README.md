# lorekeeper

A **local GraphRAG MCP server** over an Obsidian vault. It reads your notes, has a local
LLM extract entities + relationships into a **knowledge graph**, clusters that graph into
communities, and exposes it over the **Model Context Protocol** — so Claude Code, Claude
Desktop, and a small Obsidian panel can all *reason over* your knowledge. Fully offline,
no API keys, no paywall.

Built on [LightRAG](https://github.com/HKUDS/LightRAG) + [Ollama](https://ollama.com).

## Why
Flat note apps store text; they don't understand it. lorekeeper builds a graph of what you
know, answers questions by walking it, and can tell you **where a new note belongs** and
what to link it to.

## Status
MVP (M1 — index + query). See [`docs/DESIGN.md`](docs/DESIGN.md) for architecture,
tradeoffs, and roadmap.

## Hardware note (why these model choices)
Target: RTX 3060 Ti, **8 GB VRAM** (~5.6 GB free with the desktop running). `llama3.1:8b`
(Q4, ~4.9 GB) + `nomic-embed-text` (~0.3 GB) just fit. Keep `LORE_NUM_CTX` <= 6144; if VRAM
spills, Ollama offloads layers to CPU (slower, never fatal). Close the browser during the
first full index to reclaim ~1 GB.

## Setup (desktop / indexer)
```powershell
# 1. models (one-time per machine)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. install
cd D:\dev\lorekeeper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env     # edit if needed

# 3. preflight + first index
python -m lorekeeper.ollama_check
python scripts\reindex.py
```

## Use from Claude Code / Claude Desktop (stdio)
Add to your MCP client config:
```json
{ "mcpServers": { "lorekeeper": {
    "command": "D:\\dev\\lorekeeper\\.venv\\Scripts\\python.exe",
    "args": ["-m", "lorekeeper.server"]
} } }
```

## Use from the Obsidian panel (http)
```powershell
python -m lorekeeper.server --http      # serves 127.0.0.1:8765
```
Then enable the plugin in `obsidian-plugin/` (see its README — M3).

## Laptop (query-only)
Set `LORE_INDEXER=false` in `.env`, `git pull` to get the committed graph store, and query
with a smaller model (`LORE_LLM_MODEL=llama3.2:3b`) or retrieval-only.

## Tools exposed
`query_kb` · `related_notes` · `suggest_placement` · `reindex` · `graph_stats`

## First git push
```powershell
cd D:\dev\lorekeeper
git init && git add . && git commit -m "feat: lorekeeper MVP scaffold"
# create the repo on GitHub, then:
git remote add origin git@github.com:<you>/lorekeeper.git
git push -u origin main
```

## Roadmap
- **M2** — sharpen `suggest_placement` with real embedding-nearest + folder stats.
- **M3** — Obsidian chat panel + "suggest placement for this note" command.
- **M4** — index all of `D:\dev` (code + `learning-ledger`) via a `code` adapter, so the
  graph also captures what you learned building each project.
