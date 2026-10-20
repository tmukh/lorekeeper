# DESIGN — lorekeeper (local GraphRAG knowledge base)

> **Status:** Accepted · **Date:** 2026-10-20 · **Author:** Tarik Abu Mukh
> **Repo:** `D:\dev\lorekeeper` (engine/code) — reads `D:\dev\obsidian-vault` (data)

## 1. Context & goals
A fully-local, free second brain that understands notes, not just stores them: it
consolidates the Obsidian vault into a **knowledge graph**, clusters it into **subgraphs
(communities)**, and lets a **local LLM** browse and reason over it, exposed as an **MCP
server** reusable by Claude Code, Claude Desktop, and a small in-Obsidian panel.

### Non-negotiables
1. Browse the KB with a local LLM — no cloud, no API keys, no paywall.
2. Graph + catalog the knowledge (entities/relations/subgraphs), not a flat list.
3. Auto-placement: "I want to write about X" -> suggest the folder + links.

### Future (designed-for, not built)
Extend ingestion from the vault to all of `D:\dev` (code + learning-ledger), so the graph
captures what was learned building each project. Obsidian is just the first source adapter.

## 2. Non-goals (for now)
Cloud / multi-user / sharing; fine-tuning; replacing Obsidian's UI.

## 3. Constraints
- Desktop RTX 3060 Ti, **8 GB VRAM** confirmed (~5.6 GB free with display at idle).
- Laptop: assume no/weak GPU -> it queries, never indexes.
- Fully local & free (Ollama). Two machines synced via git. Windows-first.

## 4. Decision summary
| Layer | Choice | Why |
|---|---|---|
| Graph/RAG engine | **LightRAG** | Local via Ollama; **incremental** updates (no full rebuild); JSON+GraphML store; local/global/hybrid query |
| LLM runtime | Ollama (already installed) | Local, free, OpenAI-compatible |
| Generation | llama3.1:8b (Q4) | Fits 8 GB; strong enough for entity extraction (1B halts) |
| Embeddings | nomic-embed-text | 768-dim, ~0.3 GB |
| Interface | MCP server (Python, FastMCP) | Framework-independent; one server, many clients |
| In-app UI | Obsidian plugin (TS) -> MCP over localhost http | Chat + "suggest placement" |

### Alternatives rejected
- **Microsoft GraphRAG**: rebuilds the whole graph on every change — fatal for a daily vault.
- **Cognee (MCP)**: batteries-included KG memory, but heavier/opinionated and "not built by me". Fallback if LightRAG extraction quality disappoints.
- **Graphiti/Zep (MCP)**: temporal KG — more than a note vault needs.
- **Obsidian-native (links/tags as graph)**: doesn't build its own graph (the point). We still SEED LightRAG with existing `[[links]]`/tags as cheap high-confidence edges.

## 5. Architecture
```
vault (future: D:\dev\**) -> source adapters -> LightRAG indexer (Ollama) -> graph store
                                                                                  |
                                                     MCP server (query_kb, suggest_placement, ...)
                                                                                  |
                                       Claude Code  |  Claude Desktop  |  Obsidian panel
```

## 6. MCP tools (v1)
`query_kb(question, mode)` · `suggest_placement(draft, title)` · `related_notes(text, k)` ·
`graph_stats()` · `reindex(paths?)`.

## 7. Index sync across two machines  (the one hard problem)
The graph store is derived. **Decision: desktop indexes and COMMITS the store to git**;
the laptop pulls it and only queries (smaller model / retrieval-only). `LORE_INDEXER`
gates writes. `.gitignore` excludes model blobs + caches; the JSON/GraphML store IS tracked.

## 8. Milestones
M0 skeleton · **M1 index+query (here)** · M2 placement · M3 Obsidian panel · M4 code+ledger adapter.

## 9. Risks & tradeoffs
- Indexing cost on 8 GB -> keep num_ctx <= 6k; incremental after first build; desktop-only.
- Local extraction quality (8B) -> seed with wikilinks/tags; allow re-extract.
- Graph drift -> reindex on save (debounced); LightRAG dedup/merge.
- Laptop generation -> small model or retrieval-only.

## 10. Open questions
Native Windows vs WSL for the service (MVP: native Windows); laptop query model size;
whether to fold in the existing `D:\dev\obsidian-graph` folder.
