"""Central configuration, loaded from environment / .env.

The desktop (GPU) is the indexer; the laptop is query-only. One flag
(LORE_INDEXER) decides whether a machine may write the graph.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_host(h: str) -> str:
    """Make an Ollama host string safe to CONNECT to.

    Ollama commonly exports OLLAMA_HOST=0.0.0.0:11434 (a *listen* address with
    no scheme). For connecting we need a scheme, and 0.0.0.0 must become a
    loopback address.
    """
    h = (h or "").strip()
    if "://" not in h:
        h = "http://" + h
    h = h.replace("://0.0.0.0", "://127.0.0.1").replace("://[::]", "://127.0.0.1")
    return h.rstrip("/")


@dataclass(frozen=True)
class Config:
    vault_path: Path
    working_dir: Path  # where LightRAG writes the graph store
    ollama_host: str
    llm_model: str
    embed_model: str
    embed_dim: int
    num_ctx: int
    is_indexer: bool  # may this machine write the graph?

    @staticmethod
    def load() -> "Config":
        root = Path(__file__).resolve().parents[2]
        return Config(
            vault_path=Path(os.getenv("VAULT_PATH", r"D:\dev\obsidian-vault")),
            working_dir=Path(os.getenv("LORE_WORKING_DIR", str(root / "store"))),
            ollama_host=_normalize_host(os.getenv("OLLAMA_HOST", "http://localhost:11434")),
            llm_model=os.getenv("LORE_LLM_MODEL", "llama3.1:8b"),
            embed_model=os.getenv("LORE_EMBED_MODEL", "nomic-embed-text"),
            embed_dim=int(os.getenv("LORE_EMBED_DIM", "768")),
            num_ctx=int(os.getenv("LORE_NUM_CTX", "6144")),
            is_indexer=_bool(os.getenv("LORE_INDEXER"), default=True),
        )


CONFIG = Config.load()
