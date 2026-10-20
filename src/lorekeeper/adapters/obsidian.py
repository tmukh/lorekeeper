"""Obsidian vault adapter: walks .md files, strips YAML frontmatter, and
harvests [[wikilinks]] and #tags as high-confidence seed edges for the graph.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import Document

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
TAG = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-/]+)")
FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

IGNORE_DIRS = {".obsidian", ".git", ".smart-env", "Attachments", "Templates"}


class ObsidianAdapter:
    name = "obsidian"

    def __init__(self, vault_path: Path):
        self.vault = Path(vault_path)

    def documents(self) -> Iterator[Document]:
        for md in self.vault.rglob("*.md"):
            if any(part in IGNORE_DIRS for part in md.parts):
                continue
            rel = md.relative_to(self.vault).as_posix()
            raw = md.read_text(encoding="utf-8", errors="ignore")
            body = FRONTMATTER.sub("", raw, count=1)
            title = md.stem
            links = WIKILINK.findall(body)
            tags = TAG.findall(body)
            yield Document(
                doc_id=rel,
                text=f"# {title}\n\n{body}".strip(),
                metadata={"path": rel, "title": title, "tags": tags},
                seed_edges=[(title, dst.strip()) for dst in links],
            )
