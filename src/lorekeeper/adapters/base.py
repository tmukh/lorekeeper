"""Source-agnostic ingestion interface.

An adapter turns some source (an Obsidian vault today; a code tree tomorrow)
into a stream of Documents the indexer feeds to LightRAG. Adding a new source
= adding a new adapter, NOT changing the engine. This is the seam that lets
lorekeeper grow from the vault to all of D:\\dev later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Document:
    doc_id: str  # stable id (e.g. vault-relative path)
    text: str  # plain text fed to the graph extractor
    metadata: dict = field(default_factory=dict)
    seed_edges: list[tuple[str, str]] = field(default_factory=list)  # (src_title, dst_title) from [[links]]


class SourceAdapter(Protocol):
    name: str

    def documents(self) -> Iterator[Document]:
        ...
