"""Standalone (re)index entrypoint — run on the desktop/indexer machine."""
import asyncio

from lorekeeper import indexer
from lorekeeper.adapters.obsidian import ObsidianAdapter
from lorekeeper.config import CONFIG
from lorekeeper.ollama_check import check


async def main():
    if not check():
        raise SystemExit(1)
    n = await indexer.index_documents(ObsidianAdapter(CONFIG.vault_path).documents())
    print(f"Indexed/updated {n} notes from {CONFIG.vault_path}")


if __name__ == "__main__":
    asyncio.run(main())
