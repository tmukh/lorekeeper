"""Preflight: is Ollama up, and are the required models pulled?"""
from __future__ import annotations

import json
import sys
import urllib.request

from .config import CONFIG


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def check(verbose: bool = True) -> bool:
    host = CONFIG.ollama_host.rstrip("/")
    try:
        tags = _get(f"{host}/api/tags")
    except Exception as e:
        if verbose:
            print(f"[lorekeeper] Ollama not reachable at {host}: {e}", file=sys.stderr)
            print("  -> start it with `ollama serve` (or the desktop app).", file=sys.stderr)
        return False
    have_full = {m.get("name", "") for m in tags.get("models", [])}
    have_base = {n.split(":")[0] for n in have_full}
    ok = True
    for model in (CONFIG.llm_model, CONFIG.embed_model):
        if model not in have_full and model.split(":")[0] not in have_base:
            ok = False
            if verbose:
                print(f"[lorekeeper] missing model: {model}  ->  ollama pull {model}", file=sys.stderr)
    if ok and verbose:
        print(f"[lorekeeper] Ollama OK at {host}; required models present.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
