"""M2: grounded note placement (the "librarian").

Given a draft, decide where it belongs (existing folder, or a new subfolder
when nothing fits), a kebab-case filename, the matching template, and related
links. Facts the model shouldn't own are decided deterministically against the
live vault:
  - folder_exists / is_new: checked against the real folder list.
  - template: validated against the real Templates/ list (else null).
  - links: the model may only choose from a keyword-relevance-filtered
    candidate list, and anything not matching a real note is dropped — so an
    unrelated draft yields zero links instead of padded noise.
Everything is read from disk each call, so it always reflects the current vault.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import indexer
from .config import CONFIG

IGNORE_DIRS = {".obsidian", ".git", ".smart-env", "Attachments"}
FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
LINK_INNER = re.compile(r"\[\[([^\]|#]+)")
_WORD = re.compile(r"[a-zA-Z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "is", "are", "it",
    "its", "as", "by", "at", "this", "that", "be", "was", "were", "from", "into", "than",
    "then", "so", "no", "not", "you", "my", "your", "we", "can", "will", "would", "use",
    "used", "using", "about", "note", "notes", "which", "when", "what", "how", "why",
}


def _tok(s: str) -> set[str]:
    return {w for w in (x.lower() for x in _WORD.findall(s)) if len(w) > 2 and w not in _STOP}


def _purpose(md: Path) -> str:
    try:
        raw = FRONTMATTER.sub("", md.read_text(encoding="utf-8", errors="ignore"), count=1)
    except Exception:
        return ""
    for line in raw.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:120]
    return ""


def snapshot(root: Path) -> dict:
    root = Path(root)
    folders = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() and not any(x in p.parts for x in IGNORE_DIRS):
            idx = p / "_index.md"
            folders.append(
                {"path": p.relative_to(root).as_posix(),
                 "purpose": _purpose(idx) if idx.exists() else ""}
            )
    templates = []
    tdir = root / "Templates"
    if tdir.exists():
        for t in sorted(tdir.glob("*.md")):
            templates.append({"name": t.stem, "purpose": _purpose(t)})
    notes = []
    for md in sorted(root.rglob("*.md")):
        if any(x in md.parts for x in IGNORE_DIRS) or md.name == "_index.md" or "Templates" in md.parts:
            continue
        notes.append({"path": md.relative_to(root).as_posix(), "title": md.stem})
    return {"folders": folders, "templates": templates, "notes": notes[:500]}


def _link_candidates(draft: str, title: str, notes: list[dict], k: int = 15) -> list[dict]:
    dt = _tok(f"{title} {draft}")
    scored = []
    for n in notes:
        score = len(dt & _tok(f"{n['title']} {n['path'].replace('/', ' ')}"))
        if score:
            scored.append((score, n))
    scored.sort(key=lambda x: (-x[0], x[1]["title"]))
    return [n for _, n in scored[:k]]


def _fmt_folders(folders):
    return "\n".join(f"- {f['path']}" + (f"  — {f['purpose']}" if f["purpose"] else "") for f in folders) or "(none)"


def _fmt_templates(templates):
    return "\n".join(f"- {t['name']}" + (f"  — {t['purpose']}" if t["purpose"] else "") for t in templates) or "(none)"


def _fmt_candidates(cands):
    return "\n".join(f"- {n['title']}" for n in cands) or "(none — return [])"


SYSTEM = (
    "You are the librarian for a personal Obsidian vault. You decide exactly where a new "
    "note belongs. You NEVER invent folders, templates, or links beyond the lists you are "
    "given. Reply with ONE JSON object and nothing else."
)


def build_prompt(draft: str, title: str, snap: dict, candidates: list[dict]) -> str:
    return f"""EXISTING FOLDERS (path — purpose):
{_fmt_folders(snap['folders'])}

TEMPLATES (name — purpose) — 'template' MUST be exactly one of these names, or null:
{_fmt_templates(snap['templates'])}

LINK CANDIDATES — choose links ONLY from these, and ONLY if genuinely related.
0–2 links is typical; return [] if none truly fit. Do NOT pad.
{_fmt_candidates(candidates)}

DECISION RULES:
- Prefer an EXISTING folder when one fits. Propose a NEW folder only when none fits.
- A new folder goes UNDER the right top-level section (keep that section's numeric prefix),
  Title-Case name: a study topic under Learning, a language under Languages, a writeup under
  CTF, etc. Go at most one level deeper for a genuine narrower subtopic (Physics/Particle Physics).
- filename: kebab-case, no extension.

NOTE TITLE: {title or '(untitled)'}
DRAFT:
{draft}

Reply with ONLY this JSON object:
{{"folder":"<path where the note should live>","filename":"<kebab-case>","template":"<exact template name or null>","links":["[[Exact Candidate Title]]"],"reasoning":"<one sentence>"}}"""


def _parse(text: str):
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i : j + 1])
    except Exception:
        return None


async def suggest(draft: str, title: str = "") -> str:
    snap = snapshot(CONFIG.vault_path)
    existing_folders = {f["path"] for f in snap["folders"]}
    valid_templates = {t["name"] for t in snap["templates"]}
    candidates = _link_candidates(draft, title, snap["notes"])
    cand_titles = {n["title"].lower() for n in candidates}

    raw = await indexer.complete(build_prompt(draft, title, snap, candidates), system=SYSTEM)
    data = _parse(raw)
    if not data:
        return "Could not parse a placement decision. Raw model output:\n\n" + raw

    folder = str(data.get("folder", "?")).rstrip("/")
    is_new = folder not in existing_folders  # decided by us, not the model

    tmpl = data.get("template")
    if tmpl not in valid_templates:  # validate against real templates
        tmpl = None

    links = []
    for lnk in data.get("links") or []:  # keep only real, candidate-approved links
        m = LINK_INNER.match(str(lnk).strip())
        if m and m.group(1).strip().lower() in cand_titles:
            links.append(f"[[{m.group(1).strip()}]]")

    filename = (str(data.get("filename", "")).strip() or "untitled")
    full = f"{folder}/{filename}.md"
    return "\n".join([
        f"\U0001F4C1 Folder:   {folder}" + ("   (NEW — will be created)" if is_new else "   (exists)"),
        f"\U0001F4C4 File:     {full}",
        f"\U0001F9E9 Template: {tmpl or '(none)'}",
        f"\U0001F517 Links:    {', '.join(links) if links else '(none)'}",
        f"\U0001F4AD Why:      {str(data.get('reasoning','')).strip()}",
    ])
