"""Thin wrapper around LightRAG: build/update the graph, retrieve context, and
generate grounded answers.

Design note (grounding): LightRAG's built-in answer prompt pushes the model to
"generate a comprehensive answer", which a small local model obeys by padding
with its own training knowledge. So we use LightRAG only for RETRIEVAL
(aquery_data) and generate the answer ourselves with a strict prompt we fully
control. If the retrieved notes don't contain the answer, we return a fixed
"not in your notes yet" and show no sources.

LightRAG import paths move between versions; if an import fails, `pip show
lightrag-hku` and adjust ONLY this file.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Iterable

from .adapters.base import Document
from .config import CONFIG

_rag = None  # lazily-initialised singleton

NOT_IN_NOTES = "That's not in your notes yet."

STRICT_ANSWER = (
    "You answer strictly from the user's personal notes, provided below as numbered NOTES. "
    "Use ONLY the NOTES. Do not use outside or general knowledge, and do not add facts, "
    "list items, names, dates or examples that are not in the NOTES, even if you know them to be true. "
    "If the NOTES cover only part of the question, answer only that part and say nothing about "
    "the rest (a note about what is missing is added separately). "
    f"If the NOTES contain nothing that answers the question, reply with EXACTLY: {NOT_IN_NOTES} "
    "Write a direct, well-structured prose answer. Do not include citations, note numbers, "
    "reference ids, or a sources list (sources are shown separately)."
)

# Grounding pipeline (small-model friendly): instead of one big "is this
# answerable?" judgement, which an 8B model gets wrong, we ask several narrow
# questions and check the risky step in plain code:
#   1. DECOMPOSE  the question into parts (+ its subject)
#   2. EXTRACT    for each part, the exact sentence(s) from the notes that answer it
#   3. VERIFY     in code that each quote really occurs in that note (no LLM)
#   4. CONFIRM    with a YES/NO on the quote alone: does it state the answer?
#   5. ANSWER     only from the confirmed quotes, only for the confirmed parts
DECOMPOSE = (
    "Split the user's QUESTION into the separate things it asks. Most questions ask 1 thing; "
    "'what is X and who invented it' asks 2; 'which IDs are used and what does each mean' asks 2. "
    "Write each part as a short phrase that completes the sentence "
    "\"the notes say something about ...\" (e.g. \"who formulated the Third Law\", "
    "\"what the First Law says\"). Reply with JSON: "
    '{"subject": "<main topic in 1-4 words>", "parts": ["<part>", ...]}'
)

EXTRACT = (
    "You find evidence in the user's numbered NOTES. Given a QUESTION and one PART of it, "
    "copy the exact sentence(s) or table row(s) from the NOTES that directly state the answer "
    "to that PART. Copy the text verbatim - do not paraphrase, summarise, or add anything. "
    "If the NOTES do not state it, return an empty list. Never use your own knowledge. "
    'Reply with JSON: {"evidence": [{"note": <number>, "quote": "<exact text>"}]}'
)

EXTRACT_ALL = (
    "You find evidence in the user's numbered NOTES. You get a QUESTION split into numbered PARTS. "
    "For EACH part, copy the exact sentence(s) or table row(s) from the NOTES that directly state "
    "the answer to that part. Copy the text verbatim - do not paraphrase, summarise, or add anything. "
    "If the NOTES do not state the answer to a part, give that part an empty evidence list. "
    "Never use your own knowledge. Reply with JSON: "
    '{"parts": [{"part": <part number>, "evidence": [{"note": <note number>, "quote": "<exact text>"}]}]}'
)

CONFIRM = (
    "Reply with ONLY one word, YES or NO. Read the QUESTION, then the PART of it being checked. "
    "Reply YES if the EVIDENCE contains the facts that answer that PART - they may be spread over "
    "several lines, table rows or bullets, and may be worded differently from the question. "
    "Reply NO if any name, date, number or fact the PART asks for is missing from the EVIDENCE, "
    "even if you know it yourself."
)

ABOUT = "Reply with ONLY one word, YES or NO. Is the main topic of this NOTE the given SUBJECT?"


async def _build_rag():
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts):
        safe = [t if (t and t.strip()) else " " for t in texts]
        vecs = await ollama_embed.func(safe, embed_model=CONFIG.embed_model, host=CONFIG.ollama_host)
        return np.asarray(vecs, dtype=np.float32)

    CONFIG.working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(CONFIG.working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=CONFIG.llm_model,
        llm_model_kwargs={"host": CONFIG.ollama_host, "options": {"num_ctx": CONFIG.num_ctx}},
        embedding_func=EmbeddingFunc(
            embedding_dim=CONFIG.embed_dim,
            max_token_size=8192,
            func=_embed,
        ),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass
    return rag


async def get_rag():
    global _rag
    if _rag is None:
        _rag = await _build_rag()
    return _rag


def _ainsert_kwargs(rag, doc) -> dict:
    """Pass ids AND file_paths when supported. file_paths is what gives chunks a
    real source for citation — without it LightRAG stores 'unknown_source'."""
    try:
        params = inspect.signature(rag.ainsert).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if "ids" in params:
        kw["ids"] = [doc.doc_id]
    if "file_paths" in params:
        kw["file_paths"] = [doc.doc_id]
    return kw


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stored_doc_ids() -> set[str]:
    """Doc ids LightRAG currently holds on disk (kv_store_full_docs.json)."""
    try:
        data = json.loads((CONFIG.working_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))
        return {k for k in data if not k.startswith("dup-")}
    except Exception:
        return set()


async def index_documents(docs: Iterable[Document]) -> int:
    """Incremental, change-aware indexing.

    LightRAG dedups by doc id and will NOT reprocess a note whose id it has
    already seen, so edits never reach the graph. We track each note's content
    hash and, when it changed (or on first run), delete the old doc and
    reinsert it. Notes removed from the vault are pruned. Returns the number of
    notes added / updated / removed."""
    if not CONFIG.is_indexer:
        raise RuntimeError(
            "This machine is not the indexer (LORE_INDEXER=false). "
            "Indexing runs on the desktop/GPU box; the laptop queries the committed graph."
        )
    rag = await get_rag()
    hpath = CONFIG.working_dir / ".lore_hashes.json"
    try:
        hashes = json.loads(hpath.read_text(encoding="utf-8"))
    except Exception:
        hashes = {}

    seen, changed = set(), 0
    for doc in docs:
        seen.add(doc.doc_id)
        h = _hash(doc.text)
        if hashes.get(doc.doc_id) == h:
            continue  # unchanged — skip the expensive extraction
        # changed / new / first-run: delete any existing version, then insert fresh
        try:
            await rag.adelete_by_doc_id(doc.doc_id)
        except Exception:
            pass  # not present yet — fine
        await rag.ainsert(doc.text, **_ainsert_kwargs(rag, doc))
        hashes[doc.doc_id] = h
        changed += 1

    # prune notes that no longer exist in the vault (deleted or renamed).
    # Check BOTH our hash file and what LightRAG actually holds: notes indexed
    # before hash tracking existed are in the store but not in .lore_hashes.json,
    # and would otherwise linger forever as "ghost" notes the gate answers from.
    stale = (set(hashes) | _stored_doc_ids()) - seen
    for gone in sorted(stale):
        try:
            await rag.adelete_by_doc_id(gone)
            changed += 1
        except Exception:
            pass
        hashes.pop(gone, None)

    try:
        hpath.write_text(json.dumps(hashes), encoding="utf-8")
    except Exception:
        pass
    return changed


async def complete(prompt: str, system: str | None = None, temperature: float = 0.1, fmt: str | None = None) -> str:
    """Direct, non-RAG chat completion against the local Ollama model."""
    import ollama

    client = ollama.AsyncClient(host=CONFIG.ollama_host)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"format": fmt} if fmt else {}
    resp = await client.chat(
        model=CONFIG.llm_model,
        messages=messages,
        # num_predict caps the output: llama3.1 in JSON mode can otherwise emit
        # whitespace forever and block Ollama's (single) queue for every query.
        options={"num_ctx": CONFIG.num_ctx, "temperature": temperature,
                 "num_predict": 600 if fmt else 900},
        **kwargs,
    )
    message = resp["message"] if isinstance(resp, dict) else resp.message
    content = message["content"] if isinstance(message, dict) else message.content
    return content or ""


def _ref_path(item) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("file_path", "reference", "source", "doc_id", "source_id", "id"):
            v = item.get(k)
            if isinstance(v, str) and v:
                return v
    return None


async def _retrieve(question: str, mode: str = "hybrid"):
    """Use LightRAG for graph-aware retrieval only. Returns (context_text, chunks)."""
    from lightrag import QueryParam

    rag = await get_rag()
    data = await rag.aquery_data(question, param=QueryParam(mode=mode))
    payload = data.get("data", data) if isinstance(data, dict) else {}
    chunks = payload.get("chunks") or [] if isinstance(payload, dict) else []
    parts = []
    for c in chunks:
        if isinstance(c, dict):
            t = (c.get("content") or "").strip()
            if t:
                parts.append(t)
    return "\n\n---\n\n".join(parts), chunks


def _title(fp: str) -> str:
    t = fp.rsplit("/", 1)[-1]
    return t[:-3] if t.endswith(".md") else t


def _number_notes(chunks, max_chars: int = 9000):
    """Group retrieved chunks by source note and number them [1], [2]...
    Returns (numbered_context, notes) where notes[i-1] = {"path", "text"}.
    Numbering by NOTE (not chunk) is what lets chips point at the exact note used."""
    notes, index = [], {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        t = (c.get("content") or "").strip()
        if not t:
            continue
        fp = _ref_path(c) or "unknown_source"
        if fp not in index:
            index[fp] = len(notes)
            notes.append({"path": fp, "text": t})
        else:
            notes[index[fp]]["text"] += "\n\n" + t
    parts, used = [], 0
    for n, note in enumerate(notes, 1):
        block = f"[{n}] (note: {_title(note['path'])})\n{note['text']}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        if not block:
            notes = notes[: n - 1]
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts), notes


def _as_ids(v, n_notes: int) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else []:
        try:
            i = int(str(x).strip("[] "))
        except ValueError:
            continue
        if 1 <= i <= n_notes and i not in out:
            out.append(i)
    return out


import re as _re

_NORM = _re.compile(r"[^0-9a-z\u00c0-\uffff]+")


_LATEX = _re.compile(r"\\[a-zA-Z]+")


def _norm(t: str) -> str:
    """Lowercase, drop LaTeX commands, markdown and punctuation so quote matching
    ignores formatting (the model writes "0 K" where the note has \\(0 \\text{ K}\\))."""
    return " ".join(_NORM.sub(" ", _LATEX.sub(" ", t.lower())).split())


def _quote_in_note(quote: str, note_text: str) -> bool:
    """Code-level hallucination check: the quote must really come from the note.

    Exact substring after normalising, OR near-verbatim: >=85% of its words and
    >=60% of its word pairs occur in the note. That tolerates small copy slips
    (dropped symbols, a changed word) but rejects invented facts, whose names,
    dates and numbers are simply not in the note."""
    hay = _norm(note_text)
    q = _norm(quote)
    if not q:
        return False
    if q in hay:
        return True
    hw, qw = hay.split(), q.split()
    if len(qw) < 3:
        return False
    words = set(hw)
    pairs = set(zip(hw, hw[1:]))
    word_hit = sum(w in words for w in qw) / len(qw)
    qp = list(zip(qw, qw[1:]))
    pair_hit = sum(p in pairs for p in qp) / len(qp)
    return word_hit >= 0.85 and pair_hit >= 0.6


def _expand_to_lines(quote: str, note_text: str) -> str:
    """Widen a verified quote to the note line it came from, so the label that
    gives it meaning comes along: "**Third Law:** As a system's temperature..."
    instead of just "As a system's temperature...". Only the line that CONTAINS
    the quote (>=80% of its words) is used, never neighbouring lines - those can
    mix in a different fact (e.g. the 5-minute break next to the long break)."""
    lines = [l.strip() for l in note_text.splitlines() if l.strip()]
    out = []
    for qline in [q for q in quote.splitlines() if q.strip()]:
        qw = _norm(qline).split()
        if not qw:
            continue
        best, best_r = None, 0.0
        for line in lines:
            lw = set(_norm(line).split())
            r = sum(w in lw for w in qw) / len(qw)
            if r > best_r:
                best, best_r = line, r
        pick = best if best is not None and best_r >= 0.8 else qline.strip()
        if pick not in out:
            out.append(pick)
    return "\n".join(out) if out else quote


async def _json(prompt: str, system: str) -> dict:
    raw = await complete(prompt, system=system, temperature=0.0, fmt="json")
    try:
        j = json.loads(raw)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


async def _yes(prompt: str, system: str) -> bool:
    return (await complete(prompt, system=system, temperature=0.0)).strip().upper().startswith("Y")


async def _judge(question: str, context: str, notes: list[dict]) -> dict:
    d = await _json(f"QUESTION: {question}", DECOMPOSE)
    subject = str(d.get("subject") or "").strip()
    parts = [str(p).strip().rstrip("?.") for p in (d.get("parts") or []) if str(p).strip()][:4] or [question.rstrip("?")]

    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
    e_all = await _json(f"NOTES:\n{context}\n\nQUESTION: {question}\nPARTS:\n{numbered}", EXTRACT_ALL)
    by_part: dict[int, list] = {}
    if not e_all.get("parts") and e_all.get("evidence") and len(parts) == 1:
        e_all = {"parts": [{"part": 1, "evidence": e_all["evidence"]}]}  # model skipped the wrapper
    for k, entry in enumerate(e_all.get("parts") or [], 1):
        if not isinstance(entry, dict):
            continue
        ids = _as_ids([entry.get("part", k)], len(parts))
        by_part.setdefault(ids[0] if ids else k, []).extend(entry.get("evidence") or [])

    checked, rejected, retried = [], [], []
    for pi, part in enumerate(parts, 1):
        items = by_part.get(pi, [])
        if not items:
            # The one-call extraction found nothing for this part; small models
            # sometimes miss with several parts at once, so ask about it alone.
            retried.append(part)
            one = await _json(f"NOTES:\n{context}\n\nQUESTION: {question}\nPART: {part}", EXTRACT)
            items = one.get("evidence") or []
        ev = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ids = _as_ids([item.get("note")], len(notes))
            q = str(item.get("quote") or "").strip()
            if ids and q and _quote_in_note(q, notes[ids[0] - 1]["text"]):
                ev.append({"note": ids[0], "quote": _expand_to_lines(q, notes[ids[0] - 1]["text"])})
            elif q:
                rejected.append({"part": part, "note": ids[0] if ids else None, "quote": q})
        ok = bool(ev) and await _yes(
            "EVIDENCE:\n" + "\n".join(x["quote"] for x in ev) + f"\n\nQUESTION: {question}\nPART: {part}", CONFIRM)
        checked.append({"part": part, "answered": ok, "evidence": ev})

    answered = [c for c in checked if c["answered"]]
    answer_notes = []
    for c in answered:
        for x in c["evidence"]:
            if x["note"] not in answer_notes:  # one chip per note, even with several quotes from it
                answer_notes.append(x["note"])

    # Which notes are ABOUT the subject (for "you have a note on X, but...").
    subject_notes = []
    if subject and not answered:
        words = [w for w in _norm(subject).split() if len(w) >= 4]
        for i, n in enumerate(notes, 1):
            if any(w[:6] in _norm(_title(n["path"])) for w in words):
                subject_notes.append(i)
        if not subject_notes:  # badly named note? ask about the top-ranked two
            for i, n in enumerate(notes[:2], 1):
                if await _yes(f"SUBJECT: {subject}\n\nNOTE:\n{n['text'][:3000]}", ABOUT):
                    subject_notes.append(i)

    return {
        "subject": subject,
        "subject_notes": subject_notes,
        "parts": checked,
        "answered_parts": [c["part"] for c in answered],
        "missing_parts": [c["part"] for c in checked if not c["answered"]],
        "answer_notes": answer_notes,
        "rejected_quotes": rejected,  # debug: quotes that failed the in-note check
        "retried_parts": retried,     # debug: parts re-asked one at a time
        "raw_extract": e_all,         # debug: what the one-call extraction returned
    }


def _join(items: list[str]) -> str:
    items = [i[:1].lower() + i[1:] for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _sources(notes, ids) -> list[dict]:
    out = []
    for i in ids:
        fp = notes[i - 1]["path"]
        if fp != "unknown_source":
            out.append({"title": _title(fp), "path": fp})
    return out


async def answer(question: str, mode: str = "hybrid", strict: bool = True, debug: bool = False) -> dict:
    """Return {status, answer, sources}. status is one of:
      full       - the notes answer every part; chips = notes actually used
      partial    - notes answer some parts; answer those + "your notes don't say anything about X"
      topic_only - a note on the subject exists but answers no part of the question
      none       - nothing relevant in the notes; fixed refusal, no chips
    With debug=True also returns the retrieved notes and the judge's verdict."""
    if not strict:
        from lightrag import QueryParam

        rag = await get_rag()
        return {"status": "full", "answer": await rag.aquery(question, param=QueryParam(mode=mode)), "sources": []}

    try:
        _, chunks = await _retrieve(question, mode)
    except Exception:
        chunks = []
    context, notes = _number_notes(chunks)
    dbg = {"retrieved": [n["path"] for n in notes]}

    def done(r):
        try:  # last query's verdict, for diagnosing a wrong answer (git-ignored)
            (CONFIG.working_dir / ".last_query.json").write_text(
                json.dumps({"question": question, **r, "debug": dbg}, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        if debug:
            r["debug"] = dbg
        return r

    if not notes:
        return done({"status": "none", "answer": NOT_IN_NOTES, "sources": []})

    j = await _judge(question, context, notes)
    dbg["judge"] = j
    missing = _join(j["missing_parts"]) or question.rstrip("?").lower()

    def topic_only():
        titles = [_title(notes[i - 1]["path"]) for i in j["subject_notes"]]
        named = " and ".join(f"**{t}**" for t in titles)
        lead = f"Your note {named} doesn't" if len(titles) == 1 else f"Your notes {named} don't"
        return done({
            "status": "topic_only",
            "answer": f"{lead} say anything about {missing}.",
            "sources": _sources(notes, j["subject_notes"]),
        })

    if not j["answer_notes"]:
        return topic_only() if j["subject_notes"] else done({"status": "none", "answer": NOT_IN_NOTES, "sources": []})

    # Generate ONLY from the verified quotes, and only for the parts they answer.
    evidence = "\n\n".join(
        f"[{x['note']}] (note: {_title(notes[x['note'] - 1]['path'])}) {x['quote']}"
        for c in j["parts"] if c["answered"] for x in c["evidence"])
    scope = "\n\nAnswer ONLY these parts: " + "; ".join(j["answered_parts"]) + "."
    if j["missing_parts"]:
        scope += " Do NOT answer or mention: " + "; ".join(j["missing_parts"]) + "."
    text = (await complete(f"NOTES:\n{evidence}\n\nQUESTION: {question}{scope}", system=STRICT_ANSWER)).strip()
    if text.lower().startswith("that's not in your notes"):
        return topic_only() if j["subject_notes"] else done({"status": "none", "answer": NOT_IN_NOTES, "sources": []})

    if j["missing_parts"]:
        text += f"\n\n*Your notes don't say anything about {missing}.*"
        return done({"status": "partial", "answer": text, "sources": _sources(notes, j["answer_notes"])})
    return done({"status": "full", "answer": text, "sources": _sources(notes, j["answer_notes"])})


async def query_with_sources(question: str, mode: str = "hybrid", strict: bool = True):
    """Back-compat: (answer, sources)."""
    r = await answer(question, mode=mode, strict=strict)
    return r["answer"], r["sources"]


async def query(question: str, mode: str = "hybrid", strict: bool = True) -> str:
    r = await answer(question, mode=mode, strict=strict)
    return r["answer"]
