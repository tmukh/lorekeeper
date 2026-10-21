"""Answer-behaviour scenarios for the strict pipeline. Run on the indexer box:

    .venv\\Scripts\\python scripts\\eval_scenarios.py [--reindex]

Writes scripts/eval_results.json and prints PASS/FAIL per scenario. Each run is a
fresh process, so it always tests the code on disk (no server restart needed).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from lorekeeper import indexer
from lorekeeper.adapters.obsidian import ObsidianAdapter
from lorekeeper.config import CONFIG

THERMO = "thermodynamics-laws"

# (id, description, question, allowed statuses, required chip TITLES, forbidden-in-answer)
# Chips are compared by note title: LightRAG stores the file name, not the folder path.
SCENARIOS = [
    ("1-full", "note has the answer -> answer + correct chip",
     "What does the First Law of thermodynamics say?", {"full"}, {THERMO}, []),
    ("2-partial", "concept is there but the asked detail isn't -> answer what's there + say what's missing",
     "What does the Third Law of thermodynamics say, and who formulated it?", {"partial"}, {THERMO}, ["Nernst"]),
    ("3-topic-only", "note on the subject exists but says nothing about the question",
     "Who worked in the field of thermodynamics between 1995 and 1997?", {"topic_only"}, {THERMO}, []),
    ("4-none", "no note at all -> fixed refusal, no chips",
     "What is the stock market?", {"none"}, set(), []),
    ("5-misnamed", "content exists in a badly named note -> still found, chip points at it",
     "How long is the long break in the Pomodoro technique?", {"full", "partial"}, {"untitled-3"}, []),
    ("6-multi-note", "answer spans two notes -> both chips",
     "Which event IDs does my evtx-analyst project use to detect a new admin account, and what does each ID mean?",
     {"full", "partial"}, {"evtx-analyst", "windows-event-ids"}, []),
    ("7-none-nearby", "general-knowledge question near a known subject must not be answered from training",
     "What is the boiling point of water at sea level?", {"none"}, set(), ["100"]),
]


async def main():
    if "--reindex" in sys.argv:
        n = await indexer.index_documents(ObsidianAdapter(CONFIG.vault_path).documents())
        print(f"reindexed {n} notes")
    results, passed = [], 0
    for sid, desc, q, statuses, chips, forbidden in SCENARIOS:
        t = time.time()
        r = await indexer.answer(q)
        got = {s["title"] for s in r["sources"]}
        problems = []
        if r["status"] not in statuses:
            problems.append(f"status {r['status']} not in {sorted(statuses)}")
        if chips and not chips <= got:
            problems.append(f"missing chips {sorted(chips - got)}")
        if not chips and got:
            problems.append(f"unexpected chips {sorted(got)}")
        if chips and got - chips:
            problems.append(f"extra chips {sorted(got - chips)}")
        for f in forbidden:
            if f.lower() in r["answer"].lower():
                problems.append(f"answer contains outside knowledge '{f}'")
        ok = not problems
        passed += ok
        results.append({"id": sid, "desc": desc, "question": q, "ok": ok, "problems": problems,
                        "seconds": round(time.time() - t, 1), **r})
        print(f"{'PASS' if ok else 'FAIL'}  {sid}  {'; '.join(problems)}")
    out = Path(__file__).with_name("eval_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{passed}/{len(SCENARIOS)} passed -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
