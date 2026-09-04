"""
Repeatable measurements, so a claim in the docs can be checked rather than
believed.

WHY THIS EXISTS. Every strong statement in docs/architecture.md rests on a
measurement: routing went 1/5 to 5/5 when a tool description changed, fabrication
went 8/8 to 0/8 when search results carried a coverage note, a relevance floor of
0.90 produced 7 fabrications in 8 where 0.70 produced none. Each of those numbers
came from a script that was written, run once and thrown away. The conclusions
outlived the evidence for them, which is exactly the shape of problem this project
keeps finding in its own agents.

Two commands, because two shapes of question keep recurring:

    python -m eval.experiment repeat --case honest-ignorance --runs 8
    python -m eval.experiment ab --case honest-ignorance --tool search_web \\
        --strip "Note: none of these" --runs 8

`repeat` answers "how often does this happen", which needs the whole loop and
tolerates it wandering down different paths - the path taken is reported, because
"6 of 8 never reached the web" was itself the finding once.

`ab` answers "did this change cause it", which the whole loop cannot answer: the
loop reaches search_web about once in eight runs, so measuring a change to search
results through it would take thirty runs to get a handful of samples. It fetches
the evidence ONCE, then asks the model to answer from it with and without the
text from --strip onwards. Same evidence, same question, one difference.

A third shape - the distance study behind the relevance floor - needs the store's
embedder and cannot run from here. It lives in eval/distance_study.py, which is
run inside the pod; see docs/RUNBOOK.md.
"""

import argparse
import asyncio
import statistics
import sys

from eval.judge import judge
from eval.run_eval import check_subject_grounding, load_test_cases, run_case
from orchestrator.graph import SYSTEM_PROMPT
from orchestrator.llm import OllamaProvider
from orchestrator.mcp_client import MCPToolRegistry


def _case(case_id: str) -> dict:
    for case in load_test_cases():
        if case["id"] == case_id:
            return case
    sys.exit(f"no such case: {case_id}. Known: {[c['id'] for c in load_test_cases()]}")


def _verdict(case: dict, answer: str, tool_outputs: list[dict]) -> list[str]:
    return check_subject_grounding(case, answer, tool_outputs)["invented_subject_claims"]


def summarise(runs: list[dict]) -> dict:
    """
    Reduce a set of runs to the numbers worth quoting.

    Paths are counted because which route the loop took is often the finding
    rather than a detail: a change that makes the model stop calling search_web
    at all will look like a fabrication fix if only the totals are read.
    """
    paths: dict[str, int] = {}
    for run in runs:
        key = ",".join(run["tools"]) or "(no tools)"
        paths[key] = paths.get(key, 0) + 1

    fabricated = sum(1 for r in runs if r["invented"])
    seconds = [r["seconds"] for r in runs if r.get("seconds") is not None]
    return {
        "runs": len(runs),
        "fabricated": fabricated,
        "paths": dict(sorted(paths.items(), key=lambda kv: -kv[1])),
        "median_seconds": round(statistics.median(seconds), 1) if seconds else None,
    }


def _report(label: str, runs: list[dict]) -> None:
    summary = summarise(runs)
    print(f"\n  -> {label}: fabricated {summary['fabricated']}/{summary['runs']}")
    for path, count in summary["paths"].items():
        print(f"     {count}x {path}")
    if summary["median_seconds"] is not None:
        print(f"     median {summary['median_seconds']}s per run")


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------


def repeat(case_id: str, runs: int, use_judge: bool) -> list[dict]:
    case = _case(case_id)
    print(f"=== repeat: {case_id} x{runs} ===")
    if not case.get("subject"):
        print("  note: this case declares no subject, so fabrication cannot be counted")

    collected = []
    for i in range(1, runs + 1):
        result = run_case(case)
        if result.get("error"):
            print(f"  ERR run {i}: {result['error'][:90]}")
            continue

        invented = result.get("invented_subject_claims") or []
        collected.append(
            {
                "tools": result["tools_called"],
                "invented": invented,
                "seconds": result.get("seconds"),
            }
        )
        shown = (invented[0] if invented else result["answer"]) or "(empty)"
        flag = "FAB" if invented else "ok "
        extra = f" grnd={result.get('grounding')}" if use_judge else ""
        print(f"  {flag} run {i}:{extra} {shown[:88]}".replace("\n", " "))

    _report(case_id, collected)
    return collected


# ---------------------------------------------------------------------------
# ab
# ---------------------------------------------------------------------------


async def _answer_from(provider, case: dict, tool: str, evidence: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case["task"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": tool, "arguments": {"query": case["task"]}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": tool, "content": evidence},
    ]
    response = await provider.chat(messages, [])
    return response.content or ""


async def ab(case_id: str, tool: str, strip: str, runs: int, use_judge: bool) -> None:
    case = _case(case_id)
    registry = MCPToolRegistry()
    await registry.discover()

    full = await registry.call(tool, {"query": case["task"]})
    if strip not in full:
        sys.exit(
            f"{tool} output does not contain {strip!r}, so there is nothing to vary.\n"
            "Run it again - live results differ - or pass a --strip that appears in:\n"
            f"{full[:400]}"
        )
    reduced = full.split(strip)[0].rstrip()

    print(f"=== ab: {case_id} via {tool}, {runs} runs each ===")
    print(f"  evidence fetched once; the variant differs by {len(full) - len(reduced)} chars\n")

    provider = OllamaProvider()
    for label, evidence in (("without", reduced), ("with", full)):
        collected = []
        for i in range(1, runs + 1):
            answer = await _answer_from(provider, case, tool, evidence)
            outputs = [{"name": tool, "output": evidence}]
            invented = _verdict(case, answer, outputs)
            row = {"tools": [tool], "invented": invented, "seconds": None}
            if use_judge:
                scored = judge(case["task"], answer, outputs)
                row["grounding"] = scored["grounding"]
            collected.append(row)
            shown = (invented[0] if invented else answer) or "(empty)"
            print(f"  {'FAB' if invented else 'ok '} {label} run {i}: {shown[:82]}".replace("\n", " "))
        _report(f"{label} {strip!r}", collected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p_repeat = sub.add_parser("repeat", help="run one case N times through the whole loop")
    p_repeat.add_argument("--case", required=True)
    p_repeat.add_argument("--runs", type=int, default=8)
    p_repeat.add_argument("--judge", action="store_true", help="also score each run (slower)")

    p_ab = sub.add_parser("ab", help="hold the evidence fixed and vary one thing")
    p_ab.add_argument("--case", required=True)
    p_ab.add_argument("--tool", default="search_web")
    p_ab.add_argument("--strip", required=True, help="text from which the variant is cut")
    p_ab.add_argument("--runs", type=int, default=8)
    p_ab.add_argument("--judge", action="store_true")

    args = parser.parse_args()
    if args.command == "repeat":
        repeat(args.case, args.runs, args.judge)
    else:
        asyncio.run(ab(args.case, args.tool, args.strip, args.runs, args.judge))


if __name__ == "__main__":
    main()
