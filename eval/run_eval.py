"""
Evaluation harness - the piece that turns "it works on my machine" into
a numbers table you can put in the README.

Pipeline: load test_cases.json -> run each through the orchestrator ->
check automated signals (tools called, keyword match) -> run an LLM-judge pass
for quality scoring -> print a results table and save the raw detail.

The two halves answer different questions and neither is sufficient alone:

  AUTOMATED SIGNALS are cheap, deterministic and objective, but shallow. They
  can tell you search_web was called and the word "protocol" appears. They
  cannot tell you the answer was invented.

  THE LLM JUDGE is subjective and noisy, but it can read. Crucially it scores
  the answer against the tool output the agent actually received (see
  eval/judge.py), which is what catches a fluent answer with no evidence
  behind it - the exact failure this system produced when search was stubbed.

Run with the agents up (locally or port-forwarded from the cluster) and Ollama
serving the model in orchestrator/config.py:

    python -m eval.run_eval
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.judge import judge
from orchestrator.main import run_traced
from orchestrator.mcp_client import MCPToolRegistry

RESULTS_PATH = Path("eval/results.json")


def load_test_cases(path="eval/test_cases.json"):
    with open(path) as f:
        return json.load(f)


def check_automated_signals(case: dict, output: str, tools_called: list[str]) -> dict:
    """
    Tools are split into required and allowed on purpose.

    REQUIRED tools are load-bearing: without search_web there is no external
    information, so an answer produced without it is necessarily invented. That
    is worth failing on.

    ALLOWED tools are choices. Asserting them tests prompt compliance rather
    than task success, and it fights you - every prompt improvement that makes
    the system smarter would break the test. Anything called that is in neither
    list is flagged, since that is a genuine routing mistake.
    """
    called = set(tools_called)
    required = set(case.get("required_tools", []))
    allowed = set(case.get("allowed_tools", []))

    return {
        "keyword_match": all(kw.lower() in output.lower() for kw in case["must_contain"]),
        "required_tools_called": required.issubset(called),
        "unexpected_tools": sorted(called - required - allowed),
    }


def seed_corpus(case: dict) -> str | None:
    """
    Put a case's fixture documents into the index before it runs.

    WHY THIS EXISTS. cached-retrieval used to carry a note reading "Run after
    mcp-adoption-summary", and nothing enforced it. The dependency was real but
    invisible: the case could only find MCP content in the corpus if the
    earlier case had chosen to call index_documents, and the small default
    model skips indexing - correctly, from its point of view, since indexing
    helps the NEXT run and does nothing for the answer in progress.

    So the case scored a completeness of 1 against an empty corpus and looked
    like a retrieval failure. It was measuring whether an unrelated case had
    had a side effect.

    Seeding through the MCP tool rather than writing to the store directly:
    the harness has no filesystem access to the agent's volume in Kubernetes,
    and going through the real path means the fixture is embedded by the same
    model the query will be embedded with. A fixture indexed any other way
    would not be a fair test of retrieval.

    Idempotent because the store now skips text it already holds, so repeated
    eval runs do not accumulate copies of the fixture - which would crowd the
    results they are meant to support.
    """
    documents = case.get("seed_documents")
    if not documents:
        return None

    async def _seed():
        registry = MCPToolRegistry()
        await registry.discover()
        if "index_documents" not in {t["name"] for t in registry.tools}:
            raise RuntimeError(
                "cannot seed: the retrieval agent is not reachable"
            )
        return await registry.call(
            "index_documents",
            {"texts": documents, "source": case.get("seed_source", "eval-fixture")},
        )

    return asyncio.run(_seed())


def run_case(case: dict) -> dict:
    # Seeding is NOT inside the timing window - it is fixture setup, not work
    # the system under test performed.
    try:
        seeded = seed_corpus(case)
    except Exception as exc:
        return {"id": case["id"], "error": f"seeding failed: {type(exc).__name__}: {exc}"}

    started = time.time()
    try:
        trace = run_traced(case["task"])
    except Exception as exc:
        return {"id": case["id"], "error": f"{type(exc).__name__}: {exc}"}

    automated = check_automated_signals(case, trace.answer, trace.tools_called)
    scores = judge(case["task"], trace.answer, trace.tool_outputs)

    return {
        "id": case["id"],
        "seeded": seeded,
        "seconds": round(time.time() - started, 1),
        "iterations": trace.iterations,
        "tools_called": trace.tools_called,
        **automated,
        "grounding": scores["grounding"],
        "completeness": scores["completeness"],
        "relevance": scores["relevance"],
        "unsupported_claims": scores["unsupported_claims"],
        "judge_reasoning": scores["reasoning"],
        "judge_model": scores["judge_model"],
        "answer": trace.answer,
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    return str(value)


def print_table(results: list[dict]) -> None:
    header = f"{'case':<24} {'req':<5} {'kw':<5} {'grnd':<5} {'comp':<5} {'relv':<5} {'iters':<6} {'secs':<6} tools"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['id']:<24} ERROR: {r['error'][:70]}")
            continue
        print(
            f"{r['id']:<24} "
            f"{_fmt(r['required_tools_called']):<5} "
            f"{_fmt(r['keyword_match']):<5} "
            f"{_fmt(r['grounding']):<5} "
            f"{_fmt(r['completeness']):<5} "
            f"{_fmt(r['relevance']):<5} "
            f"{r['iterations']:<6} "
            f"{r['seconds']:<6} "
            f"{','.join(r['tools_called']) or '-'}"
        )

    scored = [r for r in results if r.get("grounding") is not None]
    if scored:
        print()
        for metric in ("grounding", "completeness", "relevance"):
            print(f"  mean {metric}: {statistics.mean(r[metric] for r in scored):.1f} / 5")
    flagged = [c for r in results for c in r.get("unsupported_claims", [])]
    if flagged:
        print(f"\n  unsupported claims flagged ({len(flagged)}):")
        for c in flagged[:5]:
            print(f"    - {c[:100]}")


def main():
    cases = load_test_cases()
    print(f"running {len(cases)} case(s)...")
    results = []
    for case in cases:
        print(f"  {case['id']} ...", flush=True)
        results.append(run_case(case))

    print_table(results)

    RESULTS_PATH.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "results": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nfull detail written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
