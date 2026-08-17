"""
Evaluation harness - the piece that turns "it works on my machine" into
a numbers table you can put in the README.

Pipeline: load test_cases.json -> run each through the orchestrator ->
check automated signals (tools called, no errors, keyword match) ->
run an LLM-judge pass for quality scoring -> print/save a results table.

TODO(week 2, day 12): the LLM-judge half is still stubbed. The automated
signals below are wired up.
"""

import json
from orchestrator.main import run_traced


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
        "tools_called": tools_called,
    }


def judge_with_llm(task: str, output: str) -> dict:
    """
    TODO: call an LLM with a rubric prompt (see project notes) and parse
    the returned JSON scores. Stubbed for now so the pipeline runs end-to-end.
    """
    return {"accuracy": None, "completeness": None, "relevance": None}


def main():
    cases = load_test_cases()
    results = []
    for case in cases:
        output, tools_called = run_traced(case["task"])
        automated = check_automated_signals(case, str(output), tools_called)
        judged = judge_with_llm(case["task"], str(output))
        results.append({"id": case["id"], **automated, **judged})

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
