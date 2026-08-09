"""
Evaluation harness - the piece that turns "it works on my machine" into
a numbers table you can put in the README.

Pipeline: load test_cases.json -> run each through the orchestrator ->
check automated signals (tools called, no errors, keyword match) ->
run an LLM-judge pass for quality scoring -> print/save a results table.

TODO(week 2, day 12): wire this up once orchestrator/graph.py has real
implementations instead of stubs.
"""

import json
from orchestrator.main import run


def load_test_cases(path="eval/test_cases.json"):
    with open(path) as f:
        return json.load(f)


def check_automated_signals(case: dict, output: str) -> dict:
    return {
        "keyword_match": all(kw.lower() in output.lower() for kw in case["must_contain"]),
        # TODO: pull actual tool-call history from state/metrics to check
        # against expected_tools, instead of just checking the final text.
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
        output = run(case["task"])
        automated = check_automated_signals(case, str(output))
        judged = judge_with_llm(case["task"], str(output))
        results.append({"id": case["id"], **automated, **judged})

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
