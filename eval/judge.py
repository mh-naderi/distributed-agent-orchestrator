"""
LLM-as-judge scoring for the evaluation harness.

TWO DESIGN DECISIONS WORTH UNDERSTANDING.

1. THE JUDGE SEES THE TOOL OUTPUTS, AND SCORES GROUNDING AGAINST THEM.
   The obvious design - show the judge the task and the answer, ask "is this
   good" - measures fluency, and fluency is exactly what this system has
   already faked. Earlier runs produced confident, well-written answers about a
   "Multi-Cloud Platform" complete with an invented adoption statistic, because
   the search tool was a stub returning nothing. A judge grading that text
   without the evidence would have scored it highly.

   So "accuracy" here means SUPPORTED BY THE RETRIEVED EVIDENCE, not "matches
   what the judge happens to believe". That is the failure mode this project
   actually has, so it is the one worth measuring.

2. STRUCTURED OUTPUT, NOT PARSED PROSE.
   Ollama's format= takes a JSON Schema and constrains decoding to match it, so
   the response is valid JSON by construction. This matters more than usual
   here: qwen3 tends to think out loud, and a "reply with JSON" instruction
   comes back wrapped in commentary. Schema-constrained decoding removes the
   whole class of parsing bugs.

KNOWN LIMITATION: by default the judge is the same model being judged, because
that is what fits in 4GB of VRAM. Self-evaluation is biased - models tend to
favour their own outputs. Set EVAL_JUDGE_MODEL to something else when a second
model is available. Treat these scores as a regression signal over time, not as
an absolute measure of quality.
"""

import json
import os

import ollama

from orchestrator.config import OLLAMA_HOST, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, ollama_options

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", OLLAMA_MODEL)

# Constrains decoding, so parsing cannot fail on malformed output.
SCHEMA = {
    "type": "object",
    "properties": {
        "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "ONLY claims the evidence fails to support. Leave empty when every "
                "claim is supported. Never list a claim here that the evidence backs."
            ),
        },
        "reasoning": {"type": "string"},
    },
    "required": ["grounding", "completeness", "relevance", "unsupported_claims", "reasoning"],
}

RUBRIC = """You are evaluating an AI agent's answer.

You will be given the user's task, the raw output of every tool the agent
called, and the agent's final answer.

Score three things from 1 to 5:

grounding - Is every factual claim in the answer supported by the tool output?
  5 = every claim traceable to the evidence
  3 = mostly supported, some unsupported filler
  1 = confident claims with no support in the evidence at all
  If the tools returned nothing useful and the answer says so plainly, that is
  a 5. Admitting ignorance is grounded. Inventing detail to fill the gap is a 1.

completeness - Does the answer actually address what was asked?
relevance - Is it on topic and free of padding?

unsupported_claims must contain ONLY statements the evidence fails to back up.
If a statement IS supported by the tool output, it does not belong in this list.
When grounding is 4 or 5 the list is normally empty - return [] in that case.
Do not restate the answer here; this field is for problems only.

Be strict about grounding. A fluent, plausible answer that the evidence does
not support is the failure you are looking for."""


def judge(task: str, answer: str, tool_outputs: list[dict], model: str = JUDGE_MODEL) -> dict:
    """Score one run. Returns the parsed scores, or an error dict on failure."""
    if tool_outputs:
        evidence = "\n\n".join(
            f"--- output of {t['name']} ---\n{t['output'][:2000]}" for t in tool_outputs
        )
    else:
        evidence = "(the agent called no tools, so there is no evidence to check against)"

    prompt = (
        f"TASK GIVEN TO THE AGENT:\n{task}\n\n"
        f"TOOL OUTPUT THE AGENT RECEIVED:\n{evidence}\n\n"
        f"THE AGENT'S FINAL ANSWER:\n{answer}"
    )

    try:
        response = ollama.Client(host=OLLAMA_HOST).chat(
            model=model,
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": prompt},
            ],
            format=SCHEMA,
            think=False,
            # temperature 0 so scores are repeatable; the rest bounds VRAM.
            options={**ollama_options(), "temperature": 0},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        scores = json.loads(response.message.content)
        scores["judge_model"] = model
        return scores
    except Exception as exc:
        # A failed judge must not take the whole eval run down - the automated
        # signals are still worth collecting.
        return {
            "grounding": None,
            "completeness": None,
            "relevance": None,
            "unsupported_claims": [],
            "reasoning": f"judge failed: {type(exc).__name__}: {exc}",
            "judge_model": model,
        }
