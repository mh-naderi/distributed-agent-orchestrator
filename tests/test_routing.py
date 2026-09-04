"""
The routing decisions, called directly.

These used to be reachable only by building a graph and running it, so checking
one branch meant scripting a whole provider and inferring the decision from the
messages that came out. They are pure functions of state and now live at module
level, which lets each branch be stated as what it is: given this history, this
is the next node.

The end-to-end behaviour is still covered in test_graph.py. This is the same
logic examined one decision at a time, which is what makes a wrong branch
obvious rather than something to deduce from a transcript.
"""

from orchestrator.graph import (
    MAX_ITERATIONS,
    NO_EVIDENCE,
    NUDGE_PROMPT,
    REGROUND_PROMPT,
    _acted_this_turn,
    _every_tool_came_back_empty,
    should_continue,
)


def state(messages, iterations=1, **extra):
    return {"messages": messages, "iterations": iterations, **extra}


def user(text="what did they conclude?"):
    return {"role": "user", "content": text}


def answer(text="here is the answer"):
    return {"role": "assistant", "content": text}


def tool(content="a real document about the subject", name="retrieve"):
    return {"role": "tool", "name": name, "content": content}


def wants_tools():
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "retrieve"}]}


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------


def test_a_tool_request_continues():
    assert should_continue(state([user(), wants_tools()])) == "continue"


def test_the_iteration_guardrail_wins_over_everything():
    """
    Checked first and deliberately: a model that keeps requesting tools forever
    is a real failure mode, and the guardrail has to fire even mid-request.
    """
    decision = should_continue(state([user(), wants_tools()], iterations=MAX_ITERATIONS))

    assert decision == "truncated"


def test_an_answer_backed_by_a_tool_result_ends_the_run():
    assert should_continue(state([user(), wants_tools(), tool(), answer()])) == "end"


def test_an_answer_with_nothing_behind_it_is_nudged():
    assert should_continue(state([user(), answer()], nudges=0)) == "nudge"


def test_the_nudge_fires_only_once():
    """A second nudge would be a loop, not a recovery."""
    assert should_continue(state([user(), answer()], nudges=1)) != "nudge"


def test_an_answer_from_tools_that_all_found_nothing_is_regrounded():
    decision = should_continue(
        state([user(), wants_tools(), tool(f"{NO_EVIDENCE} nothing close enough"), answer()],
              nudges=0, regrounds=0)
    )

    assert decision == "reground"


def test_regrounding_also_fires_only_once():
    decision = should_continue(
        state([user(), wants_tools(), tool(f"{NO_EVIDENCE} nothing"), answer()],
              nudges=0, regrounds=1)
    )

    assert decision == "end"


def test_a_narrated_tool_call_after_the_nudge_is_unanswered():
    narration = {"role": "assistant", "content": '{"name": "retrieve", "arguments": {}}'}

    decision = should_continue(state([user(), narration], nudges=1))

    assert decision == "unanswered"


def test_one_real_result_among_empty_ones_is_enough_to_end():
    """Partial evidence is evidence; regrounding here would argue with a real answer."""
    decision = should_continue(
        state(
            [
                user(),
                wants_tools(),
                tool(f"{NO_EVIDENCE} nothing", name="retrieve"),
                tool("MCP standardises tool access", name="search_web"),
                answer(),
            ],
            nudges=0,
            regrounds=0,
        )
    )

    assert decision == "end"


# ---------------------------------------------------------------------------
# the two helpers the decisions rest on
# ---------------------------------------------------------------------------


def test_acting_is_scoped_to_the_current_question():
    """A tool that ran for an earlier turn does not count for this one."""
    history = [user("first"), wants_tools(), tool(), answer(), user("second"), answer()]

    assert _acted_this_turn(history) is False


def test_a_guardrails_own_prompt_does_not_start_a_new_turn():
    """
    Both guardrails speak as the user. Treating their prompts as turn boundaries
    made an honest answer produced after regrounding look like a narration, and
    it was nudged for it.
    """
    history = [
        user(),
        wants_tools(),
        tool(),
        answer(),
        {"role": "user", "content": REGROUND_PROMPT},
        answer("I could not find it"),
    ]

    assert _acted_this_turn(history) is True
    assert _acted_this_turn(history[:4] + [{"role": "user", "content": NUDGE_PROMPT}]) is True


def test_empty_evidence_needs_at_least_one_tool_result():
    """A turn where nothing ran is the nudge's business, not the reground's."""
    assert _every_tool_came_back_empty([user(), answer()]) is False


def test_empty_evidence_means_every_result_reported_nothing():
    empty = [user(), tool(f"{NO_EVIDENCE} a"), tool(f"{NO_EVIDENCE} b"), answer()]
    mixed = [user(), tool(f"{NO_EVIDENCE} a"), tool("a real document"), answer()]

    assert _every_tool_came_back_empty(empty) is True
    assert _every_tool_came_back_empty(mixed) is False
