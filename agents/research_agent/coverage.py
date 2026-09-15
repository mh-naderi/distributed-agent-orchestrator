"""
Which words from a question do the results never mention?

A fact the agent can compute: it knows what was asked and what came back, so
comparing them needs no model and cannot hallucinate. It exists because distance
cannot answer this. A relevance floor separates "related to nothing here" from
"related to something here" - measured on this corpus, unrelated queries sit near
1.00 while answerable ones sit below 0.70 - but it cannot separate "about the
subject" from "about a similar subject", and that gap is where the remaining
fabrication lives. Worse, it drifts: the query that first exposed the problem
scored 0.768 when the floor was chosen and 0.676 once the corpus had grown, so a
fixed threshold silently stops excluding what it was set to exclude.

Duplicated into each agent directory, the same trade as instrumentation.py, since
each image builds from its own directory. A test asserts the copies are identical.
"""

import re

# Words the caller capitalised are the ones a search can silently fail to be
# about. "What did the Quazzlemint Foundation conclude in its 2019 report" gets
# real annual reports from real foundations, none of them Quazzlemint's, and the
# model has answered from those - the last fabrication path left after the corpus
# was cleaned up and the empty-evidence guardrail was added.
#
# Restricted to capitalised words, because proper nouns are where misattribution
# happens, and a note that fires on ordinary words would be noise the model
# learns to skip.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

# The first word is skipped ONLY when it is one of these - a word capitalised
# because it opens a sentence, not because it names anything.
#
# This used to skip the first word unconditionally, on the reasoning that it is
# capitalised by sentence position. That is true of a question and false of a
# search query. "What did the Quazzlemint Foundation conclude in its 2019
# report?" flagged Quazzlemint correctly; "Quazzlemint Foundation 2019 report"
# flagged nothing, because the subject WAS the first word and was discarded -
# and a keyword query that opens with its subject is exactly how a model writes
# a search. The warning worked for the eval's phrasing and failed for the
# model's, over identical wrong-subject results.
#
# A closed list rather than a dictionary of English, because sentence-starters
# are a small closed class and proper nouns are not. Words under three letters
# are already excluded below, so they need no entry here.
_SENTENCE_STARTERS = frozenset({
    # question words
    "what", "who", "whom", "whose", "when", "where", "why", "how", "which",
    # auxiliaries a question opens with
    "are", "was", "were", "does", "did", "can", "could", "will", "would",
    "should", "shall", "may", "might", "must", "has", "have", "had",
    # determiners and pronouns
    "the", "this", "that", "these", "those", "you", "they",
    # imperatives a request opens with
    "tell", "explain", "describe", "find", "search", "show", "give", "list",
    "summarise", "summarize", "research", "compare", "define", "look",
    "please", "about",
})


def subject_terms(text: str) -> list[str]:
    """
    The capitalised words that say what a piece of text is ABOUT.

    Separated from the comparison below because the two questions are different
    and both get asked. "Which of these are missing from the results" annotates a
    search; "does this name a subject at all" is what tells a label like
    `web` or `integration-test` - which names nothing and cannot be wrong - apart
    from one like `Quazzlemint Foundation 2019 report`, which makes a claim that
    the text can fail to support.

    Order is the order of appearance, and repeats are dropped, so a caller can
    quote the list back without repeating itself.
    """
    terms, seen = [], set()
    for position, word in enumerate(_WORD.findall(text)):
        if position == 0 and word.lower() in _SENTENCE_STARTERS:
            continue
        if not word[0].isupper() or len(word) < 3:
            continue
        lowered = word.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(word)
    return terms


def unmentioned_terms(query: str, results: str) -> list[str]:
    """
    Which capitalised words from the query appear in none of the results?

    A fact, not a judgement. The agent knows what was asked and what came back,
    and comparing them needs no model and cannot hallucinate. What the model does
    with it is the model's business; what this avoids is presenting results as
    though they were about the thing that was asked for.
    """
    haystack = results.lower()
    return [term for term in subject_terms(query) if term.lower() not in haystack]
