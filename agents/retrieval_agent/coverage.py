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
# Restricted to capitalised words, and never the first, which is capitalised only
# because it starts the sentence. Proper nouns are where misattribution happens,
# and a note that fires on ordinary words would be noise the model learns to skip.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def unmentioned_terms(query: str, results: str) -> list[str]:
    """
    Which capitalised words from the query appear in none of the results?

    A fact, not a judgement. The agent knows what was asked and what came back,
    and comparing them needs no model and cannot hallucinate. What the model does
    with it is the model's business; what this avoids is presenting results as
    though they were about the thing that was asked for.
    """
    words = _WORD.findall(query)
    haystack = results.lower()

    missing, seen = [], set()
    for word in words[1:]:  # the first word is capitalised by sentence position
        if not word[0].isupper() or len(word) < 3:
            continue
        lowered = word.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        if lowered not in haystack:
            missing.append(word)
    return missing
