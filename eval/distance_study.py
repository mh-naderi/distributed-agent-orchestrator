"""
Measure the distance distribution the relevance floor has to separate.

Runs INSIDE the retrieval pod, because it needs that agent's embedder and the
real corpus - the MCP tool applies the floor, which is the thing under test:

    kubectl exec -i retrieval-agent-0 -- python - < eval/distance_study.py

Prints the top hit for every query so the labels below can be checked rather than
trusted. A query labelled "not answerable" that the corpus actually covers would
quietly corrupt the threshold chosen from it, and the labels are the one part of
this that no code can verify.

WHAT IT SHOWED, and the trap it did not show. Answerable questions land at
0.491-0.698 and unrelated ones at 0.997-1.048, a gap so wide that a floor of 0.90
looks obviously better than 0.70. Raising it produced 7 fabrications in 8 where
0.70 produced none, because the number this study does not contain is the one
that matters:

    0.764  a real annual report belonging to somebody else
    0.877  a real annual report belonging to exactly who was asked about

The wrong subject is NEARER than the right one, so no threshold separates them.
The floor works by excluding both and leaving the model with nothing, not by
telling them apart. Read this output with that in mind: a wide gap here is not
evidence that a higher floor is safe.
"""

import sys

sys.path.insert(0, "/app")

import server  # noqa: E402

ANSWERABLE = [
    "What is the Model Context Protocol?",
    "How is MCP being adopted?",
    "What is Kubernetes?",
    "How do Kubernetes StatefulSets use persistent volumes?",
    "How do Prometheus relabel configs work?",
    "kubernetes security best practices",
    "What hydration ratio should a sourdough starter have?",
    "What did the Mellon Foundation report in 2019?",
    "What did the Gates Foundation annual report say?",
    "Who won the 2018 FIFA World Cup final?",
    "What does the Xylophone Quarks Institute study?",
    "What do community foundation annual reports cover?",
]

NOT_ANSWERABLE = [
    "How do I rebuild a carburettor on a 1974 motorcycle?",
    "best hiking trails in Patagonia in winter",
    "What is the melting point of tungsten?",
    "How does photosynthesis work?",
    "Who directed the film Casablanca?",
    "What are the rules of cricket LBW?",
    "What is the capital of Mongolia?",
    "How do I train for a marathon?",
]

# The pair that shows why a floor cannot do this job. Kept alongside the others
# so the comparison is visible rather than argued:
#
#   the first has NO right answer in the corpus, and its nearest document is a
#   real annual report belonging to somebody else - admitting it invites a
#   confident answer about a foundation that does not exist;
#
#   the second HAS its answer in the corpus and finds it, but only at a much
#   greater distance, because the question is phrased loosely.
#
# The wrong subject is nearer than the right one. Any floor admitting the second
# admits the first, which is what "admits N/2 wrong-subject" below is counting.
UNSEPARABLE = [
    "What did the Quazzlemint Foundation conclude in its 2019 report?",
    "What did the Mellon Foundation conclude?",
]


def run(label, queries):
    print(f"\n=== {label} ===")
    distances = []
    for query in queries:
        hits = server.store.retrieve(query, k=1)  # no floor: the raw neighbour
        if not hits:
            print(f"  {query[:44]:46s}   (corpus empty)")
            continue
        distances.append(hits[0]["distance"])
        snippet = hits[0]["text"][:58].replace("\n", " ")
        print(f"  {query[:44]:46s} {hits[0]['distance']:.3f}  {snippet}")
    return distances


answerable = run("CORPUS ANSWERS THIS", ANSWERABLE)
absent = run("CORPUS DOES NOT", NOT_ANSWERABLE)
near = run("WHERE A FLOOR CANNOT HELP", UNSEPARABLE)

print("\n--- summary ---")
print(f"  answerable : {min(answerable):.3f} - {max(answerable):.3f}")
print(f"  absent     : {min(absent):.3f} - {max(absent):.3f}")
print(f"  unseparable: {min(near):.3f} - {max(near):.3f}   <- straddles the answerable tail")

print(f"\n--- what each floor would do (in force: {server.MAX_MATCH_DISTANCE}) ---")
for floor in (0.70, 0.75, 0.80, 0.85, 0.90):
    kept = sum(1 for d in answerable if d <= floor)
    leaked = sum(1 for d in absent if d <= floor)
    admits = sum(1 for d in near if d <= floor)
    print(
        f"  {floor:.2f}: keeps {kept}/{len(answerable)} real, "
        f"admits {leaked}/{len(absent)} unrelated, "
        f"admits {admits}/{len(near)} wrong-subject"
    )
