"""
Tests for the retrieval agent's vector store.

Embeddings are faked with a deterministic function, so these run without Ollama
and without any server - the store takes its embedder as an argument precisely
so this is possible. What's under test is the storage and search behaviour, not
the quality of a particular embedding model.
"""

import pytest

from agents.retrieval_agent.store import EMBEDDING_DIM, VectorStore, chunk

# Keywords mapped to distinct axes in the vector space, so "nearest" is
# predictable and the assertions below mean something.
KEYWORD_AXES = {"kubernetes": 0, "protocol": 1, "database": 2}


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIM
        for keyword, axis in KEYWORD_AXES.items():
            if keyword in text.lower():
                vector[axis] = 1.0
        vector[EMBEDDING_DIM - 1] = 0.01  # keep non-matching vectors non-zero
        vectors.append(vector)
    return vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(fake_embed, db_path=str(tmp_path / "test.db"))


def test_index_then_retrieve_finds_the_matching_document(store):
    store.index(
        [
            "Kubernetes schedules pods across nodes",
            "MCP is a protocol for tool calling",
            "Postgres is a relational database",
        ],
        source="unit-test",
    )

    hits = store.retrieve("kubernetes scheduling", k=1)

    assert len(hits) == 1
    assert "Kubernetes" in hits[0]["text"]
    assert hits[0]["source"] == "unit-test"


def test_results_are_ordered_nearest_first(store):
    store.index(["a protocol document", "a kubernetes document"], source="t")

    hits = store.retrieve("protocol", k=2)

    assert len(hits) == 2
    assert hits[0]["distance"] <= hits[1]["distance"]
    assert "protocol" in hits[0]["text"]


def test_index_survives_reopening_the_same_file(tmp_path):
    """
    The whole reason this agent is a StatefulSet rather than a Deployment: the
    corpus has to outlive the process that wrote it.
    """
    path = str(tmp_path / "persist.db")

    first = VectorStore(fake_embed, db_path=path)
    first.index(["a kubernetes document"], source="first-run")

    second = VectorStore(fake_embed, db_path=path)

    assert second.count() == 1
    assert "kubernetes" in second.retrieve("kubernetes", k=1)[0]["text"]


def test_retrieve_on_an_empty_index_returns_nothing(store):
    assert store.retrieve("anything", k=5) == []


def test_k_bounds_the_number_of_results(store):
    store.index([f"kubernetes document {i}" for i in range(10)], source="t")

    assert len(store.retrieve("kubernetes", k=3)) == 3


def test_blank_and_empty_texts_are_skipped(store):
    assert store.index(["", "   ", "\n"], source="t") == 0
    assert store.count() == 0


def test_wrong_dimension_embedding_is_rejected(tmp_path):
    """Catch a mismatched embedding model at insert, not at search time."""
    store = VectorStore(lambda texts: [[0.0] * 10 for _ in texts], db_path=str(tmp_path / "d.db"))

    with pytest.raises(ValueError, match="10 dimensions"):
        store.index(["some text"], source="t")


def test_embedder_returning_wrong_count_is_rejected(tmp_path):
    store = VectorStore(lambda texts: [[0.0] * EMBEDDING_DIM], db_path=str(tmp_path / "c.db"))

    with pytest.raises(ValueError, match="1 vectors for 2 texts"):
        store.index(["one", "two"], source="t")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_indexing_the_same_text_twice_stores_it_once(store):
    """
    Duplicates are not merely untidy - they crowd the results.

    retrieve() returns the k nearest rows, and identical rows are equally
    near, so a duplicated document can occupy the whole result set. This
    corpus accumulated three copies of one test document and a top-2
    retrieval came back with the same text twice.
    """
    assert store.index(["MCP is a protocol for tool calling"], "first") == 1
    assert store.index(["MCP is a protocol for tool calling"], "second") == 0

    results = store.retrieve("protocol", k=5)
    assert len(results) == 1


def test_duplicates_within_one_batch_are_collapsed(store):
    """Counted, not set-compared - a set cannot tell one copy from three."""
    stored = store.index(["same text", "same text", "different text"], "batch")

    assert stored == 2
    assert len(store.retrieve("text", k=10)) == 2


def test_a_duplicate_no_longer_displaces_a_real_result(store):
    """The failure this fixes, stated as a retrieval outcome."""
    store.index(["MCP is a protocol", "MCP is a protocol"], "dupes")
    store.index(["Kubernetes schedules pods"], "other")

    texts = [r["text"] for r in store.retrieve("protocol", k=2)]

    # Two slots, two DIFFERENT documents - previously both went to the dupe.
    assert len(texts) == len(set(texts)) == 2


def test_known_text_is_not_re_embedded(tmp_path):
    """Filtering happens before the embed call, which is what index() spends."""
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    store = VectorStore(counting_embed, db_path=str(tmp_path / "t.db"))
    store.index(["MCP is a protocol"], "first")
    store.index(["MCP is a protocol"], "again")

    # One embed for the first index, none for the repeat.
    assert len(calls) == 1


def test_chunk_splits_search_results_into_one_document_each():
    """
    The research agent joins its results with blank lines, so passing its output
    straight through must yield one document per result rather than one blob.
    """
    search_output = (
        "Result One\nbody one\nSource: http://a"
        "\n\n"
        "Result Two\nbody two\nSource: http://b"
        "\n\n"
        "Result Three\nbody three\nSource: http://c"
    )

    chunks = chunk([search_output])

    assert len(chunks) == 3
    assert chunks[0].startswith("Result One")
    assert chunks[2].startswith("Result Three")


def test_chunk_drops_empty_blocks():
    assert chunk(["a\n\n\n\nb", "", "   "]) == ["a", "b"]


def test_chunk_leaves_single_paragraph_intact():
    assert chunk(["one coherent paragraph"]) == ["one coherent paragraph"]


def test_chunking_improves_match_quality(tmp_path):
    """
    The point of chunking, demonstrated: a blob containing an unrelated document
    is a worse match for a query than that document indexed on its own.
    """
    blob = "a kubernetes document\n\na protocol document\n\na database document"

    blobbed = VectorStore(fake_embed, db_path=str(tmp_path / "blob.db"))
    blobbed.index([blob], source="unchunked")

    chunked = VectorStore(fake_embed, db_path=str(tmp_path / "chunk.db"))
    chunked.index(chunk([blob]), source="chunked")

    assert blobbed.count() == 1
    assert chunked.count() == 3
    assert chunked.retrieve("protocol", k=1)[0]["distance"] < blobbed.retrieve("protocol", k=1)[0]["distance"]


# ---------------------------------------------------------------------------
# The relevance floor
# ---------------------------------------------------------------------------
# Nearest-neighbour search always returns k rows if the corpus holds k
# documents, so "closest" silently reads as "match". That is not a hypothetical:
# a question about an entity that does not exist came back with real documents
# at distance 0.768, and the model answered from them.


def test_without_a_floor_a_far_neighbour_is_still_returned(store):
    """The behaviour the floor exists to bound. Kept as the baseline it is."""
    store.index(["kubernetes runs containers"], "fixture")

    hits = store.retrieve("protocol")

    assert len(hits) == 1, "nearest-neighbour returns something regardless"
    assert hits[0]["distance"] > 1.0


def test_a_far_neighbour_is_dropped_when_a_floor_is_set(store):
    store.index(["kubernetes runs containers"], "fixture")

    assert store.retrieve("protocol", max_distance=0.5) == []


def test_a_genuine_match_survives_the_floor(store):
    store.index(["kubernetes runs containers"], "fixture")

    hits = store.retrieve("kubernetes", max_distance=0.5)

    assert len(hits) == 1
    assert "kubernetes" in hits[0]["text"]


def test_the_floor_keeps_the_near_and_drops_the_far_in_one_query(store):
    """
    The partial case. A query can match one document well and another barely;
    dropping only the second is the whole point, and returning k rows because
    k rows exist is what produced the fabrication.
    """
    store.index(["kubernetes runs containers", "protocol defines messages"], "fixture")

    hits = store.retrieve("kubernetes", max_distance=0.5)

    assert len(hits) == 1
    assert "kubernetes" in hits[0]["text"]


def test_the_floor_is_inclusive_at_the_boundary(store):
    """An exact match sits at distance 0, so a floor of 0 must not reject it."""
    store.index(["kubernetes runs containers"], "fixture")

    assert len(store.retrieve("kubernetes", max_distance=0.0)) == 1


def test_ordering_survives_filtering(store):
    store.index(
        ["kubernetes runs containers", "kubernetes and protocol together"], "fixture"
    )

    hits = store.retrieve("kubernetes", max_distance=1.5)

    distances = [hit["distance"] for hit in hits]
    assert distances == sorted(distances), "nearest must still come first"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# A document's source used to be whatever the caller said about the whole batch,
# and the research agent said the search QUERY. A search for something that does
# not exist therefore filed real documents under its name.


def test_a_document_that_names_its_origin_keeps_it(store):
    store.index(
        ["2019 Report\nGrants supported exemplary work.\nSource: https://example.org/a.pdf"],
        "web-search",
    )

    hit = store.retrieve("grants")[0]

    assert hit["source"] == "https://example.org/a.pdf"


def test_the_callers_label_is_only_a_fallback(store):
    store.index(["kubernetes runs containers"], "eval-fixture")

    assert store.retrieve("kubernetes")[0]["source"] == "eval-fixture"


def test_each_document_in_a_batch_gets_its_own_origin(store):
    """
    The batch is the unit the caller sends; the document is the unit that gets a
    source. Collapsing the two is what let one label speak for five results.
    """
    store.index(
        [
            "kubernetes primer\nSource: https://example.org/k8s",
            "protocol primer\nSource: https://example.net/proto",
        ],
        "web-search",
    )

    by_text = {hit["text"][:10]: hit["source"] for hit in store.retrieve("kubernetes", k=5)}

    assert by_text["kubernetes"] == "https://example.org/k8s"
    assert by_text["protocol p"] == "https://example.net/proto"


def test_a_false_premise_query_cannot_become_provenance(store):
    """
    The exact contamination, reproduced. A search for a foundation that does not
    exist returns real documents about a different one; they must not be stored
    under a label asserting they are about the fictional entity.
    """
    result = (
        "2019 Report - assets.ctfassets.net\n"
        "Foundation grants in 2019 supported exemplary work.\n"
        "Source: https://assets.ctfassets.net/mellonannualreport_2019.pdf"
    )

    store.index([result], "web-search")

    hit = store.retrieve("foundation grants 2019")[0]
    assert "Quazzlemint" not in hit["source"]
    assert hit["source"] == "https://assets.ctfassets.net/mellonannualreport_2019.pdf"


def test_a_quoted_source_line_in_the_body_does_not_win(store):
    """The real origin is appended last, so the last line is the one that counts."""
    store.index(
        [
            "kubernetes notes\nSomeone wrote Source: https://example.org/quoted\n"
            "Source: https://example.org/real"
        ],
        "web-search",
    )

    assert store.retrieve("kubernetes")[0]["source"] == "https://example.org/real"
