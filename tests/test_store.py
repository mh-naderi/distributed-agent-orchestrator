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
