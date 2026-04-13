"""Tests for document loading and FAISS vector store retrieval."""
import sys
from pathlib import Path

import pytest

# Make the package importable when pytest is run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = str(Path(__file__).parent.parent / "src" / "me_assistant" / "data")


@pytest.fixture(scope="module")
def documents():
    """Load documents once for the entire test module."""
    from me_assistant.documents.loader import load_documents  # pylint: disable=import-outside-toplevel
    return load_documents(DATA_DIR)


def test_load_documents_returns_expected_count(documents):
    """Document loading should produce between 8 and 15 chunks."""
    assert 8 <= len(documents) <= 15, (
        f"Expected 8–15 chunks, got {len(documents)}"
    )


def test_each_chunk_has_required_metadata(documents):
    """Every chunk must carry series, models_covered, and source metadata."""
    for doc in documents:
        assert "series" in doc.metadata, "Missing 'series' metadata"
        assert "models_covered" in doc.metadata, "Missing 'models_covered' metadata"
        assert "source" in doc.metadata, "Missing 'source' metadata"
        assert doc.metadata["series"] in ("ECU-700", "ECU-800")


def test_context_prefix_injected(documents):
    """Every chunk's page_content must start with the ECU context prefix."""
    for doc in documents:
        assert doc.page_content.startswith("[ECU-"), (
            f"Missing context prefix in chunk from '{doc.metadata.get('source')}'"
        )


def test_both_series_present(documents):
    """Both ECU-700 and ECU-800 series chunks must be loaded."""
    series_set = {d.metadata["series"] for d in documents}
    assert "ECU-700" in series_set, "No ECU-700 chunks loaded"
    assert "ECU-800" in series_set, "No ECU-800 chunks loaded"


def test_ecu850b_chunks_have_correct_model(documents):
    """ECU-850b chunks must list ECU-850b in models_covered, not ECU-850."""
    plus_chunks = [d for d in documents if "ECU-800_Series_Plus" in d.metadata["source"]]
    assert plus_chunks, "No ECU-800_Series_Plus chunks found"
    for doc in plus_chunks:
        assert "ECU-850b" in doc.metadata["models_covered"]


@pytest.mark.integration
def test_build_vector_stores_returns_two_indices():
    """build_vector_stores should return both ECU-700 and ECU-800 FAISS indices."""
    from me_assistant.documents.store import build_vector_stores  # pylint: disable=import-outside-toplevel
    stores = build_vector_stores(DATA_DIR)
    assert "ecu_700" in stores
    assert "ecu_800" in stores
    assert "raw_docs" in stores


@pytest.mark.integration
def test_retriever_returns_results_for_temperature_query():
    """Retriever must return at least one document for a temperature query."""
    from me_assistant.documents.store import build_vector_stores  # pylint: disable=import-outside-toplevel
    stores = build_vector_stores(DATA_DIR)
    retriever = stores["ecu_700"].as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke("operating temperature ECU-750")
    assert len(docs) > 0, "No documents returned for temperature query"
