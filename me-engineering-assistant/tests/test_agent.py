"""Tests for the LangGraph agent end-to-end behaviour."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="module")
def agent():
    """Create the agent once and reuse it across all tests in this module."""
    from me_assistant.agent.graph import create_agent  # pylint: disable=import-outside-toplevel
    return create_agent()


@pytest.mark.integration
def test_agent_creates_without_error(agent):
    """Agent initialisation must complete without raising exceptions."""
    assert agent is not None


@pytest.mark.integration
def test_single_source_query_returns_temperature(agent):
    """A single-source ECU-700 query should return the +85°C specification."""
    from me_assistant.agent.graph import query_agent  # pylint: disable=import-outside-toplevel
    answer = query_agent(agent, "What is the maximum operating temperature for the ECU-750?")
    assert answer is not None
    assert len(answer) > 20
    assert any(kw in answer for kw in ["85", "°C", "temperature"]), (
        f"Expected temperature data in answer, got: {answer[:200]}"
    )


@pytest.mark.integration
def test_cross_series_query_references_both_models(agent):
    """A cross-series query should reference both ECU-750 and ECU-850 data."""
    from me_assistant.agent.graph import query_agent  # pylint: disable=import-outside-toplevel
    answer = query_agent(agent, "Compare the CAN bus capabilities of ECU-750 and ECU-850.")
    assert any(kw in answer for kw in ["750", "ECU-750"]), (
        "Answer should reference ECU-750"
    )
    assert any(kw in answer for kw in ["850", "ECU-850"]), (
        "Answer should reference ECU-850"
    )
    assert any(kw in answer for kw in ["Mbps", "channel", "CAN"]), (
        "Answer should include CAN bus technical details"
    )


@pytest.mark.integration
def test_response_time_under_10_seconds(agent):
    """Agent must respond within the 10-second SLA for a simple query."""
    from me_assistant.agent.graph import query_agent  # pylint: disable=import-outside-toplevel
    t_start = time.monotonic()
    query_agent(agent, "How much RAM does the ECU-850 have?")
    elapsed = time.monotonic() - t_start
    assert elapsed < 10.0, (
        f"Response took {elapsed:.2f}s — exceeds the 10-second SLA"
    )
