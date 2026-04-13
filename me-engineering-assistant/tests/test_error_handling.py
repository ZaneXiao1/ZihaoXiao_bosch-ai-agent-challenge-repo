"""Unit tests for error handling — no API key required.

These tests verify that:
- validate_config() catches missing env vars for each provider
- validate_config() rejects unknown providers
- predict() returns structured error messages by category
- load_context() propagates MEAssistantError on init failure
- Input validation catches missing 'question' column/key
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from me_assistant.exceptions import (  # noqa: E402
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    LLMError,
    MEAssistantError,
    RetrievalError,
)


# ---------------------------------------------------------------------------
# validate_config() tests
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for config.validate_config()."""

    @patch.dict("os.environ", {"MODEL_PROVIDER": "openai", "OPENAI_API_KEY": ""}, clear=False)
    def test_missing_openai_key(self):
        """Should raise ConfigurationError when OPENAI_API_KEY is empty."""
        from me_assistant.config import validate_config  # noqa: E402
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            validate_config()

    @patch.dict(
        "os.environ",
        {
            "MODEL_PROVIDER": "databricks",
            "DATABRICKS_HOST": "",
            "DATABRICKS_TOKEN": "",
            "LLM_MODEL_NAME": "",
            "EMBEDDING_MODEL_NAME": "",
        },
        clear=False,
    )
    def test_missing_databricks_vars(self):
        """Should raise ConfigurationError listing all missing Databricks vars."""
        from me_assistant.config import validate_config
        with pytest.raises(ConfigurationError, match="DATABRICKS_HOST"):
            validate_config()

    @patch.dict(
        "os.environ",
        {"MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-key"},
        clear=False,
    )
    def test_valid_openai_config(self):
        """Should not raise when OpenAI config is complete."""
        from me_assistant.config import validate_config
        validate_config()  # no exception expected

    @patch.dict("os.environ", {"MODEL_PROVIDER": "invalid_provider"}, clear=False)
    def test_unknown_provider(self):
        """Should raise ConfigurationError for unsupported MODEL_PROVIDER."""
        from me_assistant.config import validate_config
        with pytest.raises(ConfigurationError, match="Unknown MODEL_PROVIDER"):
            validate_config()

    @patch.dict(
        "os.environ",
        {
            "MODEL_PROVIDER": "databricks",
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "dapi-test",
            "LLM_MODEL_NAME": "my-llm",
            "EMBEDDING_MODEL_NAME": "my-embed",
        },
        clear=False,
    )
    def test_valid_databricks_config(self):
        """Should not raise when Databricks config is complete."""
        from me_assistant.config import validate_config
        validate_config()  # no exception expected


# ---------------------------------------------------------------------------
# predict() structured error tests
# ---------------------------------------------------------------------------

class TestPredictErrorMessages:
    """Tests that predict() returns the correct error prefix per exception type."""

    def _make_model(self, side_effect):
        """Create an MEAssistantModel with a mocked agent that raises *side_effect*."""
        from me_assistant.model.mlflow_wrapper import MEAssistantModel
        model = MEAssistantModel()
        model.agent = MagicMock()
        model.agent.invoke.side_effect = side_effect
        return model

    def test_configuration_error_prefix(self):
        from me_assistant.model.mlflow_wrapper import MEAssistantModel
        model = self._make_model(ConfigurationError("bad config"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("Configuration error:")

    def test_embedding_error_prefix(self):
        model = self._make_model(EmbeddingError("embed fail"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("Embedding error:")

    def test_retrieval_error_prefix(self):
        model = self._make_model(RetrievalError("store down"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("Retrieval error:")

    def test_llm_error_prefix(self):
        model = self._make_model(LLMError("rate limited"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("LLM error:")

    def test_generic_runtime_error_wrapped_as_llm_error(self):
        """RuntimeError from agent.invoke is wrapped by query_agent into LLMError."""
        model = self._make_model(RuntimeError("something broke"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("LLM error:")

    def test_base_assistant_error_prefix(self):
        model = self._make_model(MEAssistantError("general"))
        result = model.predict(None, {"question": "test"})
        assert result[0].startswith("Error:")


# ---------------------------------------------------------------------------
# predict() input validation tests
# ---------------------------------------------------------------------------

class TestPredictInputValidation:
    """Tests that predict() validates input format."""

    def _make_model_ok(self):
        from me_assistant.model.mlflow_wrapper import MEAssistantModel
        model = MEAssistantModel()
        model.agent = MagicMock()
        return model

    def test_dataframe_missing_question_column(self):
        model = self._make_model_ok()
        bad_df = pd.DataFrame({"text": ["hello"]})
        result = model.predict(None, bad_df)
        assert "Configuration error" in result[0]
        assert "question" in result[0]

    def test_dict_missing_question_key(self):
        model = self._make_model_ok()
        result = model.predict(None, {"text": "hello"})
        assert "Configuration error" in result[0]
        assert "question" in result[0]


# ---------------------------------------------------------------------------
# load_context() failure tests
# ---------------------------------------------------------------------------

class TestLoadContextErrors:
    """Tests that load_context() propagates errors correctly."""

    @patch("me_assistant.model.mlflow_wrapper.create_agent")
    def test_load_context_propagates_config_error(self, mock_create):
        mock_create.side_effect = ConfigurationError("no API key")
        from me_assistant.model.mlflow_wrapper import MEAssistantModel
        model = MEAssistantModel()
        with pytest.raises(ConfigurationError, match="no API key"):
            model.load_context(None)

    @patch("me_assistant.model.mlflow_wrapper.create_agent")
    def test_load_context_wraps_unexpected_error(self, mock_create):
        mock_create.side_effect = RuntimeError("unexpected")
        from me_assistant.model.mlflow_wrapper import MEAssistantModel
        model = MEAssistantModel()
        with pytest.raises(ConfigurationError, match="Failed to initialise agent"):
            model.load_context(None)


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """Verify the exception hierarchy structure."""

    def test_all_exceptions_inherit_from_base(self):
        for exc_cls in (ConfigurationError, EmbeddingError,
                        RetrievalError, LLMError, DocumentLoadError):
            assert issubclass(exc_cls, MEAssistantError)

    def test_base_inherits_from_exception(self):
        assert issubclass(MEAssistantError, Exception)
