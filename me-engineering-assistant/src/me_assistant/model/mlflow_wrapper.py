"""
MLflow pyfunc wrapper for the ME Engineering Assistant agent.

This module packages the LangGraph agent as an MLflow model that can be:
1. Logged to an MLflow tracking server
2. Loaded and served via MLflow model serving
3. Called via REST API when deployed on Databricks
"""
import logging
from typing import Any, Optional

import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from me_assistant.agent.graph import create_agent, query_agent
from me_assistant.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
    MEAssistantError,
    RetrievalError,
)

logger = logging.getLogger(__name__)

# Pinned to match pyproject.toml. faiss was never actually used — the
# project uses langchain-chroma; shipping faiss-cpu would break serving.
_PIP_REQUIREMENTS = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-chroma>=0.1.0",
    "langgraph>=0.2.0",
    "chromadb>=0.5.0",
    "mlflow>=2.15.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0.0",
]


class MEAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ECU Engineering Assistant."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Initialise the agent when the model is loaded.

        Called once at model load time, not on every predict() call.
        All heavy initialisation (vector stores, model loading) is done here.

        Args:
            context: MLflow context object (required by interface).

        Raises:
            MEAssistantError: If agent creation fails (propagated so MLflow
                serving knows the model failed to load).
        """
        try:
            self.agent = create_agent()  # pylint: disable=attribute-defined-outside-init
        except MEAssistantError:
            logger.error("Agent failed to initialise", exc_info=True)
            raise
        except Exception as exc:
            logger.error("Unexpected error during agent init: %s", exc, exc_info=True)
            raise ConfigurationError(
                f"Failed to initialise agent: {exc}"
            ) from exc

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input,
        params=None,
    ) -> list[str]:
        """Process user queries and return agent responses.

        Args:
            context: MLflow context (required by interface, unused here).
            model_input: :class:`~pandas.DataFrame` with a ``question`` column,
                         or a ``dict`` with a ``"question"`` key.
            params: Optional parameters (unused).

        Returns:
            List of answer strings, one per input question.
            On failure, the error string is prefixed with the error category
            (e.g. ``"Configuration error: ..."``).
        """
        if isinstance(model_input, pd.DataFrame):
            if "question" not in model_input.columns:
                return [
                    "Configuration error: input DataFrame must contain "
                    "a 'question' column"
                ]
            questions = model_input["question"].tolist()
        elif isinstance(model_input, dict):
            if "question" not in model_input:
                return [
                    "Configuration error: input dict must contain "
                    "a 'question' key"
                ]
            questions = [model_input["question"]]
        else:
            questions = [str(model_input)]

        answers = []
        for question in questions:
            try:
                result = query_agent(self.agent, question)
                answers.append(result["answer"])
            except ConfigurationError as exc:
                logger.error("Configuration error for '%s': %s", question, exc)
                answers.append(f"Configuration error: {exc}")
            except EmbeddingError as exc:
                logger.error("Embedding error for '%s': %s", question, exc)
                answers.append(f"Embedding error: {exc}")
            except RetrievalError as exc:
                logger.error("Retrieval error for '%s': %s", question, exc)
                answers.append(f"Retrieval error: {exc}")
            except LLMError as exc:
                logger.error("LLM error for '%s': %s", question, exc)
                answers.append(f"LLM error: {exc}")
            except MEAssistantError as exc:
                logger.error("Assistant error for '%s': %s", question, exc)
                answers.append(f"Error: {exc}")
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Unexpected error for '%s': %s", question, exc,
                    exc_info=True,
                )
                answers.append(f"Error processing query: {exc}")

        return answers


def _build_signature() -> ModelSignature:
    """Return the MLflow signature: DataFrame[question:str] -> list[str]."""
    input_schema = Schema([ColSpec("string", "question")])
    output_schema = Schema([ColSpec("string")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def _build_input_example() -> pd.DataFrame:
    """Return a minimal input example for the model signature."""
    return pd.DataFrame(
        {"question": ["What is the maximum operating temperature for the ECU-750?"]}
    )


def log_model(
    artifact_path: str = "me-assistant-model",
    registered_model_name: Optional[str] = None,
) -> mlflow.models.model.ModelInfo:
    """Log the MEAssistantModel to the active MLflow run.

    Must be called inside an active ``mlflow.start_run()`` context so that
    the caller (e.g. ``train_and_log_model.py``) owns run lifecycle and
    parameter/metric logging.

    Args:
        artifact_path: Path under the run's artifact root.
        registered_model_name: If provided, register the model in MLflow
            Model Registry under this name.

    Returns:
        The :class:`~mlflow.models.model.ModelInfo` returned by
        :func:`mlflow.pyfunc.log_model`.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_model() must be called inside an active mlflow.start_run() "
            "context — caller is responsible for run lifecycle."
        )

    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=MEAssistantModel(),
        pip_requirements=_PIP_REQUIREMENTS,
        signature=_build_signature(),
        input_example=_build_input_example(),
        registered_model_name=registered_model_name,
    )
    logger.info("Model logged: uri=%s", model_info.model_uri)
    return model_info
