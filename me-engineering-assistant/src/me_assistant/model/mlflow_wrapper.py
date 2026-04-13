"""
MLflow pyfunc wrapper for the ME Engineering Assistant agent.

This module packages the LangGraph agent as an MLflow model that can be:
1. Logged to an MLflow tracking server
2. Loaded and served via MLflow model serving
3. Called via REST API when deployed on Databricks
"""
import logging

import mlflow
import pandas as pd

from me_assistant.agent.graph import create_agent, query_agent
from me_assistant.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
    MEAssistantError,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class MEAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ECU Engineering Assistant."""

    def load_context(self, context):
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

    def predict(self, context, model_input, params=None):
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
                answer = query_agent(self.agent, question)
                answers.append(answer)
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


def log_model() -> str:
    """Log the MEAssistantModel to MLflow.

    Returns:
        The MLflow ``run_id`` for the logged model.
    """
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        mlflow.pyfunc.log_model(
            artifact_path="me-assistant-model",
            python_model=MEAssistantModel(),
            pip_requirements=[
                "langchain>=0.3.0",
                "langchain-openai>=0.2.0",
                "langchain-community>=0.3.0",
                "langgraph>=0.2.0",
                "faiss-cpu>=1.7.4",
                "mlflow>=2.15.0",
            ],
            artifacts={
                "ecu_700_doc": "src/me_assistant/data/ECU-700_Series_Manual.md",
                "ecu_800_base_doc": "src/me_assistant/data/ECU-800_Series_Base.md",
                "ecu_800_plus_doc": "src/me_assistant/data/ECU-800_Series_Plus.md",
            },
        )
        logger.info("Model logged with run_id: %s", run.info.run_id)
        return run.info.run_id
