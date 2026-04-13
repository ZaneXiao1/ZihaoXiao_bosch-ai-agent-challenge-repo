"""
Configuration module with model provider abstraction.

Design Decision: All model instantiation is centralized here.
Switching between OpenAI (local dev) and Databricks (production)
requires only changing environment variables, zero code changes.

Environment Variables:
    MODEL_PROVIDER: "openai" or "databricks"
    OPENAI_API_KEY: Required when MODEL_PROVIDER=openai
    DATABRICKS_HOST: Required when MODEL_PROVIDER=databricks
    DATABRICKS_TOKEN: Required when MODEL_PROVIDER=databricks
    LLM_MODEL_NAME: Model name/endpoint (default varies by provider)
    EMBEDDING_MODEL_NAME: Embedding model name/endpoint
"""
import logging
import os
from dotenv import load_dotenv

from me_assistant.exceptions import ConfigurationError

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")

_VALID_PROVIDERS = ("openai", "databricks")


def validate_config() -> None:
    """Pre-check that all required environment variables are set.

    Reads ``MODEL_PROVIDER`` from the environment at call time (not from
    the module-level cache) so that it respects runtime changes.

    Raises:
        ConfigurationError: With an actionable message listing every
            missing variable for the current provider.
    """
    provider = os.getenv("MODEL_PROVIDER", "openai")
    missing: list[str] = []

    if provider not in _VALID_PROVIDERS:
        raise ConfigurationError(
            f"Unknown MODEL_PROVIDER '{provider}'. "
            f"Supported providers: {', '.join(_VALID_PROVIDERS)}"
        )

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    elif provider == "databricks":
        for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN",
                    "LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME"):
            if not os.getenv(var):
                missing.append(var)

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables for "
            f"MODEL_PROVIDER='{provider}': {', '.join(missing)}"
        )


def get_llm():
    """Return the LLM instance based on configured provider.

    Raises:
        ConfigurationError: If the provider package is missing or
            authentication fails at instantiation time.
    """
    if MODEL_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ConfigurationError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            max_retries=2,
        )
    if MODEL_PROVIDER == "databricks":
        try:
            from langchain_databricks import ChatDatabricks  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ConfigurationError(
                "langchain-databricks is not installed. "
                "Run: pip install langchain-databricks"
            ) from exc
        return ChatDatabricks(
            endpoint=os.getenv("LLM_MODEL_NAME"),
            temperature=0,
        )
    raise ConfigurationError(
        f"Unknown MODEL_PROVIDER: '{MODEL_PROVIDER}'. "
        f"Supported providers: {', '.join(_VALID_PROVIDERS)}"
    )


def get_embeddings():
    """Return the embeddings model based on configured provider.

    Raises:
        ConfigurationError: If the provider package is missing or
            authentication fails at instantiation time.
    """
    if MODEL_PROVIDER == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ConfigurationError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from exc
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        )
    if MODEL_PROVIDER == "databricks":
        try:
            from langchain_databricks import DatabricksEmbeddings  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ConfigurationError(
                "langchain-databricks is not installed. "
                "Run: pip install langchain-databricks"
            ) from exc
        return DatabricksEmbeddings(
            endpoint=os.getenv("EMBEDDING_MODEL_NAME"),
        )
    raise ConfigurationError(
        f"Unknown MODEL_PROVIDER: '{MODEL_PROVIDER}'. "
        f"Supported providers: {', '.join(_VALID_PROVIDERS)}"
    )
