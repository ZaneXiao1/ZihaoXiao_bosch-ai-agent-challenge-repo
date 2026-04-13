"""
Custom exception hierarchy for the ME Engineering Assistant.

Provides structured error types so callers can distinguish between
configuration issues, embedding failures, retrieval problems, and
LLM errors — enabling actionable error messages instead of raw tracebacks.
"""


class MEAssistantError(Exception):
    """Base exception for all ME Engineering Assistant errors."""


class ConfigurationError(MEAssistantError):
    """Missing environment variables, unsupported provider, or import failures."""


class EmbeddingError(MEAssistantError):
    """Embedding API failures during indexing or retrieval."""


class RetrievalError(MEAssistantError):
    """Vector store query failures."""


class LLMError(MEAssistantError):
    """LLM API failures — authentication, rate limiting, timeouts, etc."""


class DocumentLoadError(MEAssistantError):
    """Failures loading or parsing ECU documentation files."""
