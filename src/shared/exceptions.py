"""
src/shared/exceptions.py

Custom exceptions for FinSight.
Using specific exceptions makes error handling precise — you know exactly
what went wrong without reading stack traces.
"""


class FinSightError(Exception):
    """Base exception for all FinSight errors."""
    pass


class DocumentNotFoundError(FinSightError):
    """Raised when a requested document does not exist in the store."""
    pass


class DocumentProcessingError(FinSightError):
    """Raised when PDF processing fails — corrupt file, unreadable layout, etc."""
    pass


class EmbeddingError(FinSightError):
    """Raised when the embedding API call fails."""
    pass


class RetrievalError(FinSightError):
    """Raised when the vector store query fails."""
    pass


class RerankerError(FinSightError):
    """Raised when the Cohere reranker API call fails."""
    pass


class ConfigurationError(FinSightError):
    """Raised when required configuration or API keys are missing."""
    pass


class MetadataError(FinSightError):
    """Raised when document metadata is missing or invalid."""
    pass