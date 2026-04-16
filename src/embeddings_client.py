"""Provider-agnostic embeddings client with safe error handling."""

from __future__ import annotations

import os
from typing import Any

from src.llm_client import load_env_file
from src.utils import normalize_text


DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}

EMBEDDING_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class EmbeddingsClientError(Exception):
    """Base exception for safe embeddings failures."""


class UnsupportedEmbeddingProviderError(EmbeddingsClientError):
    """Raised when the requested embeddings provider is unsupported."""


class EmbeddingProviderConfigError(EmbeddingsClientError):
    """Raised when the provider is missing configuration."""


class EmbeddingProviderRequestError(EmbeddingsClientError):
    """Raised when the provider request fails."""


def get_available_embedding_providers() -> dict[str, bool]:
    """Return which embeddings providers have an API key configured."""
    load_env_file()
    return {
        provider: bool(os.getenv(env_key))
        for provider, env_key in EMBEDDING_ENV_KEYS.items()
    }


def generate_embedding(
    text: str,
    provider: str = "openai",
    model: str | None = None,
) -> list[float]:
    """Generate one embedding vector for the provided text."""
    embeddings = generate_embeddings([text], provider=provider, model=model)
    return embeddings[0]


def generate_embeddings(
    texts: list[str],
    provider: str = "openai",
    model: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts using the selected provider."""
    load_env_file()

    normalized_provider = normalize_text(provider, lowercase=True)
    if normalized_provider not in EMBEDDING_ENV_KEYS:
        raise UnsupportedEmbeddingProviderError("Proveedor de embeddings no soportado.")

    prepared_texts = [_validate_text(text) for text in texts]
    api_key = os.getenv(EMBEDDING_ENV_KEYS[normalized_provider])
    if not api_key:
        raise EmbeddingProviderConfigError(
            "El proveedor solicitado no tiene una API key configurada."
        )

    selected_model = model or DEFAULT_EMBEDDING_MODELS[normalized_provider]

    try:
        if normalized_provider == "openai":
            return _generate_with_openai(api_key, prepared_texts, selected_model)
        if normalized_provider == "gemini":
            return _generate_with_gemini(api_key, prepared_texts, selected_model)
    except EmbeddingsClientError:
        raise
    except Exception as error:
        provider_label = "OpenAI" if normalized_provider == "openai" else "Gemini"
        raise EmbeddingProviderRequestError(
            f"Fallo la solicitud al proveedor {provider_label} embeddings."
        ) from error

    raise UnsupportedEmbeddingProviderError("Proveedor de embeddings no soportado.")


def _validate_text(text: str) -> str:
    """Validate and normalize an input text before requesting embeddings."""
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Los embeddings requieren textos no vacios.")
    return normalized


def _generate_with_openai(
    api_key: str,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Generate embeddings using the OpenAI Python SDK."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise EmbeddingProviderRequestError(
            "No fue posible cargar el cliente de OpenAI embeddings."
        ) from error

    try:
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        response = client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )
        data = getattr(response, "data", None) or []
        embeddings = [_extract_openai_embedding(item) for item in data]
    except EmbeddingsClientError:
        raise
    except Exception as error:
        raise EmbeddingProviderRequestError(
            "Fallo la solicitud al proveedor OpenAI embeddings."
        ) from error

    _validate_embeddings_shape(texts, embeddings)
    return embeddings


def _generate_with_gemini(
    api_key: str,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Generate embeddings using the Gemini Python SDK."""
    try:
        from google import genai
    except ImportError as error:
        raise EmbeddingProviderRequestError(
            "No fue posible cargar el cliente de Gemini embeddings."
        ) from error

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(model=model, contents=texts)
        raw_embeddings = getattr(response, "embeddings", None) or []
        embeddings = [_extract_gemini_embedding(item) for item in raw_embeddings]
    except EmbeddingsClientError:
        raise
    except Exception as error:
        raise EmbeddingProviderRequestError(
            "Fallo la solicitud al proveedor Gemini embeddings."
        ) from error

    _validate_embeddings_shape(texts, embeddings)
    return embeddings


def _extract_openai_embedding(item: Any) -> list[float]:
    """Extract an embedding vector from an OpenAI response item."""
    vector = getattr(item, "embedding", None)
    if vector is None and isinstance(item, dict):
        vector = item.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise EmbeddingProviderRequestError("OpenAI embeddings devolvio un vector invalido.")
    return [float(value) for value in vector]


def _extract_gemini_embedding(item: Any) -> list[float]:
    """Extract an embedding vector from a Gemini response item."""
    vector = getattr(item, "values", None)
    if vector is None and isinstance(item, dict):
        vector = item.get("values")
    if vector is None:
        inner_embedding = getattr(item, "embedding", None)
        if inner_embedding is not None:
            vector = getattr(inner_embedding, "values", None)
    if vector is None and isinstance(item, dict):
        inner_embedding = item.get("embedding")
        if isinstance(inner_embedding, dict):
            vector = inner_embedding.get("values")
    if not isinstance(vector, list) or not vector:
        raise EmbeddingProviderRequestError("Gemini embeddings devolvio un vector invalido.")
    return [float(value) for value in vector]


def _validate_embeddings_shape(texts: list[str], embeddings: list[list[float]]) -> None:
    """Ensure providers return one valid vector per input text."""
    if len(texts) != len(embeddings):
        raise EmbeddingProviderRequestError(
            "El proveedor de embeddings devolvio una cantidad de vectores inconsistente."
        )

    expected_size = len(embeddings[0]) if embeddings else 0
    if expected_size == 0:
        raise EmbeddingProviderRequestError(
            "El proveedor de embeddings devolvio una respuesta vacia."
        )

    for vector in embeddings:
        if len(vector) != expected_size:
            raise EmbeddingProviderRequestError(
                "El proveedor de embeddings devolvio vectores de tamano inconsistente."
            )
