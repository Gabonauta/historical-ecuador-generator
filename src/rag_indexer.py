"""Build and persist the local RAG index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings_client import (
    DEFAULT_EMBEDDING_MODELS,
    EmbeddingsClientError,
    generate_embeddings,
    get_available_embedding_providers,
)
from src.loader import save_json
from src.rag_chunker import build_chunks
from src.utils import normalize_text


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAG_DIR = DATA_DIR / "rag"
CHUNKS_PATH = RAG_DIR / "chunks.json"
EMBEDDINGS_PATH = RAG_DIR / "embeddings.npy"
METADATA_PATH = RAG_DIR / "metadata.json"
INDEX_BACKEND = "numpy-cosine"
INDEX_VERSION = "phase4-v1"


class RAGIndexError(Exception):
    """Raised when the local RAG index cannot be built."""


def build_and_save_index(
    entities: list[dict[str, Any]],
    embedding_provider: str = "openai",
    model: str | None = None,
    *,
    chunks_path: str | Path = CHUNKS_PATH,
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> dict[str, Any]:
    """Build a local vector index from entities and persist it to disk."""
    chunks = build_chunks(entities)
    if not chunks:
        raise RAGIndexError("No se generaron chunks validos para construir el indice RAG.")

    texts = [chunk["texto"] for chunk in chunks]
    selected_provider = normalize_text(embedding_provider, lowercase=True) or "openai"
    selected_model = model or DEFAULT_EMBEDDING_MODELS.get(selected_provider)
    provider_used = selected_provider
    fallback_used = False

    try:
        embeddings = generate_embeddings(texts, provider=selected_provider, model=selected_model)
    except EmbeddingsClientError as error:
        alternative_provider = _find_fallback_provider(selected_provider)
        if alternative_provider is None:
            raise RAGIndexError("No fue posible generar embeddings para el indice RAG.") from error

        provider_used = alternative_provider
        selected_model = model or DEFAULT_EMBEDDING_MODELS[provider_used]
        fallback_used = True
        try:
            embeddings = generate_embeddings(texts, provider=provider_used, model=selected_model)
        except EmbeddingsClientError as fallback_error:
            raise RAGIndexError("No fue posible generar embeddings para el indice RAG.") from fallback_error

    matrix = _normalize_embeddings_matrix(embeddings)
    resolved_chunks_path = Path(chunks_path)
    resolved_embeddings_path = Path(embeddings_path)
    resolved_metadata_path = Path(metadata_path)

    resolved_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(resolved_chunks_path, chunks)
    np.save(resolved_embeddings_path, matrix)

    metadata = {
        "version": INDEX_VERSION,
        "backend": INDEX_BACKEND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "entity_count": len({chunk["entity_id"] for chunk in chunks}),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_provider": provider_used,
        "embedding_model": selected_model,
        "fallback_provider_used": fallback_used,
    }
    save_json(resolved_metadata_path, metadata)

    return {
        "status": "ok",
        "chunk_count": len(chunks),
        "entity_count": metadata["entity_count"],
        "embedding_dimension": metadata["embedding_dimension"],
        "embedding_provider": provider_used,
        "embedding_model": selected_model,
        "backend": INDEX_BACKEND,
        "chunks_path": str(resolved_chunks_path),
        "embeddings_path": str(resolved_embeddings_path),
        "metadata_path": str(resolved_metadata_path),
        "fallback_provider_used": fallback_used,
    }


def _normalize_embeddings_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Convert raw embeddings into a normalized float32 matrix."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RAGIndexError("Los embeddings generados no tienen una forma valida para indexar.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _find_fallback_provider(selected_provider: str) -> str | None:
    """Find a configured alternative embeddings provider."""
    available = get_available_embedding_providers()
    for provider in DEFAULT_EMBEDDING_MODELS:
        if provider != selected_provider and available.get(provider):
            return provider
    return None
