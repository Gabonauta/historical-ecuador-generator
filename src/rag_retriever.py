"""Load and query the local RAG index with cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings_client import generate_embedding
from src.loader import load_json
from src.rag_indexer import CHUNKS_PATH, EMBEDDINGS_PATH, METADATA_PATH
from src.utils import normalize_text


class RAGRetrieverError(Exception):
    """Raised when the local RAG index cannot be queried safely."""


def load_index(
    chunks_path: str | Path = CHUNKS_PATH,
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> dict[str, Any]:
    """Load the persisted RAG index from local storage."""
    resolved_chunks_path = Path(chunks_path)
    resolved_embeddings_path = Path(embeddings_path)
    resolved_metadata_path = Path(metadata_path)

    missing_paths = [
        str(path)
        for path in [resolved_chunks_path, resolved_embeddings_path, resolved_metadata_path]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Faltan archivos del indice RAG local: " + ", ".join(missing_paths)
        )

    chunks = load_json(resolved_chunks_path)
    metadata = load_json(resolved_metadata_path)
    embeddings = np.load(resolved_embeddings_path)

    if not isinstance(chunks, list):
        raise RAGRetrieverError("El archivo de chunks RAG no tiene el formato esperado.")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise RAGRetrieverError("El archivo de embeddings RAG no tiene una matriz valida.")
    if len(chunks) != embeddings.shape[0]:
        raise RAGRetrieverError("La cantidad de chunks no coincide con la matriz de embeddings.")

    return {
        "chunks": chunks,
        "embeddings": _normalize_matrix(np.asarray(embeddings, dtype=np.float32)),
        "metadata": metadata,
    }


def retrieve(
    query: str,
    top_k: int = 5,
    entity_type: str | None = None,
    provider: str = "openai",
    *,
    model: str | None = None,
    index_data: dict[str, Any] | None = None,
    chunks_path: str | Path = CHUNKS_PATH,
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a query from the local index."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    loaded_index = index_data or load_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )
    chunks = loaded_index["chunks"]
    embeddings = loaded_index["embeddings"]

    filtered_pairs = [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if not entity_type
        or normalize_text(chunk.get("tipo"), lowercase=True)
        == normalize_text(entity_type, lowercase=True)
    ]
    if not filtered_pairs:
        return []

    filtered_indices = [index for index, _ in filtered_pairs]
    filtered_chunks = [chunk for _, chunk in filtered_pairs]
    filtered_embeddings = embeddings[filtered_indices]

    query_embedding = np.asarray(
        generate_embedding(normalized_query, provider=provider, model=model),
        dtype=np.float32,
    )
    if query_embedding.ndim != 1:
        raise RAGRetrieverError("El embedding de consulta no tiene una forma valida.")
    if filtered_embeddings.shape[1] != query_embedding.shape[0]:
        raise RAGRetrieverError(
            "El embedding de consulta no coincide con la dimension del indice cargado."
        )

    normalized_query_embedding = _normalize_vector(query_embedding)
    scores = filtered_embeddings @ normalized_query_embedding

    requested_top_k = max(1, int(top_k))
    ranked_positions = np.argsort(scores)[::-1][:requested_top_k]

    results: list[dict[str, Any]] = []
    for position in ranked_positions:
        chunk = dict(filtered_chunks[int(position)])
        chunk["score"] = float(scores[int(position)])
        results.append(chunk)

    return results


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a matrix row-wise for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a single vector for cosine similarity."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise RAGRetrieverError("El embedding de consulta no puede ser un vector nulo.")
    return vector / norm
