"""Tests for local RAG retrieval."""

from __future__ import annotations

import numpy as np

from src.loader import save_json
from src.rag_retriever import load_index, retrieve


def build_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "espejo__resumen",
            "entity_id": "eugenio_espejo",
            "nombre": "Eugenio Espejo",
            "tipo": "personaje",
            "categoria_chunk": "resumen",
            "texto": "Intelectual y medico en Quito.",
            "etiquetas": ["Quito"],
        },
        {
            "chunk_id": "quito__descripcion",
            "entity_id": "quito_historico",
            "nombre": "Centro Historico de Quito",
            "tipo": "lugar",
            "categoria_chunk": "descripcion",
            "texto": "Espacio urbano patrimonial del Ecuador.",
            "etiquetas": ["Quito"],
        },
        {
            "chunk_id": "montufar__importancia",
            "entity_id": "carlos_montufar",
            "nombre": "Carlos Montufar",
            "tipo": "personaje",
            "categoria_chunk": "importancia",
            "texto": "Actor politico de la independencia.",
            "etiquetas": ["independencia"],
        },
    ]


def build_index_files(tmp_path) -> tuple[str, str, str]:
    chunks_path = tmp_path / "chunks.json"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    save_json(chunks_path, build_chunks())
    np.save(
        embeddings_path,
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    save_json(
        metadata_path,
        {
            "backend": "numpy-cosine",
            "embedding_provider": "openai",
            "embedding_dimension": 2,
        },
    )
    return str(chunks_path), str(embeddings_path), str(metadata_path)


def test_load_index_reads_local_files(tmp_path) -> None:
    chunks_path, embeddings_path, metadata_path = build_index_files(tmp_path)

    index_data = load_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    assert len(index_data["chunks"]) == 3
    assert index_data["embeddings"].shape == (3, 2)
    assert index_data["metadata"]["embedding_provider"] == "openai"


def test_retrieve_returns_top_k_sorted_by_score(tmp_path, monkeypatch) -> None:
    chunks_path, embeddings_path, metadata_path = build_index_files(tmp_path)
    monkeypatch.setattr(
        "src.rag_retriever.generate_embedding",
        lambda *args, **kwargs: [1.0, 0.0],
    )

    results = retrieve(
        query="intelectual en Quito",
        top_k=2,
        provider="openai",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "espejo__resumen"
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_filters_by_entity_type(tmp_path, monkeypatch) -> None:
    chunks_path, embeddings_path, metadata_path = build_index_files(tmp_path)
    monkeypatch.setattr(
        "src.rag_retriever.generate_embedding",
        lambda *args, **kwargs: [0.0, 1.0],
    )

    results = retrieve(
        query="patrimonio urbano",
        top_k=3,
        entity_type="lugar",
        provider="openai",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    assert len(results) == 1
    assert results[0]["tipo"] == "lugar"


def test_retrieve_works_with_mocked_embeddings(tmp_path, monkeypatch) -> None:
    chunks_path, embeddings_path, metadata_path = build_index_files(tmp_path)
    monkeypatch.setattr(
        "src.rag_retriever.generate_embedding",
        lambda *args, **kwargs: [0.7, 0.7],
    )

    results = retrieve(
        query="independencia",
        top_k=1,
        provider="openai",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    assert results[0]["chunk_id"] == "montufar__importancia"
