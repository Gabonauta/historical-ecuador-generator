"""Tests for RAG-aware generation flow."""

from __future__ import annotations

from src import generator
from src.embeddings_client import EmbeddingsClientError


def build_entity() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
        "epoca": "Siglo XVIII",
        "ubicacion": "Quito, Ecuador",
        "resumen": "Intelectual y precursor de ideas independentistas.",
        "descripcion_larga": "Figura destacada del pensamiento reformista en Quito.",
        "importancia": "Ayudo a difundir ideas de cambio social y politico.",
        "lugares_relacionados": ["Quito"],
        "personajes_relacionados": ["Carlos Montufar"],
        "eventos_relacionados": ["Primer Grito de Independencia"],
        "etiquetas": ["independencia"],
        "anio_inicio": 1747,
        "anio_fin": 1795,
        "fuente_base": "Registro base",
    }


def build_retrieved_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "eugenio_espejo__descripcion",
            "entity_id": "eugenio_espejo",
            "nombre": "Eugenio Espejo",
            "tipo": "personaje",
            "categoria_chunk": "descripcion",
            "texto": "Eugenio Espejo fue un intelectual destacado de Quito.",
            "score": 0.92,
        }
    ]


def test_generate_content_with_rag_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        generator,
        "load_index",
        lambda: {"metadata": {"embedding_provider": "openai"}},
    )
    monkeypatch.setattr(generator, "retrieve", lambda **_: build_retrieved_chunks())
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_content(
        build_entity(),
        "ficha_historica",
        provider="openai",
        use_llm=True,
        use_rag=True,
    )

    assert result["mode"] == "llm"
    assert result["use_rag"] is True
    assert result["retrieved_chunks"]
    assert "Fragmento 1" in result["retrieved_context"]


def test_generate_content_without_rag(monkeypatch) -> None:
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_content(
        build_entity(),
        "resumen_corto",
        provider="gemini",
        use_llm=True,
        use_rag=False,
    )

    assert result["mode"] == "llm"
    assert result["use_rag"] is False
    assert result["retrieved_context"] == ""
    assert result["retrieved_chunks"] == []


def test_generate_content_continues_without_rag_if_embeddings_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        generator,
        "load_index",
        lambda: {"metadata": {"embedding_provider": "openai"}},
    )

    def fail_retrieval(**_: object) -> list[dict]:
        raise EmbeddingsClientError("fallo embeddings")

    monkeypatch.setattr(generator, "retrieve", fail_retrieval)
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_content(
        build_entity(),
        "texto_turistico",
        provider="openai",
        use_llm=True,
        use_rag=True,
    )

    assert result["mode"] == "llm"
    assert result["use_rag"] is False
    assert result["generated_text"] == "Texto generado por LLM"
    assert result["error"] is not None


def test_generate_content_uses_fallback_if_llm_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        generator,
        "load_index",
        lambda: {"metadata": {"embedding_provider": "openai"}},
    )
    monkeypatch.setattr(generator, "retrieve", lambda **_: build_retrieved_chunks())

    def fail_generate_text(**_: object) -> str:
        raise generator.LLMClientError("provider down")

    monkeypatch.setattr(generator, "generate_text", fail_generate_text)

    result = generator.generate_content(
        build_entity(),
        "post_redes",
        provider="xai",
        use_llm=True,
        use_rag=True,
    )

    assert result["mode"] == "fallback"
    assert result["provider"] == "fallback"
    assert result["use_rag"] is True
    assert result["error"] is not None


def test_generate_content_returns_consistent_rag_output_dict(monkeypatch) -> None:
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_content(
        build_entity(),
        "ficha_historica",
        provider="openai",
        use_llm=True,
        use_rag=False,
    )

    assert set(result.keys()) == {
        "mode",
        "provider",
        "output_type",
        "use_rag",
        "embedding_provider",
        "prompt",
        "base_context",
        "context",
        "retrieved_context",
        "retrieved_chunks",
        "generated_text",
        "error",
    }
