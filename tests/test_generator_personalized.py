"""Tests for personalized multimodal generation flow."""

from __future__ import annotations

from src import generator


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
            "texto": "Intelectual destacado de Quito.",
            "score": 0.91,
        }
    ]


def test_generate_multimodal_content_with_audience(monkeypatch) -> None:
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")
    monkeypatch.setattr(
        "src.image_generator.generate_visual_content",
        lambda **_: {
            "provider": "fallback",
            "status": "fallback",
            "image_mode": "postal_turistica",
            "visual_style": "realista",
            "size": "1024x1024",
            "use_rag": False,
            "embedding_provider": "openai",
            "prompt": "Prompt visual turistico",
            "base_context": "Contexto base",
            "retrieved_context": "",
            "retrieved_chunks": [],
            "image_path": None,
            "image_url": None,
            "error": None,
        },
    )

    result = generator.generate_multimodal_content(
        build_entity(),
        "texto_turistico",
        use_rag=False,
        generate_image=True,
        image_provider="fallback",
        audience_id="turista_general",
    )

    assert result["text_result"]["generated_text"] == "Texto generado por LLM"
    assert result["image_result"]["prompt"] == "Prompt visual turistico"
    assert result["personalization"]["audience_id"] == "turista_general"
    assert result["personalization"]["purpose"] == "turistico"


def test_generate_multimodal_content_fallback_respects_personalization(monkeypatch) -> None:
    def fail_generate_text(**_: object) -> str:
        raise generator.LLMClientError("provider down")

    monkeypatch.setattr(generator, "generate_text", fail_generate_text)

    result = generator.generate_multimodal_content(
        build_entity(),
        "resumen_corto",
        use_llm=True,
        use_rag=False,
        generate_image=False,
        audience_id="estudiante_secundaria",
    )

    assert result["text_result"]["mode"] == "fallback"
    assert result["personalization"]["tone"] == "didactico"
    assert "Idea central:" in result["text_result"]["generated_text"]


def test_generate_multimodal_content_returns_consistent_output_dict(monkeypatch) -> None:
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_multimodal_content(
        build_entity(),
        "ficha_historica",
        use_rag=False,
        generate_image=False,
        audience_id="docente",
        tone="formal",
    )

    assert set(result.keys()) == {
        "text_result",
        "image_result",
        "entity_id",
        "output_type",
        "generate_text",
        "generate_image",
        "personalization",
    }
    assert result["personalization"]["tone"] == "formal"


def test_generate_multimodal_content_keeps_compatibility_with_rag(monkeypatch) -> None:
    monkeypatch.setattr(
        generator,
        "load_index",
        lambda: {"metadata": {"embedding_provider": "openai"}},
    )
    monkeypatch.setattr(generator, "retrieve", lambda **_: build_retrieved_chunks())
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")

    result = generator.generate_multimodal_content(
        build_entity(),
        "ficha_historica",
        use_llm=True,
        use_rag=True,
        generate_image=False,
        audience_id="docente",
    )

    assert result["text_result"]["use_rag"] is True
    assert result["personalization"]["depth"] == "alta"
