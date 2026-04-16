"""Tests for visual generation flow."""

from __future__ import annotations

from src import image_generator


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
        "etiquetas": ["independencia"],
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
            "score": 0.93,
        }
    ]


def test_generate_visual_content_with_rag(monkeypatch) -> None:
    monkeypatch.setattr(
        image_generator,
        "load_index",
        lambda: {"metadata": {"embedding_provider": "openai"}},
    )
    monkeypatch.setattr(image_generator, "retrieve", lambda **_: build_retrieved_chunks())
    monkeypatch.setattr(image_generator, "build_image_prompt", lambda **_: "PROMPT VISUAL")
    monkeypatch.setattr(
        image_generator,
        "generate_image",
        lambda **_: {
            "provider": "openai",
            "status": "success",
            "prompt": "PROMPT VISUAL",
            "image_path": "/tmp/image.png",
            "image_url": None,
            "error": None,
        },
    )

    result = image_generator.generate_visual_content(build_entity(), use_rag=True)

    assert result["status"] == "success"
    assert result["use_rag"] is True
    assert result["retrieved_chunks"]
    assert result["prompt"] == "PROMPT VISUAL"


def test_generate_visual_content_without_rag(monkeypatch) -> None:
    monkeypatch.setattr(image_generator, "build_image_prompt", lambda **_: "PROMPT VISUAL")
    monkeypatch.setattr(
        image_generator,
        "generate_image",
        lambda **_: {
            "provider": "fallback",
            "status": "fallback",
            "prompt": "PROMPT VISUAL",
            "image_path": None,
            "image_url": None,
            "error": None,
        },
    )

    result = image_generator.generate_visual_content(build_entity(), use_rag=False)

    assert result["use_rag"] is False
    assert result["retrieved_context"] == ""
    assert result["retrieved_chunks"] == []


def test_generate_visual_content_handles_provider_fallback(monkeypatch) -> None:
    monkeypatch.setattr(image_generator, "build_image_prompt", lambda **_: "PROMPT VISUAL")
    monkeypatch.setattr(
        image_generator,
        "generate_image",
        lambda **_: {
            "provider": "fallback",
            "status": "fallback",
            "prompt": "PROMPT VISUAL",
            "image_path": None,
            "image_url": None,
            "error": "Sin provider disponible",
        },
    )

    result = image_generator.generate_visual_content(build_entity(), provider="openai", use_rag=False)

    assert result["provider"] == "fallback"
    assert result["status"] == "fallback"
    assert result["error"] is not None


def test_generate_visual_content_uses_prompt_builder(monkeypatch) -> None:
    captured: dict = {}

    def fake_build_image_prompt(**kwargs: object) -> str:
        captured.update(kwargs)
        return "PROMPT VISUAL"

    monkeypatch.setattr(image_generator, "build_image_prompt", fake_build_image_prompt)
    monkeypatch.setattr(
        image_generator,
        "generate_image",
        lambda **_: {
            "provider": "fallback",
            "status": "fallback",
            "prompt": "PROMPT VISUAL",
            "image_path": None,
            "image_url": None,
            "error": None,
        },
    )

    image_generator.generate_visual_content(
        build_entity(),
        image_mode="escena_historica",
        visual_style="grabado_antiguo",
        use_rag=False,
    )

    assert captured["image_mode"] == "escena_historica"
    assert captured["visual_style"] == "grabado_antiguo"
