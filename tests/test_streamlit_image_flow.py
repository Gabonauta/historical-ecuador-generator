"""Lightweight tests for the Streamlit multimodal flow helpers."""

from __future__ import annotations

from app import streamlit_app


def build_entity() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
    }


def test_execute_generation_request_accepts_image_parameters(monkeypatch) -> None:
    captured: dict = {}

    def fake_generate_multimodal_content(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {
            "text_result": None,
            "image_result": {
                "provider": "fallback",
                "status": "fallback",
                "image_mode": "retrato_historico",
                "visual_style": "realista",
                "size": "1024x1024",
                "use_rag": False,
                "embedding_provider": "openai",
                "prompt": "PROMPT VISUAL",
                "base_context": "Contexto base",
                "retrieved_context": "",
                "retrieved_chunks": [],
                "image_path": None,
                "image_url": None,
                "error": None,
            },
            "entity_id": "eugenio_espejo",
            "output_type": "ficha_historica",
            "generate_text": False,
            "generate_image": True,
        }

    monkeypatch.setattr(streamlit_app, "generate_multimodal_content", fake_generate_multimodal_content)

    result = streamlit_app.execute_generation_request(
        entity=build_entity(),
        output_type="ficha_historica",
        llm_provider="openai",
        image_provider="openai",
        embedding_provider="openai",
        use_llm=False,
        use_rag=True,
        top_k=4,
        generate_text=False,
        generate_image=True,
        image_mode="retrato_historico",
        visual_style="realista",
        image_size="1024x1024",
        debug_mode=False,
    )

    assert captured["generate_text"] is False
    assert captured["generate_image"] is True
    assert captured["image_mode"] == "retrato_historico"
    assert result["image_result"]["status"] == "fallback"


def test_execute_generation_request_handles_fallback_image_result(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "generate_multimodal_content",
        lambda **_: {
            "text_result": {
                "mode": "fallback",
                "provider": "fallback",
                "output_type": "resumen_corto",
                "use_rag": False,
                "embedding_provider": "openai",
                "prompt": "PROMPT",
                "base_context": "Contexto base",
                "context": "Contexto base",
                "retrieved_context": "",
                "retrieved_chunks": [],
                "generated_text": "Texto fallback",
                "error": None,
            },
            "image_result": {
                "provider": "fallback",
                "status": "fallback",
                "image_mode": "ilustracion_educativa",
                "visual_style": "ilustracion_editorial",
                "size": "1024x1024",
                "use_rag": False,
                "embedding_provider": "openai",
                "prompt": "PROMPT VISUAL",
                "base_context": "Contexto base",
                "retrieved_context": "",
                "retrieved_chunks": [],
                "image_path": None,
                "image_url": None,
                "error": "Sin provider",
            },
            "entity_id": "eugenio_espejo",
            "output_type": "resumen_corto",
            "generate_text": True,
            "generate_image": True,
        },
    )

    result = streamlit_app.execute_generation_request(
        entity=build_entity(),
        output_type="resumen_corto",
        llm_provider="openai",
        image_provider="fallback",
        embedding_provider="openai",
        use_llm=False,
        use_rag=False,
        top_k=3,
        generate_text=True,
        generate_image=True,
        image_mode="ilustracion_educativa",
        visual_style="ilustracion_editorial",
        image_size="1024x1024",
        debug_mode=False,
    )

    assert result["text_result"]["generated_text"] == "Texto fallback"
    assert result["image_result"]["provider"] == "fallback"
