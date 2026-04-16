"""Lightweight tests for Streamlit personalization flow helpers."""

from __future__ import annotations

from app import streamlit_app


def build_entity() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
    }


def test_execute_generation_request_accepts_personalization_parameters(monkeypatch) -> None:
    captured: dict = {}

    def fake_generate_multimodal_content(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {
            "text_result": None,
            "image_result": None,
            "entity_id": "eugenio_espejo",
            "output_type": "ficha_historica",
            "generate_text": True,
            "generate_image": False,
            "personalization": {
                "audience_id": "docente",
                "nombre_visible": "Docente",
                "descripcion": "Perfil didactico",
                "tone": "didactico",
                "depth": "alta",
                "length": "media",
                "purpose": "educativo",
                "style_rules": ["Claridad pedagogica"],
                "forbidden_patterns": ["No exagerar"],
            },
        }

    monkeypatch.setattr(streamlit_app, "generate_multimodal_content", fake_generate_multimodal_content)

    result = streamlit_app.execute_generation_request(
        entity=build_entity(),
        output_type="ficha_historica",
        llm_provider="openai",
        image_provider="fallback",
        embedding_provider="openai",
        use_llm=True,
        use_rag=False,
        top_k=3,
        generate_text=True,
        generate_image=False,
        image_mode="retrato_historico",
        visual_style="realista",
        image_size="1024x1024",
        audience_id="docente",
        tone="didactico",
        depth="alta",
        length="media",
        purpose="educativo",
        debug_mode=False,
    )

    assert captured["audience_id"] == "docente"
    assert captured["tone"] == "didactico"
    assert result["personalization"]["audience_id"] == "docente"


def test_execute_generation_request_uses_defaults_when_personalization_missing(monkeypatch) -> None:
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
            "image_result": None,
            "entity_id": "eugenio_espejo",
            "output_type": "resumen_corto",
            "generate_text": True,
            "generate_image": False,
            "personalization": {
                "audience_id": "estudiante_universitario",
                "nombre_visible": "Audiencia general segura",
                "descripcion": "Fallback",
                "tone": "formal",
                "depth": "media",
                "length": "media",
                "purpose": "divulgacion",
                "style_rules": ["Mantener claridad"],
                "forbidden_patterns": ["No inventar datos"],
            },
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
        generate_image=False,
        image_mode="retrato_historico",
        visual_style="realista",
        image_size="1024x1024",
        audience_id="audiencia_inexistente",
        tone=None,
        depth=None,
        length=None,
        purpose=None,
        debug_mode=False,
    )

    assert result["text_result"]["generated_text"] == "Texto fallback"
    assert result["personalization"]["tone"] == "formal"
