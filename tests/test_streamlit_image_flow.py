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
        api_key_overrides={"openai": "runtime-openai-key"},
        debug_mode=False,
    )

    assert captured["generate_text"] is False
    assert captured["generate_image"] is True
    assert captured["image_mode"] == "retrato_historico"
    assert captured["api_keys"] == {"openai": "runtime-openai-key"}
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
        api_key_overrides=None,
        debug_mode=False,
    )

    assert result["text_result"]["generated_text"] == "Texto fallback"
    assert result["image_result"]["provider"] == "fallback"


def test_build_api_key_overrides_discards_empty_values() -> None:
    overrides = streamlit_app.build_api_key_overrides(
        openai_api_key=" openai-key ",
        gemini_api_key="",
        xai_api_key="   ",
    )

    assert overrides == {"openai": "openai-key"}


def test_build_generation_request_payload_never_includes_api_keys() -> None:
    payload = streamlit_app.build_generation_request_payload(
        output_type="ficha_historica",
        llm_provider="openai",
        image_provider="openai",
        embedding_provider="openai",
        use_llm=True,
        use_rag=True,
        top_k=5,
        generate_text=True,
        generate_image=True,
        image_mode="retrato_historico",
        visual_style="realista",
        image_size="1024x1024",
        debug_mode=False,
    )

    assert "api_keys" not in payload
    assert "api_key_overrides" not in payload


def test_get_provider_availability_snapshot_supports_legacy_signatures(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "get_available_providers",
        lambda: {"openai": False, "gemini": False, "xai": False},
    )
    monkeypatch.setattr(
        streamlit_app,
        "get_available_embedding_providers",
        lambda: {"openai": False, "gemini": False},
    )
    monkeypatch.setattr(
        streamlit_app,
        "get_available_image_providers",
        lambda: {"openai": False, "fallback": True},
    )

    llm, embeddings, image = streamlit_app.get_provider_availability_snapshot(
        {"openai": "runtime-openai-key"}
    )

    assert llm == {"openai": False, "gemini": False, "xai": False}
    assert embeddings == {"openai": False, "gemini": False}
    assert image == {"openai": False, "fallback": True}


def test_execute_generation_request_omits_api_keys_when_not_provided(monkeypatch) -> None:
    captured: dict = {}

    def fake_generate_multimodal_content(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {
            "text_result": None,
            "image_result": None,
            "entity_id": "eugenio_espejo",
            "output_type": "ficha_historica",
            "generate_text": False,
            "generate_image": False,
        }

    monkeypatch.setattr(streamlit_app, "generate_multimodal_content", fake_generate_multimodal_content)

    streamlit_app.execute_generation_request(
        entity=build_entity(),
        output_type="ficha_historica",
        llm_provider="openai",
        image_provider="openai",
        embedding_provider="openai",
        use_llm=False,
        use_rag=True,
        top_k=4,
        generate_text=True,
        generate_image=False,
        image_mode="retrato_historico",
        visual_style="realista",
        image_size="1024x1024",
        api_key_overrides={},
        debug_mode=False,
    )

    assert "api_keys" not in captured


def test_get_generation_run_count_supports_legacy_history_store(monkeypatch) -> None:
    monkeypatch.delattr(streamlit_app.history_store, "count_generation_runs", raising=False)
    monkeypatch.setattr(
        streamlit_app,
        "list_generation_runs",
        lambda limit=None: [{"id": 3}, {"id": 2}, {"id": 1}],
    )

    assert streamlit_app.get_generation_run_count() == 3
