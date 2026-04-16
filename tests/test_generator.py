"""Tests for hybrid generation flow."""

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


def test_generate_content_uses_fallback_when_llm_disabled() -> None:
    result = generator.generate_content(build_entity(), "ficha_historica", use_llm=False)
    assert result["mode"] == "fallback"
    assert result["provider"] == "fallback"
    assert result["generated_text"]
    assert result["use_rag"] is False


def test_generate_content_falls_back_when_provider_fails(monkeypatch) -> None:
    def fail_generate_text(**_: object) -> str:
        raise generator.LLMClientError("failure")

    monkeypatch.setattr(generator, "generate_text", fail_generate_text)
    result = generator.generate_content(build_entity(), "resumen_corto", provider="openai", use_llm=True)

    assert result["mode"] == "fallback"
    assert result["provider"] == "fallback"
    assert result["error"] is not None


def test_generate_content_returns_consistent_output_dict(monkeypatch) -> None:
    monkeypatch.setattr(generator, "generate_text", lambda **_: "Texto generado por LLM")
    result = generator.generate_content(build_entity(), "post_redes", provider="gemini", use_llm=True)

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
    assert result["mode"] == "llm"
    assert result["provider"] == "gemini"
    assert result["generated_text"] == "Texto generado por LLM"
