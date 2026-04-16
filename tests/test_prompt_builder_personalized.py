"""Tests for personalized textual prompt construction."""

from src.prompt_builder import build_prompt


def build_personalization() -> dict:
    return {
        "audience_id": "docente",
        "nombre_visible": "Docente",
        "descripcion": "Perfil para uso pedagogico.",
        "tone": "didactico",
        "depth": "alta",
        "length": "media",
        "purpose": "educativo",
        "style_rules": ["Organizar la informacion con claridad pedagogica"],
        "forbidden_patterns": ["No priorizar impacto sobre claridad"],
    }


def test_build_prompt_includes_audience_rules() -> None:
    prompt = build_prompt(
        {"nombre": "Eugenio Espejo"},
        "ficha_historica",
        "Contexto base",
        personalization_config=build_personalization(),
    )

    assert "Audiencia objetivo: Docente" in prompt
    assert "Reglas de estilo:" in prompt
    assert "No priorizar impacto sobre claridad" in prompt


def test_build_prompt_changes_with_tone_depth_length_and_purpose() -> None:
    config = build_personalization()
    prompt = build_prompt(
        {"nombre": "Centro Historico de Quito"},
        "texto_turistico",
        "Contexto base",
        personalization_config=config,
    )

    assert "Tono solicitado: didactico" in prompt
    assert "Nivel de profundidad: alta" in prompt
    assert "Longitud deseada: media" in prompt
    assert "Proposito principal: educativo" in prompt


def test_build_prompt_maintains_factuality_instructions() -> None:
    prompt = build_prompt(
        {"nombre": "Primer Grito de Independencia"},
        "post_redes",
        "Contexto base",
        retrieved_context="Fragmento 1\nTexto: Evento en Quito.",
        personalization_config=build_personalization(),
    )

    assert "Usa solo la informacion proporcionada" in prompt
    assert "No inventes datos" in prompt
    assert "prioriza el contexto recuperado" in prompt
