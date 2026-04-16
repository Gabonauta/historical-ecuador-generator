"""Tests for grounded visual prompt construction."""

from src.image_prompt_builder import build_image_prompt


def build_person() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
    }


def build_place() -> dict:
    return {
        "id": "centro_historico_quito",
        "nombre": "Centro Historico de Quito",
        "tipo": "lugar",
    }


def build_event() -> dict:
    return {
        "id": "primer_grito",
        "nombre": "Primer Grito de Independencia",
        "tipo": "evento",
    }


def test_build_image_prompt_for_character_place_and_event() -> None:
    person_prompt = build_image_prompt(
        build_person(),
        "retrato_historico",
        "realista",
        "Nombre: Eugenio Espejo\nTipo: personaje",
    )
    place_prompt = build_image_prompt(
        build_place(),
        "postal_turistica",
        "pintura_oleo",
        "Nombre: Centro Historico de Quito\nTipo: lugar",
    )
    event_prompt = build_image_prompt(
        build_event(),
        "escena_historica",
        "grabado_antiguo",
        "Nombre: Primer Grito de Independencia\nTipo: evento",
    )

    assert "Si la entidad es un personaje" in person_prompt
    assert "arquitectura" in place_prompt
    assert "escena colectiva" in event_prompt


def test_build_image_prompt_changes_with_image_mode() -> None:
    portrait_prompt = build_image_prompt(
        build_person(),
        "retrato_historico",
        "realista",
        "Contexto base",
    )
    educational_prompt = build_image_prompt(
        build_person(),
        "ilustracion_educativa",
        "realista",
        "Contexto base",
    )

    assert "pose sobria" in portrait_prompt
    assert "util para aprendizaje visual" in educational_prompt


def test_build_image_prompt_changes_with_visual_style() -> None:
    oil_prompt = build_image_prompt(
        build_place(),
        "postal_turistica",
        "pintura_oleo",
        "Contexto base",
    )
    engraving_prompt = build_image_prompt(
        build_place(),
        "postal_turistica",
        "grabado_antiguo",
        "Contexto base",
    )

    assert "pincelada visible" in oil_prompt
    assert "grabado antiguo" in engraving_prompt


def test_build_image_prompt_includes_factuality_restrictions() -> None:
    prompt = build_image_prompt(
        build_event(),
        "escena_historica",
        "ilustracion_editorial",
        "Contexto base",
        retrieved_context="Fragmento 1\nTexto: Evento en Quito.",
    )

    assert "Usa solo la informacion proporcionada" in prompt
    assert "No inventes atributos historicos" in prompt
    assert "Contexto recuperado adicional:" in prompt
