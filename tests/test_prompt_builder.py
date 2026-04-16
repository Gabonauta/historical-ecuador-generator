"""Tests for prompt construction."""

from src.prompt_builder import build_prompt


def test_prompt_contains_factuality_instructions() -> None:
    prompt = build_prompt({"nombre": "Eugenio Espejo"}, "ficha_historica", "Contexto de prueba")
    assert "Usa solo la informacion proporcionada" in prompt
    assert "No inventes datos" in prompt
    assert "Contexto base estructurado:" in prompt
    assert "Contexto recuperado adicional:" in prompt


def test_prompt_changes_with_output_type() -> None:
    prompt_short = build_prompt({"nombre": "A"}, "resumen_corto", "Contexto")
    prompt_social = build_prompt({"nombre": "A"}, "post_redes", "Contexto")

    assert "maximo 80 palabras" in prompt_short
    assert "post breve para redes sociales" in prompt_social


def test_prompt_includes_context() -> None:
    context = "Nombre: Eugenio Espejo"
    prompt = build_prompt({"nombre": "Eugenio Espejo"}, "texto_turistico", context)
    assert context in prompt


def test_prompt_includes_retrieved_context_when_present() -> None:
    retrieved_context = "Fragmento 1\nTexto: Eugenio Espejo fue un intelectual."
    prompt = build_prompt(
        {"nombre": "Eugenio Espejo"},
        "ficha_historica",
        "Contexto base",
        retrieved_context=retrieved_context,
    )

    assert retrieved_context in prompt
    assert "prioriza el contexto recuperado" in prompt
