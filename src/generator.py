"""Programmatic text generation using JSON data and lightweight templates."""

from __future__ import annotations

from typing import Any

from src.formatter import (
    build_context_block,
    build_paragraphs,
    build_section,
    clean_text,
    format_related_list,
)
from src.utils import has_text


SUPPORTED_OUTPUTS = (
    "ficha_historica",
    "resumen_corto",
    "texto_turistico",
    "post_redes",
)


def generate_content(
    entity: dict[str, Any], output_type: str, templates: dict[str, str]
) -> str:
    """Dispatch generation for the requested output type."""
    if output_type not in SUPPORTED_OUTPUTS:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

    if output_type not in templates:
        raise ValueError(f"No existe plantilla para: {output_type}")

    generators = {
        "ficha_historica": generate_ficha_historica,
        "resumen_corto": generate_resumen_corto,
        "texto_turistico": generate_texto_turistico,
        "post_redes": generate_post_redes,
    }
    return generators[output_type](entity, templates[output_type])


def generate_ficha_historica(entity: dict[str, Any], template: str) -> str:
    """Generate a structured historical card."""
    header = f"# {clean_text(entity.get('nombre'))}"
    context = build_section("Contexto general", build_context_block(entity))
    intro = clean_text(template.format(nombre=entity.get("nombre", "")))
    summary = build_section("Resumen", clean_text(entity.get("resumen")))
    description = build_section("Descripcion ampliada", clean_text(entity.get("descripcion_larga")))
    importance = build_section("Importancia historica", clean_text(entity.get("importancia")))
    relationships = build_section(
        "Relaciones clave",
        "; ".join(
            part
            for part in [
                _build_relation_line("Lugares", entity.get("lugares_relacionados")),
                _build_relation_line("Personajes", entity.get("personajes_relacionados")),
                _build_relation_line("Eventos", entity.get("eventos_relacionados")),
            ]
            if part
        ),
    )
    tags = build_section("Etiquetas", format_related_list(entity.get("etiquetas"), "Sin etiquetas"))

    return build_paragraphs([header, intro, context, summary, description, importance, relationships, tags])


def generate_resumen_corto(entity: dict[str, Any], template: str) -> str:
    """Generate a concise educational summary."""
    intro = clean_text(template.format(nombre=entity.get("nombre", "")))
    sentences = [
        clean_text(entity.get("resumen")),
        clean_text(entity.get("importancia")),
    ]
    body = " ".join(sentence for sentence in sentences if sentence)
    return build_paragraphs([intro, body])


def generate_texto_turistico(entity: dict[str, Any], template: str) -> str:
    """Generate an attractive tourism-oriented text."""
    intro = clean_text(template.format(nombre=entity.get("nombre", "")))
    opening = _compose_tourism_opening(entity)
    historical_value = clean_text(entity.get("descripcion_larga"))
    visit_value = _compose_visit_value(entity)
    return build_paragraphs([intro, opening, historical_value, visit_value])


def generate_post_redes(entity: dict[str, Any], template: str) -> str:
    """Generate a short social-media style post."""
    intro = clean_text(template.format(nombre=entity.get("nombre", "")))
    hook = f"{clean_text(entity.get('nombre'))}: {clean_text(entity.get('resumen'))}"
    significance = clean_text(entity.get("importancia"))
    hashtags = _build_hashtags(entity)
    return build_paragraphs([intro, hook, significance, hashtags])


def _build_relation_line(label: str, values: list[Any] | None) -> str:
    """Format a single relationship line when data exists."""
    text = format_related_list(values, "")
    if not text:
        return ""
    return f"{label}: {text}"


def _compose_tourism_opening(entity: dict[str, Any]) -> str:
    """Build the opening paragraph for tourism content."""
    nombre = clean_text(entity.get("nombre"))
    ubicacion = clean_text(entity.get("ubicacion"))
    resumen = clean_text(entity.get("resumen"))
    if has_text(ubicacion):
        return f"{nombre} en {ubicacion} ofrece una puerta de entrada a la memoria historica del Ecuador. {resumen}"
    return f"{nombre} invita a acercarse a una parte esencial de la historia ecuatoriana. {resumen}"


def _compose_visit_value(entity: dict[str, Any]) -> str:
    """Explain the cultural value for visitors without inventing facts."""
    importance = clean_text(entity.get("importancia"))
    related_places = format_related_list(entity.get("lugares_relacionados"), "")
    if related_places:
        return (
            f"Su valor cultural se aprecia mejor al conectarlo con lugares como {related_places}. "
            f"{importance}"
        )
    return f"Su valor cultural e historico lo convierte en un referente para comprender mejor el Ecuador. {importance}"


def _build_hashtags(entity: dict[str, Any]) -> str:
    """Convert tags into simple hashtags."""
    raw_tags = entity.get("etiquetas") or []
    hashtags = []
    for tag in raw_tags:
        cleaned = clean_text(tag).replace(" ", "")
        if cleaned:
            hashtags.append(f"#{cleaned}")
    return " ".join(hashtags)
