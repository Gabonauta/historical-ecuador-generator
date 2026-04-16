"""Fallback generation without external LLM providers."""

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


def generate_fallback_content(entity: dict[str, Any], output_type: str) -> str:
    """Generate text using deterministic templates and local entity data only."""
    generators = {
        "ficha_historica": _generate_ficha_historica,
        "resumen_corto": _generate_resumen_corto,
        "texto_turistico": _generate_texto_turistico,
        "post_redes": _generate_post_redes,
    }

    if output_type not in generators:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

    return generators[output_type](entity)


def _generate_ficha_historica(entity: dict[str, Any]) -> str:
    header = f"# {clean_text(entity.get('nombre'))}"
    context = build_section("Contexto general", build_context_block(entity))
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
    return build_paragraphs([header, context, summary, description, importance, relationships, tags])


def _generate_resumen_corto(entity: dict[str, Any]) -> str:
    return build_paragraphs(
        [
            clean_text(entity.get("resumen")),
            clean_text(entity.get("importancia")),
        ]
    )


def _generate_texto_turistico(entity: dict[str, Any]) -> str:
    nombre = clean_text(entity.get("nombre"))
    ubicacion = clean_text(entity.get("ubicacion"))
    resumen = clean_text(entity.get("resumen"))
    importance = clean_text(entity.get("importancia"))
    related_places = format_related_list(entity.get("lugares_relacionados"), "")

    opening = (
        f"{nombre} en {ubicacion} ofrece una puerta de entrada a la memoria historica del Ecuador. {resumen}"
        if has_text(ubicacion)
        else f"{nombre} invita a acercarse a una parte esencial de la historia ecuatoriana. {resumen}"
    )
    closing = (
        f"Su valor cultural se aprecia mejor al conectarlo con lugares como {related_places}. {importance}"
        if related_places
        else f"Su valor cultural e historico lo convierte en un referente para comprender mejor el Ecuador. {importance}"
    )
    return build_paragraphs([opening, clean_text(entity.get("descripcion_larga")), closing])


def _generate_post_redes(entity: dict[str, Any]) -> str:
    hook = f"{clean_text(entity.get('nombre'))}: {clean_text(entity.get('resumen'))}"
    importance = clean_text(entity.get("importancia"))
    hashtags = " ".join(
        f"#{clean_text(tag).replace(' ', '')}"
        for tag in (entity.get("etiquetas") or [])
        if clean_text(tag)
    )
    return build_paragraphs([hook, importance, hashtags])


def _build_relation_line(label: str, values: list[Any] | None) -> str:
    text = format_related_list(values, "")
    if not text:
        return ""
    return f"{label}: {text}"
