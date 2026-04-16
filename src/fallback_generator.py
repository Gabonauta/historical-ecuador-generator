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
from src.utils import has_text, normalize_text, safe_str


def generate_fallback_content(
    entity: dict[str, Any],
    output_type: str,
    personalization_config: dict[str, Any] | None = None,
) -> str:
    """Generate text using deterministic templates and local entity data only."""
    generators = {
        "ficha_historica": _generate_ficha_historica,
        "resumen_corto": _generate_resumen_corto,
        "texto_turistico": _generate_texto_turistico,
        "post_redes": _generate_post_redes,
    }

    if output_type not in generators:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

    return generators[output_type](entity, personalization_config)


def _generate_ficha_historica(
    entity: dict[str, Any],
    personalization_config: dict[str, Any] | None = None,
) -> str:
    profile = _normalize_personalization(personalization_config)
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
    source = build_section("Fuente base", clean_text(entity.get("fuente_base")))
    framing = build_section("Enfoque para la audiencia", _build_audience_framing(entity, profile))

    parts = [header, framing, context, summary]
    if profile["depth"] != "baja" or profile["length"] != "corta":
        parts.append(description)
    parts.append(importance)
    if profile["depth"] == "alta" or profile["length"] == "larga":
        parts.extend([relationships, tags])
    if profile["purpose"] == "academico" or profile["length"] == "larga":
        parts.append(source)

    return build_paragraphs(parts)


def _generate_resumen_corto(
    entity: dict[str, Any],
    personalization_config: dict[str, Any] | None = None,
) -> str:
    profile = _normalize_personalization(personalization_config)
    parts = [
        _apply_tone_prefix(clean_text(entity.get("resumen")), profile["tone"], entity.get("tipo")),
        clean_text(entity.get("importancia")),
        clean_text(entity.get("descripcion_larga")),
    ]
    max_parts = {"corta": 1, "media": 2, "larga": 3}[profile["length"]]
    if profile["depth"] == "baja":
        max_parts = min(max_parts, 1)
    elif profile["depth"] == "media":
        max_parts = min(max_parts, 2)
    return build_paragraphs(parts[:max_parts])


def _generate_texto_turistico(
    entity: dict[str, Any],
    personalization_config: dict[str, Any] | None = None,
) -> str:
    profile = _normalize_personalization(personalization_config)
    nombre = clean_text(entity.get("nombre"))
    ubicacion = clean_text(entity.get("ubicacion"))
    resumen = clean_text(entity.get("resumen"))
    importance = clean_text(entity.get("importancia"))
    related_places = format_related_list(entity.get("lugares_relacionados"), "")

    if profile["purpose"] == "academico" or profile["tone"] == "formal":
        opening = (
            f"{nombre} en {ubicacion} representa un punto relevante del patrimonio historico del Ecuador. {resumen}"
            if has_text(ubicacion)
            else f"{nombre} representa un referente del patrimonio historico ecuatoriano. {resumen}"
        )
    else:
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
    parts = [opening]
    if profile["length"] != "corta":
        parts.append(clean_text(entity.get("descripcion_larga")))
    parts.append(closing)
    return build_paragraphs(parts)


def _generate_post_redes(
    entity: dict[str, Any],
    personalization_config: dict[str, Any] | None = None,
) -> str:
    profile = _normalize_personalization(personalization_config)
    hook = _apply_tone_prefix(
        f"{clean_text(entity.get('nombre'))}: {clean_text(entity.get('resumen'))}",
        profile["tone"],
        entity.get("tipo"),
    )
    importance = clean_text(entity.get("importancia"))
    location = clean_text(entity.get("ubicacion"))
    hashtags = " ".join(
        f"#{clean_text(tag).replace(' ', '')}"
        for tag in (entity.get("etiquetas") or [])
        if clean_text(tag)
    )
    parts = [hook]
    if profile["length"] != "corta":
        parts.append(importance)
    if profile["length"] == "larga" and location:
        parts.append(location)
    parts.append(hashtags)
    return build_paragraphs(parts)


def _build_relation_line(label: str, values: list[Any] | None) -> str:
    text = format_related_list(values, "")
    if not text:
        return ""
    return f"{label}: {text}"


def _normalize_personalization(
    personalization_config: dict[str, Any] | None,
) -> dict[str, str]:
    """Normalize personalization values for deterministic fallback use."""
    config = personalization_config or {}
    return {
        "tone": normalize_text(config.get("tone"), lowercase=True) or "formal",
        "depth": normalize_text(config.get("depth"), lowercase=True) or "media",
        "length": normalize_text(config.get("length"), lowercase=True) or "media",
        "purpose": normalize_text(config.get("purpose"), lowercase=True) or "divulgacion",
    }


def _build_audience_framing(entity: dict[str, Any], profile: dict[str, str]) -> str:
    """Build a deterministic framing sentence that reflects personalization."""
    summary = clean_text(entity.get("resumen"))
    importance = clean_text(entity.get("importancia"))
    base_sentence = summary or importance
    if not base_sentence:
        return ""

    if profile["tone"] == "didactico":
        return f"Idea central: {base_sentence}"
    if profile["tone"] == "divulgativo":
        return f"Dato clave: {base_sentence}"
    if profile["tone"] == "promocional":
        return f"Vale la pena conocer esta entidad historica: {base_sentence}"
    if profile["tone"] == "narrativo":
        return f"En la historia del Ecuador destaca esta entidad: {base_sentence}"

    if profile["purpose"] == "academico":
        return f"Referencia principal para este enfoque: {base_sentence}"
    return base_sentence


def _apply_tone_prefix(text: str, tone: str, entity_type: Any) -> str:
    """Apply a light stylistic prefix without altering factual content."""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return ""

    normalized_type = normalize_text(entity_type, lowercase=True) or "entidad"
    if tone == "didactico":
        return f"Idea central: {cleaned_text}"
    if tone == "divulgativo":
        return f"Dato clave: {cleaned_text}"
    if tone == "promocional":
        return f"Vale la pena descubrir este {normalized_type}: {cleaned_text}"
    if tone == "narrativo":
        return f"En la historia del Ecuador aparece este {normalized_type}: {cleaned_text}"
    return cleaned_text
