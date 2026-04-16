"""Cleaning helpers for Phase 2 historical data curation."""

from __future__ import annotations

from typing import Any

from src.utils import (
    VALID_ENTITY_TYPES,
    has_text,
    normalize_text,
    safe_int,
    safe_list,
    unique_preserve_order,
)


TYPE_ALIASES = {
    "personaje": "personaje",
    "personajes": "personaje",
    "persona": "personaje",
    "figura": "personaje",
    "figura_historica": "personaje",
    "figura historica": "personaje",
    "lugar": "lugar",
    "lugares": "lugar",
    "sitio": "lugar",
    "sitio_historico": "lugar",
    "sitio historico": "lugar",
    "ubicacion": "lugar",
    "evento": "evento",
    "eventos": "evento",
    "acontecimiento": "evento",
    "hecho_historico": "evento",
    "hecho historico": "evento",
}

TAG_ALIASES = {
    "ilustración": "ilustracion",
    "ilustracion": "ilustracion",
    "independentista": "independencia",
    "patrimonial": "patrimonio",
    "prócer": "procer",
    "turística": "turismo",
    "turistica": "turismo",
}

LIST_FIELDS = [
    "lugares_relacionados",
    "personajes_relacionados",
    "eventos_relacionados",
]


def normalize_type(raw_type: Any) -> str:
    """Normalize entity type values into canonical project types."""
    normalized = normalize_text(raw_type, lowercase=True)
    if not normalized:
        return ""
    return TYPE_ALIASES.get(normalized, normalized if normalized in VALID_ENTITY_TYPES else normalized)


def normalize_tags(tags: Any) -> list[str]:
    """Normalize tag labels, remove blanks, and preserve order."""
    cleaned_tags: list[str] = []

    for tag in safe_list(tags):
        normalized = normalize_text(tag, lowercase=True)
        if not normalized:
            continue
        canonical = TAG_ALIASES.get(normalized, normalized)
        cleaned_tags.append(canonical)

    return unique_preserve_order(cleaned_tags)


def clean_related_list(values: Any) -> list[str]:
    """Clean related entity lists and preserve the original order."""
    cleaned_values = [normalize_text(value) for value in safe_list(values) if has_text(value)]
    return unique_preserve_order(cleaned_values)


def infer_epoca(anio_inicio: Any, anio_fin: Any) -> str:
    """Infer a broad historical period from the available years."""
    years = [year for year in (safe_int(anio_inicio), safe_int(anio_fin)) if year]
    if not years:
        return ""

    earliest = min(years)
    latest = max(years)

    if earliest >= 2001:
        return "Siglo XXI"
    if earliest >= 1901:
        return "Siglo XX"
    if earliest >= 1801:
        return "Siglo XIX"
    if earliest >= 1701:
        return "Siglo XVIII"
    if earliest >= 1601:
        return "Siglo XVII"
    if earliest >= 1501:
        if latest >= 1801:
            return "Periodo colonial y republicano"
        return "Periodo colonial"
    return "Periodo historico"


def clean_entity(entity: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned entity record ready for validation."""
    cleaned: dict[str, Any] = {
        "id": normalize_text(entity.get("id"), lowercase=True).replace(" ", "_"),
        "nombre": normalize_text(entity.get("nombre")),
        "tipo": normalize_type(entity.get("tipo")),
        "ubicacion": normalize_text(entity.get("ubicacion")),
        "resumen": normalize_text(entity.get("resumen")),
        "descripcion_larga": normalize_text(entity.get("descripcion_larga")),
        "importancia": normalize_text(entity.get("importancia")),
        "anio_inicio": safe_int(entity.get("anio_inicio")),
        "anio_fin": safe_int(entity.get("anio_fin")),
        "fuente_base": normalize_text(entity.get("fuente_base")),
    }

    for field in LIST_FIELDS:
        cleaned[field] = clean_related_list(entity.get(field))

    cleaned["etiquetas"] = normalize_tags(entity.get("etiquetas"))

    original_epoca = normalize_text(entity.get("epoca"))
    cleaned["epoca"] = original_epoca or infer_epoca(
        cleaned.get("anio_inicio"),
        cleaned.get("anio_fin"),
    )

    return cleaned


def deduplicate_entities(
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Remove duplicates by id or name while keeping the first occurrence."""
    unique_entities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_names: list[str] = []

    for entity in entities:
        entity_id = normalize_text(entity.get("id"), lowercase=True)
        entity_name = normalize_text(entity.get("nombre"), lowercase=True)

        if entity_id and entity_id in seen_ids:
            duplicate_ids.append(entity.get("id", ""))
            continue
        if entity_name and entity_name in seen_names:
            duplicate_names.append(entity.get("nombre", ""))
            continue

        if entity_id:
            seen_ids.add(entity_id)
        if entity_name:
            seen_names.add(entity_name)

        unique_entities.append(entity)

    return unique_entities, {
        "duplicate_ids": duplicate_ids,
        "duplicate_names": duplicate_names,
    }
