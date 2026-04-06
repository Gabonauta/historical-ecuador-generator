"""Utility helpers for the historical Ecuador generator project."""

from __future__ import annotations

from typing import Any, Iterable


REQUIRED_ENTITY_FIELDS = [
    "id",
    "nombre",
    "tipo",
    "epoca",
    "ubicacion",
    "resumen",
    "descripcion_larga",
    "importancia",
    "lugares_relacionados",
    "personajes_relacionados",
    "eventos_relacionados",
    "etiquetas",
    "anio_inicio",
    "anio_fin",
]


def ensure_list(value: Any) -> list[Any]:
    """Return a safe list for values that may be null or malformed."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def safe_strip(value: Any) -> str:
    """Convert a value into a normalized string."""
    if value is None:
        return ""
    return str(value).strip()


def has_text(value: Any) -> bool:
    """Check whether a value contains non-empty text."""
    return bool(safe_strip(value))


def compact_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(text.split())


def join_non_empty(parts: Iterable[str], separator: str = "\n") -> str:
    """Join only non-empty text fragments preserving order."""
    cleaned_parts = [safe_strip(part) for part in parts if has_text(part)]
    return separator.join(cleaned_parts)


def is_valid_year(value: Any) -> bool:
    """Return True when the provided year is a meaningful integer."""
    return isinstance(value, int) and value > 0


def format_year_range(start_year: Any, end_year: Any) -> str:
    """Format a start/end year pair for display."""
    has_start = is_valid_year(start_year)
    has_end = is_valid_year(end_year)

    if has_start and has_end:
        if start_year == end_year:
            return str(start_year)
        return f"{start_year} - {end_year}"
    if has_start:
        return f"Desde {start_year}"
    if has_end:
        return f"Hasta {end_year}"
    return ""


def validate_entity(entity: dict[str, Any]) -> list[str]:
    """Return the list of required keys missing from an entity."""
    if not isinstance(entity, dict):
        return REQUIRED_ENTITY_FIELDS.copy()
    return [field for field in REQUIRED_ENTITY_FIELDS if field not in entity]


def validate_entities_payload(payload: Any) -> list[str]:
    """Validate the full entities payload and report human-readable issues."""
    issues: list[str] = []

    if not isinstance(payload, list):
        return ["El archivo de entidades debe contener una lista JSON."]

    for index, entity in enumerate(payload, start=1):
        missing_fields = validate_entity(entity)
        if missing_fields:
            issues.append(
                f"La entidad #{index} tiene campos faltantes: {', '.join(missing_fields)}."
            )
    return issues


def validate_templates_payload(payload: Any) -> list[str]:
    """Validate the prompt templates payload structure."""
    expected_keys = {
        "ficha_historica",
        "resumen_corto",
        "texto_turistico",
        "post_redes",
    }

    if not isinstance(payload, dict):
        return ["El archivo de plantillas debe contener un objeto JSON."]

    missing = [key for key in expected_keys if key not in payload]
    if missing:
        return [f"Faltan plantillas requeridas: {', '.join(sorted(missing))}."]

    return []
