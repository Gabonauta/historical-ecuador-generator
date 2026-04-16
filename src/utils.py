"""Utility helpers shared across the project."""

from __future__ import annotations

from typing import Any, Iterable, TypeVar


T = TypeVar("T")

REQUIRED_ENTITY_FIELDS = [
    "id",
    "nombre",
    "tipo",
    "resumen",
    "descripcion_larga",
    "importancia",
]

VALID_ENTITY_TYPES = {"personaje", "lugar", "evento"}


def safe_str(value: Any, default: str = "") -> str:
    """Return a stripped string representation for any value."""
    if value is None:
        return default
    return str(value).strip()


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return " ".join(safe_str(text).split())


def normalize_text(value: Any, lowercase: bool = False) -> str:
    """Normalize text values for storage and comparison."""
    text = normalize_spaces(safe_str(value))
    return text.lower() if lowercase else text


def safe_list(value: Any) -> list[Any]:
    """Return a safe list for null, scalar, tuple, or list values."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def unique_preserve_order(items: Iterable[T]) -> list[T]:
    """Remove duplicates while preserving the first occurrence."""
    seen: set[Any] = set()
    result: list[T] = []

    for item in items:
        marker = item
        if isinstance(item, list):
            marker = tuple(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)

    return result


def safe_int(value: Any) -> int | None:
    """Convert values into integers when possible."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = normalize_text(value)
    if not text:
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def has_text(value: Any) -> bool:
    """Check whether a value contains non-empty text."""
    return bool(normalize_text(value))


def join_non_empty(parts: Iterable[str], separator: str = "\n") -> str:
    """Join only non-empty text fragments preserving order."""
    cleaned_parts = [normalize_text(part) for part in parts if has_text(part)]
    return separator.join(cleaned_parts)


def format_year_range(start_year: Any, end_year: Any) -> str:
    """Format a start/end year pair for display."""
    start = safe_int(start_year)
    end = safe_int(end_year)

    if start and end:
        if start == end:
            return str(start)
        return f"{start} - {end}"
    if start:
        return f"Desde {start}"
    if end:
        return f"Hasta {end}"
    return ""


def validate_entity(entity: dict[str, Any]) -> list[str]:
    """Return the list of required keys missing from an entity."""
    if not isinstance(entity, dict):
        return REQUIRED_ENTITY_FIELDS.copy()

    missing_fields: list[str] = []
    for field in REQUIRED_ENTITY_FIELDS:
        value = entity.get(field)
        if isinstance(value, list):
            if not value:
                missing_fields.append(field)
        elif not has_text(value):
            missing_fields.append(field)
    return missing_fields


def validate_entities_payload(payload: Any) -> list[str]:
    """Validate the top-level entities payload."""
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

    missing_keys = sorted(key for key in expected_keys if key not in payload)
    if missing_keys:
        return [f"Faltan plantillas requeridas: {', '.join(missing_keys)}."]

    return []


# Compatibility aliases for Phase 1 modules.
safe_strip = safe_str
compact_whitespace = normalize_spaces
ensure_list = safe_list
