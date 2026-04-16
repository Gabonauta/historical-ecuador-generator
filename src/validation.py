"""Validation helpers for Phase 2 data curation."""

from __future__ import annotations

from typing import Any

from src.utils import REQUIRED_ENTITY_FIELDS, VALID_ENTITY_TYPES, has_text, normalize_text, safe_int


MIN_SUMMARY_LENGTH = 40
MIN_DESCRIPTION_LENGTH = 80


def validate_required_fields(entity: dict[str, Any]) -> list[str]:
    """Return error messages for required fields that are missing or empty."""
    errors: list[str] = []

    for field in REQUIRED_ENTITY_FIELDS:
        value = entity.get(field)
        if isinstance(value, list):
            if not value:
                errors.append(f"Campo requerido vacio: {field}")
        elif not has_text(value):
            errors.append(f"Campo requerido faltante o vacio: {field}")

    return errors


def validate_entity_type(entity: dict[str, Any]) -> list[str]:
    """Return errors when the entity type is not valid."""
    entity_type = normalize_text(entity.get("tipo"), lowercase=True)
    if entity_type and entity_type in VALID_ENTITY_TYPES:
        return []
    return [f"Tipo invalido: {entity.get('tipo', '')}"]


def detect_year_warnings(entity: dict[str, Any]) -> list[str]:
    """Return warnings related to inconsistent year values."""
    warnings: list[str] = []
    start_year = safe_int(entity.get("anio_inicio"))
    end_year = safe_int(entity.get("anio_fin"))

    if start_year and end_year and end_year < start_year:
        warnings.append(
            f"Rango de años inconsistente: anio_fin ({end_year}) es menor que anio_inicio ({start_year})"
        )

    return warnings


def detect_text_warnings(entity: dict[str, Any]) -> list[str]:
    """Return warnings for suspiciously short text fields."""
    warnings: list[str] = []
    summary = normalize_text(entity.get("resumen"))
    description = normalize_text(entity.get("descripcion_larga"))

    if summary and len(summary) < MIN_SUMMARY_LENGTH:
        warnings.append("Resumen demasiado corto")
    if description and len(description) < MIN_DESCRIPTION_LENGTH:
        warnings.append("Descripcion larga demasiado corta")

    return warnings


def validate_entity(entity: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    """Generate a validation report for a single entity."""
    errors = []
    errors.extend(validate_required_fields(entity))
    errors.extend(validate_entity_type(entity))

    warnings = []
    warnings.extend(detect_year_warnings(entity))
    warnings.extend(detect_text_warnings(entity))

    return {
        "index": index,
        "id": entity.get("id", ""),
        "nombre": entity.get("nombre", ""),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a consolidated validation report for many entities."""
    entity_reports = [
        validate_entity(entity, index=index)
        for index, entity in enumerate(entities, start=1)
    ]

    valid_count = sum(1 for report in entity_reports if report["valid"])
    invalid_count = len(entity_reports) - valid_count
    warning_count = sum(1 for report in entity_reports if report["warnings"])

    return {
        "summary": {
            "total_entities": len(entity_reports),
            "valid_entities": valid_count,
            "invalid_entities": invalid_count,
            "entities_with_warnings": warning_count,
            "required_fields": REQUIRED_ENTITY_FIELDS,
            "valid_types": sorted(VALID_ENTITY_TYPES),
        },
        "entities": entity_reports,
    }
