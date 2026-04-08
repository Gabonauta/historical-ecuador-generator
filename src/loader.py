"""Load, save, and query project JSON data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils import validate_entities_payload, validate_templates_payload


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ENTITIES_PATH = DATA_DIR / "historical_entities.json"
TEMPLATES_PATH = DATA_DIR / "prompt_templates.json"


def load_json(path: str | Path) -> Any:
    """Load a JSON file with friendly error messages."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON mal formado en {resolved_path}: {error}") from error


def save_json(path: str | Path, data: Any) -> Path:
    """Persist JSON data creating parent directories when needed."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return resolved_path


def load_historical_entities(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the historical entities dataset."""
    resolved_path = path or ENTITIES_PATH
    payload = load_json(resolved_path)
    issues = validate_entities_payload(payload)
    if issues:
        raise ValueError(" ; ".join(issues))
    return payload


def load_prompt_templates(path: Path | None = None) -> dict[str, str]:
    """Load and validate the prompt templates dataset."""
    resolved_path = path or TEMPLATES_PATH
    payload = load_json(resolved_path)
    issues = validate_templates_payload(payload)
    if issues:
        raise ValueError(" ; ".join(issues))
    return payload


def get_entity_names(entities: list[dict[str, Any]]) -> list[str]:
    """Return entity names sorted alphabetically."""
    return sorted(entity.get("nombre", "") for entity in entities if entity.get("nombre"))


def get_entities_by_name(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lookup table keyed by entity name."""
    return {entity["nombre"]: entity for entity in entities if entity.get("nombre")}


def get_entity_by_name(
    entities: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Find an entity by its display name."""
    for entity in entities:
        if entity.get("nombre") == name:
            return entity
    return None


def get_entity_by_id(
    entities: list[dict[str, Any]], entity_id: str
) -> dict[str, Any] | None:
    """Find an entity by its unique identifier."""
    for entity in entities:
        if entity.get("id") == entity_id:
            return entity
    return None
