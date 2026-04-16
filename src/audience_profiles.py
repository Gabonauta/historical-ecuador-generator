"""Load and expose audience profiles for personalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.loader import load_json
from src.utils import has_text, safe_list, safe_str


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIENCE_PROFILES_PATH = "data/audience_profiles.json"
DEFAULT_AUDIENCE_ID = "estudiante_universitario"

REQUIRED_AUDIENCE_PROFILE_FIELDS = {
    "audience_id",
    "nombre_visible",
    "descripcion",
    "preferred_tone",
    "preferred_depth",
    "preferred_length",
    "preferred_purpose",
    "style_rules",
    "forbidden_patterns",
}


def load_audience_profiles(path: str = DEFAULT_AUDIENCE_PROFILES_PATH) -> list[dict[str, Any]]:
    """Load and validate audience profiles from JSON."""
    payload = load_json(_resolve_profiles_path(path))
    issues = _validate_profiles_payload(payload)
    if issues:
        raise ValueError(" ; ".join(issues))
    return payload


def get_audience_profile(
    audience_id: str,
    path: str = DEFAULT_AUDIENCE_PROFILES_PATH,
) -> dict[str, Any]:
    """Return a single audience profile by id."""
    normalized_target = safe_str(audience_id)
    for profile in load_audience_profiles(path):
        if safe_str(profile.get("audience_id")) == normalized_target:
            return profile
    raise KeyError(f"No se encontro el perfil de audiencia solicitado: {normalized_target}")


def list_audience_options(path: str = DEFAULT_AUDIENCE_PROFILES_PATH) -> list[dict[str, Any]]:
    """Return lightweight audience options for UI selectors."""
    profiles = load_audience_profiles(path)
    return [
        {
            "audience_id": profile["audience_id"],
            "nombre_visible": profile["nombre_visible"],
            "descripcion": profile["descripcion"],
        }
        for profile in profiles
    ]


def build_safe_audience_profile(audience_id: str = DEFAULT_AUDIENCE_ID) -> dict[str, Any]:
    """Build a safe default profile when external configuration is unavailable."""
    return {
        "audience_id": safe_str(audience_id, DEFAULT_AUDIENCE_ID) or DEFAULT_AUDIENCE_ID,
        "nombre_visible": "Audiencia general segura",
        "descripcion": "Perfil generico y seguro cuando no hay configuracion valida.",
        "preferred_tone": "formal",
        "preferred_depth": "media",
        "preferred_length": "media",
        "preferred_purpose": "divulgacion",
        "style_rules": [
            "Mantener claridad y sobriedad",
            "Usar solo informacion del contexto",
            "Priorizar rigor factual sobre estilo"
        ],
        "forbidden_patterns": [
            "No inventar datos",
            "No asumir conocimiento experto no sustentado"
        ],
    }


def _resolve_profiles_path(path: str | Path) -> Path:
    """Resolve relative data paths from the project root."""
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path


def _validate_profiles_payload(payload: Any) -> list[str]:
    """Validate top-level audience profiles payload."""
    issues: list[str] = []
    if not isinstance(payload, list):
        return ["El archivo de perfiles de audiencia debe contener una lista JSON."]

    seen_ids: set[str] = set()
    for index, profile in enumerate(payload, start=1):
        issues.extend(_validate_single_profile(profile, index))
        audience_id = safe_str(profile.get("audience_id")) if isinstance(profile, dict) else ""
        if audience_id:
            if audience_id in seen_ids:
                issues.append(f"El perfil de audiencia '{audience_id}' esta duplicado.")
            seen_ids.add(audience_id)

    return issues


def _validate_single_profile(profile: Any, index: int) -> list[str]:
    """Validate one audience profile structure."""
    if not isinstance(profile, dict):
        return [f"El perfil de audiencia #{index} debe ser un objeto JSON."]

    issues: list[str] = []
    missing = sorted(field for field in REQUIRED_AUDIENCE_PROFILE_FIELDS if field not in profile)
    if missing:
        issues.append(
            f"El perfil de audiencia #{index} tiene campos faltantes: {', '.join(missing)}."
        )

    for text_field in [
        "audience_id",
        "nombre_visible",
        "descripcion",
        "preferred_tone",
        "preferred_depth",
        "preferred_length",
        "preferred_purpose",
    ]:
        if text_field in profile and not has_text(profile.get(text_field)):
            issues.append(
                f"El perfil de audiencia #{index} tiene un campo vacio: {text_field}."
            )

    for list_field in ["style_rules", "forbidden_patterns"]:
        if list_field in profile and not isinstance(profile.get(list_field), list):
            issues.append(
                f"El perfil de audiencia #{index} debe tener '{list_field}' como lista."
            )
        elif list_field in profile:
            cleaned_values = [safe_str(value) for value in safe_list(profile.get(list_field)) if has_text(value)]
            if not cleaned_values:
                issues.append(
                    f"El perfil de audiencia #{index} debe incluir al menos un valor en '{list_field}'."
                )

    return issues
