"""Resolve audience-driven personalization settings."""

from __future__ import annotations

from typing import Any

from src.audience_profiles import build_safe_audience_profile
from src.utils import normalize_text, safe_list, safe_str


SUPPORTED_TONES = ("formal", "didactico", "divulgativo", "promocional", "narrativo")
SUPPORTED_DEPTHS = ("baja", "media", "alta")
SUPPORTED_LENGTHS = ("corta", "media", "larga")
SUPPORTED_PURPOSES = ("educativo", "turistico", "academico", "redes", "divulgacion")


def build_personalization_config(
    audience_profile: dict[str, Any],
    tone: str | None = None,
    depth: str | None = None,
    length: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Build the final personalization config from a profile plus optional overrides."""
    safe_profile = audience_profile if isinstance(audience_profile, dict) else build_safe_audience_profile()
    fallback_profile = build_safe_audience_profile(
        safe_str(safe_profile.get("audience_id")) or build_safe_audience_profile()["audience_id"]
    )

    resolved_profile = {
        "audience_id": safe_str(safe_profile.get("audience_id"), fallback_profile["audience_id"]),
        "nombre_visible": safe_str(safe_profile.get("nombre_visible"), fallback_profile["nombre_visible"]),
        "descripcion": safe_str(safe_profile.get("descripcion"), fallback_profile["descripcion"]),
        "preferred_tone": safe_str(safe_profile.get("preferred_tone"), fallback_profile["preferred_tone"]),
        "preferred_depth": safe_str(safe_profile.get("preferred_depth"), fallback_profile["preferred_depth"]),
        "preferred_length": safe_str(safe_profile.get("preferred_length"), fallback_profile["preferred_length"]),
        "preferred_purpose": safe_str(safe_profile.get("preferred_purpose"), fallback_profile["preferred_purpose"]),
        "style_rules": _clean_text_list(safe_profile.get("style_rules")) or fallback_profile["style_rules"],
        "forbidden_patterns": _clean_text_list(safe_profile.get("forbidden_patterns")) or fallback_profile["forbidden_patterns"],
    }

    resolved_tone = _resolve_value(
        override=tone,
        default=resolved_profile["preferred_tone"],
        supported=SUPPORTED_TONES,
    )
    resolved_depth = _resolve_value(
        override=depth,
        default=resolved_profile["preferred_depth"],
        supported=SUPPORTED_DEPTHS,
    )
    resolved_length = _resolve_value(
        override=length,
        default=resolved_profile["preferred_length"],
        supported=SUPPORTED_LENGTHS,
    )
    resolved_purpose = _resolve_value(
        override=purpose,
        default=resolved_profile["preferred_purpose"],
        supported=SUPPORTED_PURPOSES,
    )

    return {
        "audience_id": resolved_profile["audience_id"],
        "nombre_visible": resolved_profile["nombre_visible"],
        "descripcion": resolved_profile["descripcion"],
        "tone": resolved_tone,
        "depth": resolved_depth,
        "length": resolved_length,
        "purpose": resolved_purpose,
        "style_rules": resolved_profile["style_rules"],
        "forbidden_patterns": resolved_profile["forbidden_patterns"],
    }


def _resolve_value(override: str | None, default: str, supported: tuple[str, ...]) -> str:
    """Resolve a personalization override against supported values."""
    normalized_override = normalize_text(override, lowercase=True)
    if normalized_override in supported:
        return normalized_override

    normalized_default = normalize_text(default, lowercase=True)
    if normalized_default in supported:
        return normalized_default

    return supported[0]


def _clean_text_list(values: Any) -> list[str]:
    """Normalize a list of textual personalization rules."""
    return [safe_str(value) for value in safe_list(values) if normalize_text(value)]
