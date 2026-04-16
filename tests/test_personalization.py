"""Tests for personalization config resolution."""

from src.personalization import build_personalization_config


def build_profile() -> dict:
    return {
        "audience_id": "estudiante_secundaria",
        "nombre_visible": "Estudiante de secundaria",
        "descripcion": "Perfil de prueba.",
        "preferred_tone": "didactico",
        "preferred_depth": "media",
        "preferred_length": "media",
        "preferred_purpose": "educativo",
        "style_rules": ["Usar lenguaje claro"],
        "forbidden_patterns": ["No asumir conocimiento experto"],
    }


def test_build_personalization_config_resolves_defaults() -> None:
    config = build_personalization_config(build_profile())

    assert config["audience_id"] == "estudiante_secundaria"
    assert config["tone"] == "didactico"
    assert config["depth"] == "media"
    assert config["length"] == "media"
    assert config["purpose"] == "educativo"


def test_build_personalization_config_applies_overrides() -> None:
    config = build_personalization_config(
        build_profile(),
        tone="formal",
        depth="alta",
        length="larga",
        purpose="academico",
    )

    assert config["tone"] == "formal"
    assert config["depth"] == "alta"
    assert config["length"] == "larga"
    assert config["purpose"] == "academico"


def test_build_personalization_config_returns_consistent_payload() -> None:
    config = build_personalization_config(build_profile(), tone="divulgativo")

    assert set(config.keys()) == {
        "audience_id",
        "nombre_visible",
        "descripcion",
        "tone",
        "depth",
        "length",
        "purpose",
        "style_rules",
        "forbidden_patterns",
    }
    assert config["style_rules"] == ["Usar lenguaje claro"]
