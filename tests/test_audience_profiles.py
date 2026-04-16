"""Tests for audience profiles loading and validation."""

from __future__ import annotations

import json

import pytest

from src.audience_profiles import (
    get_audience_profile,
    list_audience_options,
    load_audience_profiles,
)


def test_load_audience_profiles_from_default_file() -> None:
    profiles = load_audience_profiles()

    assert len(profiles) >= 6
    assert any(profile["audience_id"] == "estudiante_secundaria" for profile in profiles)


def test_get_audience_profile_by_id() -> None:
    profile = get_audience_profile("docente")

    assert profile["nombre_visible"] == "Docente"
    assert profile["preferred_tone"] == "didactico"


def test_list_audience_options_returns_lightweight_payload() -> None:
    options = list_audience_options()

    assert options
    assert {"audience_id", "nombre_visible", "descripcion"} <= set(options[0].keys())


def test_load_audience_profiles_validates_minimum_structure(tmp_path) -> None:
    invalid_path = tmp_path / "audience_profiles.json"
    invalid_path.write_text(
        json.dumps(
            [
                {
                    "audience_id": "invalido",
                    "nombre_visible": "Invalido",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_audience_profiles(str(invalid_path))
