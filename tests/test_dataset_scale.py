"""Dataset scale checks for the historical entities base."""

from __future__ import annotations

from src.loader import load_historical_entities


def test_historical_entities_dataset_scales_to_one_hundred() -> None:
    entities = load_historical_entities()

    assert len(entities) == 100
    assert len({entity["id"] for entity in entities}) == 100


def test_historical_entities_dataset_keeps_supported_types() -> None:
    entities = load_historical_entities()

    supported_types = {"personaje", "lugar", "evento"}
    assert {entity["tipo"] for entity in entities} <= supported_types
