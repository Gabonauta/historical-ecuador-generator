"""Tests for context building."""

from src.context_builder import build_entity_context


def test_build_entity_context_with_complete_entity() -> None:
    entity = {
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
        "epoca": "Siglo XVIII",
        "ubicacion": "Quito, Ecuador",
        "resumen": "Intelectual y precursor independentista.",
        "descripcion_larga": "Figura clave del pensamiento ilustrado en Quito.",
        "importancia": "Difundio ideas de cambio.",
        "lugares_relacionados": ["Quito"],
        "personajes_relacionados": ["Carlos Montufar"],
        "eventos_relacionados": ["Primer Grito de Independencia"],
        "etiquetas": ["independencia", "intelectual"],
        "anio_inicio": 1747,
        "anio_fin": 1795,
        "fuente_base": "Registro base",
    }

    context = build_entity_context(entity)

    assert "Nombre: Eugenio Espejo" in context
    assert "Lugares relacionados: Quito" in context
    assert "Periodo en años: 1747 - 1795" in context


def test_build_entity_context_with_missing_optional_fields() -> None:
    entity = {
        "nombre": "Plaza Grande",
        "tipo": "lugar",
        "resumen": "Espacio emblematico de Quito.",
    }

    context = build_entity_context(entity)

    assert "Nombre: Plaza Grande" in context
    assert "Descripcion larga: No disponible" in context
    assert "Eventos relacionados: No disponible" in context
