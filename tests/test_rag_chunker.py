"""Tests for entity chunking in the RAG pipeline."""

from src.rag_chunker import build_chunks


def build_entity() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
        "epoca": "Siglo XVIII",
        "ubicacion": "Quito, Ecuador",
        "resumen": "Intelectual y precursor de ideas independentistas.",
        "descripcion_larga": "Figura destacada del pensamiento reformista en Quito y actor clave del debate ilustrado.",
        "importancia": "Ayudo a difundir ideas de cambio social y politico.",
        "lugares_relacionados": ["Quito"],
        "personajes_relacionados": ["Carlos Montufar"],
        "eventos_relacionados": ["Primer Grito de Independencia"],
        "etiquetas": ["independencia", "Quito"],
    }


def test_build_chunks_creates_expected_categories() -> None:
    chunks = build_chunks([build_entity()])

    categories = {chunk["categoria_chunk"] for chunk in chunks}
    assert categories == {"resumen", "descripcion", "importancia", "relaciones"}


def test_build_chunks_preserves_metadata() -> None:
    chunk = build_chunks([build_entity()])[0]

    assert chunk["entity_id"] == "eugenio_espejo"
    assert chunk["nombre"] == "Eugenio Espejo"
    assert chunk["tipo"] == "personaje"
    assert chunk["chunk_id"].startswith("eugenio_espejo__")
    assert chunk["etiquetas"] == ["independencia", "Quito"]


def test_build_chunks_do_not_create_empty_chunks() -> None:
    entity = {
        "id": "entidad_vacia",
        "nombre": "Entidad vacia",
        "tipo": "evento",
        "resumen": "",
        "descripcion_larga": "",
        "importancia": "",
        "lugares_relacionados": [],
        "personajes_relacionados": [],
        "eventos_relacionados": [],
        "etiquetas": [],
    }

    chunks = build_chunks([entity])

    assert chunks == []
