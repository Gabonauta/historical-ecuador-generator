"""Chunk historical entities into retrievable RAG documents."""

from __future__ import annotations

from typing import Any

from src.utils import has_text, normalize_text, safe_list, safe_str


MIN_PRIMARY_TEXT_LENGTH = 80


def build_chunks(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform entities into entity-scoped chunks with retrieval metadata."""
    chunks: list[dict[str, Any]] = []

    for entity in entities:
        entity_id = safe_str(entity.get("id"))
        nombre = safe_str(entity.get("nombre"))
        tipo = safe_str(entity.get("tipo"))
        etiquetas = _clean_list(entity.get("etiquetas"))

        if not entity_id or not nombre:
            continue

        chunk_specs = [
            (
                "resumen",
                _expand_primary_text(
                    primary=entity.get("resumen"),
                    fallback_values=[entity.get("importancia"), entity.get("descripcion_larga")],
                ),
            ),
            (
                "descripcion",
                _expand_primary_text(
                    primary=entity.get("descripcion_larga"),
                    fallback_values=[entity.get("resumen"), entity.get("importancia")],
                ),
            ),
            (
                "importancia",
                _expand_primary_text(
                    primary=entity.get("importancia"),
                    fallback_values=[entity.get("resumen"), entity.get("descripcion_larga")],
                ),
            ),
            ("relaciones", _build_relationship_text(entity)),
        ]

        for categoria_chunk, body in chunk_specs:
            texto = _build_chunk_text(entity, categoria_chunk, body)
            if not has_text(texto):
                continue

            chunks.append(
                {
                    "chunk_id": f"{entity_id}__{categoria_chunk}",
                    "entity_id": entity_id,
                    "nombre": nombre,
                    "tipo": tipo,
                    "categoria_chunk": categoria_chunk,
                    "texto": texto,
                    "etiquetas": etiquetas,
                }
            )

    return chunks


def _build_chunk_text(entity: dict[str, Any], categoria_chunk: str, body: Any) -> str:
    """Compose a chunk with lightweight entity framing plus category content."""
    section_body = normalize_text(body)
    if not section_body:
        return ""

    prefix_parts = [
        f"Entidad: {safe_str(entity.get('nombre'))}.",
        f"Tipo: {safe_str(entity.get('tipo'))}.",
    ]

    if has_text(entity.get("epoca")):
        prefix_parts.append(f"Epoca: {safe_str(entity.get('epoca'))}.")
    if has_text(entity.get("ubicacion")):
        prefix_parts.append(f"Ubicacion: {safe_str(entity.get('ubicacion'))}.")

    category_label = {
        "resumen": "Resumen",
        "descripcion": "Descripcion amplia",
        "importancia": "Importancia historica",
        "relaciones": "Relaciones y etiquetas",
    }.get(categoria_chunk, "Contenido")

    return normalize_text(" ".join(prefix_parts + [f"{category_label}: {section_body}"]))


def _expand_primary_text(primary: Any, fallback_values: list[Any]) -> str:
    """Keep chunks meaningful without mixing multiple entities."""
    parts: list[str] = []
    primary_text = normalize_text(primary)
    if primary_text:
        parts.append(primary_text)

    for fallback_value in fallback_values:
        if sum(len(part) for part in parts) >= MIN_PRIMARY_TEXT_LENGTH:
            break

        fallback_text = normalize_text(fallback_value)
        if fallback_text and fallback_text not in parts:
            parts.append(fallback_text)

    return " ".join(parts)


def _build_relationship_text(entity: dict[str, Any]) -> str:
    """Build a relationship-focused chunk from related lists and tags."""
    lines: list[str] = []

    related_groups = {
        "Lugares relacionados": entity.get("lugares_relacionados"),
        "Personajes relacionados": entity.get("personajes_relacionados"),
        "Eventos relacionados": entity.get("eventos_relacionados"),
        "Etiquetas": entity.get("etiquetas"),
    }

    for label, values in related_groups.items():
        cleaned_values = _clean_list(values)
        if cleaned_values:
            lines.append(f"{label}: {', '.join(cleaned_values)}.")

    return " ".join(lines)


def _clean_list(values: Any) -> list[str]:
    """Normalize a list field into clean, non-empty strings."""
    return [normalize_text(value) for value in safe_list(values) if has_text(value)]
