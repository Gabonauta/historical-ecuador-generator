"""Build controlled entity and retrieval context for prompting."""

from __future__ import annotations

from typing import Any

from src.utils import format_year_range, has_text, safe_list, safe_str


def _format_list(items: Any) -> str:
    values = [safe_str(item) for item in safe_list(items) if has_text(item)]
    return ", ".join(values) if values else "No disponible"


def build_entity_context(entity: dict[str, Any]) -> str:
    """Build a deterministic text context block from a local entity record."""
    lines = [
        f"Nombre: {safe_str(entity.get('nombre'), 'No disponible')}",
        f"Tipo: {safe_str(entity.get('tipo'), 'No disponible')}",
        f"Epoca: {safe_str(entity.get('epoca'), 'No disponible')}",
        f"Ubicacion: {safe_str(entity.get('ubicacion'), 'No disponible')}",
        f"Resumen: {safe_str(entity.get('resumen'), 'No disponible')}",
        f"Descripcion larga: {safe_str(entity.get('descripcion_larga'), 'No disponible')}",
        f"Importancia: {safe_str(entity.get('importancia'), 'No disponible')}",
        f"Lugares relacionados: {_format_list(entity.get('lugares_relacionados'))}",
        f"Personajes relacionados: {_format_list(entity.get('personajes_relacionados'))}",
        f"Eventos relacionados: {_format_list(entity.get('eventos_relacionados'))}",
        f"Etiquetas: {_format_list(entity.get('etiquetas'))}",
        f"Periodo en años: {format_year_range(entity.get('anio_inicio'), entity.get('anio_fin')) or 'No disponible'}",
        f"Fuente base: {safe_str(entity.get('fuente_base'), 'No disponible')}",
    ]
    return "\n".join(lines)


def build_retrieved_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved RAG chunks into a prompt-ready context block."""
    blocks: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        score = chunk.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
        lines = [
            f"Fragmento {index}",
            f"Entidad: {safe_str(chunk.get('nombre'), 'No disponible')}",
            f"Tipo: {safe_str(chunk.get('tipo'), 'No disponible')}",
            f"Categoria: {safe_str(chunk.get('categoria_chunk'), 'No disponible')}",
            f"Score: {score_text}",
            f"Texto: {safe_str(chunk.get('texto'), 'No disponible')}",
        ]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
