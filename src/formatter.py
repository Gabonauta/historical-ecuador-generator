"""Formatting helpers for building clean historical outputs."""

from __future__ import annotations

from typing import Any

from src.utils import compact_whitespace, ensure_list, format_year_range, has_text, safe_strip


def clean_text(text: Any) -> str:
    """Normalize and clean text values."""
    return compact_whitespace(safe_strip(text))


def format_related_list(items: list[Any] | None, empty_text: str = "No disponible") -> str:
    """Format related entities and tags as a comma-separated string."""
    values = [clean_text(item) for item in ensure_list(items) if has_text(item)]
    return ", ".join(values) if values else empty_text


def format_bulleted_list(items: list[Any] | None) -> str:
    """Format a list into markdown bullets when values exist."""
    values = [clean_text(item) for item in ensure_list(items) if has_text(item)]
    return "\n".join(f"- {item}" for item in values)


def build_section(title: str, content: str) -> str:
    """Return a reusable markdown section only when content is present."""
    text = clean_text(content)
    if not text:
        return ""
    return f"**{title}:** {text}"


def build_paragraphs(parts: list[str]) -> str:
    """Join non-empty paragraph fragments with blank lines."""
    paragraphs = [part.strip() for part in parts if has_text(part)]
    return "\n\n".join(paragraphs)


def format_metadata(entity: dict[str, Any]) -> str:
    """Build a readable metadata block for the selected entity."""
    year_range = format_year_range(entity.get("anio_inicio"), entity.get("anio_fin"))

    lines = [
        f"ID: {clean_text(entity.get('id'))}",
        f"Tipo: {clean_text(entity.get('tipo'))}",
        f"Epoca: {clean_text(entity.get('epoca'))}",
        f"Ubicacion: {clean_text(entity.get('ubicacion'))}",
    ]

    if year_range:
        lines.append(f"Periodo: {year_range}")
    if has_text(entity.get("fuente_base")):
        lines.append(f"Fuente base: {clean_text(entity.get('fuente_base'))}")

    return "\n".join(lines)


def build_context_block(entity: dict[str, Any]) -> str:
    """Create a concise context line with the main identifying fields."""
    period = format_year_range(entity.get("anio_inicio"), entity.get("anio_fin"))
    pieces = [
        clean_text(entity.get("tipo")),
        clean_text(entity.get("epoca")),
        clean_text(entity.get("ubicacion")),
        period,
    ]
    return " | ".join(piece for piece in pieces if piece)
