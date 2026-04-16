"""Prompt construction for controlled historical generation with personalization."""

from __future__ import annotations

from typing import Any

from src.utils import normalize_text, safe_list, safe_str


OUTPUT_INSTRUCTIONS = {
    "ficha_historica": (
        "Redacta una ficha historica formal, clara, informativa y objetiva, con secciones bien organizadas."
    ),
    "resumen_corto": (
        "Redacta un resumen corto de maximo 80 palabras, claro, directo y util para estudiantes."
    ),
    "texto_turistico": (
        "Redacta un texto turistico atractivo que resalte valor cultural, historico e interes para visitantes."
    ),
    "post_redes": (
        "Redacta un post breve para redes sociales, llamativo, informativo y facil de leer."
    ),
}

TONE_GUIDANCE = {
    "formal": "Usa un tono formal, claro y preciso.",
    "didactico": "Usa un tono didactico, claro y explicativo.",
    "divulgativo": "Usa un tono divulgativo, accesible y atractivo.",
    "promocional": "Usa un tono promocional sobrio, sin exageraciones ni invenciones.",
    "narrativo": "Usa un tono narrativo controlado, manteniendo rigor factual.",
}

DEPTH_GUIDANCE = {
    "baja": "Prioriza las ideas principales y evita exceso de detalle.",
    "media": "Mantiene un nivel de detalle intermedio y bien organizado.",
    "alta": "Desarrolla con mas detalle explicativo cuando el contexto lo permita.",
}

LENGTH_GUIDANCE = {
    "corta": "Mantiene una longitud breve y concentrada.",
    "media": "Mantiene una longitud media y equilibrada.",
    "larga": "Desarrolla una longitud amplia sin agregar informacion no sustentada.",
}

PURPOSE_GUIDANCE = {
    "educativo": "Prioriza claridad pedagogica y utilidad para aprendizaje.",
    "turistico": "Prioriza interes cultural, valor patrimonial y accesibilidad para visitantes.",
    "academico": "Prioriza precision conceptual y estructura sobria.",
    "redes": "Prioriza sintesis, claridad e impacto responsable para redes.",
    "divulgacion": "Prioriza accesibilidad, contexto y lectura fluida.",
}


def build_prompt(
    entity: dict[str, Any],
    output_type: str,
    base_context: str,
    retrieved_context: str = "",
    personalization_config: dict[str, Any] | None = None,
) -> str:
    """Build a robust Spanish prompt grounded in base context plus optional RAG."""
    if output_type not in OUTPUT_INSTRUCTIONS:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

    personalization_block = _build_personalization_block(personalization_config)

    prompt = (
        "Eres un asistente de escritura historica del Ecuador.\n"
        "Usa solo la informacion proporcionada en el contexto.\n"
        "No inventes datos, fechas, relaciones, citas ni hechos no presentes.\n"
        "Si hay ambiguedad, prioriza el contexto recuperado y el contexto base estructurado.\n"
        "Si falta informacion, omite ese detalle con elegancia.\n"
        "No menciones que faltan datos salvo que sea estrictamente necesario.\n"
        f"{OUTPUT_INSTRUCTIONS[output_type]}\n"
        f"Tipo de salida solicitado: {output_type}\n"
        f"Entidad principal: {entity.get('nombre', '')}\n\n"
        "Contexto base estructurado:\n"
        f"{base_context}\n\n"
    )

    if personalization_block:
        prompt += (
            "Personalizacion aplicada:\n"
            f"{personalization_block}\n\n"
        )

    if retrieved_context.strip():
        prompt += (
            "Contexto recuperado adicional:\n"
            f"{retrieved_context}\n\n"
        )
    else:
        prompt += "Contexto recuperado adicional:\nNo disponible.\n\n"

    prompt += (
        "Genera solo el texto final.\n"
        "No expliques el proceso ni cites instrucciones internas."
    )

    return prompt


def _build_personalization_block(personalization_config: dict[str, Any] | None) -> str:
    """Format audience-driven personalization rules for the prompt."""
    if not personalization_config:
        return ""

    tone = normalize_text(personalization_config.get("tone"), lowercase=True)
    depth = normalize_text(personalization_config.get("depth"), lowercase=True)
    length = normalize_text(personalization_config.get("length"), lowercase=True)
    purpose = normalize_text(personalization_config.get("purpose"), lowercase=True)
    audience_name = safe_str(
        personalization_config.get("nombre_visible") or personalization_config.get("audience_id"),
        "Audiencia general",
    )

    lines = [
        f"Audiencia objetivo: {audience_name}",
        f"Tono solicitado: {tone or 'formal'}",
        f"Nivel de profundidad: {depth or 'media'}",
        f"Longitud deseada: {length or 'media'}",
        f"Proposito principal: {purpose or 'divulgacion'}",
        TONE_GUIDANCE.get(tone, TONE_GUIDANCE["formal"]),
        DEPTH_GUIDANCE.get(depth, DEPTH_GUIDANCE["media"]),
        LENGTH_GUIDANCE.get(length, LENGTH_GUIDANCE["media"]),
        PURPOSE_GUIDANCE.get(purpose, PURPOSE_GUIDANCE["divulgacion"]),
    ]

    style_rules = [safe_str(rule) for rule in safe_list(personalization_config.get("style_rules")) if normalize_text(rule)]
    forbidden_patterns = [
        safe_str(rule)
        for rule in safe_list(personalization_config.get("forbidden_patterns"))
        if normalize_text(rule)
    ]

    if style_rules:
        lines.append("Reglas de estilo: " + "; ".join(style_rules) + ".")
    if forbidden_patterns:
        lines.append("Evita especialmente: " + "; ".join(forbidden_patterns) + ".")

    return "\n".join(lines)
