"""Prompt construction for controlled historical generation with optional RAG."""

from __future__ import annotations

from typing import Any


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


def build_prompt(
    entity: dict[str, Any],
    output_type: str,
    base_context: str,
    retrieved_context: str = "",
) -> str:
    """Build a robust Spanish prompt grounded in base context plus optional RAG."""
    if output_type not in OUTPUT_INSTRUCTIONS:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

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
