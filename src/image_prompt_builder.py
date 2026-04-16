"""Build grounded visual prompts for historical image generation."""

from __future__ import annotations

from typing import Any

from src.utils import normalize_text, safe_str


SUPPORTED_IMAGE_MODES = (
    "retrato_historico",
    "escena_historica",
    "postal_turistica",
    "ilustracion_educativa",
)

SUPPORTED_VISUAL_STYLES = (
    "realista",
    "pintura_oleo",
    "grabado_antiguo",
    "ilustracion_editorial",
)

MODE_INSTRUCTIONS = {
    "retrato_historico": (
        "Construye una composicion centrada en la figura principal, con pose sobria, "
        "encuadre claro y fondo historicamente plausible. Si faltan rasgos exactos, "
        "mantiene una representacion prudente y no hiper-especifica."
    ),
    "escena_historica": (
        "Construye una escena historica contextual, mostrando accion, entorno y actores "
        "solo cuando el contexto lo permita. Evita dramatizaciones extremas o simbolos no sustentados."
    ),
    "postal_turistica": (
        "Construye una imagen evocadora y accesible para divulgacion cultural, resaltando "
        "arquitectura, paisaje, espacio y atmosfera sin volverla una fantasia contemporanea."
    ),
    "ilustracion_educativa": (
        "Construye una imagen clara, ordenada y didactica, util para aprendizaje visual. "
        "Prioriza legibilidad compositiva, contexto historico y ausencia de elementos distractores."
    ),
}

STYLE_INSTRUCTIONS = {
    "realista": (
        "Estilo realista sobrio, con textura natural, luz plausible y sin hiperrealismo exagerado."
    ),
    "pintura_oleo": (
        "Estilo de pintura al oleo historica, con pincelada visible, paleta cuidada y atmosfera clasica."
    ),
    "grabado_antiguo": (
        "Estilo de grabado antiguo, con lineas definidas, contraste grafico y apariencia de documento historico."
    ),
    "ilustracion_editorial": (
        "Estilo de ilustracion editorial contemporanea sobria, con composicion limpia y enfoque divulgativo."
    ),
}

ENTITY_GUIDANCE = {
    "personaje": (
        "Si la entidad es un personaje, prioriza retrato o escena vinculada a su contexto historico. "
        "Solo describe vestimenta, accesorios o rasgos fisicos cuando el contexto los sugiera de forma razonable."
    ),
    "lugar": (
        "Si la entidad es un lugar, prioriza arquitectura, espacio, materiales, atmosfera y escala urbana o geografica."
    ),
    "evento": (
        "Si la entidad es un evento, prioriza la escena colectiva, el momento historico, el entorno y la composicion narrativa."
    ),
}


def build_image_prompt(
    entity: dict[str, Any],
    image_mode: str,
    visual_style: str,
    base_context: str,
    retrieved_context: str = "",
) -> str:
    """Build a safe and grounded Spanish prompt for historical image generation."""
    normalized_mode = normalize_text(image_mode, lowercase=True)
    normalized_style = normalize_text(visual_style, lowercase=True)

    if normalized_mode not in SUPPORTED_IMAGE_MODES:
        raise ValueError(f"Modo visual no soportado: {image_mode}")
    if normalized_style not in SUPPORTED_VISUAL_STYLES:
        raise ValueError(f"Estilo visual no soportado: {visual_style}")

    entity_name = safe_str(entity.get("nombre"), "Entidad historica")
    entity_type = normalize_text(entity.get("tipo"), lowercase=True) or "entidad"

    prompt_sections = [
        "Genera una imagen historica del Ecuador con grounding razonable.",
        f"Entidad principal: {entity_name}.",
        f"Tipo de entidad: {entity_type}.",
        f"Modo visual solicitado: {normalized_mode}.",
        f"Estilo visual solicitado: {normalized_style}.",
        "Usa solo la informacion proporcionada en el contexto.",
        "No inventes atributos historicos, uniformes, rostros exactos, insignias, arquitectura o simbolos no sustentados.",
        "Si faltan detalles visuales concretos, usa una representacion prudente y general, sin anacronismos.",
        "Prioriza fidelidad historica razonable sobre espectacularidad.",
        "No incluyas texto incrustado, marcas modernas, logos, fotografias contemporaneas ni efectos cinematograficos excesivos.",
        ENTITY_GUIDANCE.get(entity_type, "Representa la entidad con sobriedad y contexto historico plausible."),
        MODE_INSTRUCTIONS[normalized_mode],
        STYLE_INSTRUCTIONS[normalized_style],
        "Contexto base estructurado:",
        base_context,
    ]

    if normalize_text(retrieved_context):
        prompt_sections.extend(
            [
                "Contexto recuperado adicional:",
                retrieved_context,
                "Si hay ambiguedad, prioriza el contexto estructurado y recuperado sobre detalles imaginados.",
            ]
        )
    else:
        prompt_sections.append(
            "No se proporciono contexto recuperado adicional; mantente conservador en detalles no confirmados."
        )

    prompt_sections.append(
        "Devuelve una sola descripcion visual lista para un generador de imagen, en espanol y sin explicar el proceso."
    )

    return "\n".join(prompt_sections)
