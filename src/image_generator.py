"""Coordinate grounded image generation with optional RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.context_builder import build_entity_context, build_retrieved_context
from src.embeddings_client import EmbeddingsClientError
from src.image_client import DEFAULT_IMAGE_SAVE_DIR, generate_image
from src.image_prompt_builder import build_image_prompt
from src.llm_client import get_safe_error_chain
from src.rag_retriever import RAGRetrieverError, load_index, retrieve
from src.utils import has_text, safe_list, safe_str


def generate_visual_content(
    entity: dict[str, Any],
    provider: str = "openai",
    use_rag: bool = True,
    top_k: int = 5,
    image_mode: str = "retrato_historico",
    visual_style: str = "realista",
    size: str = "1024x1024",
    embedding_provider: str = "openai",
    quality: str = "standard",
    save_dir: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Generate visual content using grounded context and a safe image provider wrapper."""
    base_context = build_entity_context(entity)
    requested_top_k = max(1, int(top_k))
    retrieved_chunks: list[dict[str, Any]] = []
    retrieved_context = ""
    rag_enabled = False
    resolved_embedding_provider = embedding_provider
    notices: list[str] = []

    if use_rag:
        query = _build_visual_rag_query(entity, image_mode, visual_style)
        try:
            index_data = load_index()
            metadata_provider = safe_str(index_data.get("metadata", {}).get("embedding_provider")).lower()
            if metadata_provider:
                resolved_embedding_provider = metadata_provider
            if (
                metadata_provider
                and metadata_provider != safe_str(embedding_provider).lower()
            ):
                notices.append(
                    "El indice RAG local fue construido con otro provider de embeddings; "
                    "se uso el provider compatible con el indice."
                )

            retrieved_chunks = retrieve(
                query=query,
                top_k=requested_top_k,
                entity_type=entity.get("tipo"),
                provider=resolved_embedding_provider,
                index_data=index_data,
            )
            if retrieved_chunks:
                retrieved_context = build_retrieved_context(retrieved_chunks)
                rag_enabled = True
        except (FileNotFoundError, RAGRetrieverError, EmbeddingsClientError, ValueError) as error:
            notices.append(
                _build_runtime_notice(
                    message="No fue posible usar RAG visual; se continuo solo con el contexto base.",
                    error=error,
                    debug=debug,
                )
            )

    try:
        prompt = build_image_prompt(
            entity=entity,
            image_mode=image_mode,
            visual_style=visual_style,
            base_context=base_context,
            retrieved_context=retrieved_context,
        )
    except ValueError as error:
        return {
            "provider": "fallback",
            "status": "error",
            "image_mode": image_mode,
            "visual_style": visual_style,
            "size": size,
            "use_rag": rag_enabled,
            "embedding_provider": resolved_embedding_provider,
            "prompt": "",
            "base_context": base_context,
            "retrieved_context": retrieved_context,
            "retrieved_chunks": retrieved_chunks,
            "image_path": None,
            "image_url": None,
            "error": _build_runtime_notice(
                message="No fue posible construir el prompt visual.",
                error=error,
                debug=debug,
            ),
        }

    image_result = generate_image(
        provider=provider,
        prompt=prompt,
        size=size,
        quality=quality,
        save_dir=save_dir or str(Path(DEFAULT_IMAGE_SAVE_DIR)),
    )

    final_error = _join_notices(notices + [image_result.get("error")])
    return {
        "provider": image_result["provider"],
        "status": image_result["status"],
        "image_mode": image_mode,
        "visual_style": visual_style,
        "size": size,
        "use_rag": rag_enabled,
        "embedding_provider": resolved_embedding_provider,
        "prompt": prompt,
        "base_context": base_context,
        "retrieved_context": retrieved_context,
        "retrieved_chunks": retrieved_chunks,
        "image_path": image_result.get("image_path"),
        "image_url": image_result.get("image_url"),
        "error": final_error,
    }


def _build_visual_rag_query(entity: dict[str, Any], image_mode: str, visual_style: str) -> str:
    """Build a semantic query optimized for visual grounding."""
    query_parts = [
        safe_str(entity.get("nombre")),
        f"Tipo: {safe_str(entity.get('tipo'))}",
        f"Modo visual: {image_mode.replace('_', ' ')}",
        f"Estilo visual: {visual_style.replace('_', ' ')}",
        safe_str(entity.get("epoca")),
        safe_str(entity.get("ubicacion")),
        safe_str(entity.get("resumen")),
        safe_str(entity.get("descripcion_larga")),
        safe_str(entity.get("importancia")),
    ]

    etiquetas = ", ".join(safe_str(tag) for tag in safe_list(entity.get("etiquetas")) if has_text(tag))
    if etiquetas:
        query_parts.append(f"Etiquetas: {etiquetas}")

    return " | ".join(part for part in query_parts if has_text(part))


def _build_runtime_notice(message: str, error: Exception, debug: bool) -> str:
    """Build a safe runtime notice for degraded visual flows."""
    if debug:
        return f"{message} Diagnostico seguro: {get_safe_error_chain(error)}"
    return message


def _join_notices(notices: list[str | None]) -> str | None:
    """Join safe notices into one string when present."""
    cleaned_notices = [notice for notice in notices if notice]
    if not cleaned_notices:
        return None
    return " ".join(cleaned_notices)
