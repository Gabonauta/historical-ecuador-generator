"""Hybrid content generation coordinator for Phase 4 with optional RAG."""

from __future__ import annotations

from typing import Any

from src.context_builder import build_entity_context, build_retrieved_context
from src.embeddings_client import EmbeddingsClientError
from src.fallback_generator import generate_fallback_content
from src.llm_client import LLMClientError, generate_text, get_safe_error_chain
from src.prompt_builder import build_prompt
from src.rag_retriever import RAGRetrieverError, load_index, retrieve
from src.utils import has_text, safe_list, safe_str


SUPPORTED_OUTPUTS = (
    "ficha_historica",
    "resumen_corto",
    "texto_turistico",
    "post_redes",
)


def generate_content(
    entity: dict[str, Any],
    output_type: str,
    provider: str = "openai",
    use_llm: bool = True,
    use_rag: bool = True,
    top_k: int = 5,
    model: str | None = None,
    embedding_provider: str = "openai",
    debug: bool = False,
) -> dict[str, Any]:
    """Generate content using an LLM when available, otherwise use safe fallback."""
    if output_type not in SUPPORTED_OUTPUTS:
        raise ValueError(f"Tipo de salida no soportado: {output_type}")

    base_context = build_entity_context(entity)
    requested_top_k = max(1, int(top_k))
    retrieved_chunks: list[dict[str, Any]] = []
    retrieved_context = ""
    rag_enabled = False
    resolved_embedding_provider = embedding_provider
    notices: list[str] = []

    if use_llm and use_rag:
        query = _build_rag_query(entity, output_type)
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
                    message="No fue posible usar RAG; se continuo solo con el contexto base.",
                    error=error,
                    debug=debug,
                )
            )

    prompt = build_prompt(
        entity=entity,
        output_type=output_type,
        base_context=base_context,
        retrieved_context=retrieved_context,
    )

    if not use_llm:
        generated_text = generate_fallback_content(entity, output_type)
        return _build_result(
            mode="fallback",
            provider_name="fallback",
            output_type=output_type,
            prompt=prompt,
            base_context=base_context,
            retrieved_context=retrieved_context,
            retrieved_chunks=retrieved_chunks,
            generated_text=generated_text,
            use_rag=False,
            embedding_provider=resolved_embedding_provider,
            error=_join_notices(notices),
        )

    try:
        generated_text = generate_text(
            provider=provider,
            prompt=prompt,
            model=model,
            temperature=0.3,
        )
        return _build_result(
            mode="llm",
            provider_name=provider,
            output_type=output_type,
            prompt=prompt,
            base_context=base_context,
            retrieved_context=retrieved_context,
            retrieved_chunks=retrieved_chunks,
            generated_text=generated_text,
            use_rag=rag_enabled,
            embedding_provider=resolved_embedding_provider,
            error=_join_notices(notices),
        )
    except LLMClientError as error:
        generated_text = generate_fallback_content(entity, output_type)
        notices.append(
            _build_runtime_notice(
                message="No fue posible usar el proveedor LLM seleccionado; se uso el modo fallback.",
                error=error,
                debug=debug,
            )
        )
        return _build_result(
            mode="fallback",
            provider_name="fallback",
            output_type=output_type,
            prompt=prompt,
            base_context=base_context,
            retrieved_context=retrieved_context,
            retrieved_chunks=retrieved_chunks,
            generated_text=generated_text,
            use_rag=rag_enabled,
            embedding_provider=resolved_embedding_provider,
            error=_join_notices(notices),
        )


def _build_rag_query(entity: dict[str, Any], output_type: str) -> str:
    """Build a focused semantic query from the selected entity."""
    query_parts = [
        safe_str(entity.get("nombre")),
        f"Tipo: {safe_str(entity.get('tipo'))}",
        f"Salida: {output_type.replace('_', ' ')}",
        safe_str(entity.get("resumen")),
        safe_str(entity.get("descripcion_larga")),
        safe_str(entity.get("importancia")),
    ]

    etiquetas = ", ".join(safe_str(tag) for tag in safe_list(entity.get("etiquetas")) if has_text(tag))
    if etiquetas:
        query_parts.append(f"Etiquetas: {etiquetas}")

    return " | ".join(part for part in query_parts if has_text(part))


def _build_runtime_notice(message: str, error: Exception, debug: bool) -> str:
    """Build a safe runtime notice for degraded flows."""
    if debug:
        return f"{message} Diagnostico seguro: {get_safe_error_chain(error)}"
    return message


def _join_notices(notices: list[str]) -> str | None:
    """Join safe notices into a single field when present."""
    cleaned_notices = [notice for notice in notices if notice]
    if not cleaned_notices:
        return None
    return " ".join(cleaned_notices)


def _build_result(
    *,
    mode: str,
    provider_name: str,
    output_type: str,
    prompt: str,
    base_context: str,
    retrieved_context: str,
    retrieved_chunks: list[dict[str, Any]],
    generated_text: str,
    use_rag: bool,
    embedding_provider: str,
    error: str | None,
) -> dict[str, Any]:
    """Return a consistent output payload for UI and tests."""
    return {
        "mode": mode,
        "provider": provider_name,
        "output_type": output_type,
        "use_rag": use_rag,
        "embedding_provider": embedding_provider,
        "prompt": prompt,
        "base_context": base_context,
        "context": base_context,
        "retrieved_context": retrieved_context,
        "retrieved_chunks": retrieved_chunks,
        "generated_text": generated_text,
        "error": error,
    }
