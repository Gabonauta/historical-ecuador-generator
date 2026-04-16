"""Streamlit UI for the historical Ecuador generator with optional RAG."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.embeddings_client import get_available_embedding_providers
from src.formatter import format_metadata, format_related_list
from src.generator import SUPPORTED_OUTPUTS, generate_content
from src.llm_client import get_available_providers
from src.loader import get_entity_by_name, get_entity_names, load_historical_entities
from src.rag_retriever import RAGRetrieverError, load_index


st.set_page_config(
    page_title="Historical Ecuador Generator",
    page_icon="📚",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_cached_entities() -> list[dict]:
    """Cache the historical entities to keep the interface responsive."""
    return load_historical_entities()


@st.cache_data(show_spinner=False)
def get_cached_rag_status() -> dict:
    """Cache lightweight RAG index status for the interface."""
    try:
        index_data = load_index()
        return {
            "available": True,
            "metadata": index_data.get("metadata", {}),
            "error": None,
        }
    except (FileNotFoundError, RAGRetrieverError, ValueError) as error:
        return {
            "available": False,
            "metadata": {},
            "error": str(error),
        }


def render_entity_overview(entity: dict) -> None:
    """Render the main details of the selected entity."""
    st.subheader(entity["nombre"])
    st.caption(f"{entity['tipo'].title()} | {entity['epoca']} | {entity['ubicacion']}")
    st.write(entity["resumen"])

    with st.container(border=True):
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**Etiquetas**")
            tags = entity.get("etiquetas", [])
            if tags:
                st.markdown(" ".join(f"`{tag}`" for tag in tags))
            else:
                st.write("Sin etiquetas disponibles.")

        with right_col:
            st.markdown("**Relaciones**")
            st.write(f"Lugares: {format_related_list(entity.get('lugares_relacionados'))}")
            st.write(f"Personajes: {format_related_list(entity.get('personajes_relacionados'))}")
            st.write(f"Eventos: {format_related_list(entity.get('eventos_relacionados'))}")


def render_retrieved_chunks(chunks: list[dict]) -> None:
    """Render retrieved chunks and their scores."""
    if not chunks:
        st.info("No se recuperaron fragmentos adicionales para esta ejecucion.")
        return

    for chunk in chunks:
        score = chunk.get("score", 0.0)
        with st.container(border=True):
            st.markdown(
                f"**{chunk.get('nombre', 'Sin nombre')}** | "
                f"`{chunk.get('tipo', 'sin_tipo')}` | "
                f"`{chunk.get('categoria_chunk', 'sin_categoria')}` | "
                f"score `{float(score):.4f}`"
            )
            st.write(chunk.get("texto", ""))


def main() -> None:
    """Render the Streamlit application."""
    st.title("Historical Ecuador Generator")
    st.write(
        "Genera contenido historico del Ecuador desde una base local estructurada y, "
        "cuando esta disponible, con grounding RAG sobre un indice semantico local."
    )

    try:
        entities = get_cached_entities()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"No fue posible cargar los datos del proyecto: {error}")
        st.stop()

    entity_names = get_entity_names(entities)
    if not entity_names:
        st.warning("No hay entidades historicas disponibles para mostrar.")
        st.stop()

    selected_name = st.selectbox("Selecciona una entidad historica", entity_names)
    output_type = st.selectbox(
        "Selecciona el tipo de salida",
        options=list(SUPPORTED_OUTPUTS),
        format_func=lambda value: value.replace("_", " ").title(),
    )
    provider = st.selectbox(
        "Selecciona el proveedor LLM",
        options=["openai", "gemini", "xai"],
        format_func=lambda value: value.upper(),
    )
    embedding_provider = st.selectbox(
        "Selecciona el proveedor de embeddings",
        options=["openai", "gemini"],
        format_func=lambda value: value.upper(),
    )
    use_llm = st.checkbox("Usar LLM", value=True)
    use_rag = st.checkbox("Usar RAG", value=True, disabled=not use_llm)
    top_k = st.slider("Cantidad de chunks recuperados (top_k)", min_value=1, max_value=8, value=5)
    debug_mode = st.checkbox("Mostrar diagnostico seguro", value=False)

    entity = get_entity_by_name(entities, selected_name)
    if entity is None:
        st.error("No se pudo encontrar la entidad seleccionada.")
        st.stop()

    available_providers = get_available_providers()
    available_embedding_providers = get_available_embedding_providers()
    rag_status = get_cached_rag_status()

    with st.expander("Estado de providers e indice RAG", expanded=False):
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**LLM providers**")
            st.json(available_providers)
            st.markdown("**Embedding providers**")
            st.json(available_embedding_providers)
        with right_col:
            st.markdown("**Indice RAG local**")
            if rag_status["available"]:
                st.json(rag_status["metadata"])
            else:
                st.info(
                    "El indice RAG todavia no esta listo o no se pudo cargar. "
                    "Construyelo con `python scripts/build_rag_index.py`."
                )
                if rag_status["error"]:
                    st.caption(rag_status["error"])

    render_entity_overview(entity)

    show_metadata = st.checkbox("Mostrar metadatos de la entidad", value=False)
    if show_metadata:
        with st.expander("Metadatos", expanded=True):
            st.code(format_metadata(entity), language="text")

    with st.expander("Contexto base estructurado", expanded=False):
        st.code(format_metadata(entity), language="text")
        st.json(entity)

    if st.button("Generar contenido", type="primary"):
        try:
            result = generate_content(
                entity=entity,
                output_type=output_type,
                provider=provider,
                use_llm=use_llm,
                use_rag=use_rag,
                top_k=top_k,
                embedding_provider=embedding_provider,
                debug=debug_mode,
            )
        except ValueError as error:
            st.error(f"No se pudo generar el contenido: {error}")
            st.stop()

        st.subheader("Resultado generado")
        st.caption(
            f"Modo usado: {result['mode']} | "
            f"LLM provider: {result['provider']} | "
            f"RAG usado: {result['use_rag']} | "
            f"Embedding provider: {result['embedding_provider']}"
        )
        if result["error"]:
            st.warning(result["error"])

        with st.expander("Contexto base usado", expanded=False):
            st.code(result["base_context"], language="text")

        with st.expander("Contexto recuperado", expanded=False):
            if result["retrieved_context"]:
                st.code(result["retrieved_context"], language="text")
            else:
                st.info("Esta generacion no utilizo contexto recuperado adicional.")

        with st.expander("Chunks recuperados", expanded=False):
            render_retrieved_chunks(result["retrieved_chunks"])

        st.text_area(
            "Texto generado",
            value=result["generated_text"],
            height=320,
            help="Resultado final generado por el modo LLM o por el fallback local.",
        )
        st.download_button(
            label="Descargar resultado .txt",
            data=result["generated_text"],
            file_name=f"{entity['id']}_{output_type}.txt",
            mime="text/plain",
        )

        with st.expander("Prompt final", expanded=False):
            st.code(result["prompt"], language="text")


if __name__ == "__main__":
    main()
