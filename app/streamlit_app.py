"""Streamlit UI for the historical Ecuador generator with personalization."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.audience_profiles import (
    DEFAULT_AUDIENCE_ID,
    build_safe_audience_profile,
    load_audience_profiles,
)
from src.embeddings_client import get_available_embedding_providers
from src.formatter import format_metadata, format_related_list
from src.generator import SUPPORTED_OUTPUTS, generate_multimodal_content
from src.image_client import SUPPORTED_IMAGE_SIZES, get_available_image_providers
from src.image_prompt_builder import SUPPORTED_IMAGE_MODES, SUPPORTED_VISUAL_STYLES
from src.llm_client import get_available_providers
from src.loader import get_entity_by_name, get_entity_names, load_historical_entities
from src.personalization import (
    SUPPORTED_DEPTHS,
    SUPPORTED_LENGTHS,
    SUPPORTED_PURPOSES,
    SUPPORTED_TONES,
)
from src.rag_retriever import RAGRetrieverError, load_index


st.set_page_config(
    page_title="Historical Ecuador Generator",
    page_icon="📚",
    layout="wide",
)

DEFAULT_OVERRIDE_OPTION = "__usar_perfil__"


@st.cache_data(show_spinner=False)
def get_cached_entities() -> list[dict]:
    """Cache the historical entities to keep the interface responsive."""
    return load_historical_entities()


@st.cache_data(show_spinner=False)
def get_cached_audience_profiles() -> list[dict]:
    """Cache audience profiles, falling back to a safe local default."""
    try:
        return load_audience_profiles()
    except (FileNotFoundError, ValueError):
        return [build_safe_audience_profile(DEFAULT_AUDIENCE_ID)]


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


def render_personalization_config(
    personalization: dict,
    title: str = "Configuracion final de personalizacion",
) -> None:
    """Render the personalization config used for generation."""
    with st.expander(title, expanded=False):
        st.json(personalization)


def render_text_result(result: dict) -> None:
    """Render the textual generation result."""
    st.subheader("Resultado textual")
    st.caption(
        f"Modo usado: {result['mode']} | "
        f"LLM provider: {result['provider']} | "
        f"RAG usado: {result['use_rag']} | "
        f"Embedding provider: {result['embedding_provider']}"
    )
    if result["error"]:
        st.warning(result["error"])

    with st.expander("Contexto base textual", expanded=False):
        st.code(result["base_context"], language="text")

    with st.expander("Contexto recuperado textual", expanded=False):
        if result["retrieved_context"]:
            st.code(result["retrieved_context"], language="text")
        else:
            st.info("Esta generacion textual no utilizo contexto recuperado adicional.")

    with st.expander("Chunks recuperados para texto", expanded=False):
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
        file_name=f"{result.get('provider', 'texto')}_{result['output_type']}.txt",
        mime="text/plain",
    )

    with st.expander("Prompt textual final", expanded=False):
        st.code(result["prompt"], language="text")


def render_image_result(result: dict, entity: dict) -> None:
    """Render the visual generation result."""
    st.subheader("Resultado visual")
    st.caption(
        f"Provider visual: {result['provider']} | "
        f"Estado: {result['status']} | "
        f"Modo visual: {result['image_mode']} | "
        f"Estilo: {result['visual_style']} | "
        f"RAG usado: {result['use_rag']}"
    )
    if result["error"]:
        st.warning(result["error"])

    with st.expander("Prompt visual final", expanded=False):
        st.code(result["prompt"], language="text")

    with st.expander("Contexto base visual", expanded=False):
        st.code(result["base_context"], language="text")

    with st.expander("Contexto recuperado visual", expanded=False):
        if result["retrieved_context"]:
            st.code(result["retrieved_context"], language="text")
        else:
            st.info("Esta generacion visual no utilizo contexto recuperado adicional.")

    with st.expander("Chunks recuperados para imagen", expanded=False):
        render_retrieved_chunks(result["retrieved_chunks"])

    image_reference = result.get("image_path") or result.get("image_url")
    if image_reference:
        st.image(
            image_reference,
            caption=f"Imagen generada para {entity['nombre']}",
            use_container_width=True,
        )
    else:
        st.info("No se genero una imagen final. El prompt visual quedo listo para copiar o reutilizar.")


def render_audience_comparison(
    primary_result: dict,
    secondary_result: dict,
    primary_label: str,
    secondary_label: str,
) -> None:
    """Render a simple side-by-side comparison between two audiences."""
    st.subheader("Comparacion de audiencias")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"**{primary_label}**")
        render_personalization_config(primary_result["personalization"], title="Configuracion primaria")
        if primary_result["text_result"]:
            st.text_area(
                "Texto primario",
                value=primary_result["text_result"]["generated_text"],
                height=260,
                key=f"comparison_primary_{primary_result['entity_id']}",
            )

    with right_col:
        st.markdown(f"**{secondary_label}**")
        render_personalization_config(secondary_result["personalization"], title="Configuracion secundaria")
        if secondary_result["text_result"]:
            st.text_area(
                "Texto secundario",
                value=secondary_result["text_result"]["generated_text"],
                height=260,
                key=f"comparison_secondary_{secondary_result['entity_id']}",
            )


def _format_override_option(value: str) -> str:
    """Format personalization override options for the UI."""
    if value == DEFAULT_OVERRIDE_OPTION:
        return "Usar perfil"
    return value.replace("_", " ").title()


def _normalize_override_value(value: str) -> str | None:
    """Convert the UI sentinel value into None."""
    if value == DEFAULT_OVERRIDE_OPTION:
        return None
    return value


def execute_generation_request(
    *,
    entity: dict,
    output_type: str,
    llm_provider: str,
    image_provider: str,
    embedding_provider: str,
    use_llm: bool,
    use_rag: bool,
    top_k: int,
    generate_text: bool,
    generate_image: bool,
    image_mode: str,
    visual_style: str,
    image_size: str,
    audience_id: str = DEFAULT_AUDIENCE_ID,
    tone: str | None = None,
    depth: str | None = None,
    length: str | None = None,
    purpose: str | None = None,
    debug_mode: bool = False,
) -> dict:
    """Execute the multimodal generation flow with personalization."""
    if not generate_text and not generate_image:
        raise ValueError("Debes activar al menos texto o imagen.")

    return generate_multimodal_content(
        entity=entity,
        output_type=output_type,
        llm_provider=llm_provider,
        image_provider=image_provider,
        use_llm=use_llm,
        use_rag=use_rag,
        top_k=top_k,
        embedding_provider=embedding_provider,
        generate_image=generate_image,
        image_mode=image_mode,
        visual_style=visual_style,
        image_size=image_size,
        audience_id=audience_id,
        tone=tone,
        depth=depth,
        length=length,
        purpose=purpose,
        generate_text=generate_text,
        debug=debug_mode,
    )


def main() -> None:
    """Render the Streamlit application."""
    st.title("Historical Ecuador Generator")
    st.write(
        "Genera contenido historico del Ecuador desde una base local estructurada, "
        "con grounding RAG, generacion visual y personalizacion por audiencia."
    )

    try:
        entities = get_cached_entities()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"No fue posible cargar los datos del proyecto: {error}")
        st.stop()

    audience_profiles = get_cached_audience_profiles()
    audience_ids = [profile["audience_id"] for profile in audience_profiles]
    audience_map = {profile["audience_id"]: profile for profile in audience_profiles}

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

    audience_id = st.selectbox(
        "Selecciona la audiencia objetivo",
        options=audience_ids,
        index=audience_ids.index(DEFAULT_AUDIENCE_ID) if DEFAULT_AUDIENCE_ID in audience_ids else 0,
        format_func=lambda value: audience_map.get(value, {}).get("nombre_visible", value),
    )
    st.caption(audience_map.get(audience_id, {}).get("descripcion", ""))

    tone_override = st.selectbox(
        "Sobrescribir tono",
        options=[DEFAULT_OVERRIDE_OPTION, *SUPPORTED_TONES],
        format_func=_format_override_option,
    )
    depth_override = st.selectbox(
        "Sobrescribir profundidad",
        options=[DEFAULT_OVERRIDE_OPTION, *SUPPORTED_DEPTHS],
        format_func=_format_override_option,
    )
    length_override = st.selectbox(
        "Sobrescribir longitud",
        options=[DEFAULT_OVERRIDE_OPTION, *SUPPORTED_LENGTHS],
        format_func=_format_override_option,
    )
    purpose_override = st.selectbox(
        "Sobrescribir proposito",
        options=[DEFAULT_OVERRIDE_OPTION, *SUPPORTED_PURPOSES],
        format_func=_format_override_option,
    )

    generate_text = st.checkbox("Generar texto", value=True)
    generate_image = st.checkbox("Generar imagen", value=False)
    compare_audiences = st.checkbox(
        "Comparar dos audiencias",
        value=False,
        disabled=not generate_text,
    )
    secondary_audience_options = [value for value in audience_ids if value != audience_id] or audience_ids
    secondary_audience_id = st.selectbox(
        "Segunda audiencia para comparar",
        options=secondary_audience_options,
        format_func=lambda value: audience_map.get(value, {}).get("nombre_visible", value),
        disabled=not compare_audiences,
    )

    provider = st.selectbox(
        "Selecciona el proveedor LLM",
        options=["openai", "gemini", "xai"],
        format_func=lambda value: value.upper(),
        disabled=not generate_text,
    )
    image_provider = st.selectbox(
        "Selecciona el proveedor de imagen",
        options=["openai", "fallback"],
        format_func=lambda value: value.upper(),
        disabled=not generate_image,
    )
    embedding_provider = st.selectbox(
        "Selecciona el proveedor de embeddings",
        options=["openai", "gemini"],
        format_func=lambda value: value.upper(),
        disabled=not (generate_text or generate_image),
    )
    use_llm = st.checkbox("Usar LLM", value=True, disabled=not generate_text)
    use_rag = st.checkbox("Usar RAG", value=True, disabled=not (generate_text or generate_image))
    top_k = st.slider(
        "Cantidad de chunks recuperados (top_k)",
        min_value=1,
        max_value=8,
        value=5,
        disabled=not (generate_text or generate_image),
    )
    image_mode = st.selectbox(
        "Modo visual",
        options=list(SUPPORTED_IMAGE_MODES),
        format_func=lambda value: value.replace("_", " ").title(),
        disabled=not generate_image,
    )
    visual_style = st.selectbox(
        "Estilo visual",
        options=list(SUPPORTED_VISUAL_STYLES),
        format_func=lambda value: value.replace("_", " ").title(),
        disabled=not generate_image,
    )
    image_size = st.selectbox(
        "Tamano de imagen",
        options=list(SUPPORTED_IMAGE_SIZES),
        format_func=lambda value: value,
        disabled=not generate_image,
    )
    debug_mode = st.checkbox("Mostrar diagnostico seguro", value=False)

    entity = get_entity_by_name(entities, selected_name)
    if entity is None:
        st.error("No se pudo encontrar la entidad seleccionada.")
        st.stop()

    available_providers = get_available_providers()
    available_embedding_providers = get_available_embedding_providers()
    available_image_providers = get_available_image_providers()
    rag_status = get_cached_rag_status()

    with st.expander("Estado de providers e indice RAG", expanded=False):
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**LLM providers**")
            st.json(available_providers)
            st.markdown("**Embedding providers**")
            st.json(available_embedding_providers)
            st.markdown("**Image providers**")
            st.json(available_image_providers)
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

    with st.expander("Perfil de audiencia seleccionado", expanded=False):
        st.json(audience_map.get(audience_id, build_safe_audience_profile(audience_id)))

    if st.button("Generar contenido", type="primary"):
        try:
            result = execute_generation_request(
                entity=entity,
                output_type=output_type,
                llm_provider=provider,
                image_provider=image_provider,
                embedding_provider=embedding_provider,
                use_llm=use_llm,
                use_rag=use_rag,
                top_k=top_k,
                generate_text=generate_text,
                generate_image=generate_image,
                image_mode=image_mode,
                visual_style=visual_style,
                image_size=image_size,
                audience_id=audience_id,
                tone=_normalize_override_value(tone_override),
                depth=_normalize_override_value(depth_override),
                length=_normalize_override_value(length_override),
                purpose=_normalize_override_value(purpose_override),
                debug_mode=debug_mode,
            )
        except ValueError as error:
            st.error(f"No se pudo generar el contenido: {error}")
            st.stop()

        st.subheader("Resultado generado")
        render_personalization_config(result["personalization"])

        if result["text_result"]:
            render_text_result(result["text_result"])

        if result["image_result"]:
            render_image_result(result["image_result"], entity)

        if compare_audiences and generate_text:
            comparison_result = execute_generation_request(
                entity=entity,
                output_type=output_type,
                llm_provider=provider,
                image_provider=image_provider,
                embedding_provider=embedding_provider,
                use_llm=use_llm,
                use_rag=use_rag,
                top_k=top_k,
                generate_text=True,
                generate_image=False,
                image_mode=image_mode,
                visual_style=visual_style,
                image_size=image_size,
                audience_id=secondary_audience_id,
                tone=_normalize_override_value(tone_override),
                depth=_normalize_override_value(depth_override),
                length=_normalize_override_value(length_override),
                purpose=_normalize_override_value(purpose_override),
                debug_mode=debug_mode,
            )
            render_audience_comparison(
                primary_result=result,
                secondary_result=comparison_result,
                primary_label=audience_map.get(audience_id, {}).get("nombre_visible", audience_id),
                secondary_label=audience_map.get(secondary_audience_id, {}).get("nombre_visible", secondary_audience_id),
            )


if __name__ == "__main__":
    main()
