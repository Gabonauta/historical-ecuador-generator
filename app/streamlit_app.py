"""Streamlit UI for the historical Ecuador generator with multimodal support."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.embeddings_client import get_available_embedding_providers
from src.formatter import format_metadata, format_related_list
from src.generator import SUPPORTED_OUTPUTS, generate_multimodal_content
from src.image_client import SUPPORTED_IMAGE_SIZES, get_available_image_providers
from src.image_prompt_builder import SUPPORTED_IMAGE_MODES, SUPPORTED_VISUAL_STYLES
from src.llm_client import get_available_providers
from src.loader import get_entity_by_name, get_entity_names, load_historical_entities
from src.rag_retriever import RAGRetrieverError, load_index
import src.history_store as history_store


st.set_page_config(
    page_title="Historical Ecuador Generator",
    page_icon="📚",
    layout="wide",
)


get_generation_run = history_store.get_generation_run
init_store = history_store.init_store
list_generation_runs = history_store.list_generation_runs
save_generation_run = history_store.save_generation_run


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


def build_generation_request_payload(
    *,
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
    debug_mode: bool,
) -> dict:
    """Build a serializable snapshot of the generation request options."""
    return {
        "output_type": output_type,
        "llm_provider": llm_provider,
        "image_provider": image_provider,
        "embedding_provider": embedding_provider,
        "use_llm": use_llm,
        "use_rag": use_rag,
        "top_k": top_k,
        "generate_text": generate_text,
        "generate_image": generate_image,
        "image_mode": image_mode,
        "visual_style": visual_style,
        "image_size": image_size,
        "debug_mode": debug_mode,
    }


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
    api_key_overrides: dict[str, str] | None,
    debug_mode: bool,
) -> dict:
    """Execute the multimodal generation flow with a light validation layer."""
    if not generate_text and not generate_image:
        raise ValueError("Debes activar al menos texto o imagen.")

    generation_kwargs = {
        "entity": entity,
        "output_type": output_type,
        "llm_provider": llm_provider,
        "image_provider": image_provider,
        "use_llm": use_llm,
        "use_rag": use_rag,
        "top_k": top_k,
        "embedding_provider": embedding_provider,
        "generate_image": generate_image,
        "image_mode": image_mode,
        "visual_style": visual_style,
        "image_size": image_size,
        "generate_text": generate_text,
        "debug": debug_mode,
    }
    if api_key_overrides:
        generation_kwargs["api_keys"] = api_key_overrides

    try:
        return generate_multimodal_content(**generation_kwargs)
    except TypeError:
        generation_kwargs.pop("api_keys", None)
        return generate_multimodal_content(**generation_kwargs)


def get_provider_availability_snapshot(
    api_key_overrides: dict[str, str] | None,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool]]:
    """Load provider availability while remaining compatible with older signatures."""
    available_providers = _call_provider_availability(
        get_available_providers,
        api_key_overrides=api_key_overrides,
    )
    available_embedding_providers = _call_provider_availability(
        get_available_embedding_providers,
        api_key_overrides=api_key_overrides,
    )
    available_image_providers = _call_provider_availability(
        get_available_image_providers,
        api_key_overrides=api_key_overrides,
    )
    return (
        available_providers,
        available_embedding_providers,
        available_image_providers,
    )


def _call_provider_availability(
    loader: object,
    *,
    api_key_overrides: dict[str, str] | None,
) -> dict[str, bool]:
    """Call provider availability helpers with graceful fallback for older signatures."""
    if not api_key_overrides:
        return loader()

    try:
        return loader(api_key_overrides=api_key_overrides)
    except TypeError:
        return loader()


def build_api_key_overrides(
    openai_api_key: str,
    gemini_api_key: str,
    xai_api_key: str,
) -> dict[str, str]:
    """Build runtime-only API key overrides from UI inputs."""
    overrides = {
        "openai": openai_api_key.strip(),
        "gemini": gemini_api_key.strip(),
        "xai": xai_api_key.strip(),
    }
    return {provider: api_key for provider, api_key in overrides.items() if api_key}


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
        st.image(image_reference, caption=f"Imagen generada para {entity['nombre']}", use_container_width=True)
    else:
        st.info("No se genero una imagen final. El prompt visual quedo listo para copiar o reutilizar.")


def get_generation_run_count() -> int:
    """Return the total number of persisted runs with backward compatibility."""
    counter = getattr(history_store, "count_generation_runs", None)
    if callable(counter):
        return int(counter())

    return len(list_generation_runs(limit=None))


def render_generation_history(limit: int = 10) -> None:
    """Render a recent history list from the local SQLite store."""
    with st.expander("Historial reciente", expanded=False):
        total_runs = get_generation_run_count()
        if total_runs == 0:
            st.info("Todavia no hay corridas persistidas.")
            return

        show_all_history = st.checkbox(
            "Mostrar todo el historial",
            value=False,
            key="history_show_all",
            help="Activalo para listar todas las corridas guardadas en la base local.",
        )
        selected_limit = int(
            st.number_input(
                "Cantidad de corridas a mostrar",
                min_value=1,
                max_value=max(1, total_runs),
                value=min(limit, total_runs),
                step=1,
                disabled=show_all_history,
                key="history_limit",
            )
        )
        recent_runs = list_generation_runs(limit=None if show_all_history else selected_limit)
        st.caption(f"Mostrando {len(recent_runs)} de {total_runs} corridas persistidas.")

        for run in recent_runs:
            history_label = (
                f"#{run['id']} | {run['created_at']} | {run.get('entity_name') or 'Sin entidad'} | "
                f"{run.get('output_type') or 'sin_salida'} | "
                f"texto {run.get('text_mode') or 'n/a'}:{run.get('effective_text_provider') or 'n/a'} | "
                f"RAG {run.get('use_rag')} | imagen {run.get('generate_image')}"
            )

            with st.expander(history_label, expanded=False):
                details = get_generation_run(run["id"])
                if details is None:
                    st.warning("No se pudo cargar el detalle de esta corrida.")
                    continue

                st.markdown("**Resumen**")
                st.json(
                    {
                        "entity_id": details.get("entity_id"),
                        "entity_name": details.get("entity_name"),
                        "entity_type": details.get("entity_type"),
                        "output_type": details.get("output_type"),
                        "requested_llm_provider": details.get("requested_llm_provider"),
                        "effective_text_provider": details.get("effective_text_provider"),
                        "text_mode": details.get("text_mode"),
                        "requested_image_provider": details.get("requested_image_provider"),
                        "effective_image_provider": details.get("effective_image_provider"),
                        "generate_text": details.get("generate_text"),
                        "generate_image": details.get("generate_image"),
                        "use_llm": details.get("use_llm"),
                        "use_rag": details.get("use_rag"),
                        "embedding_provider": details.get("embedding_provider"),
                        "image_mode": details.get("image_mode"),
                        "visual_style": details.get("visual_style"),
                        "image_size": details.get("image_size"),
                    }
                )

                if details.get("text_error") or details.get("image_error"):
                    st.markdown("**Errores seguros**")
                    if details.get("text_error"):
                        st.warning(f"Texto: {details['text_error']}")
                    if details.get("image_error"):
                        st.warning(f"Imagen: {details['image_error']}")

                if details.get("generated_text"):
                    st.markdown("**Texto generado**")
                    st.text_area(
                        f"Texto corrida #{details['id']}",
                        value=details["generated_text"],
                        height=220,
                    )

                if details.get("image_path") or details.get("image_url"):
                    st.markdown("**Referencia visual**")
                    st.write(details.get("image_path") or details.get("image_url"))

                with st.expander("Prompt textual", expanded=False):
                    st.code(details.get("prompt_text") or "", language="text")

                with st.expander("Prompt visual", expanded=False):
                    st.code(details.get("prompt_image") or "", language="text")

                with st.expander("Contexto base persistido", expanded=False):
                    st.code(details.get("base_context_text") or "", language="text")

                with st.expander("Contexto recuperado persistido", expanded=False):
                    if details.get("retrieved_context_text"):
                        st.code(details["retrieved_context_text"], language="text")
                    else:
                        st.info("Esta corrida no persistio contexto recuperado adicional.")

                with st.expander("Opciones de solicitud persistidas", expanded=False):
                    st.json(details.get("request_options_json") or {})

                with st.expander("Snapshot de entidad persistido", expanded=False):
                    st.json(details.get("entity_snapshot_json") or {})

                with st.expander("Chunks recuperados persistidos", expanded=False):
                    retrieved_chunks = details.get("retrieved_chunks_json") or []
                    if retrieved_chunks:
                        st.json(retrieved_chunks)
                    else:
                        st.info("Esta corrida no guardo chunks recuperados adicionales.")

                with st.expander("Resultado textual completo persistido", expanded=False):
                    text_result_payload = details.get("text_result_json")
                    if text_result_payload:
                        st.json(text_result_payload)
                    else:
                        st.info("Esta corrida no persistio un resultado textual completo.")

                with st.expander("Resultado visual completo persistido", expanded=False):
                    image_result_payload = details.get("image_result_json")
                    if image_result_payload:
                        st.json(image_result_payload)
                    else:
                        st.info("Esta corrida no persistio un resultado visual completo.")


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

    store_ready = True
    try:
        init_store()
    except Exception as error:
        store_ready = False
        st.warning(
            "No fue posible inicializar la persistencia local del historial. "
            f"La generacion seguira funcionando. Detalle seguro: {error}"
        )

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
    generate_text = st.checkbox("Generar texto", value=True)
    generate_image = st.checkbox("Generar imagen", value=False)
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

    with st.expander("API keys temporales del usuario", expanded=False):
        st.caption(
            "Puedes introducir tus propias API keys para OpenAI, Gemini y xAI. "
            "Estas claves se usan solo durante esta sesion de la app y no se guardan "
            "en el historial local, archivos del proyecto ni mensajes de error."
        )
        openai_api_key = st.text_input(
            "OPENAI_API_KEY temporal",
            type="password",
            placeholder="sk-...",
            help="Tambien se reutiliza para embeddings OpenAI e imagen OpenAI cuando aplique.",
        )
        gemini_api_key = st.text_input(
            "GEMINI_API_KEY temporal",
            type="password",
            placeholder="AIza...",
            help="Tambien se reutiliza para embeddings Gemini cuando aplique.",
        )
        xai_api_key = st.text_input(
            "XAI_API_KEY temporal",
            type="password",
            placeholder="xai-...",
        )

    api_key_overrides = build_api_key_overrides(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        xai_api_key=xai_api_key,
    )

    entity = get_entity_by_name(entities, selected_name)
    if entity is None:
        st.error("No se pudo encontrar la entidad seleccionada.")
        st.stop()

    (
        available_providers,
        available_embedding_providers,
        available_image_providers,
    ) = get_provider_availability_snapshot(api_key_overrides)
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

    if st.button("Generar contenido", type="primary"):
        request_payload = build_generation_request_payload(
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
            debug_mode=debug_mode,
        )

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
                api_key_overrides=api_key_overrides,
                debug_mode=debug_mode,
            )
        except ValueError as error:
            st.error(f"No se pudo generar el contenido: {error}")
            st.stop()

        if store_ready:
            try:
                saved_run_id = save_generation_run(entity, request_payload, result)
                st.caption(f"Corrida guardada en historial local como #{saved_run_id}.")
            except Exception as error:
                st.warning(
                    "No fue posible guardar esta corrida en el historial local. "
                    f"La generacion si se completo. Detalle seguro: {error}"
                )

        st.subheader("Resultado generado")

        if result["text_result"]:
            render_text_result(result["text_result"])

        if result["image_result"]:
            render_image_result(result["image_result"], entity)

    if store_ready:
        render_generation_history(limit=10)


if __name__ == "__main__":
    main()
