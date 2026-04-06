"""Streamlit UI for the historical Ecuador generator MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.formatter import format_metadata, format_related_list
from src.generator import SUPPORTED_OUTPUTS, generate_content
from src.loader import get_entity_by_name, get_entity_names, load_historical_entities, load_prompt_templates


st.set_page_config(
    page_title="Historical Ecuador Generator",
    page_icon="📚",
    layout="wide",
)


def load_app_data() -> tuple[list[dict], dict[str, str]]:
    """Load the application datasets with Streamlit caching."""
    entities = load_historical_entities()
    templates = load_prompt_templates()
    return entities, templates


@st.cache_data(show_spinner=False)
def get_cached_data() -> tuple[list[dict], dict[str, str]]:
    """Cache the project data to keep the interface responsive."""
    return load_app_data()


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


def main() -> None:
    """Render the Streamlit application."""
    st.title("Historical Ecuador Generator")
    st.write(
        "Genera contenido historico del Ecuador a partir de una base estructurada en JSON, "
        "sin depender todavia de APIs externas."
    )

    try:
        entities, templates = get_cached_data()
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

    entity = get_entity_by_name(entities, selected_name)
    if entity is None:
        st.error("No se pudo encontrar la entidad seleccionada.")
        st.stop()

    render_entity_overview(entity)

    show_metadata = st.checkbox("Mostrar metadatos de la entidad", value=False)
    if show_metadata:
        with st.expander("Metadatos", expanded=True):
            st.code(format_metadata(entity), language="text")

    if st.button("Generar contenido", type="primary"):
        try:
            generated_text = generate_content(entity, output_type, templates)
        except ValueError as error:
            st.error(f"No se pudo generar el contenido: {error}")
            st.stop()

        st.subheader("Resultado generado")
        st.text_area(
            "Texto listo para copiar",
            value=generated_text,
            height=320,
            help="Selecciona el contenido y copialo facilmente desde este bloque.",
        )
        st.download_button(
            label="Descargar resultado .txt",
            data=generated_text,
            file_name=f"{entity['id']}_{output_type}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
