# historical-ecuador-generator

Proyecto en Python + Streamlit para generar contenido historico del Ecuador a partir de una base local en JSON. La Fase 3 agrego generacion hibrida con LLM multi-provider, la Fase 4 incorporo RAG local y la Fase 5 suma generacion de imagenes historicas con grounding visual razonable.

## Estructura

```text
historical-ecuador-generator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── historical_entities.json
│   ├── prompt_templates.json
│   └── rag/
│       ├── chunks.json
│       ├── embeddings.npy
│       └── metadata.json
├── outputs/
│   ├── generated_images/
│   └── sample_outputs/
├── scripts/
│   └── build_rag_index.py
├── src/
│   ├── context_builder.py
│   ├── embeddings_client.py
│   ├── fallback_generator.py
│   ├── formatter.py
│   ├── generator.py
│   ├── image_client.py
│   ├── image_generator.py
│   ├── image_prompt_builder.py
│   ├── llm_client.py
│   ├── loader.py
│   ├── prompt_builder.py
│   ├── rag_chunker.py
│   ├── rag_indexer.py
│   ├── rag_retriever.py
│   └── utils.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalacion

1. Crear y activar un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno segun los providers que vayas a usar:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

## Ejecucion de la app

Desde la carpeta del proyecto:

```bash
streamlit run app/streamlit_app.py
```

La interfaz permite:

- seleccionar una entidad historica
- elegir el tipo de salida textual
- elegir provider LLM, provider de imagen y provider de embeddings
- activar o desactivar texto, imagen, LLM y RAG
- elegir `image_mode`, `visual_style` y tamano de imagen
- revisar contexto base, contexto recuperado y chunks relevantes

## Fase 5: generacion de imagenes

La Fase 5 agrega una capacidad visual separada de la generacion textual. El objetivo es producir prompts visuales historicos bien controlados y, cuando haya provider disponible, generar imagenes apoyadas en el mismo grounding del sistema textual.

### Arquitectura visual

- `src/image_prompt_builder.py`: construye prompts visuales en espanol a partir de entidad, contexto base, contexto recuperado, modo visual y estilo.
- `src/image_client.py`: encapsula la generacion de imagenes, empezando con `openai` y un fallback seguro.
- `src/image_generator.py`: coordina grounding visual, RAG opcional, prompt visual y llamada al provider de imagen.
- `src/generator.py`: ahora expone `generate_multimodal_content()` para texto, imagen o modo combinado.
- `app/streamlit_app.py`: permite flujo textual, visual o ambos sin romper la experiencia existente.

### Texto, imagen y modo combinado

- Texto:
  usa `generate_content()` y conserva la logica de Fase 4.

- Imagen:
  usa `generate_visual_content()` y puede apoyarse en RAG para enriquecer el prompt visual.

- Texto + imagen:
  usa `generate_multimodal_content()` para coordinar ambos resultados sin acoplar las capas.

### Modos visuales

`image_mode` soporta:

- `retrato_historico`
- `escena_historica`
- `postal_turistica`
- `ilustracion_educativa`

Cada modo cambia la composicion sugerida para el provider visual.

### Estilos visuales

`visual_style` soporta:

- `realista`
- `pintura_oleo`
- `grabado_antiguo`
- `ilustracion_editorial`

Esto permite orientar el lenguaje visual sin asumir hiperrealismo por defecto.

### Grounding visual

El prompt visual usa:

1. la entidad seleccionada
2. el contexto base estructurado
3. el contexto recuperado por RAG cuando esta disponible

Las instrucciones visuales refuerzan:

- usar solo contexto disponible
- no inventar atributos historicos no sustentados
- ser prudente cuando faltan detalles visuales
- evitar anacronismos, logos modernos y texto incrustado

### Provider visual y fallback

Providers iniciales:

- `openai`
- `fallback`

Comportamiento:

- si `openai` esta disponible y la llamada funciona, se devuelve imagen local o referencia de imagen
- si falta la API key o falla la solicitud, el sistema devuelve un fallback seguro con el prompt visual listo para copiar
- la app no se rompe cuando falla la imagen

### Guardado de imagenes

Cuando el provider devuelve imagen utilizable, el sistema intenta guardarla en:

- `outputs/generated_images/`

Si no puede guardar localmente, conserva la referencia devuelta por el provider cuando exista.

## Fase 4: RAG

La Fase 4 agrega Retrieval-Augmented Generation sobre la base historica local sin introducir una vector database externa.

### Arquitectura RAG

- `src/rag_chunker.py`: transforma cada entidad del JSON en chunks recuperables por categoria.
- `src/embeddings_client.py`: abstrae la generacion de embeddings para `openai` y `gemini`.
- `src/rag_indexer.py`: genera embeddings, normaliza vectores y guarda el indice local en `data/rag/`.
- `src/rag_retriever.py`: carga el indice, embebe la consulta y recupera los chunks mas relevantes por similitud coseno.
- `src/context_builder.py`: construye contexto base estructurado y contexto recuperado.
- `src/prompt_builder.py`: combina contexto base y contexto recuperado en un prompt controlado en espanol.
- `src/generator.py`: coordina RAG, LLM, imagen y fallback.

### Construir el indice RAG

Comando base:

```bash
python scripts/build_rag_index.py
```

Tambien puedes elegir el provider de embeddings:

```bash
python scripts/build_rag_index.py --embedding-provider gemini
```

## Fase 3: Generacion hibrida multi-provider

La Fase 3 sigue vigente y se mantiene compatible:

- `llm_client.py` encapsula `openai`, `gemini` y `xai`
- `fallback_generator.py` mantiene salidas sin LLM
- `generator.py` decide entre modo `llm` y modo `fallback`

Si el provider LLM falla, el sistema no rompe la app y devuelve una salida local segura.

## Tipos de salida textual

- `ficha_historica`
- `resumen_corto`
- `texto_turistico`
- `post_redes`

## Tests

Ejecutar toda la suite:

```bash
pytest
```

La cobertura de Fase 5 incluye:

- construccion de prompts visuales
- cliente de imagen sin llamadas reales
- flujo visual con y sin RAG
- flujo ligero de la app para texto e imagen

## Evolucion futura

La base de Fase 5 queda preparada para:

- sumar nuevos providers visuales
- extenderse a edicion de imagenes
- compartir grounding entre texto e imagen con mas eficiencia
- avanzar hacia una fase multimodal mas completa sin mezclar capas de responsabilidad
