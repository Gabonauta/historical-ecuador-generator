# historical-ecuador-generator

Proyecto en Python + Streamlit para generar contenido historico del Ecuador a partir de una base local en JSON. La Fase 3 agrego generacion hibrida con LLM multi-provider, la Fase 4 incorporo RAG local, la Fase 5 sumo generacion de imagenes y la Fase 6 agrega personalizacion por audiencia.

## Estructura

```text
historical-ecuador-generator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── historical_entities.json
│   ├── prompt_templates.json
│   ├── audience_profiles.json
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
│   ├── audience_profiles.py
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
│   ├── personalization.py
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
- elegir una audiencia objetivo
- sobrescribir tono, profundidad, longitud y proposito
- elegir provider LLM, provider de imagen y provider de embeddings
- activar o desactivar texto, imagen, LLM y RAG
- comparar dos audiencias sobre la misma entidad

## Fase 6: personalizacion por audiencia

La Fase 6 agrega una capa de personalizacion controlada que adapta el contenido segun audiencia, tono, profundidad, longitud y proposito, sin relajar las restricciones de factualidad.

### Que es la personalizacion

La personalizacion no cambia los hechos ni agrega informacion nueva. Su funcion es ajustar:

- como se redacta el contenido
- cuanto detalle se ofrece
- que estructura o enfasis se prioriza
- como se formula el prompt visual en casos compatibles

Siempre se mantiene la regla central del proyecto:

- usar solo contexto base y contexto recuperado
- no inventar datos

### Perfiles de audiencia soportados

El archivo `data/audience_profiles.json` incluye perfiles iniciales como:

- `estudiante_secundaria`
- `estudiante_universitario`
- `turista_general`
- `docente`
- `divulgacion_redes`
- `investigador_inicial`

Cada perfil define:

- tono preferido
- profundidad preferida
- longitud preferida
- proposito preferido
- reglas de estilo
- patrones a evitar

### Parametros editables

La app permite usar el perfil tal cual o sobrescribir de forma puntual:

- `tone`: `formal`, `didactico`, `divulgativo`, `promocional`, `narrativo`
- `depth`: `baja`, `media`, `alta`
- `length`: `corta`, `media`, `larga`
- `purpose`: `educativo`, `turistico`, `academico`, `redes`, `divulgacion`

Los overrides del usuario se resuelven encima del perfil seleccionado y producen una configuracion final de personalizacion.

### Como se aplica al texto

La personalizacion textual afecta:

- el prompt textual
- el tono pedido al modelo
- la profundidad deseada
- la longitud orientativa
- el enfoque principal del contenido
- el fallback local cuando no se usa LLM o el provider falla

Esto se implementa en:

- `src/audience_profiles.py`
- `src/personalization.py`
- `src/prompt_builder.py`
- `src/fallback_generator.py`
- `src/generator.py`

### Como se aplica a la imagen

La personalizacion visual es ligera y controlada. No cambia los hechos visuales permitidos, pero si orienta el enfoque:

- `turista_general`: mas patrimonial y atractivo cultural
- `docente`: mas claro y explicativo
- `divulgacion_redes`: mas sintetico e impactante
- `investigador_inicial`: mas sobrio y documental

Esto se implementa en:

- `src/image_prompt_builder.py`
- `src/image_generator.py`

### Como comparar audiencias

La app incluye una opcion simple para comparar dos audiencias sobre la misma entidad.

Flujo:

1. eliges una audiencia principal
2. activas `Comparar dos audiencias`
3. eliges una segunda audiencia
4. generas el contenido

La app muestra ambas configuraciones y los textos lado a lado para una comparacion rapida.

## Fase 5: generacion de imagenes

La Fase 5 agrega una capacidad visual separada de la generacion textual.

### Arquitectura visual

- `src/image_prompt_builder.py`: construye prompts visuales en espanol.
- `src/image_client.py`: encapsula la generacion de imagenes con `openai` y fallback.
- `src/image_generator.py`: coordina grounding visual, RAG opcional y prompt visual.
- `src/generator.py`: expone `generate_multimodal_content()` para texto, imagen o ambos.

### Providers visuales

- `openai`
- `fallback`

Si el provider no esta disponible, el sistema devuelve un fallback seguro con el prompt visual listo para copiar.

## Fase 4: RAG

La Fase 4 agrega Retrieval-Augmented Generation sobre la base historica local sin introducir una vector database externa.

### Arquitectura RAG

- `src/rag_chunker.py`: transforma entidades del JSON en chunks recuperables.
- `src/embeddings_client.py`: abstrae embeddings para `openai` y `gemini`.
- `src/rag_indexer.py`: genera embeddings y guarda el indice local en `data/rag/`.
- `src/rag_retriever.py`: recupera los chunks mas relevantes por similitud coseno.

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

La Fase 3 se mantiene compatible:

- `llm_client.py` encapsula `openai`, `gemini` y `xai`
- `fallback_generator.py` mantiene salidas sin LLM
- `generator.py` coordina texto, imagen, RAG y fallback

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

La cobertura de Fase 6 incluye:

- carga y validacion de perfiles de audiencia
- resolucion de configuracion de personalizacion
- construccion de prompts personalizados
- flujo multimodal personalizado
- integracion ligera de Streamlit con audiencia y defaults seguros

## Evolucion futura

La base de Fase 6 queda preparada para:

- experimentos de evaluacion posterior por audiencia
- personalizacion mas fina de estilos visuales
- nuevas audiencias o perfiles sectoriales
- fases futuras mas multimodales sin mezclar responsabilidades
