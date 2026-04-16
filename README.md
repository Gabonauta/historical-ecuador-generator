# historical-ecuador-generator

Proyecto en Python + Streamlit para generar contenido historico del Ecuador a partir de una base local en JSON. La Fase 3 agrego generacion hibrida con LLM multi-provider y la Fase 4 incorpora una capa RAG local para recuperar contexto semantico antes de construir el prompt.

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
│   └── sample_outputs/
├── scripts/
│   └── build_rag_index.py
├── src/
│   ├── context_builder.py
│   ├── embeddings_client.py
│   ├── fallback_generator.py
│   ├── formatter.py
│   ├── generator.py
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
- elegir el tipo de salida
- elegir el provider LLM (`openai`, `gemini`, `xai`)
- elegir el provider de embeddings (`openai`, `gemini`)
- activar o desactivar `Usar LLM`
- activar o desactivar `Usar RAG`
- ajustar `top_k`
- inspeccionar contexto base, contexto recuperado y chunks recuperados

## Fase 4: RAG

La Fase 4 agrega Retrieval-Augmented Generation sobre la base historica local sin introducir una vector database externa.

### Arquitectura RAG

- `src/rag_chunker.py`: transforma cada entidad del JSON en chunks recuperables por categoria.
- `src/embeddings_client.py`: abstrae la generacion de embeddings para `openai` y `gemini`.
- `src/rag_indexer.py`: genera embeddings, normaliza vectores y guarda el indice local en `data/rag/`.
- `src/rag_retriever.py`: carga el indice, embebe la consulta y recupera los chunks mas relevantes por similitud coseno.
- `src/context_builder.py`: construye contexto base estructurado y contexto recuperado.
- `src/prompt_builder.py`: combina contexto base y contexto recuperado en un prompt controlado en espanol.
- `src/generator.py`: coordina RAG, LLM y fallback local.

### Flujo

1. Se carga `data/historical_entities.json`.
2. Cada entidad se divide en chunks como `resumen`, `descripcion`, `importancia` y `relaciones`.
3. Los chunks se convierten en embeddings con el provider seleccionado.
4. El indice se guarda localmente en `data/rag/chunks.json`, `data/rag/embeddings.npy` y `data/rag/metadata.json`.
5. Durante la generacion, el sistema construye una consulta semantica desde la entidad seleccionada.
6. Se recuperan los `top_k` chunks mas relevantes.
7. El prompt final usa:
   - contexto base estructurado: datos deterministas de la entidad seleccionada
   - contexto recuperado: fragmentos semanticamente cercanos del indice local
8. Si falla RAG, el sistema continua solo con contexto base.
9. Si falla el provider LLM, el sistema usa `fallback_generator.py`.

### Grounding y factualidad

El prompt de Fase 4 refuerza estas reglas:

- usar solo la informacion proporcionada
- no inventar datos
- priorizar contexto recuperado y estructurado si hay ambiguedad
- omitir detalles faltantes

### Construir el indice RAG

Comando base:

```bash
python scripts/build_rag_index.py
```

Tambien puedes elegir el provider de embeddings:

```bash
python scripts/build_rag_index.py --embedding-provider gemini
```

Archivos generados:

- `data/rag/chunks.json`
- `data/rag/embeddings.npy`
- `data/rag/metadata.json`

### Activar o desactivar RAG

- En la app, marca `Usar RAG` para intentar recuperar contexto adicional.
- Si `Usar LLM` esta desactivado, el sistema usa fallback local y no depende de RAG para generar la salida.
- Si el indice no existe o el provider de embeddings falla, la generacion continua con contexto base.

### Providers soportados

LLM:

- `openai`
- `gemini`
- `xai`

Embeddings:

- `openai`
- `gemini`

### Contexto base vs contexto recuperado

- Contexto base:
  se construye directamente desde la entidad seleccionada y contiene campos estructurados como nombre, tipo, resumen, descripcion e importancia.

- Contexto recuperado:
  proviene del indice RAG y contiene fragmentos semanticamente relevantes recuperados antes de llamar al LLM.

Esta separacion hace mas claro que datos vienen del registro principal y cuales llegan como apoyo semantico.

## Fase 3: Generacion hibrida multi-provider

La Fase 3 sigue vigente y se mantiene compatible:

- `llm_client.py` encapsula `openai`, `gemini` y `xai`
- `fallback_generator.py` mantiene salidas sin LLM
- `generator.py` decide entre modo `llm` y modo `fallback`

Si el provider LLM falla, el sistema no rompe la app y devuelve una salida local segura.

## Tipos de salida

- `ficha_historica`
- `resumen_corto`
- `texto_turistico`
- `post_redes`

## Tests

Ejecutar toda la suite:

```bash
pytest
```

La cobertura de Fase 4 incluye:

- chunking RAG
- cliente de embeddings sin llamadas reales
- recuperacion top-k con mocks
- flujo del generador con y sin RAG

## Evolucion futura

La implementacion de Fase 4 evita complejidad innecesaria hoy, pero deja preparada la arquitectura para:

- cambiar `numpy` por un backend vectorial dedicado
- agregar re-ranking
- agregar evaluacion automatica
- indexar nuevas fuentes historicas sin acoplar la app a un SDK concreto
