# historical-ecuador-generator

Aplicacion en Python + Streamlit para generar contenido historico del Ecuador a partir de una base local en JSON. El proyecto combina generacion textual, RAG local y generacion visual con proveedores externos opcionales.

## Nota importante

Durante esta revision no se encontro ninguna carpeta llamada `kuvho app` ni un proyecto Flutter dentro de este workspace o en `/Users/dragonborn/Documents/repo`. Tampoco aparecieron archivos tipicos de Flutter como `pubspec.yaml`, `lib/*.dart`, `android/` o `ios/`.

Eso significa que este repositorio no es una app Flutter y que los requisitos Android/iOS no aplican al codigo revisado aqui. Si la app que quieres auditar es otra, hace falta compartir esa carpeta.

## Que hace este proyecto

El sistema permite:

- seleccionar una entidad historica ecuatoriana desde un dataset local
- generar texto en varios formatos
- enriquecer el prompt con contexto recuperado desde un indice RAG local
- generar una imagen historica basada en el mismo contexto
- degradar con seguridad a un modo local si fallan los proveedores externos

Actualmente el dataset principal contiene:

- `10` entidades historicas
- `5` personajes
- `3` lugares
- `2` eventos

## Features principales

- Interfaz web con Streamlit para pruebas y exploracion rapida.
- Generacion textual en cuatro formatos: `ficha_historica`, `resumen_corto`, `texto_turistico` y `post_redes`.
- Generacion visual con modos: `retrato_historico`, `escena_historica`, `postal_turistica` e `ilustracion_educativa`.
- Estilos visuales controlados: `realista`, `pintura_oleo`, `grabado_antiguo` e `ilustracion_editorial`.
- RAG local sin base vectorial externa, usando `numpy` y archivos persistidos en `data/rag/`.
- Fallback local para que la app siga funcionando aun sin claves API o si un proveedor falla.
- Suite de pruebas automatizadas para los componentes centrales.

## Stack tecnico

- Lenguaje: Python
- UI: Streamlit
- Persistencia local: JSON, NPY, TXT, SQLite
- Recuperacion semantica: embeddings + similitud coseno con NumPy
- Proveedores LLM opcionales: OpenAI, Gemini, xAI
- Proveedor de imagen opcional: OpenAI
- Testing: Pytest

Dependencias directas declaradas en `requirements.txt`:

- `streamlit>=1.32.0`
- `pytest>=8.0.0`
- `openai>=1.30.0`
- `google-genai>=1.0.0`
- `numpy>=1.26.0`

## Estructura del proyecto

```text
historical-ecuador-generator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── app_state.sqlite3
│   ├── historical_entities.json
│   ├── prompt_templates.json
│   └── rag/
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
│   ├── history_store.py
│   ├── image_client.py
│   ├── image_generator.py
│   ├── image_prompt_builder.py
│   ├── llm_client.py
│   ├── loader.py
│   ├── prompt_builder.py
│   ├── rag_chunker.py
│   ├── rag_indexer.py
│   ├── rag_retriever.py
│   ├── utils.py
│   └── validation.py
├── tests/
├── README.md
└── requirements.txt
```

## Arquitectura funcional

### 1. Capa de datos

- `src/loader.py` carga y valida el dataset local.
- `data/historical_entities.json` actua como fuente principalf
- `data/prompt_templates.json` define formatos de salida.

### 2. Capa de contexto

- `src/context_builder.py` construye contexto base estructurado.
- `src/rag_chunker.py` transforma entidades en fragmentos recuperables.
- `src/rag_indexer.py` genera y guarda el indice local.
- `src/rag_retriever.py` consulta el indice por similitud coseno.

### 3. Capa de generacion

- `src/prompt_builder.py` arma prompts controlados en espanol.
- `src/llm_client.py` encapsula OpenAI, Gemini y xAI.
- `src/fallback_generator.py` resuelve salidas locales sin LLM.
- `src/image_prompt_builder.py` arma prompts visuales.
- `src/image_client.py` gestiona imagenes y fallback seguro.
- `src/generator.py` e `src/image_generator.py` coordinan texto, RAG e imagen.

### 4. Capa de persistencia local

- `src/history_store.py` guarda el historial de corridas en SQLite local.
- `data/app_state.sqlite3` almacena cada ejecucion como una corrida unificada.
- Se persisten prompts, contexto, providers efectivos, errores seguros y referencias a imagenes, pero nunca secretos.

### 5. Capa de interfaz

- `app/streamlit_app.py` expone el flujo completo en una UI web.

## Requisitos para correr el proyecto

### Requisitos generales

- Python 3.10 o superior
- `pip`
- Un entorno virtual recomendado

Version verificada en esta revision:

- Python `3.13.1`

### Variables de entorno opcionales

Solo son necesarias si quieres usar proveedores externos:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

El proyecto busca un archivo `.env` local y no lo sube a Git porque esta ignorado en `.gitignore`.

## Instalacion

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como ejecutar la app

### UI web

```bash
streamlit run app/streamlit_app.py
```

### Construir el indice RAG

```bash
python scripts/build_rag_index.py
```

Tambien puedes elegir el proveedor de embeddings:

```bash
python scripts/build_rag_index.py --embedding-provider gemini
```

### Ejecutar tests

```bash
./.venv/bin/pytest
```

Estado comprobado en esta revision:

- `43` tests pasaron correctamente

## Requisitos Android / iOS

No aplica para este codigo.

Este repositorio no contiene una app nativa ni Flutter. Lo que existe aqui es una aplicacion web con Streamlit. Si quieres verla desde un telefono o tablet, se accede por navegador una vez que la app esta levantada localmente o desplegada.

## Flujo de uso

1. Cargar entidades historicas desde `data/historical_entities.json`.
2. Elegir entidad, tipo de salida y opciones de generacion en la UI.
3. Construir contexto base.
4. Recuperar contexto adicional con RAG si el indice existe.
5. Generar texto con LLM o fallback local.
6. Generar imagen con proveedor externo o devolver un prompt visual reutilizable.
7. Guardar imagenes en `outputs/generated_images/` cuando aplique.

## Salidas del sistema

- Texto generado en pantalla y descargable como `.txt`
- Imagen generada local o referencia remota, segun el proveedor
- Chunks recuperados y contexto base visibles en modo diagnostico
- Indice RAG persistido en `data/rag/`

## Revision de seguridad realizada

Se hizo una revision estatica del codigo y dependencias directas presentes en este workspace.

### Hallazgos

- No se encontraron paquetes directos evidentemente maliciosos en `requirements.txt`.
- No se encontraron llamadas a `subprocess`, `os.system`, `eval`, `exec`, `curl`, `wget` ni ejecucion de shell.
- No se encontraron mecanismos de persistencia, autoarranque, instalacion silenciosa, ni cambios al sistema operativo.
- No se encontraron binarios extra, ejecutables raros o archivos sospechosos dentro del proyecto fuente.
- No se encontraron claves API hardcodeadas ni secretos expuestos en archivos trackeados.
- Las escrituras a disco estan acotadas a archivos de salida del proyecto como `outputs/generated_images/`, `data/rag/` y `data/processed/`.
- Las conexiones externas del codigo estan limitadas a SDKs de OpenAI, Gemini y xAI cuando el usuario configura una API key y ejecuta una accion que las necesite.

### Limitaciones de esta revision

- No se realizo escaneo antivirus por firmas.
- No se hizo auditoria online de CVEs o reputacion de paquetes porque la revision fue local/offline.
- No se pudo auditar la app Flutter `kuvho` porque no esta en este workspace.

## Archivos importantes

- [app/streamlit_app.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/app/streamlit_app.py)
- [src/generator.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/src/generator.py)
- [src/llm_client.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/src/llm_client.py)
- [src/image_client.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/src/image_client.py)
- [src/rag_indexer.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/src/rag_indexer.py)
- [src/rag_retriever.py](/Users/dragonborn/Documents/repo/ecuador_historical_characters/historical-ecuador-generator/src/rag_retriever.py)

## Resumen corto

`historical-ecuador-generator` es una app web educativa para generar contenido historico del Ecuador con grounding local, RAG opcional e imagenes opcionales. En el workspace revisado no hay una app Flutter ni señales obvias de codigo malicioso o comportamiento peligroso para tu computador.
