# historical-ecuador-generator

MVP en Python + Streamlit para generar contenido historico del Ecuador a partir de una base estructurada en JSON. La Fase 1 resuelve la experiencia de generacion en la interfaz y la Fase 2 agrega una capa de limpieza, validacion y exportacion de datos para preparar la base hacia integraciones futuras con LLMs, RAG y evaluacion formal.

## Demo

- App desplegada en Streamlit: [historical-ecuador-generator.streamlit.app](https://historical-ecuador-generator.streamlit.app/)

## Estructura

```text
historical-ecuador-generator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── historical_entities.json
│   ├── prompt_templates.json
│   └── processed/
├── outputs/
│   └── sample_outputs/
├── src/
│   ├── cleaning.py
│   ├── formatter.py
│   ├── generator.py
│   ├── loader.py
│   ├── pipeline_phase2.py
│   ├── utils.py
│   └── validation.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Descripcion

La aplicacion permite:

- seleccionar una entidad historica por nombre
- elegir un tipo de salida textual
- generar contenido usando un conjunto fijo de campos del JSON
- visualizar relaciones, etiquetas y metadatos
- copiar o descargar facilmente el resultado generado

## Tipos de salida

- `ficha_historica`: salida formal con contexto, relevancia y relaciones clave
- `resumen_corto`: version sintetica pensada para estudiantes
- `texto_turistico`: texto atractivo con enfoque cultural e historico
- `post_redes`: contenido breve y legible para redes sociales

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

## Ejecucion Fase 1

Desde la carpeta del proyecto:

```bash
streamlit run app/streamlit_app.py
```

## Fase 2: Limpieza y validacion

La Fase 2 agrega una capa nueva sobre `data/historical_entities.json` para:

- validar registros historicos
- normalizar tipos de entidad
- limpiar textos y listas relacionadas
- unificar etiquetas
- corregir valores nulos o inconsistentes simples
- eliminar duplicados por `id` o `nombre`
- exportar una base curada en JSON y CSV
- generar un reporte de validacion listo para inspeccion manual

### Ejecutar el pipeline

```bash
python -m src.pipeline_phase2
```

### Archivos generados por la Fase 2

- `data/processed/historical_entities_clean.json`
- `data/processed/historical_entities_clean.csv`
- `data/processed/validation_report.json`

## Datos usados

El proyecto trabaja con dos capas principales:

- `data/historical_entities.json`: base historica original
- `data/prompt_templates.json`: instrucciones base para la generacion textual

La Fase 2 produce una tercera capa procesada en `data/processed/`.

## Manejo de errores

El proyecto detecta:

- archivos JSON inexistentes
- JSON mal formado
- estructuras invalidas en entidades o plantillas
- campos requeridos faltantes
- tipos de entidad invalidos
- rangos de años inconsistentes
- textos sospechosamente cortos

## Posibles mejoras futuras

- integracion con embeddings y recuperacion semantica
- exportacion en PDF o Markdown
- filtros por tipo, epoca o ubicacion
- evaluacion automatica de calidad de registros
- integracion opcional con modelos LLM
