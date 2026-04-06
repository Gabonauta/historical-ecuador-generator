# historical-ecuador-generator

MVP en Python + Streamlit para generar contenido historico del Ecuador a partir de una base estructurada en JSON. La generacion inicial usa plantillas y composicion programatica, de modo que el proyecto quede simple, mantenible y listo para crecer hacia RAG o integracion con LLMs.

## Demo

- App desplegada en Streamlit: [historical-ecuador-generator.streamlit.app](https://historical-ecuador-generator.streamlit.app/)

## Estructura

```text
historical-ecuador-generator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── historical_entities.json
│   └── prompt_templates.json
├── outputs/
│   └── sample_outputs/
├── src/
│   ├── formatter.py
│   ├── generator.py
│   ├── loader.py
│   └── utils.py
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

## Ejecucion

Desde la carpeta del proyecto:

```bash
streamlit run app/streamlit_app.py
```

## Datos usados

El MVP carga dos archivos JSON:

- `data/historical_entities.json`: base de conocimiento con personajes, lugares y eventos
- `data/prompt_templates.json`: instrucciones base para cada tipo de salida

## Manejo de errores

El proyecto detecta:

- archivos JSON inexistentes
- JSON mal formado
- estructuras invalidas en entidades o plantillas

## Posibles mejoras futuras

- integracion con embeddings y recuperacion semantica
- exportacion en PDF o Markdown
- filtros por tipo, epoca o ubicacion
- pruebas unitarias para los modulos principales
- integracion opcional con modelos LLM
