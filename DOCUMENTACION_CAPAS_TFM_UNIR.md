# Documentacion academica de la arquitectura del evaluador multimodal

## 1. Introduccion

El sistema objeto de esta documentacion corresponde a una aplicacion de evaluacion multimodal orientada al analisis de resultados generados por inteligencia artificial. Su finalidad principal consiste en medir de forma automatizada la calidad de tres textos producidos frente a una fuente de referencia comun, complementando este nucleo con un modulo secundario de evaluacion visual. Desde la perspectiva academica, se trata de una herramienta de apoyo a la validacion cuantitativa de contenido generado, especialmente util en contextos donde se requiere contrastar similitud formal, conservacion semantica y correspondencia texto-imagen.

La solucion ha sido diseñada siguiendo un enfoque por capas. Esta decision no responde unicamente a una preferencia de estilo, sino a una necesidad metodologica concreta: separar de forma clara los componentes encargados de la interfaz, la validacion, el calculo de metricas, la persistencia, la seguridad operativa y la consulta historica. En un trabajo de fin de master, esta separacion favorece la explicabilidad del prototipo, mejora su defendibilidad tecnica y facilita la justificacion de cada decision de diseño.

El presente documento describe la arquitectura del evaluador desde una perspectiva academica, tomando como base exclusiva el sistema de evaluacion y no otros proyectos adyacentes. Para cada capa se examinan su funcionalidad, sus dependencias, la logica de su construccion y su papel dentro del flujo global de uso.

## 2. Objetivo arquitectonico del sistema

El objetivo arquitectonico principal del evaluador es proporcionar una herramienta reproducible, extensible y operativamente segura para valorar salidas de inteligencia artificial mediante metricas automaticas. Este objetivo se concreta en cuatro metas de diseño:

1. Separar claramente la interfaz de la logica de evaluacion.
2. Garantizar validacion temprana de las entradas para evitar resultados engañosos o fallos innecesarios.
3. Permitir persistencia opcional sin convertirla en requisito indispensable para la ejecucion.
4. Controlar la escritura en la base de datos aunque la aplicacion sea accesible publicamente.

La arquitectura por capas resulta adecuada para estas metas porque permite desacoplar problemas distintos: unos relacionados con experiencia de uso, otros con calculo numerico, otros con infraestructura y otros con seguridad basica de operacion.

## 3. Vision general del flujo de funcionamiento

Desde una perspectiva funcional, el sistema puede entenderse como dos modulos independientes que comparten servicios transversales. El primero es un modulo de evaluacion textual, que compara tres textos generados frente a una fuente de referencia comun. El segundo es un modulo de evaluacion visual, que calcula correspondencia texto-imagen y similitud visual relativa entre varias imagenes.

El flujo general de una evaluacion puede resumirse del siguiente modo:

1. La persona usuaria introduce los datos requeridos en la interfaz.
2. El sistema valida las entradas y las normaliza.
3. Se cargan los recursos necesarios para la metrica solicitada.
4. Se calculan las metricas correspondientes.
5. Los resultados se representan visualmente en la aplicacion.
6. Si existe configuracion de persistencia y la escritura esta autorizada, la corrida se almacena.
7. La informacion guardada puede recuperarse posteriormente desde el historial.

Este flujo dota al sistema de una estructura operativa clara y facilita tanto su analisis funcional como su mantenimiento posterior.

## 4. Descripcion de la arquitectura por capas

### 4.1. Capa de presentacion

#### Funcionalidad

La capa de presentacion constituye el punto de contacto entre la persona usuaria y el evaluador. Su funcion principal consiste en recoger entradas, iniciar procesos de evaluacion y mostrar resultados, advertencias, estados de persistencia e informacion historica. No se limita a una representacion grafica pasiva, sino que organiza el recorrido funcional de la aplicacion.

La interfaz se distribuye en tres espacios de trabajo diferenciados:

- evaluacion de texto
- evaluacion de imagenes
- historial de evaluaciones

Adicionalmente, existe un espacio lateral destinado al control del acceso de escritura.

#### Dependencias

Esta capa depende de una tecnologia de interfaz reactiva, de la capa de evaluacion para ejecutar metricas, de la capa de persistencia para mostrar el estado de la base de datos y de la capa de seguridad para gestionar el permiso de escritura.

#### Construccion

Su construccion sigue un enfoque declarativo basado en formularios y secciones separadas. Esta organizacion permite preservar la independencia entre los modulos textual y visual, evitando que la complejidad del segundo contamine el flujo principal del primero. La presentacion no contiene logica numerica pesada; se limita a coordinar entradas, disparar procesamiento y renderizar resultados.

#### Uso

Se utiliza como entorno principal de experimentacion y validacion manual. Desde el punto de vista del TFM, esta capa aporta la visibilidad necesaria para demostrar funcionalidad, interpretar resultados y justificar decisiones de evaluacion.

### 4.2. Capa de validacion y normalizacion

#### Funcionalidad

La capa de validacion y normalizacion tiene como cometido verificar que los datos introducidos por la persona usuaria cumplen unos requisitos minimos antes de activar el calculo de metricas. Su finalidad es doble: proteger la estabilidad del sistema y asegurar que los resultados obtenidos se apoyen en entradas tecnicamente validas.

En el modulo textual, la validacion se centra en la existencia de una fuente y de tres textos candidatos no vacios. En el modulo visual, se exige la presencia de tres imagenes principales y de tres textos asociados, aceptando una cuarta imagen de referencia como componente opcional. Adicionalmente, se verifican el tamaño maximo permitido y la integridad de los archivos de imagen.

#### Dependencias

Depende de utilidades de limpieza de texto, de mecanismos de lectura de archivos cargados por la interfaz y de bibliotecas de tratamiento de imagen.

#### Construccion

Esta capa ha sido construida con una estrategia de validacion temprana. Antes de iniciar procesos costosos, como la carga de modelos o el calculo de metricas multimodales, el sistema intenta identificar errores de entrada con mensajes claros y orientados al usuario. Esta decision mejora la experiencia de uso y reduce ejecuciones innecesarias.

#### Uso

Se activa automaticamente al solicitar una evaluacion y actua como puerta de entrada al resto de capas. Si detecta una condicion invalida, interrumpe el flujo y evita calculos posteriores.

### 4.3. Capa de evaluacion textual

#### Funcionalidad

La capa de evaluacion textual es el componente central del sistema. Su funcion consiste en comparar tres textos generados de manera independiente frente a una misma fuente de referencia. El evaluador no busca determinar una verdad absoluta sobre la calidad del texto, sino ofrecer medidas cuantitativas que permitan contrastar fidelidad superficial y proximidad semantica.

Las metricas empleadas son BLEU y BERTScore. BLEU aporta una medida basada en superposicion de n-gramas y resulta util para estimar cercania formal respecto a la fuente. BERTScore, por su parte, utiliza representaciones contextuales y permite captar similitud de significado aun cuando exista reformulacion o parafraseo.

#### Dependencias

Depende de bibliotecas especializadas en evaluacion textual y de un entorno de ejecucion capaz de cargar los modelos asociados a BERTScore.

#### Construccion

La construccion de esta capa responde a un esquema comparativo estable:

1. La fuente se valida y se normaliza.
2. Cada texto candidato se limpia individualmente.
3. Cada candidato se compara contra la misma fuente.
4. Se generan cuatro resultados numericos por candidato.
5. Los resultados se estructuran uniformemente para representacion y posible persistencia.

Este diseño favorece la comparabilidad entre candidatos y facilita el analisis posterior de diferencias entre modelos, prompts o reformulaciones.

#### Uso

Se utiliza en escenarios donde interesa comparar tres respuestas generadas por IA, tres versiones de una misma reescritura o tres alternativas de resumen frente a una misma fuente textual.

### 4.4. Capa de evaluacion visual

#### Funcionalidad

La capa de evaluacion visual implementa el componente multimodal del sistema. Su finalidad es medir, por una parte, la alineacion semantica entre imagen y texto y, por otra, la similitud visual relativa entre una imagen objetivo y un pequeño conjunto de referencia.

El evaluador trabaja con tres imagenes principales obligatorias y una cuarta imagen opcional como referencia adicional. Para cada una de las tres imagenes principales se proporciona un texto independiente que sirve de base para el calculo de CLIPScore. Paralelamente, se calculan tres comparaciones FID donde cada imagen principal se contrasta frente al conjunto formado por las otras imagenes disponibles.

#### Dependencias

Depende de bibliotecas de procesamiento de imagen, de metricas visuales, de modelos multimodales y del motor tensorial necesario para su ejecucion.

#### Construccion

La capa se ha construido distinguiendo dos tipos de evaluacion:

- evaluacion de correspondencia semantica texto-imagen
- evaluacion de similitud visual distribucional

La primera utiliza una proyeccion comun entre texto e imagen. La segunda requiere preparar lotes de imagenes y calcular una distancia estadistica entre distribuciones. Esta separacion conceptual resulta importante desde el punto de vista metodologico, ya que ambas metricas responden a preguntas distintas y no deben interpretarse del mismo modo.

#### Uso

Se utiliza en tareas exploratorias de evaluacion visual, especialmente cuando se desea observar si varias imagenes generadas o seleccionadas mantienen relacion con descripciones textuales y consistencia relativa con respecto a un conjunto de referencia.

### 4.5. Capa de preprocesamiento visual

#### Funcionalidad

La capa de preprocesamiento visual adapta las imagenes cargadas a los requerimientos de las metricas y modelos utilizados. Su objetivo es estandarizar la representacion de entrada y reducir errores provocados por diferencias de formato.

Entre sus tareas se encuentran la conversion cromatica, la transformacion a tensores, el redimensionado a resoluciones exigidas por determinadas metricas y la composicion de lotes para calculo estadistico.

#### Dependencias

Depende de bibliotecas de imagen y de transformaciones tensoriales.

#### Construccion

Su construccion obedece a un principio de homogeneizacion previa. Antes de que una imagen llegue a una metrica, debe poseer un formato y unas dimensiones consistentes con el resto del flujo. Esta capa evita trasladar a componentes posteriores una complejidad de preprocesamiento que no les corresponde asumir.

#### Uso

Se utiliza de forma interna dentro del modulo visual y no requiere interaccion directa por parte de la persona usuaria.

### 4.6. Capa de compatibilidad de modelos y metricas

#### Funcionalidad

Esta capa resuelve aspectos operativos asociados al uso de modelos y bibliotecas externas. Su finalidad es mantener la funcionalidad del evaluador en contextos reales donde pueden existir diferencias de version, restricciones de hardware o limites internos de los modelos.

Entre sus responsabilidades destacan la seleccion automatica del dispositivo de ejecucion, la carga cacheada de recursos, el truncamiento de textos para modelos multimodales y la gestion de caminos alternativos cuando una metrica presenta incompatibilidades con una determinada combinacion de dependencias.

#### Dependencias

Depende de los modelos de lenguaje y vision utilizados por las metricas, del motor tensorial y de las bibliotecas que encapsulan dichas metricas.

#### Construccion

Se ha construido bajo un criterio de robustez operativa. Un evaluador concebido para uso academico no debe fallar por completo ante pequeñas variaciones del entorno si existen mecanismos razonables para adaptarse. Por ello, la capa incorpora soluciones de compatibilidad que permiten mantener resultados funcionales en escenarios heterogeneos.

#### Uso

Su uso es transparente para la persona usuaria y se activa internamente durante la evaluacion textual y visual.

### 4.7. Capa de aproximacion estadistica para conjuntos pequeños

#### Funcionalidad

La evaluacion visual presenta una dificultad metodologica especifica: ciertas metricas, como FID, estan pensadas para comparar distribuciones y no muestras aisladas. Dado que el sistema trabaja con un numero muy reducido de imagenes, se introduce una capa de aproximacion que permite realizar el calculo sin invalidar el flujo operativo.

#### Dependencias

Depende del preprocesamiento visual y de la implementacion de la metrica estadistica correspondiente.

#### Construccion

La estrategia de construccion consiste en aproximar la distribucion de una imagen objetivo mediante vistas deterministicas adicionales. Esta decision no transforma la evaluacion en una medicion estadisticamente robusta, pero permite obtener un valor orientativo sin bloquear la funcionalidad del sistema. La interfaz, de manera coherente, advierte explicitamente sobre la naturaleza exploratoria de esta aproximacion.

#### Uso

Se utiliza unicamente en el modulo visual cuando el sistema calcula FID sobre conjuntos reducidos.

### 4.8. Capa de persistencia

#### Funcionalidad

La persistencia permite almacenar evaluaciones de texto e imagen para su consulta posterior. Esta capacidad transforma al sistema en una herramienta de trabajo acumulativa, en la que las corridas pueden revisarse, compararse y reutilizarse sin necesidad de repetir calculos.

En el caso textual se conservan la fuente, los tres candidatos y sus metricas. En el caso visual se conservan los textos empleados para la evaluacion multimodal, las metricas calculadas y los binarios de las imagenes cargadas.

#### Dependencias

Depende de un sistema de base de datos relacional, de un motor ORM y de mecanismos auxiliares para serializar campos compuestos o almacenar binarios de imagen.

#### Construccion

La persistencia se ha construido como un subsistema opcional. Si la configuracion de base de datos no esta disponible, la aplicacion continua funcionando en modo sin historial. Esta decision mejora la portabilidad del prototipo y evita convertir la infraestructura en un requisito imprescindible para el uso basico.

La estructura de almacenamiento se organiza mediante entidades principales y entidades hijas, lo que permite representar adecuadamente relaciones como una evaluacion textual con tres candidatos o una evaluacion visual con multiples activos de imagen asociados.

#### Uso

Se utiliza en escenarios donde interesa conservar resultados para comparacion longitudinal, auditoria o demostracion del funcionamiento del sistema.

### 4.9. Capa de control de escritura

#### Funcionalidad

La capa de control de escritura añade una medida de seguridad ligera para despliegues publicos. Su objetivo es permitir que la aplicacion pueda ser utilizada libremente para evaluar, pero que el guardado en la base de datos quede restringido a quienes posean una contraseña concreta.

#### Dependencias

Depende de la configuracion del entorno, del estado de sesion y de una comparacion segura de cadenas.

#### Construccion

Se ha construido con un modelo de desbloqueo temporal por sesion. La contraseña no se persiste como dato de trabajo, sino que se verifica en el momento y habilita un estado booleano de escritura. Esta solucion evita introducir un sistema formal de autenticacion, pero satisface una necesidad practica frecuente en prototipos desplegados en abierto.

#### Uso

Se emplea cuando se desea combinar acceso publico a la herramienta con control sobre la poblacion del historial persistente.

### 4.10. Capa de consulta historica

#### Funcionalidad

La capa de consulta historica recupera y representa las evaluaciones almacenadas. Su papel es importante no solo desde el punto de vista funcional, sino tambien metodologico, ya que permite analizar ex post el comportamiento del sistema y revisar las condiciones bajo las cuales se obtuvo cada resultado.

#### Dependencias

Depende de la persistencia, de la interfaz y de capacidades de reconstruccion de imagen a partir de binarios almacenados.

#### Construccion

Se ha construido como una lectura acotada de registros recientes, organizada por modulo. Esta limitacion controlada mejora el rendimiento de la interfaz y evita sobrecargas innecesarias en consultas de historial.

#### Uso

Se utiliza para revisar corridas anteriores, comparar metricas entre sesiones y recuperar material de evaluacion ya calculado.

### 4.11. Capa de configuracion y despliegue

#### Funcionalidad

Esta capa resuelve los parametros externos necesarios para que la aplicacion se adapte al entorno de ejecucion. Entre ellos destacan la conexion a la base de datos, el modo de seguridad de la conexion y la contraseña de escritura.

#### Dependencias

Depende del sistema de secretos de la plataforma de despliegue y de variables de entorno del sistema operativo.

#### Construccion

Su construccion sigue un criterio de flexibilidad. El sistema intenta resolver valores desde secretos gestionados por la plataforma y, en ausencia de estos, recurre a variables de entorno convencionales. Esta doble via simplifica tanto el trabajo local como el despliegue en entornos administrados.

#### Uso

Se utiliza al iniciar la aplicacion y condiciona la disponibilidad de persistencia, historial y control de escritura.

### 4.12. Capa de gestion de errores

#### Funcionalidad

La gestion de errores se ocupa de capturar excepciones y transformarlas en mensajes comprensibles para la persona usuaria. Su existencia es particularmente importante en un sistema que depende de modelos pesados, bibliotecas con multiples capas internas y una infraestructura opcional de persistencia.

#### Dependencias

Depende de las excepciones generadas por las capas de validacion, procesamiento de imagen, metricas, modelos y base de datos.

#### Construccion

Se ha construido bajo el principio de captura localizada. Cada modulo intenta encapsular sus fallos y expresarlos en su propio contexto funcional, evitando que un error de bajo nivel se propague hasta romper por completo la aplicacion.

#### Uso

Se utiliza de forma transversal durante toda la ejecucion del sistema.

## 5. Relaciones funcionales entre capas

Las capas se articulan siguiendo una dependencia jerarquica razonablemente clara. La presentacion recoge entradas y delega su analisis a la validacion. Una vez validadas, las capas de evaluacion textual o visual calculan las metricas con ayuda del preprocesamiento y de la compatibilidad de modelos. Posteriormente, la persistencia y el control de escritura determinan si la corrida puede ser almacenada. Finalmente, la consulta historica permite recuperar la informacion guardada para una nueva lectura desde la interfaz.

Esta organizacion refuerza la separacion entre las preocupaciones principales del sistema y hace posible argumentar cada bloque funcional con relativa independencia.

## 6. Ventajas del diseño adoptado

La arquitectura propuesta presenta varias ventajas relevantes en el marco de un TFM:

- separa con claridad interfaz, evaluacion, persistencia y seguridad
- permite operar sin base de datos, lo que mejora portabilidad
- conserva trazabilidad de las evaluaciones cuando el historial esta activo
- mantiene un enfoque defendible al combinar metricas superficiales y semanticas
- incorpora mecanismos de compatibilidad que mejoran la estabilidad operativa
- ofrece una medida de control de escritura suficiente para despliegues publicos sencillos

Estas ventajas favorecen no solo la funcionalidad del prototipo, sino tambien su explicabilidad y defendibilidad en una memoria academica.

## 7. Limitaciones del sistema

Pese a su utilidad, el evaluador presenta limitaciones que deben explicitarse. En el modulo textual, las metricas automaticas no sustituyen una evaluacion humana experta y pueden no captar matices discursivos o contextuales complejos. En el modulo visual, el uso de FID con conjuntos muy pequeños posee un caracter exploratorio y no debe interpretarse como una evaluacion estadistica concluyente. Asimismo, la carga inicial de modelos puede resultar costosa y la persistencia de binarios en base de datos incrementa el peso del historial.

Estas limitaciones no invalidan el sistema, pero delimitan el alcance de sus resultados y ayudan a interpretar correctamente su aportacion dentro del trabajo academico.

## 8. Lineas de evolucion futura

La arquitectura actual deja abiertas varias posibilidades de ampliacion:

- incorporacion de nuevas metricas textuales
- exportacion estructurada de resultados
- evaluacion por lotes
- autenticacion mas robusta para el acceso de escritura
- optimizacion del almacenamiento visual
- ampliacion del historial con filtros y consultas comparativas

Desde el punto de vista del diseño, estas lineas de evolucion pueden incorporarse sin romper la estructura general si se mantiene la separacion por capas adoptada.

## 9. Conclusion

La arquitectura del evaluador multimodal proporciona una base coherente para un sistema academico de analisis de salidas generadas por inteligencia artificial. Su principal fortaleza radica en la combinacion de simplicidad operativa, separacion funcional y trazabilidad del proceso de evaluacion. El modulo textual actua como eje central del sistema al permitir la comparacion de tres textos frente a una fuente comun mediante metricas automaticas ampliamente utilizadas. El modulo visual, por su parte, aporta una dimension complementaria de caracter exploratorio.

Desde la perspectiva de un Trabajo Fin de Master, la solucion resulta defendible porque no se limita a ejecutar calculos metricos, sino que articula dichos calculos dentro de una arquitectura clara, extensible y metodologicamente justificable. En consecuencia, el sistema puede entenderse no solo como una aplicacion funcional, sino como una propuesta de diseño orientada a la evaluacion cuantitativa de contenido generado por IA en entornos de uso real.
