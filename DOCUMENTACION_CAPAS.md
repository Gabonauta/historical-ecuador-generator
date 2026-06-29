# Documentacion por capas del evaluador multimodal

## Proposito general

El sistema evaluador tiene como finalidad medir de forma reproducible la calidad de salidas generadas por inteligencia artificial mediante metricas automaticas aplicadas a texto e imagen. Su foco principal recae en la comparacion de textos generados frente a una fuente de referencia, aunque incorpora ademas un modulo visual orientado a contrastar correspondencia texto-imagen y similitud visual entre conjuntos reducidos de imagenes.

La arquitectura se organiza en capas con el fin de separar la interfaz, las validaciones, el calculo de metricas, la persistencia y los mecanismos de seguridad operativa. Esta separacion facilita la evolucion del sistema, permite aislar mejor los puntos de fallo y hace mas clara la relacion entre entrada, procesamiento y resultado.

## Vista general del flujo

El sistema opera a partir de dos modulos funcionales independientes:

1. Evaluacion de texto.
2. Evaluacion de imagenes.

Ambos modulos comparten un conjunto transversal de capacidades:

- validacion de entradas
- carga controlada de modelos
- manejo seguro de errores
- persistencia opcional en base de datos
- consulta de historial
- proteccion de escritura mediante contraseña

El recorrido tipico de una evaluacion es el siguiente:

1. La persona usuaria introduce los datos requeridos.
2. El sistema limpia y valida las entradas.
3. Se calculan las metricas correspondientes.
4. Los resultados se muestran en interfaz.
5. Si la persistencia esta configurada y la escritura esta habilitada, la corrida se almacena.
6. El historial puede recuperarse posteriormente para consulta.

## Capa de presentacion

### Funcionalidad

Esta capa organiza la experiencia de uso y convierte la aplicacion en una herramienta accesible para personas no necesariamente tecnicas. Se encarga de recoger entradas, lanzar el procesamiento, mostrar mensajes de estado y renderizar resultados de forma estructurada.

La interfaz se distribuye en tres espacios funcionales:

- un modulo de evaluacion textual
- un modulo de evaluacion visual
- un espacio de historial

Adicionalmente, la barra lateral se utiliza para controlar el acceso de escritura a la base de datos cuando dicho mecanismo de proteccion esta activado.

### Dependencias

Depende de una libreria de interfaz reactiva, de la capa de persistencia para conocer el estado de la base y de la capa de calculo para ejecutar metricas. Tambien se apoya en mecanismos de cache para reducir la carga repetitiva de modelos y recursos pesados.

### Construccion

La capa de presentacion se ha construido con formularios separados por modulo y una estructura tabulada para mantener independencia funcional entre texto, imagenes e historial. La estrategia seguida evita mezclar logica de negocio compleja con renderizacion visual, delegando el calculo real a capas inferiores.

### Uso

Se utiliza para introducir una fuente textual y tres textos candidatos, asi como para cargar varias imagenes y sus textos asociados. Tambien permite revisar evaluaciones almacenadas y gestionar el permiso temporal de escritura.

## Capa de validacion y normalizacion

### Funcionalidad

Esta capa garantiza que las entradas cumplan las condiciones minimas antes de ejecutar metricas que dependen de modelos o calculo numerico. Su objetivo es prevenir fallos evitables y reducir resultados inconsistentes provocados por entradas vacias, dañadas o fuera de los limites operativos.

Las validaciones principales incluyen:

- comprobacion de que la fuente textual exista
- comprobacion de que existan los tres textos a comparar
- comprobacion de que existan las tres imagenes principales
- aceptacion opcional de una cuarta imagen de referencia
- verificacion de tamaño maximo por imagen
- verificacion de que los archivos cargados sean imagenes legibles
- conversion de imagenes a un formato cromatico comun

### Dependencias

Depende de utilidades de texto, del manejo de archivos subidos por la interfaz y de bibliotecas de imagen para lectura, conversion y deteccion de archivos dañados.

### Construccion

La construccion de esta capa responde al principio de validacion temprana. Antes de cargar modelos costosos o calcular metricas, el sistema intenta detectar problemas de entrada con mensajes claros. Esto reduce tiempos de espera innecesarios y mejora la explicabilidad de los errores.

### Uso

Se activa automaticamente cada vez que la persona usuaria pulsa una accion de evaluacion. Si la validacion falla, el flujo se detiene antes del calculo y se emite un mensaje visible.

## Capa de evaluacion textual

### Funcionalidad

La evaluacion textual constituye el nucleo principal del sistema. Su funcion es comparar tres textos generados de forma independiente frente a una misma fuente de referencia. Cada texto se evalua por separado y produce un conjunto propio de metricas.

Las metricas utilizadas son:

- BLEU
- BERTScore Precision
- BERTScore Recall
- BERTScore F1

BLEU permite medir similitud superficial basada en n-gramas, mientras que BERTScore captura proximidad semantica a partir de representaciones contextuales. La coexistencia de ambas medidas ofrece una lectura mas rica del comportamiento del texto generado.

### Dependencias

Depende de una biblioteca especializada en BLEU y de otra especializada en BERTScore, asi como de la infraestructura de tensores necesaria para ejecutar modelos de lenguaje subyacentes.

### Construccion

La construccion de esta capa responde a una logica comparativa simple y controlada:

1. Se valida la fuente textual.
2. Se validan los tres textos candidatos.
3. Cada candidato se contrasta con la misma fuente.
4. Se agregan las metricas en una estructura uniforme.
5. Los resultados se entregan a la interfaz y, opcionalmente, a la persistencia.

El diseño evita mezclar referencias multiples complejas dentro del flujo principal y privilegia un esquema estable de comparacion uno-a-uno frente a una misma fuente.

### Uso

Se utiliza para comparar distintas salidas de un mismo modelo, distintas formulaciones de prompt o respuestas de varios modelos frente a un mismo contenido base. Resulta especialmente util cuando se quiere medir tanto fidelidad lexical como conservacion de significado.

## Capa de evaluacion visual

### Funcionalidad

Esta capa implementa el modulo de imagenes. Su proposito es ofrecer una evaluacion exploratoria de correspondencia texto-imagen y de similitud visual relativa entre varias imagenes cargadas por la persona usuaria.

El flujo visual trabaja con:

- tres imagenes principales obligatorias
- una cuarta imagen opcional como referencia adicional
- tres textos independientes, uno por cada imagen principal

Las metricas calculadas son:

- un valor de CLIPScore para cada una de las tres imagenes principales
- tres valores de FID construidos mediante contrastes entre una imagen objetivo y el conjunto formado por las otras imagenes disponibles

La cuarta imagen no se evalua de forma directa con CLIPScore, pero puede enriquecer cada conjunto de referencia en FID.

### Dependencias

Depende de bibliotecas de tensores, metricas visuales, transformaciones de imagen y modelos multimodales capaces de proyectar texto e imagen en un espacio semantico comun.

### Construccion

La evaluacion visual se ha construido con dos rutas diferenciadas:

- una ruta de correspondencia semantica texto-imagen
- una ruta de similitud distribucional entre imagenes

En la primera se calcula CLIPScore para cada imagen principal frente a su texto asociado. En la segunda se calcula FID para tres comparaciones del tipo uno contra grupo:

- imagen 1 frente al conjunto formado por imagenes 2, 3 y, si existe, 4
- imagen 2 frente al conjunto formado por imagenes 1, 3 y, si existe, 4
- imagen 3 frente al conjunto formado por imagenes 1, 2 y, si existe, 4

### Uso

Se utiliza cuando se desea revisar no solo si una imagen se alinea con una descripcion textual, sino tambien si su apariencia global se mantiene razonablemente cercana al resto del conjunto de referencia cargado.

## Capa de preprocesamiento de imagenes

### Funcionalidad

Esta capa prepara las imagenes para el calculo de metricas visuales. Su cometido es homogeneizar formato, tamaño y estructura tensorial antes de entregar los datos a los modelos correspondientes.

Entre sus funciones se incluyen:

- conversion a RGB
- conversion desde imagen cargada a tensor
- redimensionado para metricas que exigen una resolucion concreta
- ensamblado de lotes de imagenes

### Dependencias

Depende de bibliotecas de imagen y de transformaciones tensoriales.

### Construccion

La construccion sigue un enfoque defensivo: todas las imagenes se convierten a un formato consistente antes de entrar en los modelos. Esto reduce la probabilidad de errores por canales incompatibles o tamaños inesperados.

### Uso

Se usa de forma interna dentro del modulo de imagenes cada vez que se calcula FID o CLIPScore.

## Capa de compatibilidad de metricas y modelos

### Funcionalidad

Esta capa aborda problemas practicos derivados del uso de bibliotecas y modelos pesados en entornos heterogeneos. Su objetivo es mantener la operatividad del evaluador incluso cuando existen diferencias de version o restricciones de ejecucion.

Entre sus responsabilidades destacan:

- seleccion automatica de CPU o GPU
- carga cacheada de modelos para evitar inicializaciones repetidas
- truncamiento de texto para ajustarse a limites del modelo multimodal
- extraccion compatible de embeddings cuando cambian las estructuras devueltas por el modelo
- uso de rutas alternativas cuando una implementacion directa de una metrica presenta incompatibilidades

### Dependencias

Depende de bibliotecas de modelos, del motor tensorial y de las bibliotecas de metricas multimodales.

### Construccion

La construccion de esta capa responde a una realidad operativa: un evaluador basado en modelos no debe depender de una unica combinacion exacta de versiones para seguir funcionando. Por ello se incorporan mecanismos de compatibilidad y degradacion local que permiten resolver diferencias en salidas de modelos o en restricciones de longitud.

### Uso

Se utiliza de forma transparente durante el calculo de BERTScore y CLIPScore, especialmente en entornos donde la primera ejecucion necesita cargar recursos externos o donde existen diferencias de implementacion entre bibliotecas.

## Capa de aproximacion numerica para FID

### Funcionalidad

Esta capa resuelve una dificultad especifica del modulo visual: FID requiere trabajar con distribuciones de imagenes y no con una unica muestra aislada. Dado que el evaluador opera con conjuntos pequeños, se introduce una aproximacion controlada para hacer posible el calculo.

### Dependencias

Depende del preprocesamiento visual y de la biblioteca de FID.

### Construccion

La estrategia consiste en construir una distribucion minima para la imagen objetivo a partir de dos vistas deterministicas. De esta forma, el sistema evita fallos numericos cuando el calculo estadistico no puede operar sobre una sola observacion.

Esta capa no pretende convertir la metrica en una evaluacion robusta desde el punto de vista estadistico. Por el contrario, la interfaz advierte explicitamente que se trata de un uso exploratorio. La capa existe para permitir comparacion orientativa, no inferencia concluyente.

### Uso

Se emplea unicamente en el modulo visual, dentro del calculo de FID sobre conjuntos reducidos.

## Capa de persistencia

### Funcionalidad

La capa de persistencia almacena el historial de evaluaciones de texto e imagen cuando la base de datos esta configurada. Su proposito es conservar tanto los insumos como los resultados, de forma que la aplicacion pueda operar como una herramienta de consulta y no solo como un evaluador efimero.

La persistencia contempla dos familias de informacion:

- evaluaciones textuales con sus tres candidatos y metricas asociadas
- evaluaciones visuales con metricas, textos usados para CLIPScore e imagenes almacenadas en binario

### Dependencias

Depende de una base de datos relacional, de un motor ORM y de mecanismos de serializacion para algunos campos compuestos. Tambien se apoya en funciones auxiliares para detectar secretos, resolver configuracion y preparar binarios de imagen.

### Construccion

La construccion se ha realizado con un modelo relacional sencillo, pero suficiente para el alcance actual:

- una entidad principal para la evaluacion textual
- una entidad hija para cada texto candidato
- una entidad principal para la evaluacion visual
- una entidad hija para cada imagen cargada

Las tablas se crean automaticamente cuando la configuracion de base esta disponible. Esto simplifica el despliegue y reduce pasos manuales en entornos como despliegues en la nube.

### Uso

Se utiliza para guardar resultados reproducibles, reconstruir historiales y revisar corridas anteriores. Si no existe configuracion de base, la aplicacion sigue funcionando, aunque sin capacidad de almacenamiento.

## Capa de seguridad de escritura

### Funcionalidad

Esta capa controla quien puede almacenar datos en la base aun cuando la aplicacion sea publica. El sistema permite evaluar libremente, pero puede exigir una contraseña especifica para habilitar escritura.

### Dependencias

Depende del mecanismo de secretos de la aplicacion, del estado de sesion y de funciones de comparacion segura de cadenas.

### Construccion

La capa se ha construido con un modelo de desbloqueo temporal por sesion. La contraseña de escritura no se persiste en la base ni se muestra en interfaz. Cuando el valor correcto se introduce, la sesion marca la escritura como habilitada hasta que se bloquee manualmente o se reinicie la sesion.

Esta aproximacion es ligera, pero eficaz para separar uso publico de capacidad de almacenamiento sin introducir un sistema completo de autenticacion.

### Uso

Se utiliza en despliegues donde interesa que muchas personas puedan probar la herramienta, pero solo algunas tengan permiso para dejar rastro persistente en la base de datos.

## Capa de consulta e historial

### Funcionalidad

La capa de historial recupera evaluaciones recientes y las presenta de forma legible. Su objetivo es transformar datos persistidos en evidencia reutilizable para analisis posterior.

En el caso textual, permite revisar:

- la fuente usada
- los tres textos evaluados
- las metricas obtenidas para cada uno

En el caso visual, permite revisar:

- los textos usados en CLIPScore
- las metricas FID y CLIPScore
- los nombres e identificadores de las imagenes
- una reconstruccion visual a partir de los binarios almacenados

### Dependencias

Depende de la capa de persistencia, de la interfaz y de bibliotecas de imagen para reconstruir vistas previas.

### Construccion

Se ha construido como una lectura acotada de registros recientes con limite fijo por modulo. Esta decision protege la fluidez de la interfaz y evita cargar un numero excesivo de evaluaciones en cada visualizacion.

### Uso

Se utiliza para auditoria, comparacion de resultados y recuperacion de evaluaciones sin necesidad de recalcular metricas.

## Capa de configuracion y despliegue

### Funcionalidad

Esta capa resuelve parametros de entorno necesarios para la operacion del evaluador en local o en despliegue remoto. Su responsabilidad principal es permitir que la aplicacion adapte su comportamiento segun exista o no infraestructura de persistencia y segun se hayan definido mecanismos de seguridad adicionales.

Los parametros mas relevantes son:

- la cadena de conexion a la base de datos
- el modo de cifrado o seguridad de la conexion
- la contraseña de escritura

### Dependencias

Depende del sistema de secretos del entorno de despliegue y de variables de entorno del sistema operativo.

### Construccion

La construccion sigue un criterio de resolucion flexible: primero se intenta leer desde secretos de la plataforma y, si no estan disponibles, se recurre a variables de entorno convencionales. La aplicacion puede degradarse a modo sin persistencia si no existe configuracion suficiente.

### Uso

Se utiliza al arrancar la aplicacion y condiciona la disponibilidad del historial y del guardado de resultados.

## Capa de manejo seguro de errores

### Funcionalidad

Esta capa se encarga de transformar fallos tecnicos en mensajes comprensibles para la persona usuaria sin comprometer la estabilidad del sistema. Afecta tanto a validaciones como a calculo de metricas, lectura de imagenes y operaciones de persistencia.

### Dependencias

Depende de las excepciones generadas por bibliotecas de texto, imagen, modelos y base de datos.

### Construccion

Se ha construido con una logica de captura localizada. En lugar de permitir que fallos de bajo nivel interrumpan toda la aplicacion, cada modulo captura y comunica el error en la interfaz correspondiente. Este enfoque mejora la experiencia de uso y favorece la depuracion.

### Uso

Se usa durante toda la ejecucion, especialmente en escenarios como:

- imagenes corruptas
- entradas vacias
- dependencias faltantes
- base de datos no accesible
- incompatibilidades de modelo o metrica

## Dependencias funcionales entre capas

La relacion entre capas sigue una secuencia clara:

1. La presentacion recoge entradas.
2. La validacion y normalizacion limpian y verifican.
3. El preprocesamiento adapta texto o imagen a las metricas.
4. Las capas de evaluacion calculan los resultados.
5. La compatibilidad de modelos resuelve detalles tecnicos transversales.
6. La persistencia almacena la corrida si corresponde.
7. El historial recupera y representa evaluaciones previas.
8. La seguridad de escritura condiciona la capacidad de guardado.

Esta estructura ayuda a mantener desacopladas las preocupaciones principales del sistema.

## Construccion del sistema completo

La construccion logica del evaluador puede resumirse asi:

1. Definir una interfaz de evaluacion simple y separada por modulos.
2. Implementar validaciones especificas para texto e imagen.
3. Integrar metricas textuales con sus modelos asociados.
4. Integrar metricas visuales con sus requisitos de preprocesamiento.
5. Añadir una capa de compatibilidad para minimizar fallos por versiones o limites de modelo.
6. Incorporar persistencia opcional y modo degradado.
7. Añadir un mecanismo de proteccion de escritura.
8. Exponer historial y visualizacion de corridas recientes.

## Uso recomendado por escenarios

### Comparacion de respuestas generadas

Conviene utilizar el modulo textual cuando se desea medir cual de tres salidas conserva mejor la forma o el significado de una fuente original.

### Revision multimodal exploratoria

Conviene utilizar el modulo visual cuando se desea observar si varias imagenes mantienen coherencia con descripciones textuales y similitud relativa entre si.

### Despliegue publico controlado

Conviene activar persistencia y contraseña de escritura cuando la aplicacion va a ser accesible por terceras personas pero se desea restringir quien puede poblar la base.

### Analisis historico o documental

Conviene utilizar principalmente el modulo textual, ya que ofrece una lectura mas robusta para comparar fidelidad de reescrituras o resúmenes generados respecto de una fuente base.

## Riesgos y limites del diseño

Existen limites que deben explicitarse:

- BLEU no agota la evaluacion semantica
- BERTScore depende del comportamiento del modelo subyacente
- CLIPScore y FID en conjuntos pequeños tienen valor exploratorio, no concluyente
- la primera carga de modelos puede ser costosa
- el almacenamiento de imagenes en la base aumenta el peso del historial

La arquitectura intenta mitigar esos limites mediante mensajes de interpretacion, validaciones y un modelo de degradacion segura, pero no los elimina por completo.

## Criterio de evolucion

La evolucion futura del evaluador deberia respetar varios principios:

- mantener independencia entre modulo textual y modulo visual
- evitar mezclar la interfaz con la logica numerica pesada
- conservar la trazabilidad de toda evaluacion almacenada
- introducir nuevas metricas como extensiones y no como sustituciones opacas
- preservar el modo sin persistencia como capacidad nativa

Con estos criterios, el sistema puede crecer hacia exportacion de resultados, evaluacion por lotes, autenticacion mas fuerte o nuevas metricas sin perder claridad estructural.
