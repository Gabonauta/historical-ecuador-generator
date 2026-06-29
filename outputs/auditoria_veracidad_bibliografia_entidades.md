# Auditoría preliminar de veracidad y bibliografía por entidad

Fecha de auditoría: 13 de mayo de 2026

## Alcance de esta auditoría

- Esta es una **auditoría preliminar de veracidad**: verifica si el núcleo factual de cada entidad es compatible con obras de referencia e instituciones consultadas, pero **no sustituye una edición historiográfica final por campo**.
- El dataset actual **no contiene citas por entidad ni por atributo**; por tanto, incluso cuando una entidad obtiene estado `Alta`, eso **no significa** que todos sus campos estén ya listos para publicación académica sin anotación adicional.
- La cadena `fuente_base` del JSON es genérica y **no cumple** como referencia bibliográfica suficiente.
- Hallazgo importante: el dominio `enciclopediadelecuador.com`, consultado el **13 de mayo de 2026**, resolvió a contenido no relacionado (apuestas/lotería). Por eso en este documento se cita **la obra** de Efrén Avilés Pino como referencia bibliográfica, pero **no** se recomienda ese dominio como fuente web activa hasta su recuperación o verificación.

## Criterio de estado

- `Alta`: el núcleo factual de la entidad es ampliamente compatible con bibliografía estándar o fuentes institucionales sólidas.
- `Media`: la entidad es correcta en términos generales, pero la redacción actual del JSON es demasiado amplia, sintética o interpretativa y requiere citas más específicas.
- `Revisar`: hay problema de modelado, simplificación fuerte o un punto historiográfico que necesita corrección antes de confiar plenamente en la ficha.

## Hallazgos globales

- La expansión a 100 entidades es **útil como base temática**, pero varias fichas agregadas después de la fase inicial son **más genéricas** y necesitan mejor respaldo bibliográfico antes de asumirse como contenido final.
- Los casos con mayor atención requerida son: **Qhapaq Ñan**, **Fundación de Guayaquil**, **Batalla Naval de Jambelí** y **Cultura Valdivia**, no porque sean falsos como tema, sino porque el **modelado** o la **formulación** actual es insuficiente o discutible.
- Los sitios patrimoniales de Quito, Cuenca y Galápagos quedan bien respaldados con UNESCO; los procesos modernos como **dolarización**, **Levantamiento Indígena de 1990** y **Cenepa** necesitan combinar historia general con fuentes institucionales especializadas.

## Fuentes marco

- `B01` Ayala Mora, Enrique. *Resumen de Historia del Ecuador*. 7.ª ed. Quito: Universidad Andina Simón Bolívar / Corporación Editora Nacional, 2022. https://www.uasb.edu.ec/publicacion/resumen-de-historia-del-ecuador/
- `B02` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época aborigen* (vol. 1). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador/
- `B03` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época aborigen II* (vol. 2). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-2/
- `B04` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época colonial I* (vol. 3). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-colonial-i/
- `B05` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época colonial II* (vol. 4). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-colonial-ii/?revistas=Procesos+Revista+ecuatoriana+de+Historia
- `B06` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época colonial III* (vol. 5). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-colonial-iii/?revistas=Procesos+Revista+ecuatoriana+de+Historia
- `B07` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Independencia y período colombiano* (vol. 6). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-independencia-y-periodo-colombiano/
- `B08` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época republicana I: El Ecuador: 1830-1895* (vol. 7). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-republicana-i-el-ecuador-1830-1895/
- `B09` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época republicana III* (vol. 9). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-republicana-iii/
- `B10` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época republicana IV* (vol. 10). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-republicana-iv/
- `B11` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador. Época republicana V* (vol. 11). Quito: UASB / Corporación Editora Nacional, 2018. https://www.uasb.edu.ec/publicacion/nueva-historia-del-ecuador-epoca-republicana-v/
- `B12` Ayala Mora, Enrique, ed. *Nueva Historia del Ecuador* (colección de 15 volúmenes). Catálogo bibliográfico: Biblioteca de la Casa de la Cultura Ecuatoriana. https://biblioteca.casadelacultura.gob.ec/bib/3593
- `B13` Encyclopaedia Britannica. Entrada correspondiente a la entidad. https://www.britannica.com/
- `B14` UNESCO World Heritage Centre. *City of Quito*. https://whc.unesco.org/en/list/2
- `B15` UNESCO World Heritage Centre. *Historic Centre of Santa Ana de los Ríos de Cuenca*. https://whc.unesco.org/en/list/863/
- `B16` UNESCO World Heritage Centre. *Galápagos Islands*. https://whc.unesco.org/en/list/1/
- `B17` UNESCO World Heritage Centre. *Main Andean Road - Qhapaq Ñan*. https://whc.unesco.org/en/qhapaqnan/
- `B18` Avilés Pino, Efrén. *Enciclopedia del Ecuador*, voz correspondiente. Nota de auditoría: el dominio enciclopediadelecuador.com, consultado el 13 de mayo de 2026, resolvió a contenido ajeno y no debe usarse como fuente web activa hasta su recuperación.
- `B19` Armada del Ecuador. *Ceremonia militar en conmemoración al Combate Naval en Jambelí*. 2023. https://www.armada.mil.ec/blog/armada-en-la-comunidad-3/izada-del-pabellon-nacional-y-ceremonia-militar-en-conmemoracion-al-octogesimo-segundo-aniversario-del-combate-naval-en-jambeli-y-glorias-navales-de-la-armada-del-ecuador-907
- `B20` Banco Central del Ecuador. *23 años de dolarización: El camino hacia la estabilidad monetaria*. 2023. https://www.bce.fin.ec/economiatricolor/publicaciones/editoriales/23-anos-de-dolarizacion-el-camino-hacia-la-estabilidad-monetaria/
- `B21` CONAIE. *1990: 30 años del Primer Gran Levantamiento Indígena*. 2020. https://conaie.org/2020/06/05/1990-30-anos-del-primer-gran-levantamiento-indigena/
- `B22` Ministerio de Defensa Nacional. *Intervención para la presentación del documental y del libro por los 21 años de la gesta del Cenepa*. 2016. https://www.defensa.gob.ec/wp-content/uploads/downloads/2016/01/02_26-ENE-2016-Intervencio%CC%81n-Libro-Cenepa.pdf
- `B23` Cancillería del Ecuador. *Ecuador y Perú conmemoran 24 años de los Acuerdos de Paz*. 2022. https://www.cancilleria.gob.ec/2022/10/26/ecuador-y-peru-conmemoran-24-anos-de-los-acuerdos-de-paz/
- `B24` Presidencia de la República del Ecuador. *Proyecto Museo de Carondelet*. 2016. https://www.presidencia.gob.ec/wp-content/uploads/2016/08/k_proyecto_museo_2016.pdf
- `B25` Visit Quito / Quito Turismo. Materiales de contexto sobre Plaza Grande y el centro histórico. Ej.: https://dev.visitquito.ec/historical-sites-of-quito/ y https://visitquito.ec/en/rutas/

## Personajes

| Entidad | Estado | Observación | Bibliografía recomendada |
| --- | --- | --- | --- |
| Eugenio Espejo | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| Carlos Montúfar | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| Manuela Sáenz | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B13` |
| Eloy Alfaro | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B09`, `B18` |
| Antonio José de Sucre | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B13` |
| Atahualpa | Media | La entidad es sólida, pero la narrativa del período inca requiere matiz historiográfico y fuentes específicas por pasaje. | `B01`, `B02`, `B03`, `B13` |
| Rumiñahui | Media | La entidad es sólida, pero buena parte de su narrativa pública mezcla memoria histórica y reconstrucción posterior. | `B01`, `B02`, `B03`, `B18` |
| Huayna Cápac | Media | Correcto a nivel general; conviene reforzar con bibliografía del período inca y del norte andino. | `B01`, `B02`, `B03`, `B13` |
| Sebastián de Benalcázar | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B04`, `B18` |
| Juan Pío Montúfar | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| Manuela Cañizares | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| Simón Bolívar | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B13` |
| Abdón Calderón | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| José Joaquín de Olmedo | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B07`, `B18` |
| Juan José Flores | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| Vicente Rocafuerte | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| Gabriel García Moreno | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| José María Urbina | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| Juan Montalvo | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| Pedro Vicente Maldonado | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B05`, `B18` |
| Manuela Espejo | Media | La entidad es válida, pero el detalle biográfico requiere respaldo puntual porque la documentación es menos abundante que en otras figuras. | `B01`, `B06`, `B18` |
| Mariana de Jesús | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B06`, `B18` |
| Juan de Velasco | Media | Figura correcta, pero su obra e interpretación del pasado deben citarse con más precisión para evitar trasladar sin filtro narrativas coloniales. | `B01`, `B06`, `B18` |
| Luis Vargas Torres | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B09`, `B18` |
| Dolores Cacuango | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B21` |
| Tránsito Amaguaña | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B21` |
| Fernando Daquilema | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| Matilde Hidalgo | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Benjamín Carrión | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| José María Velasco Ibarra | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Isidro Ayora | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Nela Martínez | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Alberto Enríquez Gallo | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Jaime Roldós Aguilera | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B11`, `B18` |
| Oswaldo Guayasamín | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B11`, `B18` |
| Jorge Icaza | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B10`, `B18` |
| Juan León Mera | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B08`, `B18` |
| La Condamine | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B05`, `B13` |
| Alexander von Humboldt | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B13`, `B18` |
| Charles Darwin | Alta | Núcleo biográfico ampliamente reconocido; la ficha actual es usable a nivel general, pero todavía carece de cita puntual por campo. | `B01`, `B13`, `B16` |

## Lugares

| Entidad | Estado | Observación | Bibliografía recomendada |
| --- | --- | --- | --- |
| Plaza Grande | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Iglesia de la Compañía de Jesús | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Centro Histórico de Quito | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Quito | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Guayaquil | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B18` |
| Cuenca | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B15`, `B18` |
| Montecristi | Media | Caracterización general plausible; conviene añadir fuentes locales o patrimoniales si se usará como ficha histórica detallada. | `B01`, `B09`, `B18` |
| Ingapirca | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B03`, `B18` |
| Palacio de Carondelet | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B24` |
| Volcán Pichincha | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B18` |
| Cerro Santa Ana | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B18` |
| Catedral Metropolitana de Quito | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Basílica del Voto Nacional | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B14`, `B25` |
| Chimborazo | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B13`, `B18` |
| Cotopaxi | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B13`, `B18` |
| Riobamba | Media | La entidad es correcta, pero la descripción actual es amplia y conviene respaldarla con historia urbana o regional específica. | `B01`, `B18` |
| Loja | Media | La entidad es correcta, pero la síntesis es panorámica y debe apoyarse con bibliografía urbana y cultural más puntual. | `B01`, `B18` |
| Otavalo | Media | Correcto a gran escala; requiere fuentes más específicas si se quiere sostener afirmaciones sobre continuidad cultural y comercio. | `B01`, `B11`, `B18` |
| Ambato | Media | Caracterización general plausible; hace falta soporte específico para enlazar historia urbana, intelectual y desastre de 1949. | `B01`, `B18` |
| Pumapungo | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B15`, `B18` |
| Cochasquí | Media | El sitio es válido, pero necesita bibliografía arqueológica más especializada si se va a usar con detalle interpretativo. | `B01`, `B03`, `B18` |
| Qhapaq Ñan | Revisar | La entidad existe y es muy importante, pero el modelado como “lugar” es limitado: en rigor es una red vial y ruta cultural serial. | `B01`, `B03`, `B17` |
| Galápagos | Alta | Caracterización general plausible; si se usará en producción conviene añadir fuente patrimonial, urbana o arqueológica más específica. | `B01`, `B13`, `B16` |
| Isla Puná | Media | Entidad correcta; conviene reforzar con bibliografía específica del litoral y de la temprana conquista costera. | `B01`, `B18` |
| Puerto de Guayaquil | Media | Caracterización general plausible, pero la ficha actual es demasiado panorámica para sostener afirmaciones económicas detalladas. | `B01`, `B18` |
| Real Alto | Media | Entidad correcta; se recomienda bibliografía arqueológica específica porque la descripción actual es muy general. | `B01`, `B02`, `B18` |
| Cayambe | Media | La relación con luchas indígenas del siglo XX es plausible, pero conviene citar historia local y movimiento campesino. | `B01`, `B11`, `B21` |
| Mitad del Mundo | Media | Entidad correcta; conviene distinguir mejor entre el símbolo turístico moderno y la historia científica del sitio. | `B01`, `B05`, `B25` |

## Eventos

| Entidad | Estado | Observación | Bibliografía recomendada |
| --- | --- | --- | --- |
| Primer Grito de Independencia | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Batalla de Pichincha | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Guerra Civil Inca | Media | Proceso histórico ampliamente reconocido, pero el resumen exige matices y fuentes especializadas del período inca. | `B01`, `B02`, `B03` |
| Fundación Española de Quito | Media | La entidad es válida, pero el lenguaje de “fundación” simplifica un proceso complejo de conquista, ocupación y trazado urbano. | `B01`, `B04`, `B14` |
| Fundación de Guayaquil | Revisar | Debe revisarse con especial cuidado: la historia fundacional de Guayaquil suele tratarse como un proceso con traslados y reformulaciones. | `B01`, `B04`, `B18` |
| Fundación de Cuenca | Media | La entidad es correcta, pero conviene verificar fecha y contexto con fuentes locales y patrimoniales antes de usarla como ficha cerrada. | `B01`, `B04`, `B15` |
| Rebelión de las Alcabalas | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B04`, `B18` |
| Revolución de los Estancos | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B05`, `B06` |
| Misión Geodésica Francesa | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B05`, `B25` |
| Expedición de Orellana | Media | Correcta en términos generales; falta especificar mejor actores, contexto y cronología en el JSON si se quiere mayor precisión. | `B01`, `B04`, `B18` |
| Masacre del 2 de Agosto de 1810 | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Independencia de Guayaquil | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Independencia de Cuenca | Media | Proceso reconocido, pero conviene reforzar la secuencia local con bibliografía del austro e independencia regional. | `B01`, `B07`, `B15` |
| Batalla de Verdeloma | Media | Evento válido, aunque menos central y con menor presencia en bibliografía general; conviene citar estudios regionales o militares. | `B01`, `B07`, `B18` |
| Batalla de Tarqui | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Separación de la Gran Colombia | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B07`, `B18` |
| Revolución Liberal | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B09`, `B18` |
| Inauguración del Ferrocarril Guayaquil-Quito | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B09`, `B18` |
| Hoguera Bárbara | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B09`, `B18` |
| Revolución Juliana | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B10`, `B18` |
| Guerra Peruano-Ecuatoriana de 1941 | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B10`, `B23` |
| Batalla Naval de Jambelí | Revisar | La conmemoración oficial es clara, pero la interpretación militar y el alcance táctico del combate suelen requerir contraste historiográfico. | `B01`, `B10`, `B19` |
| Protocolo de Río de Janeiro | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B23` |
| Terremoto de Ambato de 1949 | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B10`, `B18` |
| Retorno a la Democracia de 1979 | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B11`, `B18` |
| Guerra de Paquisha | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B11`, `B22` |
| Levantamiento Indígena de 1990 | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B11`, `B21` |
| Guerra del Cenepa | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B11`, `B22` |
| Dolarización del Ecuador | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B20` |
| Incorporación de Galápagos al Ecuador | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B16`, `B13` |
| Rebelión de Daquilema | Alta | El evento o proceso es reconocible en la historiografía general, pero la ficha debe apoyarse en citas específicas antes de publicarse como referencia cerrada. | `B01`, `B08`, `B18` |
| Cultura Valdivia | Revisar | La entidad es histórica y muy relevante, pero no corresponde bien a la categoría “evento”; sería mejor modelarla como cultura o proceso arqueológico. | `B01`, `B02`, `B03` |

## Recomendaciones para el dataset

1. Añadir a cada entidad un campo `bibliografia` o `fuentes` con una lista de referencias específicas por atributo o por ficha.
2. Separar `fuente_base` en dos niveles: `fuentes_secundarias` y `fuentes_institucionales` para no mezclar historia general con fuentes patrimoniales o oficiales.
3. Corregir el modelado de las entidades `Qhapaq Ñan` y `Cultura Valdivia`, porque su naturaleza histórica no encaja bien en el esquema actual de `lugar/evento` sin matices adicionales.
4. Revisar con bibliografía especializada la ficha de `Fundación de Guayaquil`, ya que la narrativa fundacional suele presentarse como un proceso con traslados y no como un acto único y simple.
5. Antes de reconstruir el índice RAG, conviene enriquecer primero el dataset con referencias por entidad para que el grounding semántico no herede descripciones sintéticas sin respaldo explícito.

## Nota final

Este documento sirve como base de saneamiento historiográfico del proyecto. Para una siguiente iteración, lo más recomendable es **inyectar esta bibliografía dentro del JSON por entidad** y dejar una fase posterior de **auditoría por campo** (`resumen`, `descripcion_larga`, `importancia`, relaciones y fechas).
