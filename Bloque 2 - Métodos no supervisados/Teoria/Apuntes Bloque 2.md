

# **Objetivos Bloque 2**

En primer lugar, se explica el **agrupamiento jerárquico**, que incluye dos enfoques: el **aglomerativo** (de abajo hacia arriba) y el **divisivo** (de arriba hacia abajo). Ambos generan un **dendrograma**, una representación visual en forma de árbol (generalmente binario) que muestra la estructura jerárquica de los datos. Este gráfico permite observar los distintos niveles de granularidad y facilita la identificación de grupos o particiones dentro del conjunto de datos.

En la segunda parte se aborda el **algoritmo k-means**, uno de los métodos de clustering más conocidos. Se describe su funcionamiento básico: la **inicialización de los centroides**, el cálculo de **distancias** entre puntos y centroides, y la **asignación** de cada instancia al cluster más cercano. Además, se presentan variantes del método:

- **k-medians**, que utiliza la mediana en lugar de la media para mayor robustez frente a valores atípicos.
- **k-medoids**, que selecciona como centro un punto real del conjunto de datos.  
    También se menciona el **Fuzzy C-means**, una extensión que permite **asignaciones difusas**, es decir, cada instancia pertenece en cierto grado a varios clusters en lugar de solo a uno.

Por último, se introduce el concepto de **pre-clustering**, una técnica previa al proceso de agrupamiento destinada a **reducir el coste computacional** y **acelerar** los algoritmos de clustering. Su objetivo es identificar puntos que están demasiado alejados de determinados grupos y descartarlos en fases posteriores del análisis. El **algoritmo Canopy** ejemplifica esta idea, ya que realiza una **preclasificación rápida y simple** que optimiza tiempo y recursos antes de aplicar métodos de agrupamiento más complejos.

---
# **Agrupamiento jerárquico**

##  **1. Concepto general**

El **agrupamiento jerárquico** es una técnica de **clustering no supervisado** que organiza los datos en **niveles de jerarquía**, representados mediante una estructura tipo **árbol (dendrograma)**.  
Permite **descubrir patrones** y analizar los datos a diferentes niveles de detalle (granularidad).

Existen dos enfoques principales:

- **Algoritmo aglomerativo (bottom-up):** cada instancia comienza siendo su propio grupo y los clusters se van **fusionando progresivamente** hasta formar uno solo que contiene todas las instancias.
- **Algoritmo divisivo (top-down):** parte de un único grupo con todas las instancias y lo **divide sucesivamente** hasta que cada elemento forma su propio cluster.

Ambos son **algoritmos voraces (greedy)**, es decir, eligen en cada paso la **mejor decisión local posible**, sin garantizar que el resultado final sea el **óptimo global**.

##  **2. El dendrograma**

El **dendrograma** es la representación gráfica del proceso jerárquico de agrupamiento.

- Muestra cómo los clusters se fusionan (en el método aglomerativo) o se dividen (en el divisivo).
- Permite **visualizar las relaciones jerárquicas** entre grupos y elegir el nivel de detalle deseado en los resultados.

##  **3. Algoritmos aglomerativos**

Los **algoritmos aglomerativos** comienzan con cada punto como un cluster independiente y los **unen sucesivamente** según su similitud.

#### **a) Criterios principales**

1. **Cálculo de distancia entre puntos:**
    - Se pueden usar diversas métricas (euclídea, estadística, Hamming, etc.).
    - Deben cumplir tres propiedades: **no negatividad**, **simetría** y **desigualdad triangular**.
    - La distancia euclídea es la más utilizada por su simplicidad.

2. **Cálculo de distancia entre grupos (criterio de enlace):**
    - **Enlace simple:** distancia mínima entre puntos de dos clusters.
    - **Enlace completo:** distancia máxima entre puntos de dos clusters.
    - **Enlace medio:** media de todas las distancias entre los puntos de ambos clusters (mayor coste computacional).
#### **b) Funcionamiento general**

1. Inicialmente, cada instancia es un cluster.
2. Se busca el **par de clusters más próximos** según la métrica y el criterio de enlace.
3. Se **fusionan** y se incrementa el nivel del árbol.
4. Se repite hasta que solo queda un cluster con todas las instancias.

##  **4. Algoritmos divisivos**

Los **algoritmos divisivos** operan de manera inversa: parten de un **único cluster con todos los datos** y lo **dividen iterativamente**.

#### **a) Criterios clave**

1. **Condición de parada:**
    - Se puede dividir hasta que **cada instancia quede sola**, o detener antes para **reducir el coste computacional**.
    - La parada puede basarse en criterios como el **número máximo de instancias por grupo** o el **nivel de similitud dentro de una partición**.

2. **Criterio de división:**
    - Existen múltiples formas de dividir un grupo en subgrupos.
    - Una estrategia común es usar un **algoritmo de clustering particional**, como **K-means**, para dividir un cluster en dos o más subconjuntos (por ejemplo, si se desea construir un árbol binario).

#### **b) Funcionamiento general**

1. Se inicia con un único cluster abierto con todas las instancias.
2. Se **divide en m subconjuntos disjuntos** según el criterio de división elegido.
3. Los nuevos clusters se consideran nodos inferiores en el árbol.
4. El proceso continúa hasta que se cumple la **condición de parada**.

##  **5. Ejemplo ilustrativo**

Al aplicar un **algoritmo aglomerativo** con **distancia euclídea** sobre un conjunto de datos numéricos, los resultados varían según el **criterio de enlace**:

- Con **enlace simple**, los clusters se forman uniendo los puntos más próximos.
- Con **enlace completo**, los grupos se fusionan considerando las distancias máximas.

Aunque ambos parten de los mismos datos y métrica de distancia, el tipo de enlace genera **estructuras jerárquicas distintas**.

##  **6. Conclusiones**

El **agrupamiento jerárquico** permite **descubrir patrones** y **organizar los datos de forma jerárquica**, ofreciendo una **visión multinivel** de la estructura interna del conjunto de datos.  
Sus principales **ventajas** son:

- Representación jerárquica interpretable mediante dendrogramas.
- Posibilidad de seleccionar el nivel de detalle deseado.
- Utilidad tanto **descriptiva** como **predictiva**.

Sus **limitaciones** incluyen:
- **Alto coste computacional**, especialmente en grandes volúmenes de datos.
- **Dependencia de la métrica de distancia y del criterio de enlace**.

Sin embargo, el impacto de estos problemas puede mitigarse mediante estrategias como **iniciar el proceso con un preagrupamiento (aglomerativo)** o **limitar la profundidad del árbol (divisivo)**.

---
# **El método k-means y derivados**

## **1. Introducción general**

El **método K-means** es un **algoritmo de clasificación no supervisada** que agrupa un conjunto de datos sin etiquetas conocidas.  
Su objetivo es **dividir n instancias en k grupos o particiones**, de modo que los elementos dentro de un grupo sean más similares entre sí que con los de otros grupos.

Es un **método particional**, no jerárquico: genera particiones planas y requiere que el **número de grupos k** se defina previamente, lo cual puede considerarse una **limitación** o una **ventaja**, según el problema.

## **2. Funcionamiento del algoritmo K-means**

El algoritmo consta de **dos fases principales**:
#### **a) Fase de inicialización**

- Se eligen **k puntos iniciales (centroides)**, que representan los centros de cada grupo.
- Estos puntos pueden elegirse de forma **aleatoria** (lo más común) o **determinista** (según la distribución de los datos).
    - La inicialización aleatoria introduce **variabilidad entre ejecuciones**, lo que puede llevar a **óptimos locales diferentes**, pero también permite repetir el algoritmo varias veces y elegir el mejor resultado.

#### **b) Fase iterativa**

1. **Asignación:** cada punto del conjunto de datos se asigna al **centroide más cercano**, generalmente usando la **distancia euclídea** por su simplicidad y eficacia.
2. **Recalculo de centroides:** para cada grupo, el nuevo centroide se calcula como la **media (mean)** de todos los puntos asignados.
3. El proceso se repite hasta que se cumple una **condición de parada**, es decir, hasta que los centroides dejan de variar o los cambios entre iteraciones son mínimos.

Esta convergencia suele alcanzarse en pocas iteraciones, aunque a veces se detiene antes por razones de **eficiencia computacional**.

## **3. Aspectos clave antes de aplicar K-means**

El texto destaca **cinco factores críticos** para usar correctamente el algoritmo:

1. **Inicialización de centroides:** puede ser aleatoria o basada en la distribución de los datos.
2. **Métrica de distancia:** la más común es la euclídea.
3. **Recalculado de centroides:** mediante la media aritmética de los puntos de cada grupo.
4. **Condición de parada:** cuando las asignaciones dejan de cambiar significativamente.
5. **Selección del número óptimo de grupos (k):**
    - Se suele determinar probando varios valores de k y analizando el **error de agrupamiento** (suma de distancias cuadradas dentro de los grupos).
    - El **método del codo** es una estrategia común: se elige el valor de k donde la reducción del error se estabiliza.

##  **4. Métodos derivados del K-means**

#### **a) K-medians**

- Sustituye la **media** por la **mediana** y usa la **distancia de Manhattan** en lugar de la euclídea.
- Es **más robusto frente a valores atípicos (outliers)** y adecuado para **distribuciones asimétricas**, aunque es **más costoso computacionalmente**.
#### **b) K-medoids**

- Los **centroides** deben ser **puntos reales del conjunto de datos** (no medias o medianas).
- Se eligen los puntos con **mínima disimilitud** respecto al resto del grupo.
- Reduce la sensibilidad a los **ruidos y outliers**, proporcionando resultados más **estables** que K-means.

#### **c) Fuzzy C-means**

- Similar a K-means, pero con **particiones difusas**:
    - Cada instancia no pertenece totalmente a un solo grupo, sino que tiene **grados de pertenencia** a cada partición.
    - Por ejemplo, un punto puede pertenecer a la clase A con un 60% y a la clase B con un 40%.
- Permite **modelar transiciones suaves** entre grupos y **representar la incertidumbre** en la clasificación.

## **5. Conclusiones**

El **K-means** y sus variantes son herramientas muy eficaces para **descubrir patrones y estructuras en datos sin etiquetar**.

- Son **rápidos, sencillos** y funcionan bien con **atributos continuos**.
- En aplicaciones predictivas, los **centroides** se conservan como modelo para clasificar nuevos datos.
- Sin embargo, presentan **limitaciones**: sensibilidad a la **métrica de distancia**, **dependencia del valor de k**, y en algunos casos, **falta de determinismo**.
- Los métodos derivados, especialmente **K-medians**, **K-medoids** y **Fuzzy C-means**, buscan **mayor robustez y flexibilidad**, especialmente frente a datos ruidosos o estructuras no lineales.

---
# **Algoritmo de agrupamiento Canopy**

## **1. Concepto de pre-clustering**

El **pre-clustering** es una fase previa al proceso principal de **clustering** cuyo objetivo es **reducir el coste computacional** del agrupamiento posterior.  
Su función es **organizar los datos de manera preliminar**, de forma que el algoritmo de clustering principal no tenga que calcular todas las distancias posibles entre puntos, sino solo entre aquellos que pertenezcan a los mismos grupos preliminares.

## **2. Idea general del algoritmo Canopy**

El **algoritmo Canopy** implementa este concepto generando **particiones no disjuntas** del conjunto de datos llamadas **canopies**.

- Cada **canopy** es un grupo aproximado de puntos que pueden solaparse (un punto puede pertenecer a varios canopies).
- Si dos puntos **no están en el mismo canopy**, se asume que **están demasiado lejos**, y el algoritmo de clustering posterior **no necesita calcular su distancia**.
- De este modo, se logra **una reducción significativa del número de cálculos** de distancia.


##  **3. Funcionamiento general**

El algoritmo Canopy se compone de **dos fases principales**:

### **Fase 1: Pre-agrupamiento**

1. Se escoge **una métrica sencilla y rápida** (más simple que la usada en el clustering posterior) para comparar todas las instancias del conjunto de datos.
2. Cada instancia debe pertenecer **al menos a un canopy**, y puede pertenecer a varios.
3. Se obtienen los canopies preliminares que servirán como base para el clustering final.

### **Fase 2: Agrupamiento final**

- Se aplica un algoritmo de clustering tradicional (por ejemplo, **K-means** o un método **aglomerativo**) únicamente dentro de los límites de los canopies.
- Esto evita comparar puntos de diferentes canopies y **reduce considerablemente el tiempo de cálculo**.

##  **4. Pseudocódigo y pasos del algoritmo Canopy**

1. **Seleccionar aleatoriamente un punto** del conjunto de datos (llamado centro del canopy, ( d_i )).
2. **Calcular la distancia** entre este punto y todos los demás.
3. Usar **dos umbrales (thresholds)**:
    - **T1**: distancia máxima para pertenecer al canopy.
    - **T2**: distancia menor que define el **núcleo** del canopy (puntos muy cercanos).

4. **Eliminar los puntos del núcleo (T2)** del conjunto de datos para que no sean centros de otros canopies.
5. **Repetir el proceso** con otro punto aleatorio hasta que no queden puntos sin asignar.

El resultado es un conjunto de **canopies superpuestos** que luego se utilizarán en el algoritmo de clustering final, limitando las comparaciones solo entre los puntos dentro de un mismo canopy.

##  **5. Ventajas y aplicaciones**

- **Acelera significativamente** el clustering en **grandes volúmenes de datos** (millones de instancias o muchos atributos).
- **Reduce el número de cálculos de distancia** sin afectar de forma importante la **calidad de los resultados**.
- Es un método **eficiente y práctico** para preprocesar datos antes de aplicar algoritmos más costosos computacionalmente.
## **Conclusión**

El **algoritmo Canopy** es un método de **pre-clustering** que agrupa datos de forma aproximada para **reducir el coste** de los algoritmos de clustering más precisos aplicados después.  
Mediante el uso de **dos umbrales de distancia (T1 y T2)** y **métricas simples**, permite una **clasificación preliminar eficiente**, manteniendo una **calidad de resultados cercana** a la del agrupamiento directo, pero con un **menor tiempo de procesamiento**.
