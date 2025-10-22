
# 1. [Conceptos introductorios de minería de datos](https://materials.campus.uoc.edu/cdocent/PID_00246834/)

## 1.1. ¿Qué es la minería de datos?

La **minería de datos (*Data Mining*)** es el proceso de **extraer conocimiento útil y no trivial a partir de grandes volúmenes de datos**, mediante el uso de técnicas estadísticas, matemáticas y de aprendizaje automático.

El objetivo principal es **transformar datos en información y conocimiento** para la **toma de decisiones**.

## 1.2. Etapas de un proyecto de minería de datos: **Metodología CRISP-DM (*Cross Industry Standard Process for Data Mining*)**

La metodología **CRISP-DM** (propuesta por DaimlerChrysler y SPSS) es el estándar del sector. Propone un proceso **iterativo y basado en calidad total** (ciclo PDCA: _Plan–Do–Check–Act_).  
Un proyecto exitoso requiere planificación, iteración y revisión continua, evitando concentrar esfuerzos solo en el despliegue final.

Las seis fases del proceso son:

|Fase|Descripción|
|---|---|
|**1. Comprensión del negocio**|Definir los objetivos del proyecto desde la perspectiva empresarial.|
|**2. Comprensión de los datos**|Recopilar, describir y explorar los datos iniciales para detectar problemas de calidad o patrones preliminares.|
|**3. Preparación de los datos**|Limpiar, transformar y seleccionar las variables relevantes para el modelado.|
|**4. Modelado**|Aplicar métodos estadísticos o de _machine learning_ para crear modelos predictivos o descriptivos.|
|**5. Evaluación**|Validar los modelos, analizar su rendimiento y comprobar si cumplen los objetivos de negocio.|
|**6. Despliegue**|Implementar el modelo en producción y generar reportes o sistemas automáticos de decisión.|

> 🔹 Las fases no son estrictamente secuenciales: pueden requerir iteraciones y revisiones.

![[Fases CRIPS-DM.png]]
### **1.2.1. Comprensión del negocio**

Se definen los **objetivos del negocio** y se evalúa la situación actual, recursos y limitaciones. Se concretan los objetivos específicos de minería de datos y se elabora un **plan de proyecto** con fases y tareas.

> Resultado: objetivos claros, plan estructurado.

### **1.2.2. Comprensión del negocio**

Busca familiarizarse con los datos: origen, calidad, estructura y problemas. Se analizan sus propiedades, se exploran patrones y se gestiona la calidad (limpieza y resolución de inconsistencias).

> Resultado: diagnóstico completo de la “materia prima”.

### **1.2.3. Preparación de los datos**

Se construye el **dataset final** para modelar, integrando datos relevantes, limpios y consistentes. Se documenta el proceso y se aplican técnicas de **muestreo, selección de atributos** o **reducción de dimensionalidad**.

> Resultado: conjunto de datos listo para modelado.

### **1.2.4. Modelado**

Se seleccionan las **técnicas adecuadas** (árboles, redes neuronales, SVM, k-NN, clustering, etc.) según el problema (clasificación, regresión o segmentación).  
Esta fase itera con la preparación de datos y la evaluación del modelo, ajustando parámetros para mejorar la calidad.

> Resultado: modelo ajustado y validado técnicamente.

### **1.2.5. Evaluación del modelo**

Se analiza si el modelo cumple los **objetivos de negocio** y si existen descubrimientos adicionales relevantes. Los resultados deben verificarse en entornos de prueba antes del despliegue.

> Resultado: ranking de modelos, validación y posibles líneas futuras.

### **1.2.6. Despliegue**

Se implementan los modelos y se planifica su **mantenimiento y seguimiento**. Se define cómo difundir el conocimiento obtenido y cómo medir los beneficios del despliegue.

> Resultado: modelo operativo y conocimiento transferido.

---
# 2. [Tipología de tareas y métodos en procesos de minería de datos](https://materials.campus.uoc.edu/cdocent/PID_00251983/)

## **2.1. Tipos de tareas**

Según el objetivo de nuestro análisis, podemos distinguir entre tres grandes
grupos de tareas, que revisaremos brevemente a continuación.
### **2.1.1. Clasificación**

La **clasificación** consiste en asignar cada instancia a una **clase predefinida** o categoría.  
El atributo objetivo (**variable dependiente**) es **discreto o categórico**.

**Ejemplo:**  
Predecir si un correo electrónico es _spam_ o _no spam_.

**Formalización matemática:**

$$
f: X \rightarrow Y  
$$
donde:

- ( X ) es el conjunto de **variables de entrada** o _features_,
- ( Y = {C<sub>1</sub>, C<sub>2</sub>, ..., C<sub>k</sub>} ) es el **conjunto finito de clases**,
- ( f ) es la **función de clasificación** que asigna cada instancia ( x<sub>i</sub> ) a una clase ( C<sub>j</sub> ).

**Ejemplo visual:**  
Un modelo que aprende a separar puntos rojos (clase 1) y azules (clase 2) en función de sus coordenadas.

![[Clasificacion.png]]

### **2.1.2. Regresión**

En la **regresión**, la variable objetivo es **continua o numérica**.  
El objetivo es estimar una función que relacione los predictores con un valor real.

**Ejemplo:**  
Predecir el **precio de una vivienda** según su tamaño y ubicación.

**Modelo lineal general:**

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i  
$$

donde:

- ( y<sub>i</sub> ): valor real de salida,
- ( x<sub>ij</sub> ): variables predictoras,
- (beta): coeficientes del modelo,
- (varepsilon): término de error.

![[Regresion.png]]

### **2.2.3. Agrupamiento o *clustering***

El **agrupamiento** o _clustering_ busca **descubrir estructuras ocultas** en los datos **sin conocer etiquetas previas**.  

El modelo debe determinar **cuántos grupos existen** y **qué instancias pertenecen a cada uno**.

**Ejemplo:**  
Segmentar clientes según sus hábitos de consumo sin conocer previamente los grupos.

**Formalización:**

$$
\text{Asignar } n \text{ instancias } {x_1, ..., x_n} \text{ a } k \text{ clusters } {C_1, ..., C_k}  
$$  
minimizando una función de disimilitud:

$$
J = \sum_{i=1}^k \sum_{x_j \in C_i} ||x_j - \mu_i||^2  
$$

donde ($\mu$<sub>i</sub> ) es el centroide del cluster (i).

![[clustering.png]]

## **2.2. Tipos de métodos o algoritmos**

Respecto a la tipología de métodos o algoritmos, discernimos entre dos grandes grupos:
### **2.2.1. Supervisados**

Los **métodos supervisados** requieren **datos etiquetados**, es decir, cada instancia tiene una clase o valor objetivo conocido.  
Incluyen:

- **Clasificación** → atributo objetivo **categórico**
- **Regresión** → atributo objetivo **numérico**

**Ejemplos:**  
Árboles de decisión, regresión lineal, _Support Vector Machines (SVM)_, redes neuronales.

![[Aprendizaje supervisado.png|500]]
### **2.2.2. No supervisados**

En los **métodos no supervisados**, el conjunto de datos **no tiene etiquetas**.  
Buscan estructuras internas o patrones de similitud.

![[Aprendizaje no supervisado.png|500]]

**Subtipos:**
- **Jerárquicos:**
    - Construyen una **estructura arborescente (dendrograma)**.
    - Estrategias _bottom-up_ (aglomerativas) o _top-down_ (divisivas).

- **Particionales:**
    - Dividen el conjunto de datos en **k grupos disjuntos**, sin jerarquía.
    - Ejemplo: _k-means_.

![[Tipologias de metodos de agrupamiento.png]]

Una misma técnica puede resolver distintos tipos de problemas.

> **Tabla resumen:** Árboles de decisión, redes neuronales, SVM, k-means, métodos probabilísticos y k-NN pueden aplicarse en diferentes combinaciones de tareas supervisadas o no supervisadas.


---

#  3. [Datos de entrenamiento y test en minería de datos](https://materials.campus.uoc.edu/cdocent/PID_00251987/)

## **3.1. Conjuntos entrenamiento y test**

Sirven para construir y evaluar modelos, evitando el **sobreentrenamiento (overfitting)**.  
El conjunto de **entrenamiento** entrena el modelo; el de **test** evalúa su capacidad de generalización sobre datos nunca vistos.  

> Es esencial que ambos conjuntos sean **disjuntos**.

![[proceso de entrenamiento-test.png]]

Habitualmente, los juegos de datos de entrenamiento y de test suelen ser extracciones
aleatorias del juego de datos inicial. En función del número de
datos disponibles, existen diferentes técnicas para la construcción de los conjuntos
de entrenamiento y de prueba.
## **3.2. Generación de conjuntos de entrenamiento y test**

### **3.2.1. _Leave-One-Out (LOO)_**

Para un conjunto de ( n ) instancias:
- Se entrena con ( n-1 ) y se evalúa con 1.
- Se repite ( n ) veces.
- El error final es la media de los ( n ) errores.

$$  
E = \frac{1}{n} \sum_{i=1}^{n} e_i  
$$
> Útil con pocos datos.
### **3.2.2. _Leave-p-Out (LpO)_**

- Similar al anterior, pero dejando ( p ) instancias para test.
- Computacionalmente costoso para grandes ( n ).

#### 3. _k-Fold Cross Validation_

- Divide el conjunto en ( k ) particiones disjuntas.
- Se entrena con ( k-1 ) y se prueba con la restante.
- Se repite ( k ) veces.

$$
E = \frac{1}{k} \sum_{i=1}^{k} e_i  
$$

> Valores comunes: ( k = 5 ) o ( k = 10 ).  
> Cuando ( k = n ), equivale a _Leave-One-Out_.

![[K-fold cross validation.png]]

Dividir los datos es fundamental para **evaluar la capacidad de generalización del modelo** y **evitar el sobreentrenamiento**.

## **3.3. *Overfitting* o sobreentrenamiento**

El **sobreentrenamiento (overfitting)** ocurre cuando un modelo se ajusta **demasiado** a los datos de entrenamiento, capturando **ruido o variabilidad aleatoria**.

![[overfitting.png]]

Esto provoca un bajo error de entrenamiento, pero un alto error en nuevos datos.

**Diagnóstico:**
- Error de entrenamiento ↓
- Error de test ↑

**Soluciones comunes:**
- Aumentar los datos de entrenamiento
- Regularización
- Validación cruzada
- _Early stopping_ (detener el entrenamiento antes del sobreajuste)

En la gráfica de la izquierda se observa cómo el error de entrenamiento disminuye progresivamente con el número de iteraciones, lo cual es un comportamiento esperado en cualquier proceso de aprendizaje. Sin embargo, en la gráfica de la derecha, que muestra la precisión sobre el conjunto de prueba, se aprecia que a partir de la iteración 50 el modelo deja de mejorar: la precisión se estabiliza y oscila en torno al 70-80 %.

Esto indica que, aunque el modelo sigue reduciendo el error en los datos de entrenamiento, no está mejorando su capacidad para predecir datos nuevos. En otras palabras, está **sobreajustándose**: aprende demasiado bien los patrones específicos del conjunto de entrenamiento, pero pierde capacidad de **generalización**. Esta situación, conocida como **sobreentrenamiento**, es algo que debemos evitar para obtener modelos realmente útiles y robustos.

![[evaluacion sobremuestreo.png]]

## 3.4. Conjunto de validación

El **conjunto de validación** es un tercer subconjunto usado para **ajustar hiperparámetros** o **comparar modelos**.

**Proceso típico:**
1. Entrenar con el conjunto de entrenamiento.
2. Evaluar distintas configuraciones sobre el conjunto de validación.
3. Escoger el modelo óptimo.
4. Medir el rendimiento final con el conjunto de test.

> Se recomienda usar tres conjuntos distintos: entrenamiento, validación y test, garantizando independencia para evaluar la **capacidad de generalización**.

---

# 4. [Evaluación de resultados en procesos de minería de datos](https://materials.campus.uoc.edu/cdocent/PID_00251985/)

La evaluación depende del tipo de problema: **clasificación**, **regresión** o **agrupamiento**.

El objetivo es cuantificar el grado o valor de «bondad» de la solución encontrada, permitiendo la comparación entre distintos métodos sobre los mismos conjuntos de datos.
## **4.1. Modelos de clasificación**

Se evalúan mediante **matrices de confusión** y métricas derivadas:
### **4.1.1. Matriz de confusión**

La **matriz de confusión** resume las predicciones del modelo comparando clases reales y predichas.


|                         | Predicción positiva     | Predicción negativa     |
| ----------------------- | ----------------------- | ----------------------- |
| **Clase positiva real** | Verdadero Positivo (VP) | Falso Negativo (FN)     |
| **Clase negativa real** | Falso Positivo (FP)     | Verdadero Negativo (VN) |

A partir de la matriz de confusión, definimos un conjunto de métricas que permiten cuantificar la bondad de un modelo de clasificación:

$$
\text{Exactitud (Accuracy)} = \frac{VP + VN}{VP + VN + FP + FN}  
$$

$$
\text{Precisión (Precision)} = \frac{VP}{VP + FP}  
$$

$$
\text{Exhaustividad (Recall)} = \frac{VP}{VP + FN}  
$$

$$
F1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}  
$$


**Interpretación:**
- _Accuracy_: porcentaje total de aciertos.
- _Precision_: fiabilidad de las predicciones positivas.
- _Recall_: capacidad de detectar verdaderos positivos.
- _F1_: media armónica entre precisión y recall.

 En algunos problemas nos interesarán o el error de clasificación o exactitud general del modelo ,sino evitar algún tipo de errores específicos.
 
**Ejemplo:**
- Supongamos que nuestro modelo está intentando predecir si existe la presencia de tumor o cáncer a partir de unos datos de entrenamiento. Lo que nos interesa, y el objetivo del modelo, es que no se den los casos de falsos negativos, en los cuales una instancia que debería considerarse positiva, ha sido dada como negativa. El caso contrario, aunque no nos interesa, no es tan grave. Si un paciente que no tiene cáncer, el sistema lo da como sí, un segundo análisis lo descartaría. Lo que nos interesa es maximizar este subgrupo en el cual  no haya ningún caso positivo de tumor que el sistema descarte y dé como negativo. En estos casos se usa la tasa de verdaderos positivos o tasa de verdaderos negativos.

### **4.1.2. Curvas ROC (Receiver Operating Characteristic)**

La **curva ROC** (_Receiver Operating Characteristic_) representa la relación entre:
- Eje X: tasa de falsos positivos (TFP)
- Eje Y: tasa de verdaderos positivos (TVP)

A partir de la tasa de falsos positivos (TFP) y verdaderos positivos (TVP) ,se construyen las curvas ROC. Estas curvas permiten el cálculo del área bajo esta curva, que nos da un valor de bondad o de optimización del modelo en general.

![[curva roc.png]]

La situación ideal de un modelo, vemos la figura izquierda, es cuando los verdaderos positivos alcanzan el nivel máximo mientras que los falsos positivos el nivel mínimo. Esta es una situación ideal a la que no siempre se puede llegar. En la figura del medio vemos una curva en donde el área está alrededor del 0.8. Esta es una situación más habitual, más real. La figura de la derecha presenta un ejemplo en donde esta área, que corresponde a la mitad, equivale a un modelo aleatorio. Si nuestro modelo se acerca a esta situación, estamos en un modelo de bajo entrenamiento y no estamos aportando mucho más que un modelo aleatorio de generación.

El **Área Bajo la Curva (AUC)** mide el rendimiento general:
- ( AUC = 1 ) → modelo perfecto
- ( AUC = 0.5 ) → modelo aleatorio

A modo de guía para interpretar las curvas ROC se han establecido los siguientes
intervalos para los valores de AUC:
- [0,5,0,6): Test malo
- [0,6,0,75): Test regular
- [0,75,0,9): Test bueno
- [0,9,0,97): Test muy bueno
- [0,97,1): Test excelente

## **4.2. Modelos de regresión**

En el caso de los modelos de regresión, básicamente, usamos un conjunto, distintas métricas, que nos permitan comparar el valor obtenido con el valor de la clase, el valor del conjunto de datos etiquetado.

Para problemas con variables continuas, se evalúa la **diferencia entre valores reales y predichos**.

| Métrica                                    | Fórmula                                                                 | Interpretación                                     |
| ------------------------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------- |
| **Error Absoluto Medio (MAE)**             | MAE = $\frac{1}{n}\sum_{i=1}^{n}$                                       | $y_i - \hat{y}_i$                                  |
| **Error Cuadrático Medio (MSE)**           | MSE = $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$                    | Penaliza más los errores grandes                   |
| **Raíz del Error Cuadrático Medio (RMSE)** | RMSE = $\sqrt{MSE}$                                                     | Error medio en las mismas unidades que la variable |
| **Coeficiente de determinación (R²)**      | $R^2$ = $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ ) | Proporción de la varianza explicada                |

## **4.3. Modelos de agrupamiento**

Dado que no existen etiquetas, se usan **métricas de calidad**, evaluando el rendimiento según la **cohesión interna** y la **separación entre grupos**.

### **4.3.1. Métricas de calidad por partición**

Por un lado están las métricas de calidad por partición, que evalúan, dentro de cada partición, la calidad esperada. Por ejemplo, una de las más básicas sería el diámetro. Básicamente, mide la distancia máxima entre dos instancias de una misma partición.

- **Diámetro (D):**  compactación interna.
$$
D(C_i) = \max_{x, y \in C_i} d(x, y)  
$$
Claramente, las particiones compactas, es decir, con un diámetro pequeño,
suelen ser las más deseadas. Cuando aparecen particiones con diámetros muy
dispares, puede ser un signo de que debemos ampliar (o reducir) el número de
particiones.

> Cuanto menor, mayor cohesión.

En la figura de la derecha vemos la distancia máxima que hay entre los dos puntos rojos o verdes. Esto sería el diámetro. Cuanto menor sea este valor, nos indica mayor cohesión en estas particiones, lo cual es, lógicamente, deseable.

![[distancias.png|300]]

Otra medida interesante es la separación, que mide la mínima distancia entre dos partición eso dos clusters. En este caso, nos interesa maximizar esta medida, porque nos indica que las dos particiones están separadas entre ellas.

- **Separación (S):**  distancia entre grupos.
$$  
S(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)  
$$  
Cuanto mayor, mejor separación entre grupos.

### 4.3.2. Calidad general

A partir de aquí hay otro nivel, otra categoría: las métricas de calidad general. En lugar de darnos un valor para cada partición, nos dan un valor general para todo el modelo, lo cual es deseable porque es más fácil la comparación entre distintos modelos. Aquí podemos destacar el índice Dunn el índice C.

- **Índice Dunn:**  compara mínima separación y máxima dispersión (valores altos = buena calidad).
$$
D = \frac{\min_{i \ne j} S(C_i, C_j)}{\max_k D(C_k)}  
$$ 
Valores altos indican mejor particionado.

### **4.3.3. Calidad externa**

Finalmente, hay otro tipo de medidas de calidad, que llamamos de calidad externa, que aunque estemos en un modelo no supervisado, donde no tenemos datos etiquetados, donde no tenemos variables objetivo o clase, suponemos que de alguna forma podemos obtener esta información externamente y podemos obtener una posible clasificación que luego comparamos con el resultado de nuestro modelo. Y definimos algo similar a la matriz de confusión.

En este caso, definimos los verdaderos positivos como aquellas instancias que forman parte de la misma partición, según esta evaluación externa, y también forman parte de la misma partición en nuestro modelo de agrupamiento.

De forma similar, definimos los verdaderos negativos, falsos positivos y falsos negativos. A partir de aquí, definimos el índice Rand como se muestra aquí. Básicamente es los verdaderos positivos y negativos sobre el total de posibles pares de combinaciones.

- **Índice Rand:** proporción de pares correctamente agrupados (0 = discordancia total, 1 = coincidencia perfecta).
$$
R = \frac{VP + VN}{VP + VN + FP + FN}  
$$
Evalúa la concordancia entre el agrupamiento generado y la clasificación real.


---

# **Conclusiones**

- La **minería de datos** es un proceso sistemático que combina técnicas estadísticas y de _machine learning_ para extraer conocimiento.
- La **metodología CRISP-DM** proporciona una guía estructurada y cíclica.
- Es crucial entender la **tipología del problema** (clasificación, regresión, clustering) para elegir el algoritmo adecuado.
- La **evaluación rigurosa** mediante métricas apropiadas y particiones correctas garantiza la fiabilidad del modelo.

# **Glosario**
-  **aprendizaje automático o machine learning**: Conjunto de técnicas, métodos y algoritmos que permiten a una máquina aprender de manera automática en base a experiencias pasadas.

- **aprendizaje no supervisado** Aprendizaje que se basa en el descubrimiento de patrones,
características y correlaciones en los datos de entrada, sin intervención externa.

- **aprendizaje supervisado** Aprendizaje que se basa en el uso de un componente externo,
llamado supervisor, que compara los datos obtenidos por el modelo con los datos esperados
por este, y proporciona retroalimentación al modelo para que vaya ajustándose y mejorando
las predicciones.
# Recursos adicionales

-   Python Machine Learning: Learn How to Build Powerful Python Machine Learning Algorithms to Generate Useful Data Insights with This Data Analysis Tutorial
- Data mining and knowledge discovery (Online)
- Data Mining Algorithms: Explained Using R
- Introduction to machine learning with Python : a guide for data scientists / Andreas C. Müller and Sarah Guido.
- Mastering machine learning with scikit-learn : apply effective learning algorithms to real-world problems usung scikit-learn / Gavin Hackeling.
-   Advanced machine learning with Python : solve challenging data science problems by mastering cutting-edge machine learning techniques in Python / John Hearty.
- [Repositorio de la asignatura](https://gitlab.com/UOC/eimt/datascience/MUCD/AA)

---
---

# Preparación de los datos

## **Introducción**

El módulo aborda las **técnicas fundamentales de preprocesamiento de datos**, una fase esencial en la creación de modelos de aprendizaje automático. Su propósito es **limpiar, transformar y adecuar los datos** para mejorar su calidad, eliminar ruido, corregir inconsistencias y asegurar su utilidad. La calidad de los datos determina el rendimiento del modelo (principio _GIGO: Garbage In, Garbage Out_).

---

## **Objetivos**

El texto busca que el lector:

1. Comprenda las técnicas de **limpieza, discretización, normalización y estandarización**.
2. Conozca los métodos de **reducción de la dimensionalidad**.
3. Aprenda las técnicas de **extracción y selección de atributos**.
4. Entienda cómo **tratar conjuntos desbalanceados** de datos.

---

## **1. Contextualización**

El preprocesamiento es clave para garantizar datos válidos y consistentes.  

Se usa la notación ( D_{n,m} ), donde:
- ( n ): número de instancias.
- ( m ): número de atributos.
- ( a_{i,j} ): valor del atributo ( j ) de la instancia ( i ).
- ( c_i ): clase o etiqueta (solo en aprendizaje supervisado).    

Los datos sin etiqueta (no supervisados) mantienen la misma estructura pero sin ( c_i ).

---

## **2. Conceptos preliminares**

### **2.1. Análisis estadístico**

Introduce los fundamentos de la estadística aplicados a la minería de datos: distribución de una muestra, media, mediana, varianza, desviación estándar y distribución normal.  

Se explica cómo la **distribución normal estándar (Z)** permite comparar valores distintos mediante la conversión ( $Z = (X - \mu)/\sigma$ ).

### **2.2. Métricas de distancias o similitud**

Define las propiedades de una función de distancia (no negatividad, simetría y desigualdad triangular).  

Ejemplos:
- **Euclídea**: más usada en espacios numéricos.
- **Manhattan**, **Chebyshev**, **Minkowski** y **coseno**: aplicables según el tipo de datos.  

 La elección de la métrica es crucial para los algoritmos de segmentación y clasificación.

### **2.3. Entropía y ganancia de información**

La **entropía** mide el grado de desorden o incertidumbre ($H(X) = -\sum p(x_i) \log_2 p(x_i)$).  

La **ganancia de información (IG)** evalúa cuánto reduce un atributo la entropía total:  
 $IG(X,a) = H(X) - H(X|a)$.  
 
Un atributo con alta IG es más relevante para la predicción.

---

## **3. Preparación de los datos**

### **3.1. Limpieza de datos**

Detecta y corrige errores, valores ausentes, inconsistentes o atípicos (_outliers_).  

La integración de múltiples fuentes puede generar incoherencias que deben resolverse.
### **3.2. Normalización y estandarización**

Ambas transforman variables numéricas para mejorar la precisión de los modelos:

- **Normalización**: ajusta valores a un rango fijo (ej. [0,1]).
- **Estandarización**: centra los datos en media 0 y varianza 1.  
    
Evitan que una variable con gran escala domine sobre las demás.
### **3.3. Discretización**

Convierte variables continuas en categóricas.  

Métodos:
- **Basados en intervalos**: igual amplitud o igual frecuencia.
- **Chi-Merge**: usa el estadístico χ² para unir o dividir intervalos.
- **Basados en entropía**: buscan particiones homogéneas minimizando la entropía interna.

### **3.4. Reducción de la dimensionalidad**

Permite manejar datasets grandes reduciendo columnas (atributos) o filas (casos).

- **Reducción de atributos**:
    - **Selección de atributos**: identifica los más relevantes (por significación o correlación).
    - **Creación de atributos**: combina variables (p. ej., **PCA**: análisis de componentes principales).
- **Reducción de casos**: se basa en muestras representativas y distribuciones muestrales.

---

## **4. Extracción y selección de atributos**

### **4.1. Selección de atributos**

Busca eliminar variables redundantes o irrelevantes para reducir la complejidad.  

Se emplean:
- **Métodos filtro**: usan medidas estadísticas (correlación, información mutua).
- **Métodos envolventes (wrapper)**: evalúan subconjuntos con modelos predictivos.
- **Métodos integrados**: el modelo selecciona atributos automáticamente (ej. LASSO).

### **4.2. Extracción de atributos**

Crea nuevas características a partir de las existentes:

- **PCA (Análisis de Componentes Principales)**: combina variables para maximizar varianza y reducir error cuadrático.
- **SVD (Descomposición en Valores Singulares)**: descompone matrices para representar la información en menos dimensiones.
- **NMF (Factorización de Matrices No Negativas)**: descompone datos no negativos en factores interpretables.

---

## **5. Conjuntos desbalanceados de datos**

### **5.1. Definición y problemas**

Ocurre cuando una clase tiene muchas más instancias que otra (ej. fraude bancario 0.1%).  

Consecuencias:
- Sesgo hacia clases mayoritarias.
- Baja precisión/sensibilidad en clases minoritarias.
- Riesgo de **sobreajuste** y **evaluaciones engañosas** (la exactitud puede parecer alta sin serlo).

### **5.2. Técnicas de sobremuestreo**

Aumentan el número de ejemplos de la clase minoritaria:
- **Duplicación aleatoria** o **síntesis artificial (SMOTE)**.

### **5.3. Técnicas de submuestreo**

Reducen el número de ejemplos de la clase mayoritaria:
- **Eliminación aleatoria** o selección basada en distancias.

### **5.4. Métricas de evaluación**

Cuando los datos están desbalanceados, se prefieren métricas como:
- **Precisión**, **sensibilidad (recall)**, **F1-Score**, y **ROC-AUC**.  
    Se recomienda **validación cruzada estratificada** para conservar la proporción de clases.

---

## **Resumen final**

El módulo presenta una visión integral del **preprocesamiento de datos**: desde la comprensión estadística y las métricas de similitud hasta la limpieza, normalización, discretización y reducción de la dimensionalidad.  

También aborda la **selección/extracción de atributos** y las **estrategias para manejar conjuntos desbalanceados**, destacando su impacto directo en la calidad y fiabilidad de los modelos de aprendizaje automático.
