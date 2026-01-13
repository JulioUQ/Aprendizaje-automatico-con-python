# Análisis detallado del artículo XGBoost

## 1. INTRODUCTION (Introducción)

### ¿Qué dice?

Esta sección establece el **contexto y la importancia** de XGBoost.

**Puntos clave:**

1. **El problema:** Gradient tree boosting era efectivo pero **no escalaba bien** a grandes volúmenes de datos
2. **El impacto:** En 2015, **17 de 29 soluciones ganadoras en Kaggle** usaron XGBoost
3. **Aplicaciones:** Detección de spam, publicidad, detección de fraude, física de partículas, etc.

**Factores de éxito mencionados:**

- Modelos estadísticos efectivos
- Sistemas de aprendizaje escalables

**Por qué es importante:**

> "XGBoost corre más de 10 veces más rápido que soluciones populares existentes y escala a billones de ejemplos"

### Contribuciones principales del paper:

1. ✅ Sistema end-to-end altamente escalable
2. ✅ Weighted quantile sketch teóricamente justificado
3. ✅ Algoritmo sparsity-aware para datos sparse
4. ✅ Estructura cache-aware para out-of-core learning

---

## 2. TREE BOOSTING IN A NUTSHELL

Esta es la **sección más importante teóricamente**. Voy a explicarla paso a paso.

### 2.1 Regularized Learning Objective (Función objetivo regularizada)

**El modelo ensemble:**

```
ŷᵢ = φ(xᵢ) = Σ fₖ(xᵢ)    [Ecuación 1]
              k=1 hasta K
```

**Interpretación:**

- Tenemos K árboles
- La predicción final es la **suma** de las predicciones de cada árbol
- Cada árbol fₖ mapea un ejemplo a una hoja con un peso

**Función objetivo:**

```
L(φ) = Σ l(ŷᵢ, yᵢ) + Σ Ω(fₖ)    [Ecuación 2]
       i              k

donde Ω(f) = γT + ½λ||w||²
```

**Desglose:**

- **Primer término:** Error de predicción (loss)
- **Segundo término:** Regularización (penaliza complejidad)
    - `γT`: Penaliza número de hojas T
    - `½λ||w||²`: Penaliza magnitud de pesos (L2 regularization)

**¿Por qué regularizar?**

- Evita overfitting
- Favorece modelos simples y predictivos
- Si λ = 0 y γ = 0 → volvemos a gradient boosting tradicional

---

### 2.2 Gradient Tree Boosting

**Proceso iterativo:**

En la iteración t, queremos añadir un nuevo árbol fₜ que minimize:

```
L⁽ᵗ⁾ = Σ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)) + Ω(fₜ)
       i=1 hasta n
```

**Aproximación de segundo orden (Taylor):**

```
L⁽ᵗ⁾ ≈ Σ [l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)
       i
```

**Donde:**

- **gᵢ = ∂L/∂ŷᵢ**: Primera derivada (gradiente)
- **hᵢ = ∂²L/∂ŷᵢ²**: Segunda derivada (Hessian)

**Eliminando constantes:**

```
L̃⁽ᵗ⁾ = Σ [gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)    [Ecuación 3]
       i
```

**Expandiendo la regularización:**

```
L̃⁽ᵗ⁾ = Σ [(Σ gᵢ)wⱼ + ½(Σ hᵢ + λ)wⱼ²] + γT    [Ecuación 4]
       j  i∈Iⱼ      i∈Iⱼ
```

**Donde Iⱼ** = conjunto de ejemplos en la hoja j

---

**Cálculo del peso óptimo de cada hoja:**

Para una estructura de árbol fija q(x), el peso óptimo de la hoja j es:

```
wⱼ* = - (Σ gᵢ) / (Σ hᵢ + λ)    [Ecuación 5]
        i∈Iⱼ     i∈Iⱼ
```

**Interpretación:**

- Numerador: suma de gradientes en esa hoja
- Denominador: suma de hessianos + regularización

**Valor de loss óptimo (score de calidad del árbol):**

```
L̃⁽ᵗ⁾(q) = -½ Σ (Σ gᵢ)² / (Σ hᵢ + λ) + γT    [Ecuación 6]
              j  i∈Iⱼ      i∈Iⱼ
```

**Este score se usa para evaluar qué tan buena es una estructura de árbol.**

---

**Ganancia al hacer un split:**

Cuando dividimos un nodo en hijo izquierdo (IL) y derecho (IR):

```
Lsplit = ½[(Σ gᵢ)²/(Σ hᵢ+λ) + (Σ gᵢ)²/(Σ hᵢ+λ) - (Σ gᵢ)²/(Σ hᵢ+λ)] - γ
          i∈IL    i∈IL      i∈IR    i∈IR      i∈I     i∈I
```

**Interpretación:**

- Si **Lsplit > 0** → Vale la pena dividir
- Si **Lsplit < 0** → No dividir (penalización γ es mayor que la ganancia)

**El término -γ** penaliza añadir complejidad (una hoja nueva)

---

### 2.3 Shrinkage and Column Subsampling

**Dos técnicas adicionales para evitar overfitting:**

**1. Shrinkage (η):**

```
Predicción nueva = Predicción anterior + η × nuevo_árbol
```

- η típicamente entre 0.01 y 0.3
- Reduce influencia de cada árbol individual
- Similar a learning rate en SGD

**2. Column Subsampling:**

- En cada árbol, usar solo un subconjunto aleatorio de features
- Típicamente √m o log₂(m) features
- Inspirado en Random Forest
- **Ventaja extra:** acelera el entrenamiento

---

## 3. SPLIT FINDING ALGORITHMS

Esta sección explica **cómo encontrar el mejor split** en cada nodo.

### 3.1 Basic Exact Greedy Algorithm

**Algoritmo:**

```
Para cada feature k:
    Ordenar datos por valores de feature k
    Para cada valor posible:
        GL = suma de gradientes a la izquierda
        GR = suma de gradientes a la derecha
        Calcular score usando Ecuación 7
    Seleccionar split con mejor score
```

**Características:**

- ✅ Encuentra el mejor split posible
- ❌ Computacionalmente costoso
- ❌ Requiere ordenar datos (O(n log n))
- ❌ No funciona si datos no caben en memoria

---

### 3.2 Approximate Algorithm

**Problema:** Con datos muy grandes, no es factible enumerar todos los splits posibles.

**Solución:** Proponer **candidatos de split** basados en percentiles.

**Dos variantes:**

**1. Global:**

- Propone candidatos al inicio
- Usa los mismos candidatos para todos los niveles del árbol
- Necesita más candidatos

**2. Local:**

- Re-propone candidatos después de cada split
- Refina candidatos en cada nivel
- Necesita menos candidatos
- Mejor para árboles profundos

**Resultados (Figura 3):**

- Local con eps=0.3 necesita menos buckets que global
- Global puede ser tan preciso como local con suficientes candidatos
- Ambos alcanzan precisión similar al exact greedy

---

### 3.3 Weighted Quantile Sketch

**Problema:** Necesitamos encontrar percentiles en **datos ponderados**.

**¿Por qué ponderados?**

Recordemos la ecuación 3:

```
L̃⁽ᵗ⁾ = Σ [gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)] + Ω(fₜ)
```

Esto se puede reescribir como:

```
L̃⁽ᵗ⁾ = Σ ½hᵢ(fₜ(xᵢ) - gᵢ/hᵢ)² + Ω(fₜ) + constante
```

**¡Es un problema de squared loss ponderado!**

- Labels: gᵢ/hᵢ
- Pesos: hᵢ

**Función de rango ponderado:**

```
rₖ(z) = (1 / Σhᵢ) × Σ hᵢ    donde xᵢₖ < z
```

**Objetivo:** Encontrar candidatos {s₁, s₂, ..., sₗ} tales que:

```
|rₖ(sⱼ) - rₖ(sⱼ₊₁)| < ε
```

**Interpretación:** Distribuir candidatos uniformemente según peso acumulado.

**Innovación de XGBoost:**

- Primer algoritmo de quantile sketch para datos ponderados
- Con garantías teóricas probables
- Detalles matemáticos en el Apéndice

---

### 3.4 Sparsity-aware Split Finding

**Problema:** En datos reales es común tener:

- Valores faltantes (missing values)
- Muchos ceros (ej: después de one-hot encoding)
- Features sparse

**Solución de XGBoost:**

Añadir una **dirección por defecto** en cada nodo:

```
Si feature está presente:
    Usar valor para decidir izquierda/derecha
Si feature falta (missing):
    Ir a la dirección por defecto
```

**¿Cómo aprende la dirección por defecto?**

**Algoritmo:**

1. Probar enviar missing values → derecha
2. Probar enviar missing values → izquierda
3. Elegir la que da mejor score

**Solo visita entradas no-missing (Iₖ):**

```
Complejidad = O(número de entradas no-missing)
```

**Resultado (Figura 5):**

- Sparsity-aware: **50x más rápido** que versión naive
- Esencial para datos con one-hot encoding

---

## 4. SYSTEM DESIGN

Esta sección describe las **optimizaciones a nivel de sistema**.

### 4.1 Column Block for Parallel Learning

**Problema:** Ordenar datos es la parte más costosa.

**Solución: Block Structure**

**Estructura:**

- Datos almacenados en **bloques en memoria**
- Cada bloque en formato **CSC** (Compressed Sparse Column)
- **Cada columna pre-ordenada** por valor de feature

**Ventajas:**

1. **Se ordena solo una vez** (antes del entrenamiento)
2. **Reutilizable** en todas las iteraciones
3. **Permite paralelización:** procesar columnas en paralelo
4. **Facilita column subsampling**

**Proceso (Figura 6):**

```
Datos originales → Bloques con columnas ordenadas
                ↓
    Para cada split, escaneo lineal por columna
                ↓
         Encuentra mejor split
```

**Análisis de complejidad:**

**Sin block structure:**

- Exact greedy: O(Kd||x||₀ log n)
- Approximate: O(Kd||x||₀ log q)

**Con block structure:**

- Exact greedy: O(Kd||x||₀ + ||x||₀ log n)
- Approximate: O(Kd||x||₀ + ||x||₀ log B)

**Ahorro:** Factor log n o log q (significativo cuando n es grande)

---

### 4.2 Cache-aware Access

**Problema:** Cache misses

Cuando procesamos columnas ordenadas por feature value, accedemos a los gradientes **en orden no continuo** (por row index).

**Patrón problemático (Figura 8):**

```
Columna ordenada: [feature_val₁, feature_val₂, ...]
                     ↓              ↓
Indices de fila:   [100,         37,  ...]
                     ↓              ↓
Gradientes:      g[100],        g[37], ...
                     ↑
              No continuo en memoria → Cache miss
```

**Solución: Cache-aware Prefetching**

```
Para exact greedy:
1. Asignar buffer interno por thread
2. Pre-cargar gradientes en el buffer (prefetch)
3. Acumular en mini-batches
```

**Efecto:** Cambia dependencia directa read/write → dependencia más larga

**Resultados (Figura 7):**

- Cache-aware es **2x más rápido** en datasets grandes (10M ejemplos)
- Impacto mayor cuando gradientes no caben en CPU cache

**Para approximate algorithm:**

Elegir **tamaño de bloque correcto** (B):

- **Muy pequeño:** Poca carga por thread → paralelización ineficiente
- **Muy grande:** Gradientes no caben en cache → cache misses

**Óptimo (Figura 9):** B = 2¹⁶ = 65,536 ejemplos por bloque

---

### 4.3 Blocks for Out-of-core Computation

**Objetivo:** Procesar datos que **no caben en RAM**.

**Estrategia:** Dividir datos en bloques y almacenar en disco.

**Dos técnicas:**

#### **1. Block Compression**

**Proceso:**

- Comprimir bloques por columnas
- Descomprimir on-the-fly en thread independiente
- Cargar en buffer de memoria mientras se computa

**Detalles de compresión:**

- **Feature values:** Algoritmo de compresión general
- **Row indices:**
    - Restar índice inicial del bloque
    - Usar entero de 16 bits para offset
    - Requiere 2¹⁶ ejemplos por bloque

**Ratio de compresión:** 26%-29%

**Trade-off:** Computación de descompresión ↔ Costo de lectura de disco

#### **2. Block Sharding**

**Configuración:**

- Distribuir datos en **múltiples discos**
- Manera alternada (striping)
- Un thread pre-fetcher por disco
- Thread de entrenamiento lee alternativamente de cada buffer

**Ventaja:** Aumenta throughput de lectura de disco

**Combinación de técnicas:**

- Compresión: 3x speedup
- Sharding + Compresión: 2x adicional = **6x total**

---

## 5. RELATED WORKS

Comparación con trabajos previos y otros sistemas.

**Tabla 1: Comparación de sistemas:**

|Sistema|Exact Greedy|Approx Global|Approx Local|Out-of-core|Sparsity-aware|Parallel|
|---|---|---|---|---|---|---|
|**XGBoost**|✅|✅|✅|✅|✅|✅|
|pGBRT|❌|❌|✅|❌|❌|✅|
|Spark MLLib|❌|✅|❌|❌|Parcial|✅|
|H2O|❌|✅|❌|❌|Parcial|✅|
|scikit-learn|✅|❌|❌|❌|❌|❌|
|R GBM|✅|❌|❌|❌|Parcial|❌|

**Contribuciones únicas de XGBoost:**

1. Único con todas las características
2. Primero en explorar: out-of-core, cache-aware, sparsity-aware
3. Sistema end-to-end completo

---

## 6. END TO END EVALUATIONS

Evaluación experimental completa.

### 6.2 Datasets

**Tabla 2:**

|Dataset|n|m|Tarea|
|---|---|---|---|
|Allstate|10M|4,227|Clasificación (seguros)|
|Higgs Boson|10M|28|Clasificación (física)|
|Yahoo LTRC|473K|700|Learning to Rank|
|Criteo|1.7B|67|CTR prediction|

### 6.3 Classification (Higgs-1M)

**Tabla 3: 500 árboles**

|Método|Tiempo/árbol (seg)|Test AUC|
|---|---|---|
|XGBoost|0.68|0.8304|
|XGBoost (colsample=0.5)|0.64|0.8245|
|scikit-learn|28.51|0.8302|
|R.gbm|1.03|0.6224|

**Conclusiones:**

- XGBoost: **10x más rápido que scikit-learn**
- Precisión similar o mejor
- Column subsampling reduce tiempo con mínima pérdida de precisión

### 6.4 Learning to Rank (Yahoo LTRC)

**Tabla 4:**

|Método|Tiempo/árbol (seg)|NDCG@10|
|---|---|---|
|XGBoost|0.826|0.7892|
|XGBoost (colsample=0.5)|0.506|0.7913|
|pGBRT|2.576|0.7915|

**Conclusiones:**

- XGBoost **3x más rápido** que pGBRT
- Column subsampling **mejora** precisión (evita overfitting)

### 6.5 Out-of-core (Criteo)

**Figura 11:**

**Resultados:**

- Algoritmo básico: Solo maneja 200M ejemplos
- - Compresión: **3x speedup**
- - Sharding (2 discos): **2x adicional = 6x total**
- Procesa **1.7 billones** de ejemplos en una máquina

**Sistema sale de file cache a partir de 400M ejemplos** → Realmente usa disco

### 6.6 Distributed (Criteo, 32 nodos EC2)

**Figura 12:**

**Comparación end-to-end:**

- Spark MLLib: Falla con memoria insuficiente
- H2O: Lento cargando datos
- XGBoost: Escala suavemente a 1.7B ejemplos

**Por iteración:**

- XGBoost: **10x más rápido que Spark**
- XGBoost: **2.2x más rápido que H2O**

**Figura 13: Escalabilidad**

|Máquinas|Tiempo/iteración|
|---|---|
|4|~2000 seg|
|8|~1000 seg|
|16|~500 seg|
|32|~250 seg|

**Escalamiento casi lineal** (ligeramente super-linear por más cache)

---

## 7. CONCLUSION

**Lecciones aprendidas:**

1. **Algoritmos:**
    
    - Sparsity-aware split finding
    - Weighted quantile sketch
2. **Sistema:**
    
    - Cache access patterns son críticos
    - Data compression es esencial
    - Sharding mejora throughput
3. **Resultado:**
    
    - Sistema escalable end-to-end
    - Procesa billones de ejemplos con recursos mínimos
    - Lecciones aplicables a otros sistemas ML

---

## APPENDIX: Weighted Quantile Sketch

**Contribución matemática detallada:**

Algoritmo para encontrar quantiles en datos ponderados con garantías teóricas probables.

**Operaciones principales:**

1. **Merge:** Combinar dos summaries
2. **Prune:** Reducir tamaño manteniendo precisión

**Teoremas demostrados:**

- Preservación de error en merge
- Error controlado en prune: ε → ε + 1/b

---

## 🎯 Resumen de conceptos clave por sección

|Sección|Concepto clave|Impacto|
|---|---|---|
|2.1|Regularización explícita|Evita overfitting|
|2.2|Aproximación 2º orden|Convergencia más rápida|
|3.3|Weighted quantile|Splits óptimos en approx mode|
|3.4|Sparsity-aware|50x speedup en datos sparse|
|4.1|Block structure|Ahorra factor log n|
|4.2|Cache-aware|2x speedup en datasets grandes|
|4.3|Out-of-core|Procesa datos > RAM|

