
# Planteamiento y descripción (Bloque 2)

En este bloque nos centraremos en los métodos no supervisados. El aprendizaje no supervisado es un tipo de aprendizaje automático en el que el modelo se entrena con datos sin salidas etiquetadas. El objetivo es encontrar patrones, estructuras o relaciones ocultas en los datos. A diferencia del aprendizaje supervisado, el modelo no se entrena con ejemplos explícitos de respuestas correctas, sino que aprende identificando semejanzas o agrupaciones presentes en los datos de entrada. Las tareas habituales en el aprendizaje no supervisado incluyen la agrupación, donde el modelo agrupa puntos de datos similares, y la reducción de la dimensionalidad, que simplifica los datos reduciendo su complejidad y capturando su naturaleza.

En concreto, veremos los siguientes modelos o algoritmos:

- Agrupamiento jerárquico
- Método k-means y derivados
- Agrupamiento Canopy

Este bloque se evalúa mediante un test que mide el grado de conocimiento de los conceptos clave introducidos en los materiales docentes y una PEC que valida su aplicación práctica mediante un ejercicio guiado.

El objetivo de esta segunda PEC (Prueba de Evaluación Continua) es que el estudiantado desarrolle habilidades en distintas ramas del aprendizaje no supervisado como: reducción de dimensionalidad tanto mediante proyecciones como optimización, y clustering (K-means, clustering jerárquico, DBSCAN y Mean-Shift) para analizar la composición de los clusters e identificar temáticas comunes dentro de los datos.

---

## Objetivos generales

Los objetivos relacionados con este bloque son:

- Conocer los fundamentos teóricos de los principales métodos no supervisados
- Comprender los detalles y parámetros empleados en los distintos métodos no supervisados
- Saber aplicar los principales métodos no supervisados en el lenguaje Python

## Objetivos de la PEC2

Los objetivos concretos de esta PEC son:

- Aplicar algoritmos no supervisados a conjuntos de datos utilizando librerías específicas de Python.
- Conocer aplicaciones reales en las que se utilizan métodos no supervisados sobre conjuntos de datos no etiquetados.

---
## Contenidos y recursos

Empezaremos este bloque con un vídeo que introduce los conceptos básicos y objetivos ligados a los métodos no supervisados, seguido de la lectura de los materiales docentes propios de este bloque. Después, unos vídeos describen los algoritmos de agrupación jerárquicos, el algoritmo k-means y derivados, y el concepto de canopy cluster que representa un avance con respecto a los anteriores.

**Ejemplos prácticos**:

Se recomienda consultar los ejemplos prácticos, disponibles en lenguaje Python, en el [repositorio de código abiertoEnlaces a un sitio externo.](https://gitlab.com/UOC/eimt/datascience/MUCD/AA) de la asignatura.

Algunos recursos externos (entre muchos otros) que os pueden ser de utilidad para resolver esta actividad son:

- Introducción a scikit-learn: [https://www.oreilly.com/ideas/intro-to-scikit-learn](https://www.oreilly.com/ideas/intro-to-scikit-learn)
- Introducción a matplotlib: [https://matplotlib.org/tutorials/introductory/pyplot.html](https://matplotlib.org/tutorials/introductory/pyplot.html)
- Múltiples figuras en matplotlib: [https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
- Clustering jerárquico en python: [https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)
- Introducción a t-SNE: [https://github.com/oreillymedia/t-SNE-tutorial](https://github.com/oreillymedia/t-SNE-tutorial)
- Visualización animada para entender los hiperparámetros de t-SNE: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
- Tutorial de manejo de imágenes en numpy: [https://scikit-image.org/docs/dev/user_guide/numpy_images.html](https://scikit-image.org/docs/dev/user_guide/numpy_images.html)





---
---

## 2. *Clustering* y reducción de dimensionalidad culinaria: (6 puntos)

En este ejercicio exploraremos cómo mediante clustering podemos descubrir patrones ocultos en miles de recetas reales de Food.com. Partiremos de un conjunto de unas 30.000 recetas con su nombre, descripción e ingredientes, y nuestro objetivo será agruparlas por similitud para ver qué tipos de platos emergen sin supervisión: ¿se juntan los postres? ¿los platos con ajo? ¿las recetas mediterráneas?

Probaremos distintas formas de representar las recetas: desde una codificación one-hot basada en los ingredientes, hasta embeddings semánticos generados con modelos de lenguaje (como MiniLM). Con ello veremos cómo cambia la estructura de los datos según el tipo de representación y qué tipo de similitud capturan mejor.

Finalmente aplicaremos técnicas de reducción de dimensionalidad con UMAP y varios métodos de clustering que podrás probar (KMeans, DBSCAN, HDBSCAN, Mean-shift, jerárquico...) para visualizar y analizar los grupos resultantes. Terminaremos interpretando los clusters más grandes a través de sus ingredientes y recetas representativas, comparando qué enfoque refleja mejor la *lógica culinaria* del conjunto.

```python
# Carga del dataset de recetas
df = pd.read_csv(r'../Data/recipes.csv')

# Visualización de las primeras filas
df.head(3)
```

||id|name|description|ingredients|ingredients_raw_str|serving_size|servings|steps|tags|search_terms|
|---|---|---|---|---|---|---|---|---|---|---|
|0|56594|Sun-Dried Tomato Palmiers|from "INVITATION TO DINNER" cookbook|['garlic cloves', 'pesto sauce', 'frozen puff ...|["2 garlic cloves, peeled and finely min...|1 (675 g)|1|['Preheat the oven to 350 degrees F.', 'Stir t...|['60-minutes-or-less', 'time-to-make', 'course...|{'appetizer'}|
|1|311362|Char Siu Pork Corn and Bok Choy Stir Fry|Here is another wok recipe I like stir fry's a...|['peanut oil', 'pork fillets', 'garlic cloves'...|["2 tablespoons peanut oil","600 g p...|1 (364 g)|4|['Heat 1 tabelspoon of oil in a wok and cook p...|['60-minutes-or-less', 'time-to-make', 'course...|{'pork', 'dinner'}|
|2|248335|Black Cat Fudge|These adorable fudge peices will be a huge hit...|['semisweet chocolate', 'butter', 'light corn ...|["8 ounces semisweet chocolate, coarsely ...|1 (42 g)|24|['Line 11x7 inches pan with foil, extending fo...|['time-to-make', 'course', 'preparation', 'occ...|{'dessert'}|

||Column|Data Type|Non-null Count|% Null Values|Unique Values|Shape|mean|median|std|min|25%|75%|max|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|id|int64|50000|0.00|50000|50000 rows, 10 columns|270316.64732|269245.5|154109.531409|41.0|137629.75|403126.5|537804.0|
|1|name|object|50000|0.00|47154|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|2|description|object|49053|1.89|48245|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|3|ingredients|object|50000|0.00|49939|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|4|ingredients_raw_str|object|50000|0.00|49984|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|5|serving_size|object|50000|0.00|2605|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|6|servings|int64|50000|0.00|89|50000 rows, 10 columns|6.68870|4.0|8.609285|1.0|4.00|8.0|390.0|
|7|steps|object|50000|0.00|49967|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|8|tags|object|50000|0.00|46201|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|9|search_terms|object|50000|0.00|11451|50000 rows, 10 columns|NaN|NaN|NaN|NaN|NaN|NaN|NaN|

### 2.1. Limpieza y normalizacion

Algunas columnas vienen con listas codificadas como texto ("['a','b']") o conjuntos ("{'x','y'}"). Esto significa que la lista está almacenada como texto, no como lista real de Python.

De modo que para evitar errores las convierto a texto legible, creando la funcion safe_to_text(). Posteriormente construyo la nueva columna con las columnas indicadas en el enunciado, completando los valores nulos (si los hubiera) con un espacio vacio.

```python
import ast  # para convertir strings de listas a listas reales
  
# =============================================================
# Limpieza y normalizacion
# =============================================================
  
def safe_to_text(x):
    """
    Convierte listas o conjuntos codificados como texto en una cadena legible.
    Gestiona valores nulos y errores de parseo.
    """
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, set)):
                return ", ".join(map(str, val))
        except (ValueError, SyntaxError):
            pass
    return str(x)
 
# --- Construcción robusta de la columna unificada ---
sep = " | "  # Separador seguro que no aparece en el texto
  
df['text_full'] = (
    df['name'].fillna('') + sep +
    df['description'].fillna('') + sep +
    df['ingredients'].apply(safe_to_text) + sep +
    df['steps'].apply(safe_to_text) + sep +
    df['tags'].apply(safe_to_text) + sep +
    df['search_terms'].apply(safe_to_text)
)

# Aseguramos limpieza final (sin separadores repetidos, espacios extra, etc.)
df['text_full'] = (
    df['text_full']
      .str.replace(r'\s+', ' ', regex=True)
      .str.replace(r'(\|\s*){2,}', '| ', regex=True)  # evita duplicados de separador
      .str.strip('| ')
      .str.strip()
)

print(f"Primeros valores de 'text_full':\n{df['text_full'].unique()[:5]}")
```

Primeros valores de 'text_full': ['Sun-Dried Tomato Palmiers | from "INVITATION TO DINNER" cookbook | garlic cloves, pesto sauce, frozen puff pastry, fontina, parmigiano-reggiano cheese, sun-dried tomato, fresh ground black pepper | Preheat the oven to 350 degrees F., Stir the minced garlic into the prepared pesto., Unfold the pastry sheets and on a lightly floured board roll them slightly with a rolling pin to make a rectangle that measures 11 by 12 inches., Cut each pastry sheet in half lengthwise., Spread 1/4 of the pesto (2 tablespoons) over the entire surface of each pastry sheet., Divide and sprinkle the cheeses and sun-dried tomatoes over the pesto on each pastry sheet and season each with a pinch of pepper., Tightly roll each sheet, starting at a long side, into a 1-inch-wide log., Using a sharp serrated knife, cut each log into about twenty-four 1/2-inch slices and place then on ungreased baking sheets 1 1/2 inches apart., (The palmiers may be baked right away or refrigerated for up to 2 days. The unbaked palmiers can be frozen, covered, for up to 2 weeks.) If frozen, defrost the palmiers in the refrigerator before baking., Bake the palmiers in the preheated oven for 10-15 minutes, until golden brown., Let them cool for 3 minutes on the baking sheets, then transfer them to cooling racks. | 60-minutes-or-less, time-to-make, course, preparation, appetizers, dietary, number-of-servings | appetizer' "Char Siu Pork Corn and Bok Choy Stir Fry | Here is another wok recipe I like stir fry's as they are quick, easy and reasonably priced to prepare and they are a great way to use up your left over vegetables in a nutritious meal. I served as is but it would go well with steamed rice, as I said yesterday we are watching our carb intake at the moment. This again would serve 3 good portions 4 smaller ones as is, or 4 good portions if served with rice. Note: Char Siu Sauce also know as chinese BBQ sauce. It is a paste like ingredient that is dark-red-brown in colour and has a sweet and spicy flavour. Made from fermented soy beans, honey and various spices. | peanut oil, pork fillets, garlic cloves, red chilies, onion, soy sauce, lime juice, carrot, baby corn, baby bok choy, snow peas, bean sprouts, char siu sauce | Heat 1 tabelspoon of oil in a wok and cook pork until browned all over, remove, set to one side., Heat remaining oil in same wok add onion, garlic and chili stir fry a couple of Min's until onion softens, add soy, lime juice and carrot stir fry 1 minute., Return pork to pan, stir fry 1 minute more, add bok choy, corn and char siu sauce, stir fry a couple of minutes, add peas and sprouts, stir fry until vegetables are just tender and heated through., To Serve: Serve as is for a low carb meal or over rice. | 60-minutes-or-less, time-to-make, course, main-ingredient, cuisine, preparation, main-dish, pork, asian, meat, pork-loins | dinner, pork" 'Black Cat Fudge | These adorable fudge peices will be a huge hit with the kids. | semisweet chocolate, butter, light corn syrup, whipping cream, vanilla, salt, icing sugar, white chocolate chips, black food coloring | Line 11x7 inches pan with foil, extending foil beyond edges of pan; grease foil., Melt chocolate and butter in saucepan over low heat; stir in corn syrup, cream, vanilla and salt., Remove from heat and gradually stir in powdered sugar until smooth., Add food coloring to desired color., Spread evenly in prepared pan., Refrigerate until firm, 1 to 2 hours, Using foil as handles, remove fudge from pan; peel off foil., Using cat cookie cutters cut out cat shapes in fudge., Place 2 vanilla milk chips on each cat for eyes., Score feet to make claws., Cover; refrigerate until ready to serve. | time-to-make, course, preparation, occasion, for-large-groups, fudge, desserts, holiday-event, candy, dietary, halloween, number-of-servings, 4-hours-or-less | dessert' "Hot Pickled Vegetable Medley | This has such a colorful presentation and so good! When I was small, my Dad came home from a trip to Texas with a gallon of these pickles. It has taken me years to replicate the the flavors I remember. It may not be exact but it's close! =) | carrots, cauliflower, jalapeno peppers, habanero peppers, pickling salt, cold water, water, white vinegar, cider vinegar, sugar, pickling salt, prepared horseradish, garlic | In a medium bowl, add carrots sprinkle with 1/4 cup of pickling salt and cover with cold water., (Keep carrots separate or they will tint the cauliflower orange)., Add cauliflower and jalapeño peppers in a separate bowl., Sprinkle with 3/4 cup pickling salt and cover with cold water., Let these set for 1 hour., Drain and rinse vegetables., Bring brine to a boil., Let simmer for 5 minutes., Sterilize 14 pint or 7 quart jars and lids., Into each hot jar layer the vegetables in order of:, Carrots, 2-3 red habanaros, cauliflower and 3-4 green jalapeños., (Repeat for quart jars)., Top with hot brine (stir brine occasionally to distribute the garlic)., Seal with hot lids and process in a hot water bath for 10 minutes., Let set for at least two week before using. | time-to-make, course, main-ingredient, preparation, occasion, for-large-groups, low-protein, healthy, jams-and-preserves, canning, condiments-etc, vegetables, fall, low-fat, summer, dietary, low-cholesterol, seasonal, low-saturated-fat, low-calorie, low-carb, low-in-something, carrots, cauliflower, peppers, number-of-servings, technique, water-bath, 4-hours-or-less | low-fat, healthy, low-carb, low-calorie" "Shopska Salad | The salad is named after the people living in the Sofia area in Bulgaria (the shopi). It's a great salad and easy to prepare. | tomatoes, onions, cucumber, red bell peppers, extra virgin olive oil, vinegar, chili peppers, parsley, salt, feta cheese, black olives | Combine all ingredients except parsley and feta cheese., If individual servings are preferred, pack a coffee cup and turn it upside down on the plate., Grate feta cheese on top and decorate with a parsley sprig., Put a few olives around. | 15-minutes-or-less, time-to-make, course, preparation, low-protein, healthy, salads, easy, dietary, low-sodium, low-cholesterol, low-saturated-fat, low-calorie, low-carb, healthy-2, low-in-something | salad, healthy, low-carb, low-calorie, low-sodium"]

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Implementación:</strong> Genera los <strong>embeddings semánticos</strong> de las recetas utilizando el modelo 
    <code>all-MiniLM-L6-v2</code> de la librería <code>sentence-transformers</code>.
    Este modelo es una versión ligera de BERT que transforma frases completas en vectores numéricos
    que capturan su significado.
  <p>
    Usa como entrada la columna con toda la información de la receta (título, descripción, ingredientes, pasos y etiquetas)
    para obtener una representación semántica de cada una.
  </p>
  <p>
    Incrementa el parámetro <code>batch_size</code> en <code>encode()</code> para aprovechar mejor los recursos 
    y activa <code>show_progress_bar=True</code> para visualizar el progreso durante la generación de embeddings.
  </p>
</div>

A continuación genero los [embeddings](https://openwebinars.net/blog/embeddings/) semánticos de las recetas usando el modelo [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) de `sentence-transformers`, tomando como entrada la columna `text_full` que unifique en la celda anterior para generar:
- `embeddings`: tipo array de numpy de forma (n_recetas, 384)   .
- `df[embeddings]`: Columna con cada vector para futuras búsquedas semánticas.

```python
from sentence_transformers import SentenceTransformer

  

# =============================================================

# Carga del modelo de embeddings semánticos

# =============================================================

  

# all-MiniLM-L6-v2 es un modelo ligero y eficiente (384 dimensiones)

model = SentenceTransformer('all-MiniLM-L6-v2')

  

# --- Generación de embeddings a partir del texto unificado ---

# Uso batch_size alto para eficiencia y barra de progreso para seguimiento

embeddings = model.encode(

    df['text_full'].tolist(),

    batch_size=64,              

    show_progress_bar=True,

    convert_to_numpy=True

)

  

# --- Almacenamos los embeddings como columna separada o matriz ---

df['embedding'] = list(embeddings)

  

# Verificamos tamaño y forma

print(f"Embeddings generados: {embeddings.shape}")

print(df[['name', 'embedding']].head(3))
```

Embeddings generados: (50000, 384) name \ 0 Sun-Dried Tomato Palmiers 1 Char Siu Pork Corn and Bok Choy Stir Fry 2 Black Cat Fudge embedding 0 [-0.087734625, 0.0032690035, -0.057603437, 0.0... 1 [-0.11362725, -0.025649644, -0.02073739, 0.014... 2 [0.024063792, -0.03868193, 0.024933118, 0.0333...

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Implementación:</strong> Reduce los <strong>embeddings</strong> obtenidos a un espacio de <strong>2 dimensiones</strong> utilizando 
    <code>UMAP</code> (<em>Uniform Manifold Approximation and Projection</em>), una técnica de reducción de dimensionalidad
    que conserva la estructura local de los datos en espacios de alta dimensión.
  <p>
    UMAP busca representar los datos en un espacio más pequeño manteniendo la relación entre puntos similares,
    lo que lo hace ideal para visualizar embeddings en 2D y detectar grupos o patrones de manera intuitiva.
    En comparación con <em>t-SNE</em>, UMAP es más rápido, escalable, permite la proyección de nuevos puntos y preserva mejor la estructura global de los datos.
    Puedes consultar una comparación visual detallada en 
    <a href="https://pair-code.github.io/understanding-umap/" target="_blank">esta página</a>.
  </p>
  <p>
    Aplica UMAP sobre la matriz de embeddings y guarda el resultado en una nueva variable que contenga las coordenadas 2D
    de cada receta para su posterior visualización o análisis.
  </p>
</div>
Una vez obtenidos los embeddings semánticos, se reducen a 2 las dimensiones utilizando [`UMAP`](https://pair-code.github.io/understanding-umap/) (*Uniform Manifold Approximation and Projection*) para poder visualizarlos y/o detectar patrones de similitud entre recetas. Consiguiendo un dataframe con 2 coordenadas 2D (`df['umap_x]` y `df['umap]`), listo para graficar.

```python
# =============================================================

# Configuración y aplicación de UMAP

# =============================================================

  

# n_neighbors controla la preservación de la estructura local

# min_dist regula la compactación del mapa (valores bajos agrupan más los puntos)

# random_state fija la semilla para reproducibilidad

reducer = umap.UMAP(

    n_neighbors=15,

    min_dist=0.1,

    n_components=2,

    metric='cosine', # Siguiendo documentacion de all-MiniLM-L6-v2

    random_state=42

)

  

# --- Reducción de dimensionalidad ---

embeddings_2d = reducer.fit_transform(embeddings)

  

# --- Almaceno las coordenadas 2D en el DataFrame ---

df['umap_x'] = embeddings_2d[:, 0] # type: ignore

df['umap_y'] = embeddings_2d[:, 1] # type: ignore

  

# --- Verificación del resultado ---

print("Reducción UMAP completada.")

print(f"Shape del embedding reducido: {embeddings_2d.shape}") # type: ignore

print(df[['name', 'umap_x', 'umap_y']].head(3))
```

[c:\Users\jubeda2\AppData\Local\Programs\Python\Python313\Lib\site-packages\umap\umap_.py:1952](file:///C:/Users/jubeda2/AppData/Local/Programs/Python/Python313/Lib/site-packages/umap/umap_.py:1952): UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism. warn(

Reducción UMAP completada. Shape del embedding reducido: (50000, 2) name umap_x umap_y 0 Sun-Dried Tomato Palmiers 0.989972 2.574067 1 Char Siu Pork Corn and Bok Choy Stir Fry -2.948875 -1.674589 2 Black Cat Fudge 0.681737 13.058722

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Implementación:</strong> Visualiza el resultado de la reducción a 2D con <code>UMAP</code> mediante un <strong>scatter plot</strong>.
    Representa cada receta como un punto en el plano y observa si se forman grupos o regiones con mayor densidad.
  <p>
    Para mejorar la legibilidad del gráfico:
  </p>
  <ul>
    <li>
      Ajusta el parámetro <code>alpha</code> para controlar la <strong>transparencia</strong> de los puntos:
      valores más bajos (por ejemplo, 0.3) permiten visualizar mejor las zonas donde los puntos se solapan,
      haciendo más visibles las regiones densas.
    </li>
    <li>
      Reduce el parámetro <code>s</code> (tamaño de los puntos) para evitar que se superpongan en exceso
      y poder distinguir mejor la estructura general del conjunto.
    </li>
  </ul>
</div>

```python
# =============================================================

# Visualización de los embeddings reducidos con UMAP

# =============================================================

  

# Ajusto el tamaño de los puntos (más pequeño = menos solapamiento)

# Transparencia (menor = más fácil ver densidad)

plt.figure(figsize=(10, 8))

plt.scatter(

    df['umap_x'],

    df['umap_y'],

    s=5,        

    alpha=0.3,    

    c='steelblue',

    edgecolors='none'

)

  

plt.title("Visualización 2D de embeddings de recetas (UMAP)", fontsize=14)

plt.xlabel("UMAP 1")

plt.ylabel("UMAP 2")

plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig(r'../Visualizaciones/embeddings_reducidos_con_UMAP.png', dpi=300)

plt.show()
```

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Análisis:</strong> <strong>sobre qué datos realizar el clustering</strong>: 
    ¿sobre los datos reducidos en 2D o sobre los embeddings originales en alta dimensionalidad?
  <p>
    Justifica tu elección explicando las posibles consecuencias de aplicar el algoritmo 
    de clustering sobre una representación reducida en comparación con la original.
  </p>
</div>
La **elección correcta para el clustering** es hacerlo sobre los **embeddings originales** de alta dimensionalidad, no sobre la versión reducida en 2D. Esta decisión es clave para obtener un agrupamiento que sea verdaderamente representativo del significado de los textos.

La razón principal radica en la **preservación de la información semántica** y el propósito fundamental de cada técnica:

* **Embeddings Originales (384D):** Contienen la **estructura semántica completa** que el modelo ha aprendido. Al hacer el *clustering* sobre estas 384 dimensiones, el algoritmo (como `K-Means` o `HDBSCAN`) está agrupando los textos basándose en su **significado real** y sus relaciones de similitud precisas. Esto resulta en una **alta precisión** del *clustering*.
* **Embeddings Reducidos por UMAP (2D):** UMAP es una técnica de reducción de dimensionalidad **no lineal** cuyo propósito principal es la **visualización y exploración de patrones**. Aunque es excelente para dibujar los datos, en el proceso **pierde parte de la información global y distorsiona algunas distancias**. Por lo tanto, si se agrupase sobre la versión 2D, el algoritmo estará trabajando con una **proyección aproximada**, lo que puede llevar a límites de clusters distorsionados o incorrectos.

En resumen, para un análisis riguroso y una visualización clara, el flujo de trabajo recomendado es:

1.  **Fase de Clustering:** Utiliza los **embeddings originales (384D)** como entrada para tu algoritmo de clustering.
2.  **Fase de Visualización:** Una vez se tienen las etiquetas del cluster, se proyectan sobre el plano **UMAP 2D** para poder **interpretar y visualizar** los clusters y sus centroides en un gráfico fácil de entender. 