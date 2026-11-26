<div align="center">

![Curso](https://img.shields.io/badge/CURSO-6A1B9A?style=for-the-badge) ![Ruta de ciencia de datos 2/3](https://img.shields.io/badge/RUTA_DE_CIENCIA_DE_DATOS-2%2F3-8E24AA?style=for-the-badge)

# Aprendizaje automático con Python

Proyectos de machine learning con la teoría necesaria para resolverlos: preparación de datos, aprendizaje no supervisado, modelos supervisados y combinación de modelos.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<sub>Ruta de ciencia de datos: [1. Estadística básica](https://github.com/JulioUQ/Curso-de-Estadistica-Basica) → **2. Aprendizaje automático** → [3. Deep learning](https://github.com/JulioUQ/curso-deep-learning)</sub>

</div>

---

## Descripción general

Segundo paso de la ruta de ciencia de datos, después de [estadística básica](https://github.com/JulioUQ/Curso-de-Estadistica-Basica). Cada bloque reúne la teoría del tema y un proyecto resuelto en un cuaderno de Jupyter con datos reales.

## Contenido

| Bloque | Proyecto | Técnicas |
| --- | --- | --- |
| [1. Introducción](Bloque%201%20-%20Introducci%C3%B3n%20al%20aprendizaje%20autom%C3%A1tico) | Accidentes de tráfico en Barcelona combinados con la climatología diaria de AEMET | Integración y limpieza de fuentes, creación de variables, desbalanceo de clases (imbalanced-learn), mapas con Folium |
| [2. Métodos no supervisados](Bloque%202%20-%20M%C3%A9todos%20no%20supervisados) | Agrupamiento de miles de recetas | Embeddings de texto (sentence-transformers), UMAP, HDBSCAN, k-means y DBSCAN con distancia de Jaccard |
| [3. Métodos supervisados](Bloque%203%20-%20M%C3%A9todos%20supervisados) | Exploración y caso práctico de clasificación | KNN, SVM, árboles de decisión, optimización de métricas |
| [4. Combinación de modelos](Bloque%204%20-%20Combinaci%C3%B3n-de-modelos) | Ensembles sobre un problema de clasificación | Bagging, boosting, stacking y cascading |
| [5. Estado del arte](Bloque%205%20-%20Lectura%20de%20articulos%20del%20estado%20del%20arte) | Lectura crítica de artículos | Síntesis y presentación |

Cada bloque tiene una carpeta `Teoria/` con los apuntes y otra con el proyecto.

## Tecnologías y herramientas

| Ámbito | Herramientas |
| --- | --- |
| Modelado | scikit-learn, imbalanced-learn, HDBSCAN, UMAP, sentence-transformers |
| Datos | pandas, NumPy |
| Visualización | matplotlib, seaborn, Plotly, Folium |
| Entorno | Jupyter |

## Instalación y uso

Con Python 3 y Jupyter:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn plotly folium umap-learn hdbscan sentence-transformers
jupyter lab
```

Abre el cuaderno del bloque que quieras; los datos de cada proyecto están en su propia carpeta. Algunos conjuntos grandes se guardan con Git LFS.

## Aspectos técnicos destacados

- Integración de fuentes heterogéneas (registros municipales y climatología) antes de modelar.
- Aprendizaje no supervisado sobre texto: de las recetas a embeddings, reducción de dimensionalidad y clustering por densidad.
- Comparación sistemática de clasificadores y de estrategias de ensemble.

## Autor

**Julio Úbeda Quesada** · Data Scientist

[![Portfolio](https://img.shields.io/badge/Portfolio-0B2545?style=flat-square&logo=astro&logoColor=white)](https://juliouq.github.io/Portfolio-Web---JulioUQ/es/) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square)](https://www.linkedin.com/in/juliouq/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/JulioUQ)

<sub>Material elaborado durante el Máster Universitario en Ciencia de Datos.</sub>
