# Planteamiento y descripción (Bloque 4)

En este bloque veremos los principales métodos de combinación de modelos, con el objetivo de mejorar el rendimiento general, ofreciendo una mayor precisión, robustez y generalización en comparación con modelos individuales. Mediante la agregación de modelos diversos, como árboles de decisión, redes neuronales o máquinas vectoriales de soporte, los modelos resultantes (_ensembles_) pueden reducir el riesgo de ajuste excesivo (_overfitting_), gestionar diferentes complejidades de datos y capturar varios patrones que los modelos únicos pueden perder. Técnicas como _bagging_, _boosting_, _stacking_ y _cascading_ permiten la creación de estos modelos combinados, dando lugar a predicciones más estables y fiables, aprovechando los puntos fuertes de cada modelo, al tiempo que mitigan sus debilidades individuales. Esto se traduce en un rendimiento mejorado en diversas tareas, incluida la clasificación, regresión y detección de anomalías.

Este bloque se evalúa mediante un test que mide el grado de conocimiento de los conceptos clave introducidos en los materiales docentes y una PEC que valida su aplicación práctica mediante un ejercicio guiado.

Esta cuarta Práctica de Evaluación Continua (PEC 4) tiene como objetivo principal consolidar los conocimientos adquiridos en las PECs anteriores e introducir y profundizar en técnicas avanzadas de _Ensemble Learning_ (Aprendizaje por Conjuntos).

Al finalizar esta PEC, serás capaz de:

- Implementar y comparar diversas técnicas de _ensemble learning_ como Bagging/Random Forest, Boosting (AdaBoost, Gradient Boosting) y Stacking sobre conjuntos de datos específicos.
- Analizar y explicar la influencia de los hiperparámetros clave en el comportamiento y rendimiento de los métodos _ensemble_.
- Evaluar y comparar la importancia de las características (_feature importance_) obtenida a través de diferentes modelos _ensemble_.
- Justificar la elección de un método _ensemble_ sobre otro o sobre un modelo individual, conectando con conceptos como el _trade-off_ sesgo-varianza.
- Aplicar correctamente metodologías de validación, como la validación cruzada, para una evaluación robusta de los modelos.

---

## Objetivos generales

Los objetivos relacionados con este bloque son:

- Conocer los principales métodos de combinación de modelos
- Saber aplicar el método de combinación de modelos más adecuado para cada escenario o problema

## Objetivos de la PEC4

Los objetivos concretos de esta PEC son:

- Aplicar y analizar técnicas de combinación paralela de clasificadores base similares: Bagging y Boosting.
- Aplicar y analizar técnicas de combinación secuencial de clasificadores base diferentes: Stacking y Cascading.