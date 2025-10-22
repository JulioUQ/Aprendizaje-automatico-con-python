
El objetivo de este PEC1 es la aplicación de técnicas fundamentales para la transformación de datos, como etapa previa a la construcción de modelos de aprendizaje automático. El caso de estudio se fundamenta en el uso de datos reales.

La estructura de la PEC tiene las siguientes fases:

**Carga y Fusión de Datos:** Se efectúa la ingesta y combinación de dos _datasets_: uno correspondiente a la siniestralidad viaria en Barcelona  (numero de accidentes de tráfico que resultan en victimas, mortales o heridas) y otro con datos climatológicos de la AEMET.

**Análisis Exploratorio de Datos (EDA):** Se aplican estadísticos descriptivos y técnicas de visualización para caracterizar la distribución de los datos, identificar patrones, detectar valores anómalos (_outliers_) y analizar la correlación entre variables.

**Preprocesamiento y Limpieza:** Se implementan técnicas para el tratamiento de casuísticas habituales, como la imputación de valores nulos y la estandarización de formatos.

**Reducción de la Dimensionalidad:** Se utilizan métodos para reducir el espacio de características del modelo, seleccionando los atributos más informativos.

**Gestión de Datos Desbalanceados:** Se aborda el tratamiento de clases minoritarias en la variable objetivo para mitigar el sesgo en el modelado.

---

En esta actividad no está permitido el uso de herramientas basadas en inteligencia artificial, como asistentes de código o chatbots. En el plan docente y en la [web sobre integridad académica y plagio de la UOC](https://campus.uoc.edu/estudiant/microsites/plagi/es/index.html) encontraréis información sobre qué se considera conducta irregular en la evaluación y las consecuencias que puede tener.

**Se debe entregar tanto el archivo ipynb completado como su exportación en HTML, asegurándose de que ambos presenten la estructura y el contenido completos y organizados correctamente.**

---

A continuación os facilitamos los ficheros necesarios para la realización de la actividad:

- Enunciado (el [Notebook](https://aula.uoc.edu/courses/69096/files/8882484?wrap=1 "M2.891_20251_PEC1-Enunciado.ipynb") [Download Notebook](https://aula.uoc.edu/courses/69096/files/8882484/download?download_frd=1) donde trabajaréis y una versión [HTML](https://aula.uoc.edu/courses/69096/files/8882488?wrap=1 "M2.891_20251_PEC1-Enunciado.html") [Download HTML](https://aula.uoc.edu/courses/69096/files/8882488/download?download_frd=1)para que conservéis a modo de consulta).
- [Fichero](https://aula.uoc.edu/courses/69096/files/8882491?wrap=1 "environment_uoc20251pec1.yml") [Download Fichero](https://aula.uoc.edu/courses/69096/files/8882491/download?download_frd=1)de definición del entorno virtual para el desarrollo de la PEC, con las librerías mínimas necesarias para su desarrollo.
- [Guía](https://aula.uoc.edu/courses/69096/files/8882493?wrap=1 "M2.891 - Instalacion Anaconda.pdf") [Download Guía](https://aula.uoc.edu/courses/69096/files/8882493/download?download_frd=1)para la instalación de entornos virtuales en Anaconda.
- Ficheros de datos:

- Dos ficheros con los accidentes de tráfico acaecidos en Barcelona para los años [2023](https://aula.uoc.edu/courses/69096/files/8882497?wrap=1 "2023_accidents_gu_bcn.csv") [Download 2023](https://aula.uoc.edu/courses/69096/files/8882497/download?download_frd=1)y [2024](https://aula.uoc.edu/courses/69096/files/8882500?wrap=1 "2024_accidents_gu_bcn.csv") [Download 2024](https://aula.uoc.edu/courses/69096/files/8882500/download?download_frd=1).
- [Callejero](https://aula.uoc.edu/courses/69096/files/8882503?wrap=1 "carrerer.csv") [Download Callejero](https://aula.uoc.edu/courses/69096/files/8882503/download?download_frd=1)de la ciudad de Barcelona.
- [Datos](https://aula.uoc.edu/courses/69096/files/8882505?wrap=1 "opendata_aemet_es.csv") [Download Datos](https://aula.uoc.edu/courses/69096/files/8882505/download?download_frd=1)meteorológicos para la ciudad de Barcelona.