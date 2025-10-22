import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
seed = 100

#%matplotlib inline

# --------------------------------- #

# --- 1. Cargad por separado los distintos ficheros que conforman el conjunto de datos de accidentes ---
# -- Concatenad en el mismo DataFrame los accidentes de los años 2023 y 2024.

# Lista de rutas a los archivos CSV con los accidentes de los años 2023 y 2024
rutas_csv = [
    r"C:\Users\usuario\Desktop\Directorio de trabajo Julio\Bloque 1 - Introducción al aprendizaje automático\Data\2023_accidents_gu_bcn.csv",
    r"C:\Users\usuario\Desktop\Directorio de trabajo Julio\Bloque 1 - Introducción al aprendizaje automático\Data\2024_accidents_gu_bcn.csv",
    r"C:\Users\usuario\Desktop\Directorio de trabajo Julio\Bloque 1 - Introducción al aprendizaje automático\Data\carrerer.csv"
]

# Diccionario donde se guardarán los DataFrames
dataframes = {}

for ruta in rutas_csv:
    try:
        # Leer CSV con separador coma y encabezado
        df = pd.read_csv(ruta, sep=",", encoding="utf-8")

        # Obtener nombre de archivo sin extensión
        nombre_archivo = os.path.splitext(os.path.basename(ruta))[0]

        # Guardar en el diccionario
        dataframes[nombre_archivo] = df

        print(f"Cargado: {nombre_archivo} ({len(df)} filas, {len(df.columns)} columnas)")

    except Exception as e:
        print(f"Error al leer {ruta}: {e}")

# --- 2. Concatenar los accidentes de 2023 y 2024 ---
acc_2023 = dataframes['2023_accidents_gu_bcn']
acc_2024 = dataframes['2024_accidents_gu_bcn']

accidentes = pd.concat([acc_2023, acc_2024], ignore_index=True)

print(f"Accidentes totales: {len(accidentes)} filas, {len(accidentes.columns)} columnas)")

print(f"\nEl dataframe accidentes concatenado contiene las siguientes columnas:\n{accidentes.columns.tolist()}\n")

# --------------------------------- #

# --- 3. Crear la columna 'gravedad' según muertos / heridos / graves / no victimas ---
# -- Generad una nueva columna llamada "gravedad" que tomará los valores numéricos 3, 2, 1 ó 0 en función de si hay muertos, heridos graves, leves o ningún tipo de víctima respectivamente.
# Si hay muertos => 3
# Si hay heridos graves => 2
# Si hay heridos leves => 1
# Si no hay víctimas => 0

accidentes["gravedad"] = 0
accidentes.loc[accidentes["Numero_lesionats_lleus"] > 0, "gravedad"] = 1
accidentes.loc[accidentes["Numero_lesionats_greus"] > 0, "gravedad"] = 2
accidentes.loc[accidentes["Numero_morts"] > 0, "gravedad"] = 3

# Mostrar algunas filas para inspección visual
print(f"Muestra de 10 registros de columna objetivo segun victimas mortales, graves y/o heridas.\n {accidentes[["Numero_morts", "Numero_lesionats_greus", "Numero_lesionats_lleus", "gravedad"]].head(10)})\n")

# --------------------------------- #

# --- 4. Incorporar el tipo de vía a través del código de calle ---
# -- Incorporad a dicho DataFrame la información del tipo de vía que se encuentra en el callejero utilizando para cruzarla el código de la calle.
callejero = dataframes['carrerer']  

# Comprobar el resultado
print(f"El dataframe carrerer contiene las siguientes columnas:\n{callejero.columns.tolist()}")

# Unir accidentes_es con callejero_es 
accidentes_carrerer = accidentes.merge(
    callejero[["codi_via", "tipus_via"]],
    left_on="Codi_carrer",     # columna del dataframe accidentes
    right_on="codi_via",    # columna del dataframe callejero
    how="left"
)

# Comprobar el resultado
print(f"\nEl dataframe accidentes vinculado a carrerer contiene las siguientes columnas:\n{accidentes.columns.tolist()}\n, sus dimensiones son {len(accidentes)} filas, {len(accidentes.columns)} columnas)")

# --------------------------------- #

# --- 5. Seleccionar solo las columnas relevantes ---
# -- Quedaros con las columnas referentes a la latitud y longitud en sistema WGS84, tipo de vía, día de la semana, fecha y hora, así como con la nueva columna que indicará la gravedad del accidente.
columnas_finales = [
    "Latitud_WGS84",
    "Longitud_WGS84",
    "tipus_via",
    "Descripcio_dia_setmana",
    "Dia_mes",
    "Mes_any",
    "NK_Any",
    "Hora_dia",
    "gravedad"
]

accidentes_ex1 = accidentes_carrerer[columnas_finales].copy()

print(f"\nEl dataframe final de accidentes contiene las siguientes columnas:\n{accidentes_ex1.columns.tolist()}\n, sus dimensiones son {len(accidentes_ex1)} filas, {len(accidentes_ex1.columns)} columnas)")

# --------------------------------- #

# --- 6.1 Variables descriptivas y número de muestras ---
# -- El número y nombre de los atributos descriptivos (variables que podrían ser usadas para predecir la variable objetivo "y").
variables_descriptivas = accidentes_ex1.columns.drop("gravedad")
n_variables = len(variables_descriptivas)
n_muestras = len(accidentes_ex1)

print(f"\nNúmero de variables descriptivas: {n_variables}")
print("\t- Variables:", list(variables_descriptivas))

# --- 6. El número de filas (muestras) del conjunto de datos.
print(f"Número de filas (muestras) del conjunto de datos: {n_muestras}")

# --------------------------------- #

# --- 7. Comprobación de valores faltantes ---
# -- Verificad si hay o no "missing values" y en qué columnas. En caso de exisitir, imputarlos utilizando 0 o el valor que creáis más conveniente.
def null_summary(data): # Añadir tipo de retorno -> pd.DataFrame
    """
    Devuelve un resumen simple del DataFrame con:
    Column, Non-null Count, Null Count, % Null Values y TotalCount.
    """
    total = len(data)
    summary = pd.DataFrame({
        'Column': data.columns,
        'Non-null Count': data.count().values,
        'Null Count': data.isnull().sum().values,
        '% Null Values': ((data.isnull().sum() / total) * 100).round(2).values,
        'TotalCount': total
    })
    return summary

null_summary(accidentes_ex1)

# ===============================
# IMPUTACIÓN DE VALORES NULOS
# ===============================

# 1. Imputar con un valor genérico como "Desconocido"
accidentes_ex1.loc[:, "tipus_via"] = accidentes_ex1["tipus_via"].fillna("Desconocido")

print(f"\nSe han imputado los valores nulos de la columna tipus_via como 'Desconocido' quedandose como: \n{null_summary(accidentes_ex1[['tipus_via']])}")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


# --- 1. Cargad el conjunto de datos con las condiciones meteorológicas con la ayuda de pandas
ruta_aemet = r"C:\Users\usuario\Desktop\Directorio de trabajo Julio\Bloque 1 - Introducción al aprendizaje automático\Data\opendata_aemet_es.csv"
aemet = pd.read_csv(ruta_aemet, decimal=',', sep=",", encoding="utf-8")

# Convertir la columna de fecha a datetime
aemet['fecha'] = pd.to_datetime(aemet['fecha'], format='%Y-%m-%d', errors='coerce')


# Calcular solo los percentiles de la columna 'prec'
percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
prec_percentiles = aemet['prec'].quantile(percentiles).to_frame().T

# Mostrar el resultado
print(prec_percentiles)

# -- Analisis visual de su distribucion
plt.figure(figsize=(8,4))
plt.hist(aemet['prec'].dropna(), bins=30, edgecolor='black')
plt.title("Distribución de la precipitación (prec)")
plt.xlabel("prec (mm)")
plt.ylabel("Frecuencia")
plt.show()

# --- 2.2. y decidid algún criterio mediante el cual podáis categorizarla en un reducido número de opciones.
bins = [-0.01, 0, 2, 10, 30, aemet['prec'].max()]
labels = ['Sin lluvia', 'Débil', 'Moderada', 'Fuerte', 'Muy fuerte']

aemet['prec_cat'] = pd.cut(aemet['prec'], bins=bins, labels=labels)
aemet['prec_cat'].value_counts()

# --- 3. Incorporad al conjunto de datos de accidentes que habéis preparado en la sección anterior la información metereológica relevante cruzándola a través de la fecha.
# Creo una copia del dataset del ejercicio anterior para sentar una base del ejercicio 2
accidentes_ex2 = accidentes_ex1.copy()

# Genero una columna 'fecha' tipo datetime a partir del año, mes y dia
accidentes_ex2['fecha'] = pd.to_datetime(
    accidentes_ex2[['Ano', 'Mes_ano', 'Dia_mes']]
    .rename(columns={'Ano':'year', 'Mes_ano':'month', 'Dia_mes':'day'})
)

# Unir accidentes_es con callejero_es 
accidentes_meteo = accidentes_ex2.merge(
    aemet,
    on = 'fecha',
    how="inner" # Solo quiero registros que compartan ambos datasets
)

# Eliminar columna 'Ano' (ya no es necesaria)
accidentes_meteo = accidentes_meteo.drop(columns=['Ano'])

# --- 1. Variables descriptivas y número de muestras ---
# -- El número y nombre de los atributos descriptivos (variables que podrían ser usadas para predecir la variable objetivo "y").
X = accidentes_meteo.drop(columns=['gravedad']) #  (atributos descriptivos)
y = accidentes_meteo['gravedad']                #  (variable objetivo)

n_variables = len(X.columns)
n_muestras = len(accidentes_meteo)

print(f"\nNúmero de variables descriptivas: {n_variables}")
print("\t- Variables:", list(X))

# --- 2. El número de filas (muestras) del conjunto de datos.
print(f"Número de filas (muestras) del conjunto de datos: {n_muestras}")

# ===============================
# IMPUTACIÓN DE VALORES NULOS
# ===============================

# 1 Precipitación numérica: si no hay dato, asumimos 0.0 mm
accidentes_meteo['prec'] = accidentes_meteo['prec'].fillna(0.0)

# 2️ Hora de temperatura mínima: texto simbólico
accidentes_meteo['horatmin'] = accidentes_meteo['horatmin'].fillna('Desconocido')

# 3️ Dirección del viento: reemplazar con media
accidentes_meteo['dir'] = accidentes_meteo['dir'].fillna(accidentes_meteo['dir'].mean())

# 4. Velocidad máxima (racha): reemplazar con media
accidentes_meteo['racha'] = accidentes_meteo['racha'].fillna(accidentes_meteo['racha'].mean())

# 5️. Hora de la racha máxima: texto simbólico
accidentes_meteo['horaracha'] = accidentes_meteo['horaracha'].fillna('Desconocido')

# 6️. Categoría de precipitación: si faltan, asignar 'sin_lluvia'
# Añadimos categoría si aún no existe (para evitar error con tipo category)
if 'sin_lluvia' not in accidentes_meteo['prec_cat'].cat.categories:
    accidentes_meteo['prec_cat'] = accidentes_meteo['prec_cat'].cat.add_categories(['sin_lluvia'])

accidentes_meteo['prec_cat'] = accidentes_meteo['prec_cat'].fillna('sin_lluvia')

null_summary(accidentes_meteo)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

