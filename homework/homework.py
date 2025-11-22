# flake8: noqa: E501
#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}

# ## 1. Importación de librerías

# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import pickle
import zipfile
import gzip
import json
import os
import pandas as pd


# %% [markdown]
# ## 2. Funciones de preprocesamiento

# %%
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna Age (2021 - Year), elimina Year y Car_Name
    y descarta filas con valores faltantes.
    """
    datos = df.copy()

    # Edad del vehículo
    datos["Age"] = 2021 - datos["Year"]

    # Columnas que ya no se usan de forma directa
    datos = datos.drop(columns=["Year", "Car_Name"])

    # Asegurar que no queden NaN
    datos = datos.dropna()

    return datos


def _cargar_desde_zip(path_zip: str, nombre_interno: str) -> pd.DataFrame:
    """
    Carga un CSV que está dentro de un archivo ZIP.
    """
    with zipfile.ZipFile(path_zip, "r") as zf:
        with zf.open(nombre_interno) as f:
            return pd.read_csv(f)


# %% [markdown]
# ## 3. Definición del modelo y búsqueda de hiperparámetros

# %%
def modelo() -> Pipeline:
    """
    Crea el pipeline:
    - One-Hot para categóricas
    - MinMaxScaler para numéricas
    - SelectKBest
    - LinearRegression
    """
    columnas_categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
    columnas_numericas = ["Selling_Price", "Driven_kms", "Age", "Owner"]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", MinMaxScaler(), columnas_numericas),
        ],
        remainder="passthrough",
    )

    selector = SelectKBest(score_func=f_regression)

    pipe = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("seleccionar_mejores", selector),
            ("clasificador", LinearRegression()),
        ]
    )
    return pipe


def hiperparametros(
    modelo: Pipeline,
    n_divisiones: int,
    x_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    puntuacion: str,
) -> GridSearchCV:
    """
    Optimiza el número k de características en SelectKBest
    usando GridSearchCV y validación cruzada.
    """
    rejilla_param = {
        "seleccionar_mejores__k": list(range(1, 13)),
    }

    buscador = GridSearchCV(
        estimator=modelo,
        param_grid=rejilla_param,
        cv=n_divisiones,
        refit=True,
        scoring=puntuacion,
    )

    buscador.fit(x_entrenamiento, y_entrenamiento)
    return buscador


# %% [markdown]
# ## 4. Métricas y guardado de resultados

# %%
def metricas(
    modelo,
    x_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    x_prueba: pd.DataFrame,
    y_prueba: pd.Series,
):
    """
    Calcula r2, mse y mad para train y test.
    """
    y_train_pred = modelo.predict(x_entrenamiento)
    y_test_pred = modelo.predict(x_prueba)

    metricas_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_entrenamiento, y_train_pred),
        "mse": mean_squared_error(y_entrenamiento, y_train_pred),
        "mad": median_absolute_error(y_entrenamiento, y_train_pred),
    }

    metricas_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_prueba, y_test_pred),
        "mse": mean_squared_error(y_prueba, y_test_pred),
        "mad": median_absolute_error(y_prueba, y_test_pred),
    }

    return metricas_train, metricas_test


def guardar_modelo(modelo) -> None:
    """
    Guarda el modelo en files/models/model.pkl.gz (pickle + gzip).
    """
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)


def guardar_metricas(metricas: list[dict]) -> None:
    """
    Guarda cada diccionario de métricas en una línea de
    files/output/metrics.json.
    """
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for registro in metricas:
            f.write(json.dumps(registro) + "\n")


# %% [markdown]
# ## 5. Carga de los datos desde ZIP

# %%
df_Prueba = _cargar_desde_zip("files/input/test_data.csv.zip", "test_data.csv")
df_Entrenamiento = _cargar_desde_zip("files/input/train_data.csv.zip", "train_data.csv")


# %% [markdown]
# ## 6. Ejecución principal del pipeline

# %%
if __name__ == "__main__":
    print("Limpiando datos...")
    df_Prueba = limpiar_datos(df_Prueba)
    df_Entrenamiento = limpiar_datos(df_Entrenamiento)

    print("Dividiendo en X e y...")
    x_entrenamiento = df_Entrenamiento.drop("Present_Price", axis=1)
    y_entrenamiento = df_Entrenamiento["Present_Price"]

    x_prueba = df_Prueba.drop("Present_Price", axis=1)
    y_prueba = df_Prueba["Present_Price"]

    print("Creando pipeline del modelo...")
    pipe = modelo()

    print("Buscando hiperparámetros óptimos...")
    pipe = hiperparametros(
        pipe,
        n_divisiones=10,
        x_entrenamiento=x_entrenamiento,
        y_entrenamiento=y_entrenamiento,
        puntuacion="neg_mean_absolute_error",
    )

    print("Guardando modelo...")
    guardar_modelo(pipe)

    print("Calculando métricas...")
    m_train, m_test = metricas(pipe, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

    print("Guardando métricas...")
    guardar_metricas([m_train, m_test])

    print("Proceso completado exitosamente.")
