import numpy as np
import pandas as pd
from imputacion._knn import knn_impute_group

GRUPO_A = [
    "indice gini",
    "P80P20",
    "Renta media hogar",
    "renta media unidad consumo",
    "renta media persona",
]

DISTANCE_COLS = ["latitud", "longitud", "poblacion", "densidad poblacional"]


def imputar_grupo_a(df: pd.DataFrame, n_neighbors: int = 9) -> pd.DataFrame:
    """
    Imputa Gini, P80P20 y las tres variables de renta mediante KNN espacial.

    Reference set: municipios con las 5 variables no nulas.
    Distance metric: coordenadas geográficas + población + densidad (normalizados).
    """
    n_antes = df[GRUPO_A].isna().any(axis=1).sum()

    df, mask = knn_impute_group(
        df=df,
        distance_cols=DISTANCE_COLS,
        target_cols=GRUPO_A,
        n_neighbors=n_neighbors,
    )
    df["imputado_grupo_a"] = mask

    n_residual = df[GRUPO_A].isna().any(axis=1).sum()
    print(f"  Grupo A: {n_antes} municipios con nulos -> {n_residual} residuales tras KNN (k={n_neighbors})")

    return df
