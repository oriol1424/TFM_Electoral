import numpy as np
import pandas as pd
from imputacion._knn import knn_impute_group

GRUPO_B = [
    "salarios",
    "pensiones",
    "otros ingresos",
    "otras prestaciones",
    "desempleo",
]

DISTANCE_COLS = ["latitud", "longitud", "poblacion", "densidad poblacional"]


def imputar_grupo_b(df: pd.DataFrame, n_neighbors: int = 9) -> pd.DataFrame:
    """
    Imputa las fuentes de ingresos (datos composicionales) mediante KNN espacial
    seguido de renormalización a 100% para preservar la consistencia composicional.

    Reference set: municipios con las 5 variables no nulas.
    Renormalización: solo se aplica a las filas imputadas para no alterar los datos originales.
    """
    n_antes = df[GRUPO_B].isna().any(axis=1).sum()

    df, mask = knn_impute_group(
        df=df,
        distance_cols=DISTANCE_COLS,
        target_cols=GRUPO_B,
        n_neighbors=n_neighbors,
    )
    df["imputado_grupo_b"] = mask

    filas_imp = mask[mask].index
    if len(filas_imp) > 0:
        suma = df.loc[filas_imp, GRUPO_B].sum(axis=1)
        suma = suma.replace(0, np.nan)
        for col in GRUPO_B:
            df.loc[filas_imp, col] = df.loc[filas_imp, col] / suma * 100

    n_residual   = df[GRUPO_B].isna().any(axis=1).sum()
    suma_post    = df[GRUPO_B].sum(axis=1)
    n_incons     = ((suma_post < 90) | (suma_post > 110)).sum()

    print(f"  Grupo B: {n_antes} municipios con nulos -> {n_residual} residuales tras KNN (k={n_neighbors})")
    if n_incons > 0:
        print(f"  AVISO: {n_incons} municipios con suma composicional fuera de [90, 110]%")

    return df
