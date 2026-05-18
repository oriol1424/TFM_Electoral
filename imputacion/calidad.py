import numpy as np
import pandas as pd

BINS   = [0, 100, 500, 1000, 2000, np.inf]
LABELS = [0, 1, 2, 3, 4]
ETIQUETAS = {
    0: "sin_datos",
    1: "muy_bajo",
    2: "bajo",
    3: "medio",
    4: "completo",
}

def crear_variable_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'calidad_datos' (0-4) basada en el umbral poblacional
    que determina el secreto estadístico del INE.

    Escala ordinal:
        0 = sin_datos  (pob <= 100)   → secreto estadístico total
        1 = muy_bajo   (101-500)      → datos parciales
        2 = bajo       (501-1000)     → datos con supresiones
        3 = medio      (1001-2000)    → datos casi completos
        4 = completo   (> 2000)       → datos íntegros
    """
    df = df.copy()
    df["calidad_datos"] = pd.cut(
        df["poblacion"],
        bins=BINS,
        labels=LABELS,
        right=True,
    ).astype(int)
    return df
