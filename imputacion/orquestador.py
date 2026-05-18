import os

import numpy as np
import pandas as pd

from imputacion.calidad import crear_variable_calidad
from imputacion.grupo_a import imputar_grupo_a
from imputacion.grupo_b import imputar_grupo_b


GRUPO_A = ["indice gini", "P80P20", "Renta media hogar", "renta media unidad consumo", "renta media persona"]
GRUPO_B = ["salarios", "pensiones", "otros ingresos", "otras prestaciones", "desempleo"]


def imputar_datos_v2(df: pd.DataFrame, anyo: int, n_neighbors: int = 9) -> pd.DataFrame:
    """
    Orquestador de imputación socioeconómica v2.

    Reemplaza imputar_datos_socioeconomicos() de EDA/funciones_generales.py.
    Entrada: dataset unificado con nulos (datos_unificados_{anyo}.csv).
    Salida:  dataset imputado + variables de calidad.

    Pasos:
        1. calidad_datos: indicador ordinal 0-4 basado en umbral poblacional
        2. Grupo A (Gini, P80P20, Rentas): KNN espacial multivariante
        3. Grupo B (fuentes de ingresos): KNN espacial + renormalización composicional
        4. edad_media: excluida (98.2% son medias provinciales sin valor predictivo)
        5. Guardado en data_processed/data_end/{anyo}/
    """
    anyo_str = str(anyo)
    print(f"IMPUTACION KNN ESPACIAL ({anyo_str}) - {len(df)} municipios")

    n_nulos_inicio = df[GRUPO_A + GRUPO_B].isna().any(axis=1).sum()
    print(f"Municipios con al menos un nulo al inicio: {n_nulos_inicio} ({n_nulos_inicio/len(df)*100:.1f}%)")
    print()

    df = crear_variable_calidad(df)
    print(f"[1/3] calidad_datos creada (distribucion):")
    print(df["calidad_datos"].value_counts().sort_index().to_string())
    print()

    print(f"[2/3] Imputando Grupo A (Gini, P80P20, Rentas)...")
    df = imputar_grupo_a(df, n_neighbors=n_neighbors)
    print()

    print(f"[3/3] Imputando Grupo B (Fuentes de ingresos composicionales)...")
    df = imputar_grupo_b(df, n_neighbors=n_neighbors)
    print()

    df["imputado"] = df["imputado_grupo_a"] | df["imputado_grupo_b"]

    n_nulos_edad = df["edad media"].isna().sum() if "edad media" in df.columns else 0
    if n_nulos_edad > 0:
        print(f"NOTA: edad_media tiene {n_nulos_edad} nulos ({n_nulos_edad/len(df)*100:.1f}%).")
        print("  Se excluye de la imputacion: el valor imputado seria la media provincial,")
        print("  informacion ya contenida en la variable 'provincia'. No usar como feature ML.")

    n_nulos_fin = df[GRUPO_A + GRUPO_B].isna().any(axis=1).sum()
    print()
    print(f"RESUMEN IMPUTACION ({anyo_str})")
    print(f"  Municipios imputados (cualquier grupo): {df['imputado'].sum()}")
    print(f"  Nulos residuales en features:           {n_nulos_fin}")
    print(f"  calidad_datos 0 (sin_datos):            {(df['calidad_datos']==0).sum()}")
    print(f"  calidad_datos 4 (completo):             {(df['calidad_datos']==4).sum()}")

    folder_path = os.path.join("data_processed", "data_end", anyo_str)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"datos_unificados_imputados_{anyo_str}.csv")
    df.to_csv(file_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en: {file_path}")

    return df
