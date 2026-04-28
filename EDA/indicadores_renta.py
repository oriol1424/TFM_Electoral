import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from EDA.visuals import (
    mapear_nombres_provincias,
    plot_missing_demographics,
    check_missing_values,
    plot_distribution_analysis
)

def filtrar_columnas_por_anyo(df: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Filtra el DataFrame para quedarse solo con las columnas identificadoras
    y las que corresponden al año especificado.
    """
    anyo_str = str(anyo)
    cols_base = ['Cod_Muni', 'Nombre_Muni']
    cols_anyo = [col for col in df.columns if col.endswith(anyo_str)]
    
    return df[cols_base + cols_anyo].copy()

def eda_indicadores_renta(df_renta_completo: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Función principal para el análisis exploratorio de los indicadores de renta.
    Filtra por año, analiza nulos y distribuciones.
    """
    anyo_str = str(anyo)
    print(f"\n--- INICIANDO ANÁLISIS DE INDICADORES DE RENTA ({anyo_str}) ---")
    
    df_anyo = filtrar_columnas_por_anyo(df_renta_completo, anyo_str)
    print(f"\nAUDITORÍA DE DATOS FALTANTES INDICADORES RENTA ({anyo_str})")
    total_municipios = len(df_anyo)
    
    col_ref = [col for col in df_anyo.columns if 'Renta neta media por persona' in col]
    
    if col_ref:
        col_renta_persona = col_ref[0]
        nulos_renta = df_anyo[col_renta_persona].isna().sum()
        print(f"Total de municipios en el dataset: {total_municipios}")
        print(f"Municipios sin datos de renta: {nulos_renta} ({(nulos_renta/total_municipios)*100:.1f}%)")
        
        check_missing_values(df_anyo, title=f"Porcentaje de Nulos por Indicador ({anyo_str})")
        df_nulls = df_anyo[df_anyo[col_renta_persona].isna()]
        if not df_nulls.empty:
            plot_missing_demographics(df_nulls, df_pob, anyo_str, title_suffix=f"Indicadores Renta")
    else:
        print(f"Advertencia: No se encontraron columnas de indicadores para el año {anyo_str}")

    return df_anyo

def visualizar_histogramas_renta(df_anyo: pd.DataFrame, anyo: str):
    """
    Genera histogramas y boxplots para cada indicador de renta del año.
    """
    anyo_str = str(anyo)
    print(f"\nGENERANDO VISUALIZACIONES DE DISTRIBUCIÓN PARA {anyo_str}")
    
    cols_num = [col for col in df_anyo.columns if col.endswith(anyo_str)]
    
    for col in cols_num:
        plot_distribution_analysis(
            df_anyo, 
            col, 
            title=f"Distribución: {col}",
            figsize=(10, 6),
            color="skyblue"
        )
