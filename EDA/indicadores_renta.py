import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA.visuals import (
    mapear_nombres_provincias,
    plot_missing_demographics,
    check_missing_values,
    plot_distribution_analysis
)
from EDA.funciones_generales import resolver_col_id

def _resolver_identificadores_renta(df):
    """Detecta las columnas de ID y Nombre del municipio de forma robusta."""
    col_id = resolver_col_id(df)
    col_nombre = None
    for n in ['Nombre_Muni', 'Nombre', 'Municipio', 'Nombre del municipio', 'nombre_muni']:
        if n in df.columns and n != col_id:
            col_nombre = n
            break
    if col_nombre is None:
        posibles = [c for c in df.columns if c != col_id]
        col_nombre = posibles[0] if posibles else col_id
    return col_id, col_nombre

def filtrar_columnas_por_anyo(df: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Filtra el DataFrame para quedarse solo con las columnas identificadoras
    y las que corresponden al año especificado (o todas si no hay sufijos).
    """
    anyo_str = str(anyo)
    col_id, col_nombre = _resolver_identificadores_renta(df)
    
    cols_anyo = [col for col in df.columns if col.endswith(anyo_str)]
    
    if not cols_anyo:
        cols_anyo = [c for c in df.columns if c not in [col_id, col_nombre]]
    
    df_result = df[[col_id, col_nombre] + cols_anyo].copy()
    
    df_result.rename(columns={
        col_id: 'Cod_Muni',
        col_nombre: 'Nombre_Muni'
    }, inplace=True)
    
    df_result['Cod_Muni'] = df_result['Cod_Muni'].astype(str).str.strip().str.zfill(5)
    
    return df_result

def eda_indicadores_renta(df_renta_completo: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Función principal para el análisis exploratorio de los indicadores de renta.
    """
    anyo_str = str(anyo)
    print(f"\nINICIANDO ANÁLISIS DE INDICADORES DE RENTA ({anyo_str})")

    df_anyo = filtrar_columnas_por_anyo(df_renta_completo, anyo_str)

    if df_anyo['Nombre_Muni'].isna().any() or df_anyo['Nombre_Muni'].dtype != 'object':
        col_id_pob, col_nom_pob = _resolver_identificadores_renta(df_pob)

        df_pob_map = df_pob[[col_id_pob, col_nom_pob]].copy()
        df_pob_map[col_id_pob] = df_pob_map[col_id_pob].astype(str).str.zfill(5)
        nombres_map = dict(zip(df_pob_map[col_id_pob], df_pob_map[col_nom_pob]))
        if df_anyo['Nombre_Muni'].dtype != 'object':
             df_anyo['Nombre_Muni'] = df_anyo['Cod_Muni'].map(nombres_map)
        else:
             df_anyo['Nombre_Muni'] = df_anyo['Nombre_Muni'].fillna(df_anyo['Cod_Muni'].map(nombres_map))

    print(f"\nAUDITORÍA DE DATOS FALTANTES INDICADORES RENTA ({anyo_str})")

    total_municipios = len(df_anyo)
    
    posibles_renta = [col for col in df_anyo.columns if 'Renta neta media por persona' in col]
    
    if posibles_renta:
        col_renta_persona = posibles_renta[0]
        nulos_renta = df_anyo[col_renta_persona].isna().sum()
        print(f"Total de municipios en el dataset: {total_municipios}")
        print(f"Municipios sin datos de renta: {nulos_renta} ({(nulos_renta/total_municipios)*100:.1f}%)")
        
        check_missing_values(df_anyo, title=f"Porcentaje de Nulos por Indicador ({anyo_str})")
        
        df_nulls = df_anyo[df_anyo[col_renta_persona].isna()].copy()
        if not df_nulls.empty:
            plot_missing_demographics(df_nulls, df_pob, anyo_str, title_suffix=f"Indicadores Renta")
    else:
        check_missing_values(df_anyo, title=f"Porcentaje de Nulos por Indicador ({anyo_str})")

    return df_anyo

def visualizar_histogramas_renta(df_anyo: pd.DataFrame, anyo: str):
    """
    Genera histogramas y boxplots para cada indicador de renta del año.
    """
    anyo_str = str(anyo)
    print(f"\nGENERANDO VISUALIZACIONES DE DISTRIBUCIÓN PARA {anyo_str}")
    
    cols_num = [c for c in df_anyo.columns if c not in ['Cod_Muni', 'Nombre_Muni']]
    
    for col in cols_num:
        plot_distribution_analysis(
            df_anyo, 
            col, 
            title=f"Distribución: {col}",
            figsize=(10, 6),
            color="skyblue"
        )
