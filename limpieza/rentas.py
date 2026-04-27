import pandas as pd
import numpy as np
import os
from typing import Any, Union
from .funciones_genericas_limpieza import limpiar_valor_numerico, formatear_serie_codigo,leer_archivo_csv, normalizar_nombres_columnas, guardar_dataframe_csv


def crear_lista_maestra(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la lista única de los municipios.
    Args:
        df: DataFrame original del INE con la columna de municipios en la primera posición.
    Returns:
        pd.DataFrame: DataFrame con columnas 'Cod_Muni' y 'Nombre_Muni' sin duplicados.
    """
    col_muni = df.columns[0]
    df_temp = pd.DataFrame()
    df_temp['Cod_Muni'] = df[col_muni].astype(str).str.split(' ').str[0]
    df_temp['Nombre_Muni'] = df[col_muni].astype(str).str.split(' ', n=1).str[1]
    maestra = df_temp[df_temp['Cod_Muni'].str.isdigit() == True].drop_duplicates('Cod_Muni')
    return maestra[['Cod_Muni', 'Nombre_Muni']].copy()


def generar_datos_pivotados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia, promedia por municipio y pivota los datos de renta.
    Args:
        df: DataFrame original con datos de indicadores de renta media.
    Returns:
        pd.DataFrame: DataFrame transformado a formato ancho con promedios municipales.
    """
    df['Total_Num'] = df['Total'].apply(lambda x: limpiar_valor_numerico(x, to_nan=True))
    df['Cod_Muni'] = formatear_serie_codigo(df[df.columns[0]], 5)

    df_clean = df.dropna(subset=['Total_Num'])
    df_agrupado = df_clean.groupby(['Cod_Muni', 'Indicadores de renta media', 'Periodo'])['Total_Num'].mean().reset_index()
    df_pivot = df_agrupado.pivot_table(index='Cod_Muni', columns=['Indicadores de renta media', 'Periodo'], values='Total_Num')
    df_pivot.columns = [f"{col[0]} {int(col[1])}" for col in df_pivot.columns]
    
    return df_pivot.reset_index()


def procesar_archivo_ine(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Función principal que orquesta el proceso de renta y crea carpetas si no existen.
    Args:
        path_entrada: Ruta del archivo CSV original del INE.
        path_salida: Ruta de destino para guardar el archivo procesado.
    Returns:
        pd.DataFrame: El DataFrame final resultante de la unión y limpieza.
    Raises:
        Exception: Si ocurre un error crítico durante la lectura o escritura del archivo.
    """
    directorio_salida = os.path.dirname(path_salida)
    if directorio_salida and not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida, exist_ok=True)
    try:
        df = leer_archivo_csv(path_entrada)
        df = normalizar_nombres_columnas(df)
        
        maestra = crear_lista_maestra(df)
        datos_renta = generar_datos_pivotados(df)
        
        resultado_final = pd.merge(maestra, datos_renta, on='Cod_Muni', how='left')
        guardar_dataframe_csv(resultado_final, path_salida)
        
        return resultado_final
    except Exception as e:
        print(f"Error crítico procesando archivo INE: {e}")
        raise


def generar_datos_gini_pivotados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia, promedia por municipio y pivota los datos de Gini/P80P20.
    Args:
        df: DataFrame original con indicadores de desigualdad.
    Returns:
        pd.DataFrame: DataFrame transformado a formato ancho.
    """
    col_indicador = 'Indicadores de renta media'
    df['Total_Num'] = df['Total'].apply(lambda x: limpiar_valor_numerico(x, to_nan=True))
    df['Cod_Muni'] = formatear_serie_codigo(df[df.columns[0]], 5)
    
    df_clean = df.dropna(subset=['Total_Num'])
    df_agrupado = df_clean.groupby(['Cod_Muni', col_indicador, 'Periodo'])['Total_Num'].mean().reset_index()
    df_pivot = df_agrupado.pivot_table(index='Cod_Muni', columns=[col_indicador, 'Periodo'], values='Total_Num')
    df_pivot.columns = [f"{col[0]} {int(col[1])}" for col in df_pivot.columns]
    
    return df_pivot.reset_index()


def procesar_archivo_gini(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Orquesta el proceso para Gini y Distribución P80/P20.
    Args:
        path_entrada: Ruta del archivo CSV original de Gini.
        path_salida: Ruta de destino para guardar el archivo procesado.
    Returns:
        pd.DataFrame: DataFrame final con indicadores de desigualdad por municipio.
    Raises:
        Exception: Si ocurre un error crítico durante el procesamiento.
    """
    try:
        df = leer_archivo_csv(path_entrada)
        df = normalizar_nombres_columnas(df)
        
        maestra = crear_lista_maestra(df)
        datos_gini = generar_datos_gini_pivotados(df)
        
        resultado_final = pd.merge(maestra, datos_gini, on='Cod_Muni', how='left')
        guardar_dataframe_csv(resultado_final, path_salida)
        return resultado_final
    except Exception as e:
        print(f"Error crítico procesando archivo Gini: {e}")
        raise


def generar_datos_fuente_pivotados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los nombres de las fuentes, promedia por municipio y pivota.
    Args:
        df: DataFrame original con distribución por fuente de ingresos.
    Returns:
        pd.DataFrame: DataFrame transformado a formato ancho con fuentes limpias.
    """
    col_indicador = 'Distribución por fuente de ingresos'
    df[col_indicador] = df[col_indicador].astype(str).str.split(': ').str[-1]
    
    df['Total_Num'] = df['Total'].apply(lambda x: limpiar_valor_numerico(x, to_nan=True))
    df['Cod_Muni'] = formatear_serie_codigo(df[df.columns[0]], 5)
    
    df_clean = df.dropna(subset=['Total_Num'])
    df_agrupado = df_clean.groupby(['Cod_Muni', col_indicador, 'Periodo'])['Total_Num'].mean().reset_index()
    df_pivot = df_agrupado.pivot_table(index='Cod_Muni', columns=[col_indicador, 'Periodo'], values='Total_Num')
    df_pivot.columns = [f"{col[0]} {int(col[1])}" for col in df_pivot.columns]
    
    return df_pivot.reset_index()


def procesar_fuente_ingresos(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Orquesta el proceso para el archivo de Fuente de Ingresos (30825.csv).
    Args:
        path_entrada: Ruta del archivo CSV original de fuentes de ingresos.
        path_salida: Ruta de destino para guardar el archivo procesado.
    Returns:
        pd.DataFrame: DataFrame final con fuentes de ingresos por municipio.
    Raises:
        Exception: Si ocurre un error crítico durante el procesamiento.
    """
    try:
        df = leer_archivo_csv(path_entrada)
        df = normalizar_nombres_columnas(df)
        
        maestra = crear_lista_maestra(df)
        datos_fuente = generar_datos_fuente_pivotados(df)
        
        resultado_final = pd.merge(maestra, datos_fuente, on='Cod_Muni', how='left')
        guardar_dataframe_csv(resultado_final, path_salida)
        return resultado_final
    except Exception as e:
        print(f"Error crítico procesando fuentes de ingresos: {e}")
        raise