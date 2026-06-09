import pandas as pd
import os
from .funciones_genericas_limpieza import formatear_serie_codigo, limpiar_serie_numerica, guardar_dataframe_csv, leer_archivo_csv

def procesar_geografia_municipios(path_entrada, path_salida):
    """Procesa el dataset geográfico municipal para normalizar IDs y campos numéricos."""
    if not os.path.exists(path_entrada):
        raise FileNotFoundError(f"El archivo no se encuentra en {path_entrada}")
    
    df = leer_archivo_csv(
        path_entrada, 
        sep=";", 
        encoding_principal="latin-1", 
        dtype={"COD_INE": str, "COD_PROV": str},
        decimal=','
    )

    columnas_map = {
        'COD_INE': 'id_municipio',
        'NOMBRE_ACTUAL': 'nombre_municipio',
        'COD_PROV': 'id_provincia',
        'SUPERFICIE': 'superficie_km2',
        'PERIMETRO': 'perimetro',
        'LONGITUD_ETRS89': 'longitud',
        'LATITUD_ETRS89': 'latitud',
        'ALTITUD': 'altitud'
    }

    df = df[list(columnas_map.keys())].rename(columns=columnas_map)
    
    df['id_municipio'] = formatear_serie_codigo(df['id_municipio'], 5)
    df['id_provincia'] = formatear_serie_codigo(df['id_provincia'], 2)
    
    cols_a_limpiar = ['superficie_km2', 'perimetro', 'longitud', 'latitud', 'altitud']
    for col in cols_a_limpiar:
        df[col] = limpiar_serie_numerica(df[col])
    
    df['superficie_km2'] = df['superficie_km2'] / 100
    df = df.drop_duplicates(subset=['id_municipio'])
    
    try:
        guardar_dataframe_csv(df, path_salida)
    except Exception as e:
        print(f"Error crítico al guardar el archivo: {e}")
        raise
        
    return df