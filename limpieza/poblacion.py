import os
import json
import re
from typing import Optional
import pandas as pd
from .funciones_genericas_limpieza import leer_datos_mixtos, formatear_serie_codigo, guardar_dataframe_csv, guardar_json

def extraer_anyo(texto_titulo: str) -> Optional[int]:
    """
    Extrae un año de un texto dado utilizando expresiones regulares.
    Busca un patrón de cuatro dígitos que represente un año comenzando por 19 o 20.
    Args:
        texto_titulo (str): El texto del cual extraer el año.
    Returns:
        Optional[int]: El año extraído como entero, o None si no se encuentra 
        ninguna coincidencia.
    """
    match = re.search(r'\b(19|20)\d{2}\b', str(texto_titulo))
    if match:
        return int(match.group(0))
    return None


def procesar_poblacion_maestro(path_lectura: str, path_salida_json: str, path_salida_csv: str) -> None:
    """
    Ejecuta el pipeline ETL completo del Padrón Municipal.
    Lee el archivo de origen, limpia y transforma los datos asignando 
    identificadores unificados, calcula totales poblacionales por provincia 
    y a nivel nacional, y exporta los resultados a formatos CSV plano y 
    JSON jerárquico.
    Args:
        path_lectura (str): Ruta del archivo de entrada (Excel o CSV).
        path_salida_json (str): Ruta de destino para el archivo JSON jerárquico.
        path_salida_csv (str): Ruta de destino para el archivo CSV plano.
    Returns:
        None
    """
    df_raw = leer_datos_mixtos(path_lectura, header=None)
    titulo = df_raw.iloc[0, 0]
    anyo_padron = extraer_anyo(titulo)

    df = df_raw.iloc[2:].copy()
    df.columns = ['CPRO', 'PROVINCIA', 'CMUN', 'NOMBRE', 'POB', 'HOMBRES', 'MUJERES']
    df.dropna(subset=['CPRO', 'CMUN'], inplace=True)

    df['CPRO'] = formatear_serie_codigo(df['CPRO'], 2)
    df['CMUN'] = formatear_serie_codigo(df['CMUN'], 3)
    df['id_municipio'] = df['CPRO'] + df['CMUN']

    for col in ['POB', 'HOMBRES', 'MUJERES']:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    total_nacional = int(df['POB'].sum())

    df_csv = df[['id_municipio', 'NOMBRE', 'POB', 'HOMBRES', 'MUJERES']].copy()
    df_csv.rename(columns={
        'NOMBRE': 'nombre_municipio',
        'POB': 'poblacion_total',
        'HOMBRES': 'total_hombres',
        'MUJERES': 'total_mujeres'
    }, inplace=True)
    
    guardar_dataframe_csv(df_csv, path_salida_csv)

    estructura_json = {
        "metadatos": {
            "anyo": anyo_padron,
            "total_nacional_poblacion": total_nacional
        },
        "provincias": {}
    }

    agrupacion_provincial = df.groupby('CPRO')
    
    for cpro, group in agrupacion_provincial:
        nombre_prov = group['PROVINCIA'].iloc[0].strip()
        total_provincial = int(group['POB'].sum())
        
        municipios_lista = []
        for _, row in group.iterrows():
            municipios_lista.append({
                "id_municipio": row['id_municipio'],
                "nombre": row['NOMBRE'].strip(),
                "poblacion": row['POB'],
                "hombres": row['HOMBRES'],
                "mujeres": row['MUJERES']
            })
            
        estructura_json["provincias"][cpro] = {
            "nombre_provincia": nombre_prov,
            "total_provincial": total_provincial,
            "municipios": municipios_lista
        }

    guardar_json(estructura_json, path_salida_json)