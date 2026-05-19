import os
import json
import numpy as np
import pandas as pd
from typing import Any, Optional, Union, Dict, List

def asegurar_directorio(path_archivo: str) -> None:
    """
    Crea el directorio padre para un archivo si este no existe.
    Args:
        path_archivo (str): Ruta completa al archivo.
    """
    directorio = os.path.dirname(path_archivo)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio, exist_ok = True)

def guardar_dataframe_csv(df: pd.DataFrame, path_salida: str, sep: str = ";", encoding: str = 'utf-8-sig', index: bool = False) -> None:
    """
    Guarda un DataFrame en formato CSV asegurando que exista el directorio.
    Args:
        df (pd.DataFrame): DataFrame a exportar.
        path_salida (str): Ruta de destino del archivo CSV.
        sep (str, optional): Separador de columnas. Por defecto ';'.
        encoding (str, optional): Codificación del archivo. Por defecto 'utf-8-sig'.
        index (bool, optional): Indica si se debe guardar el índice. Por defecto False.
    """
    asegurar_directorio(path_salida)
    df.to_csv(path_salida, index = index, sep = sep, encoding = encoding)

def guardar_json(datos: Union[Dict, List], path_salida: str, indent: int = 4) -> None:
    """
    Guarda un diccionario o lista en formato JSON asegurando el directorio previo.
    Args:
        datos (Union[Dict, List]): Datos a serializar en JSON.
        path_salida (str): Ruta de destino del archivo JSON.
        indent (int, optional): Nivel de indentación. Por defecto 4.
    """
    asegurar_directorio(path_salida)
    with open(path_salida, 'w', encoding = 'utf-8') as f:
        json.dump(datos, f, ensure_ascii = False, indent = indent)

def leer_json(path_archivo: str) -> Union[Dict, List]:
    """
    Lee un archivo JSON con codificación UTF-8 y devuelve su contenido.
    
    Args:
        path_archivo (str): Ruta completa al archivo JSON.
        
    Returns:
        Union[Dict, List]: Los datos cargados desde el archivo JSON 
        (normalmente un diccionario o una lista de diccionarios).
    """
    with open(path_archivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def leer_archivo_csv(path_entrada: str, sep: str = ';', 
                             encoding_principal: str = 'utf-8-sig', 
                             encoding_secundario: str = 'latin-1', 
                             header: Any = 'infer', dtype: Any = None, 
                             decimal: str = ',') -> pd.DataFrame:
    """
    Intenta leer un archivo CSV con una codificación primaria y usa una secundaria 
    como respaldo si falla, previniendo errores críticos de codificación.
    Por defecto, está optimizado para el estándar español (sep=';', decimal=',').
    Args:
        path_entrada (str): Ruta del archivo CSV.
        sep (str, optional): Separador de columnas. Por defecto ';'.
        encoding_principal (str, optional): Codificación inicial. Por defecto 'utf-8-sig'.
        encoding_secundario (str, optional): Codificación de respaldo. Por defecto 'latin-1'.
        header (Any, optional): Fila a usar como cabecera. Por defecto 'infer'.
        dtype (Any, optional): Tipos de datos para las columnas.
        decimal (str, optional): Carácter decimal. Por defecto ','.
    Returns:
        pd.DataFrame: DataFrame con los datos procesados.
    """
    try:
        return pd.read_csv(path_entrada, sep=sep, encoding=encoding_principal, 
                           header=header, dtype=dtype, decimal=decimal)
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(path_entrada, sep=sep, encoding=encoding_secundario, 
                           header=header, dtype=dtype, decimal=decimal)

def leer_datos_mixtos(path_lectura: str, header: Any = None, 
                      dtype: Any = str, sep: str = ';') -> pd.DataFrame:
    """
    Lee datos tabulares detectando dinámicamente si es formato Excel o CSV.
    Soluciona casos donde un CSV mal formado se agrupa en una sola columna.
    Args:
        path_lectura (str): Ruta del archivo a leer.
        header (Any, optional): Fila(s) a usar como cabecera. Por defecto None.
        dtype (Any, optional): Tipo de datos a forzar. Por defecto str.
        sep (str, optional): Separador esperado en CSV. Por defecto ';'.
    Returns:
        pd.DataFrame: DataFrame con los datos extraídos en columnas separadas.
    """
    if path_lectura.lower().endswith('.csv'):
        df_raw = leer_archivo_csv(path_lectura, sep=sep, header=header, dtype=dtype)
    else:
        engine = 'xlrd' if path_lectura.lower().endswith('.xls') else 'openpyxl'
        df_raw = pd.read_excel(path_lectura, header=header, dtype=dtype, engine=engine)

    if len(df_raw.columns) == 1 and len(df_raw) > 1 and isinstance(df_raw.iloc[1, 0], str) and sep in df_raw.iloc[1, 0]:
        df_raw = df_raw[df_raw.columns[0]].str.split(sep, expand=True)
        
    return df_raw

def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los espacios en blanco sobrantes en los nombres de las columnas.
    Args:
        df (pd.DataFrame): DataFrame objetivo.
    Returns:
        pd.DataFrame: DataFrame con los nombres limpios.
    """
    if isinstance(df.columns, pd.Index):
        df.columns = df.columns.astype(str).str.strip()
    return df

def formatear_serie_codigo(serie: pd.Series, longitud: int = 5) -> pd.Series:
    """
    Aplica la normalización de códigos geográficos (municipio/provincia) 
    de manera vectorizada. Limpia decimales, recorta texto y rellena con ceros.
    
    Args:
        serie (pd.Series): Serie con identificadores geográficos.
        longitud (int, optional): Longitud objetivo de relleno. Por defecto 5.
        
    Returns:
        pd.Series: Serie estandarizada con valores numéricos rellenos de ceros.
    """
    def _clean(x: Any) -> Any:
        if pd.isna(x) or str(x).strip().lower() == 'nan' or str(x).strip() == '':
            return np.nan
        s = str(x).split(' ')[0].split('.')[0].strip()
        if len(s) > longitud:
            s = s[:longitud]
        return s.zfill(longitud) if s.isdigit() else np.nan
        
    return serie.apply(_clean)

def limpiar_valor_numerico(val: Any, to_nan: bool = False, handle_less_than_5: bool = False) -> Union[int, float]:
    """
    Transforma texto monetario/numérico complejo del formato europeo al formato float 
    estándar de Python (1.234,50 -> 1234.50).
    Args:
        val (Any): Valor original a evaluar.
        to_nan (bool, optional): Retorna np.nan en vez de 0 ante errores. Por defecto False.
        handle_less_than_5 (bool, optional): Transforma la cadena '<5' al valor 2 (Típico del SEPE).
    Returns:
        Union[int, float]: Valor flotante limpio.
    """
    if pd.isna(val):
        return np.nan if to_nan else 0.0
        
    v_str = str(val).strip()

    if v_str in ['', '-', '.', '..', 'nan', 'NaN']:
        return np.nan if to_nan else 0.0
        
    if handle_less_than_5 and v_str == '<5':
        return 2.0

    try:
        if isinstance(val, (int, float)):
            return float(val)

        val_clean = v_str.replace('.', '').replace(',', '.')
        return float(val_clean)
    except (ValueError, TypeError):
        return np.nan if to_nan else 0.0

def limpiar_serie_numerica(serie: pd.Series) -> pd.Series:
    """
    Convierte vectorizadamente una serie tipo string con comas en tipo float.
    Args:
        serie (pd.Series): Serie original.
    Returns:
        pd.Series: Serie convertida numéricamente.
    """
    if serie.dtype == 'object':
        return pd.to_numeric(serie.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    return serie
