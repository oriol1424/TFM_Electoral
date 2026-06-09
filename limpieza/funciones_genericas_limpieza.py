import os
import json
import numpy as np
import pandas as pd


def asegurar_directorio(path_archivo):
    directorio = os.path.dirname(path_archivo)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio, exist_ok=True)


def guardar_dataframe_csv(df, path_salida, sep=";", encoding='utf-8-sig', index=False):
    asegurar_directorio(path_salida)
    df.to_csv(path_salida, index=index, sep=sep, encoding=encoding)


def guardar_json(datos, path_salida, indent=4):
    asegurar_directorio(path_salida)
    with open(path_salida, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=indent)


def leer_json(path_archivo):
    with open(path_archivo, 'r', encoding='utf-8') as f:
        return json.load(f)


def leer_archivo_csv(path_entrada, sep=';', encoding_principal='utf-8-sig',
                     encoding_secundario='latin-1', header='infer', dtype=None, decimal=','):
    # intenta utf-8-sig primero, si falla usa latin-1
    try:
        return pd.read_csv(path_entrada, sep=sep, encoding=encoding_principal,
                           header=header, dtype=dtype, decimal=decimal)
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(path_entrada, sep=sep, encoding=encoding_secundario,
                           header=header, dtype=dtype, decimal=decimal)


def leer_datos_mixtos(path_lectura, header=None, dtype=str, sep=';'):
    # detecta Excel o CSV automáticamente
    if path_lectura.lower().endswith('.csv'):
        df_raw = leer_archivo_csv(path_lectura, sep=sep, header=header, dtype=dtype)
    else:
        engine = 'xlrd' if path_lectura.lower().endswith('.xls') else 'openpyxl'
        df_raw = pd.read_excel(path_lectura, header=header, dtype=dtype, engine=engine)

    if len(df_raw.columns) == 1 and len(df_raw) > 1 and isinstance(df_raw.iloc[1, 0], str) and sep in df_raw.iloc[1, 0]:
        df_raw = df_raw[df_raw.columns[0]].str.split(sep, expand=True)

    return df_raw


def normalizar_nombres_columnas(df):
    if isinstance(df.columns, pd.Index):
        df.columns = df.columns.astype(str).str.strip()
    return df


def formatear_serie_codigo(serie, longitud=5):
    # normaliza códigos de municipio/provincia a string con ceros a la izquierda
    def _clean(x):
        if pd.isna(x) or str(x).strip().lower() == 'nan' or str(x).strip() == '':
            return np.nan
        s = str(x).split(' ')[0].split('.')[0].strip()
        if len(s) > longitud:
            s = s[:longitud]
        return s.zfill(longitud) if s.isdigit() else np.nan

    return serie.apply(_clean)


def limpiar_valor_numerico(val, to_nan=False, handle_less_than_5=False):
    # convierte "1.234,50" → 1234.50; maneja puntos y comas del formato europeo
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


def limpiar_serie_numerica(serie):
    if serie.dtype == 'object':
        return pd.to_numeric(serie.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    return serie
