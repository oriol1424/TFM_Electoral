import os
import numpy as np
import pandas as pd
from .funciones_genericas_limpieza import formatear_serie_codigo, limpiar_valor_numerico, guardar_dataframe_csv


def procesar_hoja_sepe(df, tipo='paro'):
    """Identifica la fila de inicio de datos y limpia una hoja del Excel del SEPE."""
    start_row = 0
    for i in range(len(df)):
        val = str(df.iloc[i, 0]).split('.')[0]
        if val.isdigit() and 4 <= len(val) <= 5:
            start_row = i
            break

    df_clean = df.iloc[start_row:].copy()
    if tipo == 'paro':
        cols = [
            'id_municipio', 'nombre_municipio', 'total_paro',
            'p_h_u25', 'p_h_25_44', 'p_h_o45',
            'p_m_u25', 'p_m_25_44', 'p_m_o45',
            'p_agr', 'p_ind', 'p_con', 'p_ser', 'p_sin_empleo'
        ]
    else:
        cols = [
            'id_municipio', 'nombre_municipio', 'total_contratos',
            'c_h_indef', 'c_h_temp', 'c_h_conv',
            'c_m_indef', 'c_m_temp', 'c_m_conv',
            'c_agr', 'c_ind', 'c_con', 'c_ser'
        ]

    faltan_columnas = len(cols) - df_clean.shape[1]
    if faltan_columnas > 0:
        for i in range(faltan_columnas):
            df_clean[f'col_vacia_{i}'] = np.nan

    df_clean = df_clean.iloc[:, :len(cols)]
    df_clean.columns = cols
    df_clean['id_municipio'] = formatear_serie_codigo(df_clean['id_municipio'], 5)
    df_clean = df_clean.dropna(subset=['id_municipio']).reset_index(drop=True)

    for col in cols[2:]:
        df_clean[col] = df_clean[col].apply(limpiar_valor_numerico, to_nan=False, handle_less_than_5=True)

    return df_clean


def limpiar_y_exportar_sepe(path_entrada, path_salida):
    """Procesa un Excel del SEPE emparejando hojas PARO/CONTRATOS y exporta a CSV."""
    try:
        engine = 'xlrd' if path_entrada.endswith('.xls') else 'openpyxl'
        print(f"DEBUG: Abriendo archivo con motor {engine}: {path_entrada}")

        if not os.path.exists(path_entrada):
            print(f"ERROR: No se encuentra el archivo de entrada: {path_entrada}")
            return None

        xls = pd.ExcelFile(path_entrada, engine=engine)
        sheets = xls.sheet_names
        print(f"DEBUG: Hojas encontradas: {len(sheets)}")

        pares_a_procesar = []
        i = 0
        while i < len(sheets):
            s = sheets[i]
            if 'PARO' in s.upper():
                s_paro = s
                s_cont = None
                if (i + 1) < len(sheets):
                    next_sheet = sheets[i + 1].upper()
                    if 'CONTRAT' in next_sheet or 'CONTRTOS' in next_sheet:
                        s_cont = sheets[i + 1]
                        i += 1
                if s_cont:
                    pares_a_procesar.append((s_paro, s_cont))
            i += 1

        print(f"DEBUG: Pares (PARO/CONTRATOS) detectados para procesar: {len(pares_a_procesar)}")
        all_data = []

        for s_paro, s_cont in pares_a_procesar:
            df_p = pd.read_excel(xls, sheet_name=s_paro)
            df_c = pd.read_excel(xls, sheet_name=s_cont)

            if df_p.empty or df_c.empty:
                continue

            clean_p = procesar_hoja_sepe(df_p, tipo='paro')
            clean_c = procesar_hoja_sepe(df_c, tipo='contratos')
            merged = pd.merge(clean_p, clean_c, on='id_municipio', how='outer', suffixes=('', '_cont'))

            if 'nombre_municipio_cont' in merged.columns:
                merged['nombre_municipio'] = merged['nombre_municipio'].fillna(merged['nombre_municipio_cont'])
                merged = merged.drop(columns=['nombre_municipio_cont'])
            all_data.append(merged)

        if not all_data:
            print("WARNING: No se pudo extraer ningún dato de las hojas encontradas.")
            return None

        df_ml = pd.concat(all_data, ignore_index=True)
        cols_numericas = df_ml.columns.drop(['id_municipio', 'nombre_municipio'])
        df_ml[cols_numericas] = df_ml[cols_numericas].fillna(0).astype(float).astype(int)

        print(f"DEBUG: Exportando {len(df_ml)} filas a {path_salida}")
        guardar_dataframe_csv(df_ml, path_salida)
        return df_ml

    except Exception as e:
        print(f"\nError crítico en limpieza_sepe: {e}")
        return None
