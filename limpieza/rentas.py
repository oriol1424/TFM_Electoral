import pandas as pd
from typing import Callable
from .funciones_genericas_limpieza import (
    limpiar_valor_numerico, formatear_serie_codigo,
    leer_archivo_csv, normalizar_nombres_columnas, guardar_dataframe_csv
)


def crear_lista_maestra(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la lista única de municipios con código y nombre."""
    col_muni = df.columns[0]
    df_temp = pd.DataFrame()
    df_temp['Cod_Muni'] = df[col_muni].astype(str).str.split(' ').str[0]
    df_temp['Nombre_Muni'] = df[col_muni].astype(str).str.split(' ', n=1).str[1]
    maestra = df_temp[df_temp['Cod_Muni'].str.isdigit()].drop_duplicates('Cod_Muni')
    return maestra[['Cod_Muni', 'Nombre_Muni']].copy()


def _pivotar_ine(df: pd.DataFrame, col_indicador: str) -> pd.DataFrame:
    """Limpia valores, agrupa por municipio y pivota un CSV de renta del INE."""
    df['Total_Num'] = df['Total'].apply(lambda x: limpiar_valor_numerico(x, to_nan=True))
    df['Cod_Muni'] = formatear_serie_codigo(df[df.columns[0]], 5)
    df_clean = df.dropna(subset=['Total_Num'])
    df_agrupado = df_clean.groupby(['Cod_Muni', col_indicador, 'Periodo'])['Total_Num'].mean().reset_index()
    df_pivot = df_agrupado.pivot_table(index='Cod_Muni', columns=[col_indicador, 'Periodo'], values='Total_Num')
    df_pivot.columns = [f"{col[0]} {int(col[1])}" for col in df_pivot.columns]
    return df_pivot.reset_index()


def _procesar_csv_ine(path_entrada: str, path_salida: str, fn_pivot: Callable) -> pd.DataFrame:
    """Lee un CSV del INE, extrae lista maestra, aplica fn_pivot y guarda el resultado."""
    try:
        df = leer_archivo_csv(path_entrada)
        df = normalizar_nombres_columnas(df)
        maestra = crear_lista_maestra(df)
        datos = fn_pivot(df)
        resultado = pd.merge(maestra, datos, on='Cod_Muni', how='left')
        guardar_dataframe_csv(resultado, path_salida)
        return resultado
    except Exception as e:
        print(f"Error crítico procesando archivo INE: {e}")
        raise


def procesar_archivo_ine(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Orquesta el proceso de indicadores de renta media (30824.csv)."""
    return _procesar_csv_ine(path_entrada, path_salida,
                             lambda df: _pivotar_ine(df, 'Indicadores de renta media'))


def procesar_archivo_gini(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Orquesta el proceso para Gini y Distribución P80/P20 (37677.csv)."""
    return _procesar_csv_ine(path_entrada, path_salida,
                             lambda df: _pivotar_ine(df, 'Indicadores de renta media'))


def procesar_fuente_ingresos(path_entrada: str, path_salida: str) -> pd.DataFrame:
    """Orquesta el proceso para el archivo de Fuente de Ingresos (30825.csv)."""
    col = 'Distribución por fuente de ingresos'
    def _pivot(df):
        df[col] = df[col].astype(str).str.split(': ').str[-1]
        return _pivotar_ine(df, col)
    return _procesar_csv_ine(path_entrada, path_salida, _pivot)
