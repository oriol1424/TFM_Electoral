import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from EDA.visuals import (
    plot_smart_bar, 
    plot_distribution_analysis, 
    plot_scatter_regression,
    mapear_nombres_provincias,
    plot_missing_demographics,
    plot_histogram
)

def auditar_nulos_desigualdad(df_gini: pd.DataFrame, anyo: str) -> None:
    """
    Imprime la cantidad exacta de valores nulos (NaN) para las columnas 
    de Gini y P80/P20 en un año específico antes de limpiar el dataset.
    """
    col_gini = f'Índice de Gini {anyo}'
    col_p80p20 = f'Distribución de la renta P80/P20 {anyo}'
    
    print(f"\nAUDITORÍA DE DATOS FALTANTES DESIGUALDAD ({anyo})")
    total_filas = len(df_gini)
    print(f"Total de municipios en el dataset: {total_filas}")
    
    if col_gini in df_gini.columns:
        nulos_gini = df_gini[col_gini].isna().sum()
        pct_gini = (nulos_gini / total_filas) * 100
        print(f"Nulos en Gini:      {nulos_gini} municipios ({pct_gini:.1f}% del total)")
    
    if col_p80p20 in df_gini.columns:
        nulos_p80p20 = df_gini[col_p80p20].isna().sum()
        pct_p80p20 = (nulos_p80p20 / total_filas) * 100
        print(f"Nulos en P80/P20:   {nulos_p80p20} municipios ({pct_p80p20:.1f}% del total)")


def municipios_nulos(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Analiza los municipios con valores nulos en el dataset de desigualdad
    utilizando la función genérica de visuals.py.
    """
    col_gini = f'Índice de Gini {anyo}'
    
    if col_gini not in df_gini.columns:
        print(f"Error: No existe la columna '{col_gini}' en el dataset.")
        return
        
    df_gini_nulos = df_gini[df_gini[col_gini].isna()].copy()
    
    if not df_gini_nulos.empty:
        plot_missing_demographics(df_gini_nulos, df_pob, anyo, title_suffix="Desigualdad")
    else:
        print(f"No se detectaron nulos en desigualdad para el año {anyo}.")


def verificar_datos_municipios_pequenos(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Busca si existe algún municipio con 100 o menos habitantes que tenga datos de Gini
    y muestra un resumen por provincia.
    """
    col_gini = f'Índice de Gini {anyo}'
    col_pob = 'poblacion' if 'poblacion' in df_pob.columns else 'poblacion_total'
    
    if col_gini not in df_gini.columns:
        return

    # Preparar datos
    df_gini_temp = df_gini.copy()
    df_gini_temp['Cod_Muni'] = df_gini_temp['Cod_Muni'].astype(str).str.zfill(5)
    
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    df_pob_temp = df_pob.copy()
    df_pob_temp[col_id_pob] = df_pob_temp[col_id_pob].astype(str).str.zfill(5)

    # Filtrar municipios pequeños y cruzar
    pequenos = df_pob_temp[df_pob_temp[col_pob] <= 100].copy()
    df_merge = pd.merge(
        pequenos,
        df_gini_temp[['Cod_Muni', col_gini, 'Nombre_Muni']],
        left_on=col_id_pob,
        right_on='Cod_Muni',
        how='inner'
    )
    
    # Filtrar los que TIENEN datos
    con_datos = df_merge[df_merge[col_gini].notna()].copy()

    print(f"\nANÁLISIS DE EXCEPCIONES AL SECRETO ESTADÍSTICO (<= 100 hab. en {anyo})")
    
    if con_datos.empty:
        print("No hay información de ningún municipio con menos de 100 habitantes (Secreto Estadístico aplicado correctamente).")
    else:
        con_datos = mapear_nombres_provincias(con_datos, anyo)
        
        # Resumen por provincia
        resumen_prov = con_datos['nombre_provincia'].value_counts()
        print(f"Se han encontrado {len(con_datos)} municipios con datos a pesar de su pequeño tamaño (<= 100 hab).")
        print("\nRESUMEN DE EXCEPCIONES POR PROVINCIA:")
        for prov, cant in resumen_prov.items():
            print(f" - {prov}: {cant} municipio(s)")


def gestion_nulos_desigualdad(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Función orquestadora para la gestión y diagnóstico de los valores nulos.
    Llama primero a la auditoría general y luego cruza los datos con la demografía.
    """
    auditar_nulos_desigualdad(df_gini, anyo)
    municipios_nulos(df_gini, df_pob, anyo)
    verificar_datos_municipios_pequenos(df_gini, df_pob, anyo)
    

def procesar_desigualdad_anyo(df_gini: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Filtra las columnas del año específico, limpia los valores nulos debido al 
    secreto estadístico del INE y estandariza los nombres de las columnas.
    """
    col_gini = f'Índice de Gini {anyo}'
    col_p80p20 = f'Distribución de la renta P80/P20 {anyo}'
    
    if col_gini not in df_gini.columns or col_p80p20 not in df_gini.columns:
        print(f"Error: No se encuentran las columnas de Gini o P80/P20 para el año {anyo}.")
        print("Columnas disponibles:", df_gini.columns.tolist())
        return pd.DataFrame()
        
    df_filtrado = df_gini[['Cod_Muni', 'Nombre_Muni', col_gini, col_p80p20]].copy()
    
    df_filtrado.rename(columns={
        col_gini: 'gini',
        col_p80p20: 'p80_p20'
    }, inplace=True)
    
    nulos_iniciales = df_filtrado['gini'].isna().sum()
    total_munis = len(df_filtrado)
    
    print(f"Info: {nulos_iniciales} de {total_munis} municipios NO tienen datos de renta en {anyo}.")
    df_clean = df_filtrado.dropna(subset=['gini', 'p80_p20']).copy()
    
    return df_clean


def mostrar_extremos_desigualdad(df_clean: pd.DataFrame, anyo: str) -> None:
    """
    Imprime los 10 municipios más desiguales y los 10 más igualitarios.
    """
    top_desiguales = df_clean.nlargest(10, 'gini')
    top_igualitarios = df_clean.nsmallest(10, 'gini')
    
    print(f"\nTOP 10 MUNICIPIOS MÁS DESIGUALES ({anyo})")
    for i, row in enumerate(top_desiguales.itertuples(), 1):
        print(f"{i:2d}. {row.Nombre_Muni:<25} | Gini: {row.gini:.1f} | Los ricos ganan {row.p80_p20:.1f}x más que los pobres")
        
    print(f"\nTOP 10 MUNICIPIOS MÁS IGUALITARIOS ({anyo})")
    for i, row in enumerate(top_igualitarios.itertuples(), 1):
        print(f"{i:2d}. {row.Nombre_Muni:<25} | Gini: {row.gini:.1f} | Los ricos ganan {row.p80_p20:.1f}x más que los pobres")
    print("-" * 50)


def graficar_desigualdad(df_clean: pd.DataFrame, anyo: str) -> None:
    """
    Genera un histograma de la distribución del Gini y un scatter plot 
    para ver la relación Gini vs P80/P20.
    """
    # 1. Histograma detallado de Gini
    plot_histogram(
        df=df_clean,
        num_col='gini',
        title=f'Frecuencia de Municipios por Índice de Gini ({anyo})',
        xlabel='Índice de Gini (Más alto = Más desigualdad)',
        ylabel='Número de Municipios',
        bins=35,
        color='indigo'
    )

    # 2. Análisis combinado (Boxplot + Histograma)
    plot_distribution_analysis(
        df=df_clean,
        num_col='gini',
        title=f'Distribución de la Desigualdad (Gini) en {anyo}',
        color='purple'
    )
    
    # 3. Correlación y Dispersión 
    plot_scatter_regression(
        df=df_clean,
        x_col='gini',
        y_col='p80_p20',
        title=f'Polarización Económica: Gini vs Ratio P80/P20 ({anyo})',
        color='teal'
    )

def eda_gini_p80p20(df_gini_completo: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Orquestador del análisis de Desigualdad.
    """
    print(f" INICIANDO ANÁLISIS DE DESIGUALDAD Y RENTA ({anyo})")
    
    df_clean = procesar_desigualdad_anyo(df_gini_completo, anyo)
    
    if df_clean.empty:
        return df_clean
        
    mostrar_extremos_desigualdad(df_clean, anyo)
    graficar_desigualdad(df_clean, anyo)
    
    print("="*50 + "\n")
    return df_clean
