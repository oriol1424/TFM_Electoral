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

def _resolver_cols_desigualdad(df: pd.DataFrame, anyo: str) -> tuple[str, str, str]:
    """
    Resuelve los nombres de las columnas de Gini, P80/P20 e ID buscando 
    primero con el año y luego el nombre base.
    """
    col_gini_anyo = f'Índice de Gini {anyo}'
    col_p80p20_anyo = f'Distribución de la renta P80/P20 {anyo}'
    
    col_gini = col_gini_anyo if col_gini_anyo in df.columns else 'Índice de Gini'
    col_p80p20 = col_p80p20_anyo if col_p80p20_anyo in df.columns else 'Distribución de la renta P80/P20'
    
    col_id = None
    for posible_id in ['Cod_Muni', 'cod_Muni', 'Código', 'Codigo', 'id_municipio', 'index', 'CPROCMUN', 'id']:
        if posible_id in df.columns:
            col_id = posible_id
            break
            
    if col_id is None:
        for c in df.columns:
            if any(key in c.lower() for key in ['cod', 'muni', 'id']):
                col_id = c
                break
                
    if col_id is None:
        posibles = [c for c in df.columns if c not in [col_gini, col_p80p20]]
        if posibles:
            col_id = posibles[0]
            
    return col_gini, col_p80p20, col_id


def auditar_nulos_desigualdad(df_gini: pd.DataFrame, anyo: str) -> None:
    """
    Imprime la cantidad exacta de valores nulos (NaN) para las columnas 
    de Gini y P80/P20 en un año específico antes de limpiar el dataset.
    """
    col_gini, col_p80p20, _ = _resolver_cols_desigualdad(df_gini, anyo)
    
    print(f"\nAUDITORÍA DE DATOS FALTANTES DESIGUALDAD ({anyo})")
    total_filas = len(df_gini)
    print(f"Total de municipios en el dataset: {total_filas}")
    
    if col_gini in df_gini.columns:
        nulos_gini = df_gini[col_gini].isna().sum()
        pct_gini = (nulos_gini / total_filas) * 100
        print(f"Nulos en Gini ({col_gini}):      {nulos_gini} municipios ({pct_gini:.1f}% del total)")
    
    if col_p80p20 in df_gini.columns:
        nulos_p80p20 = df_gini[col_p80p20].isna().sum()
        pct_p80p20 = (nulos_p80p20 / total_filas) * 100
        print(f"Nulos en P80/P20 ({col_p80p20}):   {nulos_p80p20} municipios ({pct_p80p20:.1f}% del total)")


def municipios_nulos(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Analiza los municipios con valores nulos en el dataset de desigualdad
    utilizando la función genérica de visuals.py.
    """
    col_gini, _, col_id = _resolver_cols_desigualdad(df_gini, anyo)
    
    if col_gini not in df_gini.columns:
        print(f"Error: No existe la columna de Gini en el dataset.")
        return
        
    df_gini_nulos = df_gini[df_gini[col_gini].isna()].copy()
    
    if not df_gini_nulos.empty:
        if 'Cod_Muni' not in df_gini_nulos.columns:
            if col_id:
                df_gini_nulos['Cod_Muni'] = df_gini_nulos[col_id]
            else:
                df_gini_nulos = df_gini_nulos.reset_index().rename(columns={'index': 'Cod_Muni'})
            
        plot_missing_demographics(df_gini_nulos, df_pob, anyo, title_suffix="Desigualdad")
    else:
        print(f"No se detectaron nulos en desigualdad para el año {anyo}.")


def verificar_datos_municipios_pequenos(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Busca si existe algún municipio con 100 o menos habitantes que tenga datos de Gini
    y muestra un resumen por provincia.
    """
    col_gini, _, col_id = _resolver_cols_desigualdad(df_gini, anyo)
    col_pob = 'poblacion' if 'poblacion' in df_pob.columns else 'poblacion_total'
    
    if col_gini not in df_gini.columns:
        return

    df_gini_temp = df_gini.copy()
    
    if col_id:
        df_gini_temp['Cod_Muni'] = df_gini_temp[col_id].astype(str).str.zfill(5)
    else:
        print("Aviso: No se pudo identificar la columna de código municipal.")
        return
    
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    df_pob_temp = df_pob.copy()
    df_pob_temp[col_id_pob] = df_pob_temp[col_id_pob].astype(str).str.zfill(5)

    pequenos = df_pob_temp[df_pob_temp[col_pob] <= 100].copy()
    
    col_nombre = 'Nombre_Muni' if 'Nombre_Muni' in df_gini_temp.columns else (df_gini_temp.columns[1] if len(df_gini_temp.columns) > 1 else col_id)

    df_merge = pd.merge(
        pequenos,
        df_gini_temp[['Cod_Muni', col_gini, col_nombre]],
        left_on=col_id_pob,
        right_on='Cod_Muni',
        how='inner'
    )
    
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
    col_gini, col_p80p20, col_id = _resolver_cols_desigualdad(df_gini, anyo)
    
    if col_gini not in df_gini.columns or col_p80p20 not in df_gini.columns:
        print(f"Error: No se encuentran las columnas de Gini o P80/P20 para el año {anyo}.")
        print("Columnas disponibles:", df_gini.columns.tolist())
        return pd.DataFrame()
        
    col_nombre = None
    prioritarios_nombre = ['Nombre_Muni', 'Nombre', 'Municipio', 'Nombre del municipio', 'nombre_muni']
    
    for n in prioritarios_nombre:
        if n in df_gini.columns and n != col_id:
            col_nombre = n
            break
            
    if col_nombre is None:
        posibles = [c for c in df_gini.columns if c not in [col_id, col_gini, col_p80p20]]
        if posibles:
            col_nombre = posibles[0]
        else:
            col_nombre = col_id 
            
    if col_id == col_nombre:
        df_filtrado = df_gini[[col_id, col_gini, col_p80p20]].copy()
        df_filtrado.insert(1, 'Nombre_Temp', df_filtrado[col_id]) # Duplicamos la columna para tener nombre
        mapping = {col_id: 'Cod_Muni', 'Nombre_Temp': 'Nombre_Muni', col_gini: 'gini', col_p80p20: 'p80_p20'}
    else:
        df_filtrado = df_gini[[col_id, col_nombre, col_gini, col_p80p20]].copy()
        mapping = {col_id: 'Cod_Muni', col_nombre: 'Nombre_Muni', col_gini: 'gini', col_p80p20: 'p80_p20'}

    df_filtrado.rename(columns=mapping, inplace=True)

    df_filtrado['Cod_Muni'] = df_filtrado['Cod_Muni'].astype(str).str.zfill(5)
    
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
    for i, (_, row) in enumerate(top_desiguales.iterrows(), 1):
        nombre = row['Nombre_Muni']
        print(f"{i:2d}. {nombre:<25} | Gini: {row['gini']:.1f} | Los ricos ganan {row['p80_p20']:.1f}x más que los pobres")
        
    print(f"\nTOP 10 MUNICIPIOS MÁS IGUALITARIOS ({anyo})")
    for i, (_, row) in enumerate(top_igualitarios.iterrows(), 1):
        nombre = row['Nombre_Muni']
        print(f"{i:2d}. {nombre:<25} | Gini: {row['gini']:.1f} | Los ricos ganan {row['p80_p20']:.1f}x más que los pobres")


def graficar_desigualdad(df_clean: pd.DataFrame, anyo: str) -> None:
    """
    Genera un histograma de la distribución del Gini y un scatter plot 
    para ver la relación Gini vs P80/P20.
    """
    plot_histogram(
        df=df_clean,
        num_col='gini',
        title=f'Frecuencia de Municipios por Índice de Gini ({anyo})',
        xlabel='Índice de Gini (Más alto = Más desigualdad)',
        ylabel='Número de Municipios',
        bins=35,
        color='indigo'
    )

    plot_distribution_analysis(
        df=df_clean,
        num_col='gini',
        title=f'Distribución de la Desigualdad (Gini) en {anyo}',
        color='purple'
    )
    
    plot_scatter_regression(
        df=df_clean,
        x_col='gini',
        y_col='p80_p20',
        title=f'Polarización Económica: Gini vs Ratio P80/P20 ({anyo})',
        color='teal'
    )

def analizar_similitud_vecinal_economica(df: pd.DataFrame, w, anyo: str):
    """
    Calcula la diferencia porcentual media entre cada municipio y sus vecinos físicos.
    IMPORTANTE: Los municipios sin datos son eliminados del análisis y no cuentan 
    para el promedio de sus vecinos.
    """
    import numpy as np
    from libpysal.weights import w_subset
    from libpysal.weights import lag_spatial
    
    print(f"\nANÁLISIS DE COHESIÓN VECINAL FILTRADO ({anyo})")
    
    cols_estudio = {
        'renta media persona': 'Renta Persona',
        'indice gini': 'Índice Gini',
        'salarios': '% Salarios'
    }
    
    df_sp = df.copy()
    if 'municipio' in df_sp.columns:
        df_sp['municipio'] = df_sp['municipio'].astype(str).str.zfill(5)
        df_sp = df_sp.set_index('municipio')

    ids_con_datos = df_sp[df_sp['renta media persona'].notna()].index.tolist()
    ids_validos = [m for m in ids_con_datos if m in w.id_order]
    
    print(f"Municipios con datos económicos encontrados en el grafo: {len(ids_validos)}")
    w_sub = w_subset(w, ids_validos, silence_warnings=True)
    w_sub.transform = 'r' 
    
    df_analisis = df_sp.loc[w_sub.id_order].copy()
    
    col_plot_names = []
    
    for col, alias in cols_estudio.items():
        if col not in df_analisis.columns:
            continue
        
        y = df_analisis[col].values
        
        lag = lag_spatial(w_sub, y)
        
        diff_col = f'Similitud {alias}'
        df_analisis[diff_col] = np.where(y != 0, (abs(y - lag) / y) * 100, 0)
        col_plot_names.append(diff_col)

    islas_finales = w_sub.islands
    df_analisis = df_analisis[~df_analisis.index.isin(islas_finales)].copy()
    
    print(f"Municipios finales analizados (excluyendo islas resultantes): {len(df_analisis)}")
    resumen = df_analisis.groupby('rango tamaño población', observed=False)[col_plot_names].mean().reset_index()
    
    df_melted = resumen.melt(id_vars='rango tamaño población', value_vars=col_plot_names, 
                             var_name='Indicador', value_name='Diferencia %')
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_melted, x='rango tamaño población', y='Diferencia %', hue='Indicador', palette='viridis')
    
    plt.title(f'Cohesión Territorial: Diferencia con Vecinos con Datos ({anyo})\n(Excluye vecinos sin información del promedio)', fontsize=14)
    plt.ylabel('Desviación respecto a los vecinos (%)')
    plt.xlabel('Tamaño del Municipio')
    plt.xticks(rotation=45)
    plt.legend(title='Indicadores', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

    return resumen



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
    
    return df_clean

