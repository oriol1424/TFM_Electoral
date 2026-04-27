import pandas as pd
import seaborn as sns
from EDA.visuals import plot_smart_bar, plot_distribution_analysis, plot_scatter_regression

def auditar_nulos_desigualdad(df_gini: pd.DataFrame, anyo: str) -> None:
    """
    Imprime la cantidad exacta de valores nulos (NaN) para las columnas 
    de Gini y P80/P20 en un año específico antes de limpiar el dataset.
    """
    col_gini = f'Índice de Gini {anyo}'
    col_p80p20 = f'Distribución de la renta P80/P20 {anyo}'
    
    print(f"\nAUDITORÍA DE DATOS FALTANTES ({anyo})")
    total_filas = len(df_gini)
    print(f"Total de municipios en el dataset: {total_filas}")
    
    if col_gini in df_gini.columns:
        nulos_gini = df_gini[col_gini].isna().sum()
        pct_gini = (nulos_gini / total_filas) * 100
        print(f"Nulos en Gini:      {nulos_gini} municipios ({pct_gini:.1f}% del total)")
    else:
        print(f"La columna '{col_gini}' NO existe en el DataFrame.")
        
    if col_p80p20 in df_gini.columns:
        nulos_p80p20 = df_gini[col_p80p20].isna().sum()
        pct_p80p20 = (nulos_p80p20 / total_filas) * 100
        print(f"Nulos en P80/P20:   {nulos_p80p20} municipios ({pct_p80p20:.1f}% del total)")
    else:
        print(f"La columna '{col_p80p20}' NO existe en el DataFrame.")
        

def municipios_nulos(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Analiza los municipios con valores nulos en el dataset de desigualdad,
    cruzándolos con los datos demográficos para ver a qué tamaño de población pertenecen,
    añade porcentajes al gráfico y muestra a qué provincias pertenecen.
    """
    col_gini = f'Índice de Gini {anyo}'
    
    if col_gini not in df_gini.columns:
        print(f"Error: No existe la columna '{col_gini}' en el dataset.")
        return
        
    df_gini_nulos = df_gini[df_gini[col_gini].isna()].copy()
    df_gini_nulos['Cod_Muni'] = df_gini_nulos['Cod_Muni'].astype(str).str.zfill(5)
    
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    df_pob_temp = df_pob.copy()
    df_pob_temp[col_id_pob] = df_pob_temp[col_id_pob].astype(str).str.zfill(5)
    
    codigos_pob = set(df_pob_temp[col_id_pob])
    fantasmas = df_gini_nulos[~df_gini_nulos['Cod_Muni'].isin(codigos_pob)]
    
    print(f"\nMUNICIPIOS FANTASMA DETECTADOS ({anyo})")
    if fantasmas.empty:
        print("No se han detectado municipios en el histórico que falten en el padrón.")
    else:
        print(f"Hay {len(fantasmas)} municipios en la lista de Renta que NO existen en Población.")
        print("Se descartarán automáticamente para el análisis.")
            
    cols_a_cruzar = [col_id_pob, 'tamano_municipio']
    if 'id_provincia' in df_pob_temp.columns:
        cols_a_cruzar.append('id_provincia')
    if 'nombre_provincia' in df_pob_temp.columns:
        cols_a_cruzar.append('nombre_provincia')
        
    nulos_reales = df_gini_nulos[df_gini_nulos['Cod_Muni'].isin(codigos_pob)]
    
    if nulos_reales.empty:
        print("\nNo hay municipios nulos reales cruzables con el padrón poblacional.")
        return
        
    df_merge = pd.merge(
        nulos_reales, 
        df_pob_temp[cols_a_cruzar], 
        left_on='Cod_Muni', 
        right_on=col_id_pob, 
        how='inner'
    )
    
    conteo_tamano = df_merge['tamano_municipio'].value_counts().reset_index()
    conteo_tamano.columns = ['tamano_municipio', 'cantidad']
    
    plot_smart_bar(
        df=conteo_tamano, 
        cat_col='tamano_municipio',
        val_col='cantidad',
        orientation='h', 
        title=f'Perfil Demográfico de los Municipios Ocultos ({anyo})',
        xlabel='Cantidad de Municipios sin datos de renta',
        ylabel='Categoría de Tamaño Poblacional',
        palette='magma'
    )

    print(f"\nIMPACTO DEL SECRETO ESTADÍSTICO POR PROVINCIA ({anyo})")
    
    col_prov = 'nombre_provincia' if 'nombre_provincia' in df_merge.columns else 'id_provincia'
    
    if col_prov in df_merge.columns:
        conteo_prov = df_merge[col_prov].value_counts()
        total_nulos_reales = conteo_prov.sum()
        print("Top 15 provincias con más municipios ocultados por el INE:")
        for prov, cant in conteo_prov.head(15).items():
            pct_prov = (cant / total_nulos_reales) * 100
            print(f" - {str(prov):<20}: {cant:4d} municipios ({pct_prov:.1f}% de los ocultos)")
    else:
        print("No se encontró información provincial en los datos demográficos.")


def gestion_nulos_desigualdad(df_gini: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Función orquestadora para la gestión y diagnóstico de los valores nulos.
    Llama primero a la auditoría general y luego cruza los datos con la demografía.
    """
    auditar_nulos_desigualdad(df_gini, anyo)
    municipios_nulos(df_gini, df_pob, anyo)
    

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
    plot_distribution_analysis(
        df=df_clean,
        num_col='gini',
        title=f'Distribución de la Desigualdad (Gini) en {anyo}',
        color='purple'
    )
    
    # 2. Correlación y Dispersión 
    plot_scatter_regression(
        df=df_clean,
        x_col='gini',
        y_col='p80_p20',
        title=f'Polarización Económica: Gini vs Ratio P80/P20 ({anyo})',
        color='teal'
    )
    """"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df_clean['gini'], bins=50, kde=True, color='purple', ax=axes[0])
    axes[0].set_title(f'Distribución de la Desigualdad (Gini) en {anyo}')
    axes[0].set_xlabel('Índice de Gini (Más alto = Más desigualdad)')
    axes[0].set_ylabel('Número de Municipios')
    
    media_gini = df_clean['gini'].mean()
    axes[0].axvline(media_gini, color='red', linestyle='--', label=f'Media Nac: {media_gini:.1f}')
    axes[0].legend()
    
    sns.scatterplot(
        data=df_clean, x='gini', y='p80_p20', 
        alpha=0.5, color='teal', edgecolor=None, ax=axes[1]
    )
    axes[1].set_title(f'Polarización Económica: Gini vs Ratio P80/P20 ({anyo})')
    axes[1].set_xlabel('Índice de Gini')
    axes[1].set_ylabel('Ratio P80/P20\n(Veces que los ingresos del top 20% superan al bottom 20%)')
    
    municipio_max_gini = df_clean.loc[df_clean['gini'].idxmax()]
    axes[1].text(municipio_max_gini['gini'], municipio_max_gini['p80_p20'], 
                 municipio_max_gini['Nombre_Muni'], fontsize=9, color='red')
    plt.tight_layout()
    plt.show()
"""
    
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