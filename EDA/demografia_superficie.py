import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from EDA.visuals import plot_smart_bar

def mostrar_distribucion_poblacion(df_csv: pd.DataFrame, data_json: Dict[str, Any]) -> None:
    """
    Extrae la población total del JSON y genera un histograma en escala logarítmica 
    usando el DataFrame CSV para visualizar la distribución de la población.
    Args:
        df_csv (pd.DataFrame): DataFrame con la población a nivel municipal.
        data_json (Dict[str, Any]): Diccionario extraído del JSON con metadatos nacionales.
    Returns:
        None: Solo imprime por pantalla y muestra el gráfico.
    """
    poblacion_total = data_json.get('metadatos', {}).get('total_nacional_poblacion', 0)
    
    print(f"RESUMEN DEMOGRÁFICO")
    print(f"Población total: {poblacion_total:,.0f} habitantes\n")
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.histplot(df_csv['poblacion_total'], bins=50, log_scale=True, color='skyblue')
    plt.title('Distribución de la Población por Municipio (Escala Logarítmica)')
    plt.xlabel('Población Total (escala logarítmica)')
    plt.ylabel('Cantidad de Municipios')
    plt.tight_layout()
    plt.show()


def mostrar_extremos_poblacion(df_csv: pd.DataFrame) -> None:
    """
    Extrae e imprime los 10 municipios con mayor y menor población.
    Args:
        df_csv (pd.DataFrame): DataFrame con la población a nivel municipal.
    Returns:
        None: Imprime los resultados formateados por consola.
    """
    top_10 = df_csv.nlargest(10, 'poblacion_total')[['nombre_municipio', 'poblacion_total']]
    bottom_10 = df_csv.nsmallest(10, 'poblacion_total')[['nombre_municipio', 'poblacion_total']]
    
    print("10 MUNICIPIOS MÁS POBLADOS")
    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"{i:2d}. {row.nombre_municipio:<30} | {row.poblacion_total:>10,.0f} habs.")
        
    print("\n10 MUNICIPIOS MENOS POBLADOS")
    for i, row in enumerate(bottom_10.itertuples(), 1):
        print(f"{i:2d}. {row.nombre_municipio:<30} | {row.poblacion_total:>10,.0f} habs.")


def procesar_superficie_y_densidad(df_pob: pd.DataFrame, df_sup: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza los datos de población y superficie, calcula la densidad poblacional 
    y segmenta los municipios según los estratos oficiales del Padrón del INE.
    Args:
        df_pob (pd.DataFrame): DataFrame con la población municipal.
        df_sup (pd.DataFrame): DataFrame con la superficie y coordenadas.
    Returns:
        pd.DataFrame: Un nuevo DataFrame cruzado con las columnas 'densidad_pob' 
                      y 'tamano_municipio' añadidas.
    """
    df_merged = pd.merge(df_pob, df_sup, on='id_municipio', how='inner')
    df_merged['superficie_km2'] = pd.to_numeric(
        df_merged['superficie_km2'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    
    df_merged['densidad_pob'] = np.where(
        df_merged['superficie_km2'] > 0, 
        df_merged['poblacion_total'] / df_merged['superficie_km2'], 
        np.nan
    )
    
    bins = [0, 101, 501, 1001, 2001, 5001, 10001, 20001, 50001, 100001, 500001, np.inf]
    labels = [
        '<101', '101-500', '501-1.000', '1.001-2.000', '2.001-5.000', 
        '5.001-10.000', '10.001-20.000', '20.001-50.000', '50.001-100.000', 
        '100.001-500.000', '>500.000'
    ]
    
    df_merged['tamano_municipio'] = pd.cut(
        df_merged['poblacion_total'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    return df_merged


def mostrar_analisis_superficie(df_geo: pd.DataFrame) -> None:
    """
    Genera gráficos de barras comparando el porcentaje de municipios frente 
    al porcentaje de población total por tamaño (estratos INE), y realiza 
    comprobaciones de calidad.
    Args:
        df_geo (pd.DataFrame): DataFrame ya procesado con superficie y densidad.
    Returns:
        None: Muestra los gráficos e imprime las alertas de calidad de datos.
    """
    total_municipios = len(df_geo)
    total_poblacion = df_geo['poblacion_total'].sum()
    
    resumen = df_geo.groupby('tamano_municipio', observed=False).agg(
        num_municipios=('poblacion_total', 'count'),
        pob_total=('poblacion_total', 'sum')
    ).reset_index()
    
    resumen['pct_municipios'] = (resumen['num_municipios'] / total_municipios) * 100
    resumen['pct_poblacion'] = (resumen['pob_total'] / total_poblacion) * 100
    
    plot_smart_bar(
        df=resumen,
        cat_col='tamano_municipio',
        val_col='pct_municipios',
        orientation='v',
        title='Distribución de Municipios (%)',
        xlabel='Estratos de Población (INE)',
        ylabel='Porcentaje del Total de Municipios (%)',
        palette='viridis',
        is_percentage=True,
        rotation=45
    )
    
    plot_smart_bar(
        df=resumen,
        cat_col='tamano_municipio',
        val_col='pct_poblacion',
        orientation='v',
        title='Distribución de la Población (%)',
        xlabel='Estratos de Población (INE)',
        ylabel='Porcentaje de la Población Total (%)',
        palette='magma',
        is_percentage=True,
        rotation=45
    )
    
    municipios_sup_cero = df_geo[df_geo['superficie_km2'] <= 0]
    lat_fuera = df_geo[(df_geo['latitud'] < 27.0) | (df_geo['latitud'] > 44.0)]
    lon_fuera = df_geo[(df_geo['longitud'] < -18.5) | (df_geo['longitud'] > 4.5)]
    
    print("CONTROL DE CALIDAD GEOGRÁFICO")
    pob_total_nacional = df_geo['poblacion_total'].sum()
    sup_total_nacional = df_geo['superficie_km2'].sum()
    densidad_nacional = pob_total_nacional / sup_total_nacional
    print(f"Densidad media nacional: {densidad_nacional:.2f} hab/km2")
    print(f"Municipios con superficie 0 o negativa: {len(municipios_sup_cero)}")
    print(f"Municipios con latitud fuera de rango: {len(lat_fuera)}")
    print(f"Municipios con longitud fuera de rango: {len(lon_fuera)}")

def analizar_distribucion_provincial(df_municipal: pd.DataFrame, data_json: Any) -> pd.DataFrame:
    """
    Analiza la distribución provincial, filtrando Ceuta y Melilla del gráfico 
    para normalizar la escala de densidades.
    """
    provincias_dict = data_json.get('provincias', {})
    if isinstance(data_json, pd.DataFrame):
        data_json = data_json.to_dict()
        provincias_dict = data_json.get('provincias', {})

    lista_para_df = []
    for codigo_prov, info in provincias_dict.items():
        if isinstance(info, dict):
            lista_para_df.append({
                'id_provincia': str(codigo_prov).zfill(2),
                'nombre_provincia': info.get('nombre_provincia')
            })
    
    df_nombres_prov = pd.DataFrame(lista_para_df)

    df_municipal['id_provincia'] = df_municipal['id_provincia'].astype(str).str.zfill(2)
    df_merged = pd.merge(df_municipal, df_nombres_prov, on='id_provincia', how='left')

    df_prov = df_merged.groupby(['id_provincia', 'nombre_provincia'], observed=False).agg(
        num_municipios=('id_municipio', 'count'),
        poblacion_total=('poblacion_total', 'sum'),
        superficie_total=('superficie_km2', 'sum')
    ).reset_index()

    total_pob_nac = df_prov['poblacion_total'].sum()
    total_sup_nac = df_prov['superficie_total'].sum()
    df_prov['pct_poblacion'] = (df_prov['poblacion_total'] / total_pob_nac) * 100
    df_prov['pct_superficie'] = (df_prov['superficie_total'] / total_sup_nac) * 100
    df_prov['densidad_provincial'] = df_prov['poblacion_total'] / df_prov['superficie_total']

    df_top15 = df_prov.sort_values('poblacion_total', ascending=False).head(15)
    
    plot_smart_bar(
        df=df_top15,
        cat_col='nombre_provincia',
        val_col='pct_poblacion',
        orientation='h',
        title='% de Población Nacional por Provincia (Top 15)',
        xlabel='Porcentaje de Población (%)',
        ylabel='Provincia',
        palette='viridis',
        is_percentage=True
    )

    df_prov_peninsular = df_prov[~df_prov['id_provincia'].isin(['51', '52'])].copy()
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.scatterplot(
        data=df_prov_peninsular, x='pct_superficie', y='pct_poblacion', 
        size='densidad_provincial', hue='densidad_provincial',
        palette='flare', sizes=(100, 2000), alpha=0.7
    )
    
    max_val = max(df_prov_peninsular['pct_superficie'].max(), df_prov_peninsular['pct_poblacion'].max())
    ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, label='Equilibrio (%Pob = %Sup)')
    
    for i, row in df_prov_peninsular.iterrows():
        if row['pct_poblacion'] > 4 or row['pct_superficie'] > 6 or row['nombre_provincia'] in ['Soria', 'Teruel']:
            ax.text(row['pct_superficie']+0.1, row['pct_poblacion'], row['nombre_provincia'], fontsize=9)

    ax.set_title('Desequilibrio Territorial (Excl. Ceuta y Melilla)')
    ax.set_xlabel('% de Superficie Nacional')
    ax.set_ylabel('% de Población Nacional')
    ax.legend(title='Densidad (hab/km2)', loc='lower right')

    plt.tight_layout()
    plt.show()

    print("RESUMEN ESTADÍSTICO: CIUDADES AUTÓNOMAS (Excluidas del gráfico por densidad extrema)")
    for code in ['51', '52']:
        row = df_prov[df_prov['id_provincia'] == code]
        if not row.empty:
            r = row.iloc[0]
            print(f"- {r.nombre_provincia}: {r.poblacion_total:,.0f} hab | "
                  f"{r.superficie_total:.2f} km2 | Densidad: {r.densidad_provincial:,.2f} hab/km2")

    return df_prov

def eda_demografia_superficie(df_pob: pd.DataFrame, data_json: Dict[str, Any], df_sup: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que ejecuta todo el bloque 1 del EDA secuencialmente.
    Args:
        df_pob (pd.DataFrame): DataFrame con la población municipal.
        data_json (Dict[str, Any]): Diccionario extraído del JSON.
        df_sup (pd.DataFrame): DataFrame con la superficie geográfica.
    Returns:
        pd.DataFrame: El DataFrame unificado y enriquecido listo para la Fase 2.
    """
    mostrar_distribucion_poblacion(df_pob, data_json)
    mostrar_extremos_poblacion(df_pob)
    
    df_enriquecido = procesar_superficie_y_densidad(df_pob, df_sup)
    
    mostrar_analisis_superficie(df_enriquecido)

    df_provincial = analizar_distribucion_provincial(df_enriquecido, data_json)
    
    return df_enriquecido

