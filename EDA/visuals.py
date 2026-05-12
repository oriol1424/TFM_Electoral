import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from scipy.stats import pearsonr
import json
import os

def obtener_mapeo_provincias(anyo: str) -> dict:
    """
    Lee el archivo JSON de población para obtener un mapeo de id_provincia a nombre_provincia.
    """
    path_json = f"data_processed/demografia/poblacion_{anyo}.json"
    if not os.path.exists(path_json):
        return {}
    
    try:
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        provincias = data.get('provincias', {})
        return {id_prov: info.get('nombre_provincia', id_prov) for id_prov, info in provincias.items()}
    except Exception:
        return {}

def mapear_nombres_provincias(df: pd.DataFrame, anyo: str) -> pd.DataFrame:
    """
    Añade o completa la columna 'nombre_provincia' en el DataFrame usando el mapeo del JSON.
    """
    # Estandarizar nombre de columna de entrada
    col_id = None
    for c in ['id_provincia', 'Cod_Muni', 'cod_muni', 'id_municipio']:
        if c in df.columns:
            col_id = c
            break
            
    if col_id and col_id != 'id_provincia':
        df['id_provincia'] = df[col_id].astype(str).str.zfill(5).str[:2]
        
    if 'id_provincia' in df.columns:
        mapeo = obtener_mapeo_provincias(anyo)
        if 'nombre_provincia' not in df.columns:
            df['nombre_provincia'] = df['id_provincia'].map(mapeo)
        else:
            df['nombre_provincia'] = df['nombre_provincia'].fillna(df['id_provincia'].map(mapeo))
    
    return df

def _limpiar_caracteres_especiales(texto: str) -> str:
    """
    Limpia caracteres corruptos (como el replacement character) que causan errores en fuentes de texto.
    """
    if not isinstance(texto, str): return texto
    return texto.replace('\ufffd', '').replace('', '')

def plot_votos_apilados_provincial(df_plot: pd.DataFrame, title: str):
    """
    Genera un gráfico de barras apiladas mostrando el peso relativo (%) de cada candidatura por provincia.
    """
    sns.set_theme(style="whitegrid")
    
    cols_evitar = ['id_provincia', 'nombre_provincia']
    cols_votos = [c for c in df_plot.columns if c.lower() not in cols_evitar]
    
    df_pct = df_plot.copy()
    index_col = 'nombre_provincia' if 'nombre_provincia' in df_pct.columns else 'id_provincia'
    df_pct = df_pct.set_index(index_col)

    df_pct.columns = [_limpiar_caracteres_especiales(c) for c in df_pct.columns]
    cols_votos = df_pct.columns.tolist()

    df_pct = df_pct[cols_votos].div(df_pct[cols_votos].sum(axis=1), axis=0) * 100
    df_pct = df_pct.loc[:, (df_pct != 0).any(axis=0)]
    
    ax = df_pct.plot(kind='barh', stacked=True, figsize=(14, 20), colormap='tab20')
    
    plt.title(_limpiar_caracteres_especiales(title), fontsize=16, pad=20)
    plt.xlabel('Porcentaje de Votos (%)')
    plt.ylabel('Provincia')
    plt.legend(title='Candidaturas', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_votos_individuales_por_provincia(df_final: pd.DataFrame):
    """
    Genera un gráfico de barras por cada provincia con etiquetas de porcentaje al lado.
    """
    cols_evitar = ['id_provincia', 'nombre_provincia', 'id_municipio', 'Nombre_Muni', 'nombre_muni', 'fecha_eleccion']
    cols_votos = [c for c in df_final.columns if c.lower() not in cols_evitar]
    
    print("\n--- Generando Gráficos Individuales por Provincia ---")
    
    for _, row in df_final.iterrows():
        prov_name = row['nombre_provincia'] if pd.notna(row['nombre_provincia']) else row['id_provincia']
        prov_name = _limpiar_caracteres_especiales(str(prov_name))
        
        data = row[cols_votos].sort_values(ascending=False)
        data = data[data > 0]
        if data.empty: continue
            
        data_pct = (data / data.sum()) * 100
        # Limpiar nombres de índices (partidos)
        data_pct.index = [_limpiar_caracteres_especiales(str(i)) for i in data_pct.index]
        
        plt.figure(figsize=(10, len(data_pct) * 0.5 + 1))
        ax = sns.barplot(x=data_pct.values, y=data_pct.index, palette='viridis', hue=data_pct.index, legend=False)
        
        plt.title(f"Distribución de Votos en {prov_name} (%)", fontsize=14, pad=15)
        plt.xlabel("Porcentaje de Votos (%)")
        plt.xlim(0, max(data_pct.values) * 1.2)
        
        for i, v in enumerate(data_pct.values):
            ax.text(v + 0.5, i, f'{v:.2f}%', color='black', va='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()

def plot_histogram(
    df: pd.DataFrame, 
    num_col: str, 
    title: str = "Histograma",
    xlabel: Optional[str] = None,
    ylabel: str = "Número de Municipios",
    bins: int = 30,
    color: str = "skyblue",
    figsize: tuple = (10, 6)
) -> None:
    """
    Genera un histograma sencillo con línea KDE (densidad kernel) para una variable numérica.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    
    ax = sns.histplot(data=df, x=num_col, bins=bins, color=color, kde=True)
    
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel if xlabel else num_col)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    plt.show()

def plot_smart_bar(
    df: pd.DataFrame, 
    cat_col: str, 
    val_col: Optional[str] = None,
    orientation: str = 'v', 
    title: str = "Gráfico de Barras", 
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    palette: str = "viridis",
    is_percentage: bool = False,
    width: float = 0.8,
    rotation: int = 0,
    decimals: int = 1
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    
    if val_col is None:
        plot_data = df[cat_col].value_counts().reset_index()
        plot_data.columns = [cat_col, 'valor']
        total = plot_data['valor'].sum()
    else:
        plot_data = df.copy()
        plot_data['valor'] = df[val_col]
        total = 100 if is_percentage else plot_data['valor'].sum()

    if orientation == 'v':
        ax = sns.barplot(
            data=plot_data, x=cat_col, y='valor', 
            palette=palette, hue=cat_col, legend=False, 
            dodge=False, width=width
        )
        ax.set_xlabel(xlabel if xlabel else cat_col)
        ax.set_ylabel(ylabel if ylabel else "Frecuencia")
        
        if rotation != 0:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
            
    elif orientation == 'h':
        ax = sns.barplot(
            data=plot_data, x='valor', y=cat_col, 
            palette=palette, hue=cat_col, legend=False, 
            dodge=False, width=width
        )
        ax.set_xlabel(xlabel if xlabel else "Frecuencia")
        ax.set_ylabel(ylabel if ylabel else cat_col)

    ax.set_title(title, pad=15)

    for p in ax.patches:
        val = p.get_height() if orientation == 'v' else p.get_width()
        if val > 0:
            if is_percentage:
                texto = f'{val:.{decimals}f}%'
            else:
                pct = (val / total) * 100
                texto = f'{int(val)}\n({pct:.{decimals}f}%)' if orientation == 'v' else f'{int(val)} ({pct:.{decimals}f}%)'
                
            if orientation == 'v':
                ax.annotate(texto, (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=9)
            else:
                ax.annotate(texto, (val, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_boxplot(
    df: pd.DataFrame, 
    cat_col: str, 
    num_col: str, 
    title: str = "Boxplot",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    palette: str = "Set2",
    rotation: int = 45
) -> None:
    """
    Genera un boxplot para comparar una variable numérica a través de categorías.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    
    # En Seaborn 0.13+, si se usa palette se debe asignar hue.
    # En algunas versiones 0.13.x, legend=False en boxplot puede dar UnboundLocalError: 'boxprops'
    # Por ello, lo generamos y lo quitamos manualmente si existe.
    ax = sns.boxplot(
        data=df, 
        x=cat_col, 
        y=num_col, 
        palette=palette,
        hue=cat_col, # Si lo mantienes, asegúrate de que no haya legend=False dentro
        legend=False # A veces, ponerlo explícitamente como True y luego borrarlo funciona
    )
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel if xlabel else cat_col)
    ax.set_ylabel(ylabel if ylabel else num_col)
    
    if rotation != 0:
        plt.xticks(rotation=rotation, ha='right')
        
    plt.tight_layout()
    plt.show()

def plot_distribution_analysis(
    df: pd.DataFrame, 
    num_col: str, 
    title: str = "Análisis de Distribución",
    figsize: tuple = (10, 6),
    color: str = "skyblue"
) -> None:
    """
    Genera una figura combinada con un Boxplot superior y un Histograma con KDE inferior
    compartiendo el eje X, ideal para analizar distribuciones y outliers.
    """
    sns.set_theme(style="whitegrid")
    
    data_clean = df[num_col].dropna()
    
    fig, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": (.15, .85)}
    )
    
    sns.boxplot(x=data_clean, ax=ax_box, color=color, fliersize=4)
    ax_box.set(xlabel='')
    ax_box.set_title(title, pad=15)
    
    sns.histplot(data=data_clean, kde=True, ax=ax_hist, color=color, bins=30)
    ax_hist.set_xlabel(num_col)
    ax_hist.set_ylabel("Frecuencia")
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    cols: Optional[List[str]] = None,
    title: str = "Mapa de Calor de Correlaciones",
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm"
) -> None:
    """
    Genera un mapa de calor de correlaciones aplicando una máscara 
    para ocultar el triángulo superior.
    """
    sns.set_theme(style="whitegrid")
    
    if cols:
        corr_data = df[cols].corr()
    else:
        corr_data = df.select_dtypes(include=[np.number]).corr()
        
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_data, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .75}
    )
    plt.title(title, pad=15)
    plt.tight_layout()
    plt.show()

def check_missing_values(df: pd.DataFrame, title: str = "Porcentaje de Valores Nulos por Columna") -> None:
    """
    Calcula el porcentaje de nulos y delega el gráfico en plot_smart_bar.
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).reset_index()
    
    if missing_pct.empty:
        print("No hay valores nulos en el DataFrame.")
        return
        
    missing_pct.columns = ['Columna', 'Porcentaje_Nulos']
    
    plot_smart_bar(
        df=missing_pct,
        cat_col='Columna',
        val_col='Porcentaje_Nulos', 
        orientation='h',
        title=title,
        xlabel='Porcentaje de Nulos (%)',
        ylabel='Columnas',
        palette='rocket',
        is_percentage=True
    )

def plot_scatter_regression(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str = "Análisis de Dispersión y Regresión",
    figsize: tuple = (10, 6),
    color: str = "teal"
) -> None:
    """
    Genera un gráfico de dispersión con línea de regresión y añade 
    el coeficiente de correlación de Pearson a la leyenda.
    """
    sns.set_theme(style="whitegrid")
    
    valid_data = df[[x_col, y_col]].dropna()
    
    r_coef, _ = pearsonr(valid_data[x_col], valid_data[y_col])
    
    plt.figure(figsize=figsize)
    
    ax = sns.regplot(
        data=valid_data, 
        x=x_col, 
        y=y_col, 
        color=color,
        scatter_kws={'alpha':0.5},
        line_kws={'label': f'Correlación Pearson (r): {r_coef:.3f}', 'color': 'red'}
    )
    
    ax.set_title(title, pad=15)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def plot_missing_demographics(df_nulls: pd.DataFrame, df_pob: pd.DataFrame, anyo: str, title_suffix: str = "") -> None:
    """
    Analiza y grafica el perfil demográfico (tamaño de municipio y provincia) de los nulos.
    """
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    df_pob_temp = df_pob.copy()
    df_pob_temp[col_id_pob] = df_pob_temp[col_id_pob].astype(str).str.zfill(5)
    
    df_nulls_temp = df_nulls.copy()
    df_nulls_temp['Cod_Muni'] = df_nulls_temp['Cod_Muni'].astype(str).str.zfill(5)
    
    cols_a_cruzar = [col_id_pob]
    if 'tamano_municipio' in df_pob_temp.columns:
        cols_a_cruzar.append('tamano_municipio')
    if 'id_provincia' in df_pob_temp.columns: 
        cols_a_cruzar.append('id_provincia')
    if 'nombre_provincia' in df_pob_temp.columns: 
        cols_a_cruzar.append('nombre_provincia')
        
    df_merge = pd.merge(df_nulls_temp, df_pob_temp[cols_a_cruzar], left_on='Cod_Muni', right_on=col_id_pob, how='inner')
    
    if df_merge.empty:
        print(f"No hay datos cruzables para el análisis demográfico de nulos {title_suffix}.")
        return

    if 'tamano_municipio' in df_merge.columns:
        conteo_tamano = df_merge['tamano_municipio'].value_counts().reset_index()
        conteo_tamano.columns = ['tamano_municipio', 'cantidad']
        
        plot_smart_bar(
            df=conteo_tamano, cat_col='tamano_municipio', val_col='cantidad', orientation='h', 
            title=f'Perfil Demográfico de Nulos {title_suffix} ({anyo})',
            xlabel='Cantidad de Municipios', ylabel='Tamaño Poblacional', palette='magma'
        )
    else:
        print(f"Nota: No se puede mostrar el perfil por tamaño de municipio para {title_suffix} porque la columna no existe en los datos proporcionados.")

    print(f"\nIMPACTO POR PROVINCIA {title_suffix} ({anyo})")
    df_merge = mapear_nombres_provincias(df_merge, anyo)
    col_prov = 'nombre_provincia' if 'nombre_provincia' in df_merge.columns else 'id_provincia'
    
    if col_prov in df_merge.columns:
        conteo_prov = df_merge[col_prov].value_counts()
        total = conteo_prov.sum()
        print(f"Top 10 provincias con más nulos {title_suffix}:")
        for prov, cant in conteo_prov.head(10).items():
            print(f" - {str(prov):<20}: {cant:4d} municipios ({(cant/total)*100:.1f}%)")

def plot_adjacency_graph(gdf, w, title: str = "Mapa de Adyacencia Municipal (Grafo Queen)"):
    """
    Visualiza el GeoDataFrame de municipios y superpone el grafo de adyacencia.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    
    if 'municipio' in gdf.columns:
        gdf_to_plot = gdf.set_index('municipio')
    else:
        gdf_to_plot = gdf

    fig, ax = plt.subplots(figsize=(14, 12))
    
    gdf_to_plot.plot(ax=ax, color='#f0f0f0', edgecolor='#bdbdbd', linewidth=0.3)
    
    w.plot(gdf_to_plot, ax=ax, 
           edge_kws=dict(color='#e31a1c', linewidth=0.5, alpha=0.4),
           node_kws=dict(marker='', color='#e31a1c'))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
