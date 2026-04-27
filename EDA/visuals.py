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
    if 'id_provincia' not in df.columns and 'Cod_Muni' in df.columns:
        df['id_provincia'] = df['Cod_Muni'].astype(str).str.zfill(5).str[:2]
        
    if 'id_provincia' in df.columns:
        mapeo = obtener_mapeo_provincias(anyo)
        if 'nombre_provincia' not in df.columns:
            df['nombre_provincia'] = df['id_provincia'].map(mapeo)
        else:
            df['nombre_provincia'] = df['nombre_provincia'].fillna(df['id_provincia'].map(mapeo))
    
    return df

def categorizar_municipios_tfm(pob: int) -> str:
    """
    Categoriza los municipios según los rangos específicos para el TFM.
    """
    if pd.isna(pob): return "Sin Datos"
    if pob <= 100: return "<100"
    elif pob <= 500: return "101-500"
    elif pob <= 1000: return "501-1000"
    elif pob <= 2000: return "1001-2000"
    elif pob <= 5000: return "2001-5000"
    elif pob <= 10000: return "5001-10000"
    elif pob <= 20000: return "10001-20000"
    elif pob <= 50000: return "20001-50000"
    elif pob <= 100000: return "50000-100000"
    elif pob <= 500000: return "100001-500000"
    else: return ">500000"

def plot_missing_demographics(df_nulls: pd.DataFrame, df_pob: pd.DataFrame, anyo: str, title_suffix: str = "") -> None:
    """
    Analiza y grafica el perfil demográfico (tamaño de municipio y provincia) de los nulos.
    """
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    df_pob_temp = df_pob.copy()
    df_pob_temp[col_id_pob] = df_pob_temp[col_id_pob].astype(str).str.zfill(5)
    
    df_nulls_temp = df_nulls.copy()
    df_nulls_temp['Cod_Muni'] = df_nulls_temp['Cod_Muni'].astype(str).str.zfill(5)
    
    # Verificamos qué columnas existen realmente para evitar KeyError
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