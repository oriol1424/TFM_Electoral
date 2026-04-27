import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from scipy.stats import pearsonr

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