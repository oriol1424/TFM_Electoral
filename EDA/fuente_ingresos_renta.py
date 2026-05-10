import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from EDA.visuals import (
    plot_smart_bar, 
    plot_missing_demographics, 
    check_missing_values
)

def categorizar_municipios_tfm(pob: int) -> str:
    """
    Categoriza los municipios según los rangos específicos solicitados.
    """
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

def graficar_fuentes_por_tamano(df_clean: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    """
    Prepara los datos y genera boxplots de las fuentes de ingresos filtrando por el tamaño del municipio.
    """
    cols_fuentes = ['salario', 'pensiones', 'desempleo', 'otras_prestaciones', 'otros_ingresos']
    
    df_renta_copy = df_clean.copy()
    df_renta_copy['Cod_Muni'] = df_renta_copy['Cod_Muni'].astype(str).str.zfill(5)
    
    col_id_pob = 'id_municipio' if 'id_municipio' in df_pob.columns else df_pob.columns[0]
    col_pob_val = 'poblacion' if 'poblacion' in df_pob.columns else 'poblacion_total'
    
    df_pob_copy = df_pob[[col_id_pob, col_pob_val]].copy()
    df_pob_copy[col_id_pob] = df_pob_copy[col_id_pob].astype(str).str.zfill(5)
    
    df_merge = pd.merge(df_renta_copy, df_pob_copy, left_on='Cod_Muni', right_on=col_id_pob, how='inner')
    
    df_merge['rango_poblacion'] = df_merge[col_pob_val].apply(categorizar_municipios_tfm)
    
    orden_categorias = [
        "<100", "101-500", "501-1000", "1001-2000", "2001-5000", 
        "5001-10000", "10001-20000", "20001-50000", "50000-100000", 
        "100001-500000", ">500000"
    ]
    df_merge['rango_poblacion'] = pd.Categorical(df_merge['rango_poblacion'], categories=orden_categorias, ordered=True)

    print(f"\nANÁLISIS ESTRUCTURAL: Fuentes de Ingresos por Tamaño de Municipio ({anyo})")
    
    df_melted = df_merge.melt(
        id_vars=['rango_poblacion'], 
        value_vars=cols_fuentes, 
        var_name='Fuente', 
        value_name='Porcentaje'
    )

    g = sns.FacetGrid(df_melted, col="rango_poblacion", col_wrap=3, height=4, aspect=1.2, sharey=True)
    g.map_dataframe(sns.boxplot, x="Fuente", y="Porcentaje", palette="muted", hue="Fuente", legend=False)
    
    g.set_axis_labels("Fuente", "Porcentaje (%)")
    g.set_titles(col_template="{col_name} hab.")
    
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
    
    plt.subplots_adjust(top=0.92, hspace=0.4)
    g.fig.suptitle(f'Estructura de Ingresos según Tamaño del Municipio ({anyo})', fontsize=16)
    plt.show()

def auditar_nulos_fuentes(df_fuentes: pd.DataFrame, anyo: str) -> None:
    cols_interes = [f'salario {anyo}', f'pensiones {anyo}', f'prestaciones por desempleo {anyo}', f'otras prestaciones {anyo}', f'otros ingresos {anyo}']
    print(f"\nAUDITORÍA DE DATOS FALTANTES FUENTES INGRESOS ({anyo})")
    check_missing_values(df_fuentes[cols_interes], title=f"Porcentaje de Nulos en Fuentes de Ingresos ({anyo})")

def gestion_nulos_fuentes(df_fuentes: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> None:
    auditar_nulos_fuentes(df_fuentes, anyo)
    col_ref = f'salario {anyo}'
    df_nulls = df_fuentes[df_fuentes[col_ref].isna()].copy()
    if not df_nulls.empty:
        plot_missing_demographics(df_nulls, df_pob, anyo, title_suffix="Fuentes Ingresos")

def procesar_fuentes_anyo(df_fuentes: pd.DataFrame, anyo: str) -> pd.DataFrame:
    mapeo_cols = {f'salario {anyo}': 'salario', f'pensiones {anyo}': 'pensiones', f'prestaciones por desempleo {anyo}': 'desempleo', f'otras prestaciones {anyo}': 'otras_prestaciones', f'otros ingresos {anyo}': 'otros_ingresos'}
    cols_existentes = [c for c in mapeo_cols.keys() if c in df_fuentes.columns]
    df_filtrado = df_fuentes[['Cod_Muni', 'Nombre_Muni'] + cols_existentes].copy()
    df_filtrado.rename(columns=mapeo_cols, inplace=True)
    return df_filtrado.dropna(subset=['salario']).copy()

def graficar_peso_total_fuentes(df_clean: pd.DataFrame, anyo: str) -> None:
    """
    Calcula y grafica el peso medio de cada fuente de ingresos a nivel nacional.
    """
    cols_fuentes = ['salario', 'pensiones', 'desempleo', 'otras_prestaciones', 'otros_ingresos']
    
    peso_medio = df_clean[cols_fuentes].mean().reset_index()
    peso_medio.columns = ['Fuente', 'Peso_Medio']
    peso_medio = peso_medio.sort_values('Peso_Medio', ascending=False)

    print(f"\nRESUMEN NACIONAL: Peso Medio de las Fuentes de Ingresos ({anyo})")
    for row in peso_medio.itertuples():
        print(f" - {row.Fuente:<20}: {row.Peso_Medio:.1f}%")

    plot_smart_bar(
        df=peso_medio,
        cat_col='Fuente',
        val_col='Peso_Medio',
        title=f'Peso Medio de las Fuentes de Ingresos en España ({anyo})',
        xlabel='Fuente de Ingreso',
        ylabel='Porcentaje Medio del Ingreso (%)',
        palette='magma',
        is_percentage=True,
        decimals=1
    )

def graficar_distribucion_fuentes(df_clean: pd.DataFrame, anyo: str) -> None:
    cols_fuentes = ['salario', 'pensiones', 'desempleo', 'otras_prestaciones', 'otros_ingresos']
    plt.figure(figsize=(12, 6))
    df_melted = df_clean.melt(id_vars=['Cod_Muni', 'Nombre_Muni'], value_vars=cols_fuentes, var_name='Fuente', value_name='Porcentaje')
    sns.boxplot(data=df_melted, x='Fuente', y='Porcentaje', palette='Set2', hue='Fuente', legend=False)
    plt.title(f'Distribución de Fuentes de Ingresos Nacional ({anyo})')
    plt.ylabel('% sobre ingresos')
    plt.show()

def eda_fuente_ingresos(df_fuentes_completo: pd.DataFrame, df_pob: pd.DataFrame, anyo: str) -> pd.DataFrame:
    print(f"\n{'='*20} INICIANDO ANÁLISIS DE FUENTES DE INGRESOS ({anyo}) {'='*20}")
    gestion_nulos_fuentes(df_fuentes_completo, df_pob, anyo)
    df_clean = procesar_fuentes_anyo(df_fuentes_completo, anyo)
    if not df_clean.empty:
        graficar_peso_total_fuentes(df_clean, anyo)
        graficar_distribucion_fuentes(df_clean, anyo)
        graficar_fuentes_por_tamano(df_clean, df_pob, anyo)
    return df_clean
