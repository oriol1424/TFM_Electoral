import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA.visuals import plot_smart_bar

def procesar_mercado_laboral(df_lab: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratios clave de precariedad, brechas demográficas y estructura 
    sectorial a partir de los datos absolutos del mercado laboral.
    """
    df = df_lab.copy()
    
    df['tasa_temporalidad'] = np.where(
        df['total_contratos'] > 0, 
        ((df['c_h_temp'] + df['c_m_temp']) / df['total_contratos']) * 100, 
        np.nan 
    )
    
    df['peso_paro_fem'] = np.where(
        df['total_paro'] > 0, 
        ((df['p_m_u25'] + df['p_m_25_44'] + df['p_m_o45']) / df['total_paro']) * 100, 
        np.nan
    )
    
    df['peso_paro_juv'] = np.where(
        df['total_paro'] > 0, 
        ((df['p_h_u25'] + df['p_m_u25']) / df['total_paro']) * 100, 
        np.nan
    )
    
    df['peso_paro_o45'] = np.where(
        df['total_paro'] > 0, 
        ((df['p_h_o45'] + df['p_m_o45']) / df['total_paro']) * 100, 
        np.nan
    )
    
    sectores = ['p_agr', 'p_ind', 'p_con', 'p_ser']
    df['sector_mayoritario'] = df[sectores].idxmax(axis=1)
    
    mapa_sectores = {
        'p_agr': 'Agrario', 
        'p_ind': 'Industria', 
        'p_con': 'Construcción', 
        'p_ser': 'Servicios'
    }
    df['sector_mayoritario'] = df['sector_mayoritario'].map(mapa_sectores)
    
    return df


def mostrar_analisis_laboral(df_lab: pd.DataFrame) -> None:
    """
    Genera visualizaciones para entender la salud del mercado laboral y 
    la estructura productiva mayoritaria de los municipios.
    """
    plot_smart_bar(
        df=df_lab,
        cat_col='sector_mayoritario',
        orientation='v',
        title='Estructura Productiva: ¿Qué sector domina el municipio?',
        xlabel='Sector Mayoritario',
        ylabel='Cantidad de Municipios',
        palette='Set2'
    )

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(
        data=df_lab, x='sector_mayoritario', y='tasa_temporalidad', 
        hue='sector_mayoritario', palette='Set2', legend=False
    )
    plt.title('Precariedad Laboral: Tasa de Temporalidad por Sector')
    plt.xlabel('Sector Mayoritario del Municipio')
    plt.ylabel('Tasa de Temporalidad (%)')
    plt.tight_layout()
    plt.show()

    print("RADIOGRAFÍA DEL MERCADO LABORAL")
    tot_temp = df_lab['c_h_temp'].sum() + df_lab['c_m_temp'].sum()
    tot_contratos = df_lab['total_contratos'].sum()
    tasa_real_nacional = (tot_temp / tot_contratos) * 100 if tot_contratos > 0 else 0

    print(f"Temporalidad real nacional (ponderada): {tasa_real_nacional:.1f}% de los contratos")    
    print(f"Peso medio del paro Femenino: {df_lab['peso_paro_fem'].mean():.1f}% del total de parados")
    print(f"Peso medio del paro Juvenil (<25): {df_lab['peso_paro_juv'].mean():.1f}% del total de parados")
    print(f"Peso medio del paro Senior (>45): {df_lab['peso_paro_o45'].mean():.1f}% del total de parados")


def analizar_brecha_genero_temporalidad(df_lab: pd.DataFrame) -> None:
    """
    Analiza y grafica la tasa de temporalidad diferenciada por género 
    a nivel nacional.
    """
    df = df_lab.copy()
    
    total_temp_h = df['c_h_temp'].sum()
    total_c_h = df['c_h_indef'].sum() + df['c_h_temp'].sum() + df['c_h_conv'].sum()
    
    total_temp_m = df['c_m_temp'].sum()
    total_c_m = df['c_m_indef'].sum() + df['c_m_temp'].sum() + df['c_m_conv'].sum()

    tasa_temp_h = (total_temp_h / total_c_h) * 100 if total_c_h > 0 else 0
    tasa_temp_m = (total_temp_m / total_c_m) * 100 if total_c_m > 0 else 0

    df_plot = pd.DataFrame({
        'Género': ['Hombres', 'Mujeres'],
        'Tasa_Temporalidad': [tasa_temp_h, tasa_temp_m]
    })
    
    plot_smart_bar(
        df=df_plot,
        cat_col='Género',
        val_col='Tasa_Temporalidad',
        orientation='v',
        title='Brecha de Género: Tasa de Temporalidad (Nacional)',
        ylabel='% de Contratos Temporales',
        palette=['#1f77b4', '#e377c2'],
        is_percentage=True,
        figsize=(6, 5),
        decimals=3
    )


def analizar_primer_empleo(df_lab: pd.DataFrame) -> None:
    """
    Analiza y grafica la proporción de personas paradas que están 
    buscando su primer empleo (sin experiencia laboral previa).
    """
    tot_paro = df_lab['total_paro'].sum()
    tot_sin_empleo = df_lab['p_sin_empleo'].sum()
    tot_con_empleo = tot_paro - tot_sin_empleo

    pct_sin_empleo = (tot_sin_empleo / tot_paro) * 100 if tot_paro > 0 else 0
    pct_con_empleo = (tot_con_empleo / tot_paro) * 100 if tot_paro > 0 else 0

    df_plot = pd.DataFrame({
        'Situación': ['Con empleo anterior', 'Sin empleo anterior\n(Primer Empleo)'],
        'Porcentaje': [pct_con_empleo, pct_sin_empleo]
    })
    
    plot_smart_bar(
        df=df_plot,
        cat_col='Situación',
        val_col='Porcentaje',
        orientation='v',
        title='El "Muro" del Primer Empleo: Composición del Paro',
        ylabel='% del Total de Parados',
        palette='magma',
        is_percentage=True,
        figsize=(7, 5)
    )

    print(f"PRIMER EMPLEO: {tot_sin_empleo:,.0f} parados ({pct_sin_empleo:.1f}% del total nacional) nunca han trabajado.")


def analizar_contratos_por_sector(df_lab: pd.DataFrame) -> None:
    """
    Suma de forma estricta los contratos registrados en cada sector 
    y genera un gráfico de pastel con la distribución real a nivel nacional.
    """
    totales_sector = {
        'Agricultura': df_lab['c_agr'].sum(),
        'Industria': df_lab['c_ind'].sum(),
        'Construcción': df_lab['c_con'].sum(),
        'Servicios': df_lab['c_ser'].sum()
    }

    plt.figure(figsize=(6, 6))
    plt.pie(
        totales_sector.values(), 
        labels=totales_sector.keys(), 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=['#2ca02c', '#8c564b', '#7f7f7f', '#1f77b4'],
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title('Distribución Real de Contratos por Sector (Total Nacional)')
    plt.tight_layout()
    plt.show()

    print("ESTRUCTURA DE CONTRATACIÓN NACIONAL:")
    for sector, total in totales_sector.items():
        print(f"{sector}: {total:,.0f} contratos")


def eda_mercado_laboral(df_paro_contratos: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que ejecuta todo el bloque de análisis del mercado laboral.
    """
    df_lab_enriquecido = procesar_mercado_laboral(df_paro_contratos)
    
    mostrar_analisis_laboral(df_lab_enriquecido)
    
    print("\nMÉTRICAS ESTRUCTURALES Y BRECHAS")
    analizar_brecha_genero_temporalidad(df_lab_enriquecido)
    analizar_primer_empleo(df_lab_enriquecido)
    analizar_contratos_por_sector(df_lab_enriquecido)

    return df_lab_enriquecido