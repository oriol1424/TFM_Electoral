import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA.visuals import mapear_nombres_provincias

def limpieza_columnas_votos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa los nombres de las columnas para eliminar el prefijo 'V_' y el sufijo '_2019'.
    """
    nuevas_columnas = {}
    for col in df.columns:
        nueva_col = col
        if col.upper().startswith('V_'):
            nueva_col = nueva_col[2:]
        if col.upper().endswith('_2019'):
            nueva_col = nueva_col[:-5]
        nuevas_columnas[col] = nueva_col
    return df.rename(columns=nuevas_columnas)

def agrupar_minorias_provincial(df_prov: pd.DataFrame, umbral: float = 0.03) -> pd.DataFrame:
    """
    Para cada provincia, identifica los partidos que no alcancen un umbral (3% por defecto)
    y los agrupa en la categoría 'Otros'.
    """
    cols_identificadores = ['id_provincia', 'nombre_provincia', 'id_municipio', 'Nombre_Muni', 'nombre_muni', 'fecha_eleccion']
    cols_votos = [c for c in df_prov.columns if c.lower() not in cols_identificadores]
    
    df_resultado = []
    
    for idx, row in df_prov.iterrows():
        total_votos = row[cols_votos].sum()
        if total_votos == 0:
            df_resultado.append(row)
            continue
            
        votos_otros = 0
        nueva_fila = row.copy()
        
        for col in cols_votos:
            if row[col] / total_votos < umbral:
                votos_otros += row[col]
                nueva_fila[col] = 0
        
        if 'Otros' in nueva_fila:
            nueva_fila['Otros'] += votos_otros
        else:
            nueva_fila['Otros'] = votos_otros
            
        df_resultado.append(nueva_fila)
        
    return pd.DataFrame(df_resultado)

def visualizar_distribucion_votos_provincia(df_plot: pd.DataFrame, title: str = "Distribución de Votos por Provincia (%)"):
    """
    Genera un gráfico de barras apiladas mostrando el peso relativo (%) de cada candidatura.
    """
    sns.set_theme(style="whitegrid")
    
    cols_evitar = ['id_provincia', 'nombre_provincia']
    cols_votos = [c for c in df_plot.columns if c.lower() not in cols_evitar]
    
    df_pct = df_plot.copy()
    if 'nombre_provincia' in df_pct.columns:
        df_pct = df_pct.set_index('nombre_provincia')
    else:
        df_pct = df_pct.set_index('id_provincia')

    df_pct = df_pct[cols_votos].div(df_pct[cols_votos].sum(axis=1), axis=0) * 100
    
    df_pct = df_pct.loc[:, (df_pct != 0).any(axis=0)]
    cols_votos = df_pct.columns.tolist()

    ax = df_pct.plot(
        kind='barh', stacked=True, figsize=(14, 20), colormap='tab20'
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Porcentaje de Votos (%)')
    plt.ylabel('Provincia')
    plt.legend(title='Candidaturas', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualizar_votos_individual_provincia(df_final: pd.DataFrame):
    """
    Genera un gráfico de barras por cada provincia mostrando el peso de cada partido 
    que superó el umbral (o la categoría 'Otros').
    """
    cols_evitar = ['id_provincia', 'nombre_provincia', 'id_municipio', 'Nombre_Muni', 'nombre_muni', 'fecha_eleccion']
    cols_votos = [c for c in df_final.columns if c.lower() not in cols_evitar]
    
    print("\nGenerando Gráficos Individuales por Provincia")
    
    for _, row in df_final.iterrows():
        prov_name = row['nombre_provincia'] if pd.notna(row['nombre_provincia']) else row['id_provincia']
        
        data = row[cols_votos].sort_values(ascending=False)
        data = data[data > 0]
        
        if data.empty:
            continue
            
        total = data.sum()
        data_pct = (data / total) * 100
        
        plt.figure(figsize=(10, len(data_pct) * 0.5 + 1))
        ax = sns.barplot(
            x=data_pct.values, 
            y=data_pct.index, 
            palette='viridis', 
            hue=data_pct.index, 
            legend=False
        )
        
        plt.title(f"Distribución de Votos en {prov_name} (%)", fontsize=14, pad=15)
        plt.xlabel("Porcentaje de Votos (%)")
        plt.ylabel("Partido / Candidatura")
        plt.xlim(0, max(data_pct.values) * 1.15) # Espacio para el texto
        
        for i, v in enumerate(data_pct.values):
            ax.text(v + 0.5, i, f'{v:.2f}%', color='black', va='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()

def resumen_estadistico_votos(df_prov: pd.DataFrame):
    """
    Muestra un resumen estadístico global del recuento de votos nacional.
    """
    cols_evitar = ['id_provincia', 'nombre_provincia', 'id_municipio', 'Nombre_Muni', 'nombre_muni', 'fecha_eleccion']
    cols_votos = [c for c in df_prov.columns if c.lower() not in cols_evitar]
    
    total_por_partido = df_prov[cols_votos].sum().sort_values(ascending=False)
    votos_totales_nacionales = total_por_partido.sum()
    
    print("\n" + "="*60)
    print("RESUMEN ESTADÍSTICO NACIONAL DE CANDIDATURAS")
    print("="*60)
    
    print(f"Total de candidaturas analizadas: {len(cols_votos)}")
    
    partidos_cero = total_por_partido[total_por_partido == 0].index.tolist()
    if partidos_cero:
        print(f"\nPartidos con 0 votos en todo el territorio ({len(partidos_cero)}):")
        print(f" - {', '.join(partidos_cero)}")
    else:
        print("\nNo hay partidos con 0 votos en el recuento.")
        
    top_5 = total_por_partido.head(5)
    print("\nTOP 5 - CANDIDATURAS MÁS VOTADAS:")
    for partido, votos in top_5.items():
        pct = (votos / votos_totales_nacionales) * 100
        print(f" - {partido:<25} | {int(votos):>10,} votos | {pct:>6.2f}%")
        
    bottom_5 = total_por_partido[total_por_partido > 0].tail(5).sort_values(ascending=True)
    print("\nTOP 5 - CANDIDATURAS MENOS VOTADAS (con al menos 1 voto):")
    for partido, votos in bottom_5.items():
        pct = (votos / votos_totales_nacionales) * 100
        print(f" - {partido:<25} | {int(votos):>10,} votos | {pct:>8.5f}%")
        
    print("="*60 + "\n")

def eda_votos_granularidad_total(df_votos_total: pd.DataFrame, anyo: str = "2019", individual: bool = True):
    """
    Orquestador del EDA de votos con granularidad total.
    """
    print(f"Iniciando EDA de Resultados Electorales ({anyo})")
    
    df = df_votos_total.copy()
    df.columns = [c.lower() for c in df.columns]
    
    df = limpieza_columnas_votos(df)
    
    col_muni = 'id_municipio' if 'id_municipio' in df.columns else 'cod_muni'
    if col_muni in df.columns:
        df['id_provincia'] = df[col_muni].astype(str).str.zfill(5).str[:2]
    else:
        raise KeyError(f"No se encontró la columna de municipio.")
    
    cols_identificadores = ['id_provincia', col_muni, 'nombre_muni', 'nombre_provincia', 'fecha_eleccion']
    cols_votos = [c for c in df.columns if c not in cols_identificadores]
    df_prov = df.groupby('id_provincia')[cols_votos].sum().reset_index()
    
    df_prov = mapear_nombres_provincias(df_prov, anyo)
    
    resumen_estadistico_votos(df_prov)
    
    df_final = agrupar_minorias_provincial(df_prov, umbral=0.03)
    
    visualizar_distribucion_votos_provincia(df_final, title=f"Distribución del Peso del Voto por Provincia (Nov {anyo})")
    
    if individual:
        visualizar_votos_individual_provincia(df_final)
    
    return df_final
