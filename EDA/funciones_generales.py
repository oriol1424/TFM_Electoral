import os
import json
import pandas as pd
import numpy as np
from typing import Optional
from EDA.fuente_ingresos_renta import categorizar_municipios_tfm

def unificar_datos_eda(anyo: int) -> pd.DataFrame:
    """
    Unifica todos los datos procesados (demografía, superficie, renta, ingresos y votos)
    en un único DataFrame usando las rutas de config_path.json.
    Crea un CSV en data_processed/data_end/<anyo>/ y devuelve el DataFrame.
    """
    anyo_str = str(anyo)
    
    # 1. Cargar configuración de rutas
    with open('config_path.json', 'r', encoding='utf-8') as f:
        config_total = json.load(f)
    
    if anyo_str not in config_total:
        raise ValueError(f"El año {anyo_str} no está configurado en config_path.json")
    
    config = config_total[anyo_str]["processed"]
    folder_path = config["data_end"]
    file_path = os.path.join(folder_path, f"datos_unificados_{anyo_str}.csv")

    # 2. Si el archivo ya existe, lo cargamos y devolvemos
    if os.path.exists(file_path):
        print(f"Cargando datos unificados existentes desde: {file_path}")
        return pd.read_csv(file_path, sep=';', dtype={'municipio': str})

    print(f"Iniciando proceso de unificación para el año {anyo_str}...")

    # 3. Cargar mapeo de provincias desde el JSON (para asegurar el nombre de la provincia)
    with open(config["poblacion_json"], 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    # Diccionario { "01": "Araba/Álava", ... }
    map_prov = {
        str(cp).zfill(2): info.get('nombre_provincia') 
        for cp, info in data_json.get('provincias', {}).items()
    }

    # 4. Carga de datos base (Población y Geografía)
    df_pob = pd.read_csv(config["poblacion_csv"], sep=';')
    df_sup = pd.read_csv(config["geografia"], sep=';')
    
    # Asegurar IDs a 5 dígitos para cruces limpios
    df_pob['id_municipio'] = df_pob['id_municipio'].astype(str).str.zfill(5)
    df_sup['id_municipio'] = df_sup['id_municipio'].astype(str).str.zfill(5)

    # --- PROCESO DE UNIFICACIÓN (MERGE) ---
    
    # Unión Base: Población + Geografía
    df_final = pd.merge(df_pob, df_sup.drop(columns=['nombre_municipio'], errors='ignore'), on='id_municipio', how='inner')
    
    # Añadir nombre de provincia usando el JSON
    df_final['id_provincia_temp'] = df_final['id_municipio'].str[:2]
    df_final['nombre_provincia'] = df_final['id_provincia_temp'].map(map_prov)

    # 5. Carga de datos de Renta y Desigualdad (INE + Navarra)
    df_gini = pd.read_csv(config["GINI_P80P20"], sep=';')
    df_ind = pd.read_csv(config["renta_disponible"], sep=';')
    df_fuentes = pd.read_csv(config["fuente_ingresos"], sep=';')
    df_navarra = pd.read_csv(config["renta_navarra"], sep=';')

    # Normalización de IDs en renta
    for df in [df_gini, df_ind, df_fuentes, df_navarra]:
        col_id = next((c for c in ['Cod_Muni', 'id_municipio', 'Código', 'ID_MUNICIPIO'] if c in df.columns), None)
        if col_id:
            df.rename(columns={col_id: 'Cod_Muni'}, inplace=True)
            df['Cod_Muni'] = df['Cod_Muni'].astype(str).str.zfill(5)

    # 6. Carga de Votos
    votos_path = os.path.join(config["carpeta_votos"], f"Votos_Granularidad_Total_{anyo_str}.csv")
    if not os.path.exists(votos_path):
         votos_path = os.path.join(config["carpeta_votos"], "Votos_Granularidad_Total_2019.csv")
    
    df_votos = pd.read_csv(votos_path, sep=';')
    col_id_v = next((c for c in ['ID_MUNICIPIO', 'id_municipio', 'Cod_Muni'] if c in df_votos.columns), None)
    df_votos.rename(columns={col_id_v: 'id_municipio'}, inplace=True)
    df_votos['id_municipio'] = df_votos['id_municipio'].astype(str).str.zfill(5)
    
    # Calcular votos totales
    cols_v = [c for c in df_votos.columns if c not in ['id_municipio', 'nombre_muni', 'fecha_eleccion', 'FECHA_ELECCION']]
    df_votos['votos totales'] = df_votos[cols_v].sum(axis=1)

    # --- MERGES RESTANTES ---
    df_final = pd.merge(df_final, df_gini[['Cod_Muni', f'Índice de Gini {anyo_str}', f'Distribución de la renta P80/P20 {anyo_str}']], 
                        left_on='id_municipio', right_on='Cod_Muni', how='left')
    
    df_final = pd.merge(df_final, df_ind[['Cod_Muni', f'Renta neta media por hogar {anyo_str}', 
                                         f'Media de la renta por unidad de consumo {anyo_str}', 
                                         f'Renta neta media por persona {anyo_str}']], 
                        left_on='id_municipio', right_on='Cod_Muni', how='left')

    df_final = pd.merge(df_final, df_fuentes[['Cod_Muni', f'salario {anyo_str}', f'pensiones {anyo_str}', 
                                             f'otros ingresos {anyo_str}', f'otras prestaciones {anyo_str}', 
                                             f'prestaciones por desempleo {anyo_str}']], 
                        left_on='id_municipio', right_on='Cod_Muni', how='left')

    # Integración de Navarra
    df_final = pd.merge(df_final, df_navarra, left_on='id_municipio', right_on='Cod_Muni', how='left', suffixes=('', '_nav'))
    map_nav = {
        f'Índice de Gini {anyo_str}': 'Índice de Gini',
        f'Renta neta media por hogar {anyo_str}': 'Renta neta media por hogar',
        f'Media de la renta por unidad de consumo {anyo_str}': 'Media de la renta por unidad de consumo',
        f'Renta neta media por persona {anyo_str}': 'Renta neta media por persona'
    }
    for col_ine, col_nav in map_nav.items():
        if col_nav in df_final.columns:
            df_final[col_ine] = df_final[col_ine].fillna(df_final[col_nav])

    df_final = pd.merge(df_final, df_votos[['id_municipio', 'votos totales']], on='id_municipio', how='left')

    # --- CÁLCULOS Y RENOMBRADO ---
    df_final['superficie_km2'] = pd.to_numeric(df_final['superficie_km2'].astype(str).str.replace(',', '.'), errors='coerce')
    df_final['densidad poblacional'] = df_final['poblacion_total'] / df_final['superficie_km2']
    df_final['rango tamaño población'] = df_final['poblacion_total'].apply(categorizar_municipios_tfm)

    df_final = df_final.rename(columns={
        'id_municipio': 'municipio', 'nombre_municipio': 'nombre', 'nombre_provincia': 'provincia',
        'superficie_km2': 'superficie', 'latitud': 'latitud', 'longitud': 'longitud', 'altitud': 'altitud',
        'poblacion_total': 'poblacion', 'total_hombres': 'poblacion hombres', 'total_mujeres': 'poblacion mujeres',
        f'Índice de Gini {anyo_str}': 'indice gini', f'Distribución de la renta P80/P20 {anyo_str}': 'P80P20',
        f'Renta neta media por hogar {anyo_str}': 'Renta media hogar',
        f'Media de la renta por unidad de consumo {anyo_str}': 'renta media unidad consumo',
        f'Renta neta media por persona {anyo_str}': 'renta media persona',
        f'salario {anyo_str}': 'salarios', f'pensiones {anyo_str}': 'pensiones',
        f'otros ingresos {anyo_str}': 'otros ingresos', f'otras prestaciones {anyo_str}': 'otras prestaciones',
        f'prestaciones por desempleo {anyo_str}': 'desempleo'
    })
    
    # Orden final
    columnas_ordenadas = [
        "municipio", "nombre", "provincia", "superficie", "latitud", "longitud", "altitud",
        "rango tamaño población", "poblacion", "poblacion hombres", "poblacion mujeres",
        "densidad poblacional", "indice gini", "P80P20", "Renta media hogar",
        "renta media unidad consumo", "renta media persona", "salarios", "pensiones",
        "otros ingresos", "otras prestaciones", "desempleo", "votos totales"
    ]
    df_final = df_final[columnas_ordenadas]

    os.makedirs(folder_path, exist_ok=True)
    df_final.to_csv(file_path, sep=';', index=False, encoding='utf-8-sig')
    print(f"Éxito: Datos unificados guardados en {file_path}")

    return df_final
