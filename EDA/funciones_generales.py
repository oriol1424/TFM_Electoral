import os
import json
import pandas as pd
import numpy as np
from typing import Optional, List
from EDA.fuente_ingresos_renta import categorizar_municipios_tfm

def auditar_municipios_extintos(df_origen: pd.DataFrame, df_maestro: pd.DataFrame, nombre_dataset: str, col_id: str = 'Cod_Muni') -> None:
    """
    Compara los municipios de un dataset contra el maestro de población.
    Detecta e informa de aquellos que existen en el dataset pero no en el maestro.
    """
    ids_maestro = set(df_maestro['id_municipio'].unique())
    ids_origen = set(df_origen[col_id].unique())
    
    extintos = ids_origen - ids_maestro
    
    print(f" -> Dataset '{nombre_dataset}': {len(extintos)} municipios extintos/fusionados detectados.")
    
    if len(extintos) > 0:
        nombres_extintos = df_origen[df_origen[col_id].isin(extintos)]
        col_nombre = next((c for c in ['Nombre_Muni', 'nombre', 'NOMBRE', 'nombre_muni'] if c in nombres_extintos.columns), None)
        
        for idx_ext in sorted(list(extintos)):
            nombre = ""
            if col_nombre:
                nombres = nombres_extintos[nombres_extintos[col_id] == idx_ext][col_nombre].values
                nombre = f"({nombres[0]})" if len(nombres) > 0 else ""
            print(f"    - Descartado ID: {idx_ext} {nombre}")

def unificar_datos_eda(anyo: int) -> pd.DataFrame:
    """
    Unifica todos los datos procesados (demografía, superficie, renta, ingresos y votos)
    en un único DataFrame usando las rutas de config_path.json.
    Detecta y excluye automáticamente municipios que ya no existen en el padrón.
    """
    anyo_str = str(anyo)
    
    with open('config_path.json', 'r', encoding='utf-8') as f:
        config_total = json.load(f)
    
    if anyo_str not in config_total:
        raise ValueError(f"El año {anyo_str} no está configurado en config_path.json")
    
    config = config_total[anyo_str]["processed"]
    folder_path = config["data_end"]
    file_path = os.path.join(folder_path, f"datos_unificados_{anyo_str}.csv")

    if os.path.exists(file_path):
        print(f"Cargando datos unificados existentes desde: {file_path}")
        print("Nota: Si quieres volver a ver la auditoría de extintos, borra este archivo CSV.")
        return pd.read_csv(file_path, sep=';', dtype={'municipio': str})


    print(f"INICIANDO UNIFICACIÓN Y AUDITORÍA DE INTEGRIDAD ({anyo_str})")
    with open(config["poblacion_json"], 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    map_prov = {
        str(cp).zfill(2): info.get('nombre_provincia') 
        for cp, info in data_json.get('provincias', {}).items()
    }

    df_pob = pd.read_csv(config["poblacion_csv"], sep=';')
    df_sup = pd.read_csv(config["geografia"], sep=';')
    
    df_pob['id_municipio'] = df_pob['id_municipio'].astype(str).str.zfill(5)
    df_sup['id_municipio'] = df_sup['id_municipio'].astype(str).str.zfill(5)
    
    df_final = pd.merge(df_pob, df_sup.drop(columns=['nombre_municipio'], errors='ignore'), on='id_municipio', how='inner')
    
    df_final['id_provincia_temp'] = df_final['id_municipio'].str[:2]
    df_final['nombre_provincia'] = df_final['id_provincia_temp'].map(map_prov)

    print("\nAuditoría de municipios extintos/fusionados contra el padrón:")
    
    df_gini = pd.read_csv(config["GINI_P80P20"], sep=';')
    df_ind = pd.read_csv(config["renta_disponible"], sep=';')
    df_fuentes = pd.read_csv(config["fuente_ingresos"], sep=';')
    df_navarra = pd.read_csv(config["renta_navarra"], sep=';')
    for df_name, df_obj in [("Gini/P80P20", df_gini), ("Renta Disponible", df_ind), 
                           ("Fuente Ingresos", df_fuentes), ("Renta Navarra", df_navarra)]:
        col_id = next((c for c in ['Cod_Muni', 'id_municipio', 'Código', 'ID_MUNICIPIO'] if c in df_obj.columns), None)
        if col_id:
            df_obj.rename(columns={col_id: 'Cod_Muni'}, inplace=True)
            df_obj['Cod_Muni'] = df_obj['Cod_Muni'].astype(str).str.zfill(5)
            auditar_municipios_extintos(df_obj, df_pob, df_name, 'Cod_Muni')

    votos_path = os.path.join(config["carpeta_votos"], f"Votos_Granularidad_Total_{anyo_str}.csv")
    if not os.path.exists(votos_path):
         votos_path = os.path.join(config["carpeta_votos"], "Votos_Granularidad_Total_2019.csv")
    
    df_votos = pd.read_csv(votos_path, sep=';')
    col_id_v = next((c for c in ['ID_MUNICIPIO', 'id_municipio', 'Cod_Muni'] if c in df_votos.columns), None)
    df_votos.rename(columns={col_id_v: 'id_municipio'}, inplace=True)
    df_votos['id_municipio'] = df_votos['id_municipio'].astype(str).str.zfill(5)
    
    auditar_municipios_extintos(df_votos, df_pob, "Votos", 'id_municipio')
    
    cols_v = [c for c in df_votos.columns if c not in ['id_municipio', 'nombre_muni', 'fecha_eleccion', 'FECHA_ELECCION']]
    df_votos['votos totales'] = df_votos[cols_v].sum(axis=1)

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

def imputar_datos_socioeconomicos(df: pd.DataFrame, w, anyo: int) -> pd.DataFrame:
    """
    Realiza la imputación de datos socioeconómicos utilizando lógica espacial y provincial.
    
    Lógica:
    - Grupo A (Estables - Gini y Rentas): Media de vecinos físicos con población < 2.000 hab.
    - Grupo B (Volátiles - Salarios, Pensiones, Desempleo): Mediana de la provincia para el estrato rural (< 2.000 hab).
    - Fallback: Mediana de la provincia por estrato poblacional.
    """
    anyo_str = str(anyo)
    df_imputado = df.copy()
    
    df_imputado['municipio'] = df_imputado['municipio'].astype(str).str.zfill(5)
    df_imputado['imputado'] = False
    
    grupo_a = ['indice gini', 'P80P20', 'Renta media hogar', 'renta media unidad consumo', 'renta media persona']
    grupo_b = ['salarios', 'pensiones', 'otros ingresos', 'otras prestaciones', 'desempleo']
    
    df_imputado['es_rural'] = df_imputado['poblacion'] < 2000
    
    print(f"INICIANDO PROCESO DE IMPUTACIÓN SOCIOECONÓMICA ({anyo_str})")

    df_imputado = df_imputado.set_index('municipio')
    medias_dict = df_imputado.groupby(['provincia', 'es_rural'])[grupo_a + grupo_b].median().to_dict('index')

    def get_fallback_value(municipio_idx, col):
        prov = df_imputado.at[municipio_idx, 'provincia']
        rural = df_imputado.at[municipio_idx, 'es_rural']
        return medias_dict.get((prov, rural), {}).get(col, np.nan)

    print(" -> Imputando Grupo B (Fuentes volátiles) mediante mediana provincial rural")
    for col in grupo_b:
        mask_b = df_imputado[col].isna() & (df_imputado['es_rural'] == True)
        if mask_b.any():
            df_imputado.loc[mask_b, col] = [get_fallback_value(idx, col) for idx in df_imputado.index[mask_b]]
            df_imputado.loc[mask_b, 'imputado'] = True

    print(" -> Imputando Grupo A (Indicadores estables) mediante adyacencia física...")
    for col in grupo_a:
        indices_nan = df_imputado.index[df_imputado[col].isna()]
        
        for idx in indices_nan:
            if idx in w.neighbors:
                vecinos_ids = w.neighbors[idx]
                
                vecinos_filtrados = df_imputado.loc[
                    (df_imputado.index.isin(vecinos_ids)) & 
                    (df_imputado['es_rural'] == True) &
                    (df_imputado[col].notna())
                ]
                
                if not vecinos_filtrados.empty:
                    df_imputado.at[idx, col] = vecinos_filtrados[col].mean()
                    df_imputado.at[idx, 'imputado'] = True
                else:
                    df_imputado.at[idx, col] = get_fallback_value(idx, col)
                    df_imputado.at[idx, 'imputado'] = True
            else:
                df_imputado.at[idx, col] = get_fallback_value(idx, col)
                df_imputado.at[idx, 'imputado'] = True

    df_imputado = df_imputado.reset_index()
    df_imputado = df_imputado.drop(columns=['es_rural'])
    
    folder_path = f"data_processed/data_end/{anyo_str}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"datos_unificados_imputados_{anyo_str}.csv")
    df_imputado.to_csv(file_path, sep=';', index=False, encoding='utf-8-sig')

    print(f"\nResultados de Imputación por Provincia:")
    conteo_prov = df_imputado[df_imputado['imputado']].groupby('provincia').size()
    if not conteo_prov.empty:
        for prov, cant in conteo_prov.items():
            print(f" - {prov}: {cant} municipios imputados.")
    else:
        print(" - Ningún municipio requirió imputación.")

    print(f"\nÉxito: Datos imputados guardados en {file_path}")
    print(f"Total municipios imputados: {df_imputado['imputado'].sum()} de {len(df_imputado)}.")
    
    return df_imputado

def analizar_correlaciones_eda(df: pd.DataFrame, anyo: int):
    """
    Filtra variables numéricas relevantes y genera un mapa de calor de correlaciones.
    Excluye identificadores, nombres y coordenadas.
    """
    from EDA.visuals import plot_correlation_heatmap
    
    cols_excluir = ['municipio', 'nombre', 'provincia', 'latitud', 'longitud', 'altitud', 'imputado']
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in cols_excluir if c in df.columns], errors='ignore')
    
    df_clean = df_numeric.dropna()
    
    if df_clean.empty:
        print(f"Aviso: No hay suficientes datos limpios para calcular correlaciones en el año {anyo}.")
        return

    print(f"\nANALIZANDO CORRELACIONES ({anyo})")
    print(f"Variables analizadas: {list(df_clean.columns)}")
    print(f"Registros completos utilizados: {len(df_clean)}")
    
    plot_correlation_heatmap(
        df_clean, 
        title=f"Mapa de Correlaciones Socioeconómicas - Año {anyo}",
        figsize=(12, 10)
    )
