import pandas as pd
import json
import os
from typing import Dict, Union

# Limpieza de datos (mocking imports or using the real ones)
from limpieza.votos import limpieza_votos_partidos
from limpieza.rentas import procesar_archivo_ine, procesar_archivo_gini, procesar_fuente_ingresos
from limpieza.paro_contratos import limpiar_y_exportar_sepe
from limpieza.superficie import procesar_geografia_municipios
from limpieza.poblacion import procesar_poblacion_maestro
from limpieza.funciones_genericas_limpieza import leer_json, leer_archivo_csv
from limpieza.renta_navarra import generar_renta_con_navarra

def ETL_limpieza(year: Union[int, str], config_path: str = "config_path.json") -> Dict[str, pd.DataFrame]:
    year_str = str(year)
    config = leer_json(config_path)
    if year_str not in config:
        raise ValueError(f"El año {year_str} no existe en el archivo {config_path}.")
    raw_paths = config[year_str]["raw"]
    proc_paths = config[year_str]["processed"]
    dataframes: Dict[str, pd.DataFrame] = {}
    for clave, ruta_procesada in proc_paths.items():
        nombre_df = f"df_{year_str}_{clave}"
        es_carpeta = "carpeta" in clave.lower()
        ya_procesado = False
        archivos_votos_esperados = []
        if clave == "carpeta_votos":
            archivos_votos_esperados = [
                f"arbol_candidaturas_{year_str}.json",
                f"candidaturas_ideologia_{year_str}.json",
                f"maestro_ideologia_{year_str}.json",
                f"Votos_Agrupados_Padres_{year_str}.csv",
                f"Votos_Granularidad_Total_{year_str}.csv"
            ]
            if os.path.exists(ruta_procesada):
                ya_procesado = all(
                    os.path.exists(os.path.join(ruta_procesada, arch)) 
                    for arch in archivos_votos_esperados
                )
        else:
            ya_procesado = os.path.exists(ruta_procesada)
        if not ya_procesado:
            if es_carpeta:
                os.makedirs(ruta_procesada, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(ruta_procesada), exist_ok=True)
            try:
                if clave in ["poblacion_csv", "poblacion_json"]:
                    # (Skipping for brevity or implementing if needed)
                    pass
                elif clave == "paro_contratos":
                    print(f"DEBUG: Llamando a limpiar_y_exportar_sepe con {raw_paths['paro_contratos']} y {ruta_procesada}")
                    res = limpiar_y_exportar_sepe(raw_paths["paro_contratos"], ruta_procesada)
                    print(f"DEBUG: Resultado de limpiar_y_exportar_sepe: {res is not None}")
                # ... other elifs ...
            except Exception as e:
                print(f"Error al procesar '{clave}': {e}")
        if not es_carpeta:
            if os.path.exists(ruta_procesada):
                try:
                    if ruta_procesada.endswith('.csv'):
                        dataframes[nombre_df] = leer_archivo_csv(ruta_procesada, decimal='.')
                    elif ruta_procesada.endswith('.json'):
                        dataframes[nombre_df] = pd.read_json(ruta_procesada)
                except Exception as e:
                    print(f"Error al cargar archivo '{nombre_df}': {e}")
    return dataframes

print("Ejecutando ETL_limpieza(2019)...")
dfs = ETL_limpieza(2019)
if "df_2019_paro_contratos" in dfs:
    print("df_2019_paro_contratos se creó correctamente en el diccionario.")
else:
    print("df_2019_paro_contratos NO se creó en el diccionario.")
