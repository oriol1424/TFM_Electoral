import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from .funciones_genericas_limpieza import guardar_dataframe_csv, guardar_json, leer_json

# =============================================================================
# CANDIDATURAS GENERALES E IDEOLOGÍA
# =============================================================================

def generar_json_desde_cis(fichero_cis: str, output_json: str) -> Dict[str, Any]:
    """
    Lee un archivo Excel del CIS, extrae las notas ideológicas y genera un archivo JSON.
    Args:
        fichero_cis (str): Ruta del archivo Excel del barómetro del CIS.
        output_json (str): Ruta donde se guardará el archivo JSON generado.
    Returns:
        Dict[str, Any]: Diccionario con la configuración maestra de los partidos.
    """
    df_cis = pd.read_excel(fichero_cis, header=None)
    fila_nombres = df_cis[df_cis.eq('PSOE').any(axis=1)].iloc[0].tolist()
    fila_media = df_cis[df_cis[0] == 'Media'].iloc[0].tolist()
    
    notas_cis: Dict[str, float] = {}
    for nombre, nota in zip(fila_nombres, fila_media):
        if pd.notna(nombre) and isinstance(nombre, str) and nombre != 'Media':
            try:
                notas_cis[nombre.strip()] = float(nota)
            except ValueError:
                pass

    config_partidos = {
        "000002": {"cis_name": "PSOE", "voto_retrospectivo_id": "bloque_socialista"},
        "000005": {"cis_name": "PP", "voto_retrospectivo_id": "bloque_popular"},
        "000011": {"cis_name": "VOX", "voto_retrospectivo_id": "bloque_derecha_radical"},
        "000010": {"cis_name": "Unidas Podemos", "voto_retrospectivo_id": "bloque_izquierda_radical"},
        "000050": {"cis_name": "ERC", "voto_retrospectivo_id": "bloque_nacionalista_cat"},
        "000030": {"cis_name": "EAJ-PNV", "voto_retrospectivo_id": "bloque_nacionalista_vasco"},
        "000057": {"cis_name": "JxCat", "voto_retrospectivo_id": "bloque_nacionalista_cat"},
        "000071": {"cis_name": "EH Bildu", "voto_retrospectivo_id": "bloque_nacionalista_vasco"},
        "000053": {"cis_name": "CUP", "voto_retrospectivo_id": "bloque_nacionalista_cat"}, 
        "000065": {"cis_name": "BNG", "voto_retrospectivo_id": "bloque_nacionalista_gal"},
        "000031": {"cis_name": "CCa-PNC-NC", "voto_retrospectivo_id": "bloque_nacionalista_canario"}, 
        "000032": {"cis_name": "CCa-PNC-NC", "voto_retrospectivo_id": "bloque_nacionalista_canario"},
        "000023": {"cis_name": "Teruel Existe", "voto_retrospectivo_id": "bloque_nacionalista_canario"}
    }

    json_maestro: Dict[str, Any] = {}
    for codigo, config in config_partidos.items():
        nombre_cis = config["cis_name"]
        ideologia = notas_cis.get(nombre_cis, -1.0)
        nombre_final = "SUMAR" if codigo == "000010" else nombre_cis
        json_maestro[codigo] = {
            "nombre_unificado": nombre_final,
            "ideologia": ideologia,
            "voto_retrospectivo_id": config["voto_retrospectivo_id"]
        }
        
    guardar_json(json_maestro, output_json)
    return json_maestro


def leer_fichero_candidaturas(fichero_03: str) -> pd.DataFrame:
    """
    Lee el fichero de candidaturas en formato de ancho fijo (FWF).
    Args:
        fichero_03 (str): Ruta al archivo .DAT de candidaturas.
    Returns:
        pd.DataFrame: DataFrame con los datos de las candidaturas extraídas.
    """
    colspecs = [(8, 14), (14, 64), (226, 232)]
    names = ['COD_CANDIDATURA', 'SIGLAS', 'COD_NACIONAL']
    
    df = pd.read_fwf(fichero_03, colspecs=colspecs, names=names, dtype=str, encoding='latin-1')
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def construir_relaciones_jerarquicas(fichero_03: str) -> List[List[Any]]:
    """
    Construye las relaciones jerárquicas entre candidaturas nacionales y sus filiales.
    Args:
        fichero_03 (str): Ruta al archivo .DAT de candidaturas.
    Returns:
        List[List[Any]]: Lista de listas con la estructura jerárquica de los partidos.
    """
    df = leer_fichero_candidaturas(fichero_03).copy()
    df['COD_NACIONAL'] = df['COD_NACIONAL'].replace(['', '000000', 'nan'], pd.NA)
    df['ID_PADRE_ASIGNADO'] = df['COD_NACIONAL'].fillna(df['COD_CANDIDATURA'])
    
    diccionario_nombres = dict(zip(df['COD_CANDIDATURA'], df['SIGLAS']))
    
    lista_final = []
    for id_padre, grupo in df.groupby('ID_PADRE_ASIGNADO'):
        lista_codigos = grupo['COD_CANDIDATURA'].tolist()
        lista_siglas = grupo['SIGLAS'].tolist()
        nombre_padre = diccionario_nombres.get(id_padre, lista_siglas[0])
        lista_final.append([id_padre, nombre_padre, lista_codigos, lista_siglas])
    return lista_final


def exportar_a_json(fichero_03: str, output_file: str) -> None:
    """
    Exporta las relaciones jerárquicas de candidaturas a un archivo JSON.
    Args:
        fichero_03 (str): Ruta al archivo .DAT de candidaturas.
        output_file (str): Ruta del archivo JSON de salida.
    """
    lista_jerarquia = construir_relaciones_jerarquicas(fichero_03)
    json_output: Dict[str, Any] = {}
    for item in lista_jerarquia:
        json_output[item[0]] = {
            "siglas_generales": item[1],
            "vinculadas": [{"codigo": c, "sigla": s} for c, s in zip(item[2], item[3])]
        }
    guardar_json(json_output, output_file)


def fusionar_ideologia_y_arbol(fichero_maestro_ideologia: str, fichero_arbol: str, output_file: str) -> Optional[Dict[str, Any]]:
    """
    Fusiona el árbol de candidaturas con el diccionario maestro de ideología.
    Args:
        fichero_maestro_ideologia (str): Ruta al archivo JSON con datos de ideología.
        fichero_arbol (str): Ruta al archivo JSON con el árbol de candidaturas.
        output_file (str): Ruta del archivo JSON de salida consolidado.
    Returns:
        Optional[Dict[str, Any]]: Diccionario fusionado, o None si no se encuentran los archivos.
    """
    try:
        dic_ideologia = leer_json(fichero_maestro_ideologia)
        dic_arbol = leer_json(fichero_arbol)
    except FileNotFoundError as e:
        print(f"Error crítico de lectura de archivos: {e}")
        return None
    
    json_final: Dict[str, Any] = {}
    for cod_padre, datos_arbol in dic_arbol.items():
        if cod_padre in dic_ideologia:
            nombre_unificado = dic_ideologia[cod_padre]["nombre_unificado"]
            ideologia = dic_ideologia[cod_padre]["ideologia"]
            voto_retro = dic_ideologia[cod_padre]["voto_retrospectivo_id"]
        else:
            nombre_unificado = datos_arbol["siglas_generales"]
            ideologia = -1.0
            voto_retro = "desconocido"
            
        json_final[cod_padre] = {
            "nombre_unificado": nombre_unificado,
            "ideologia": ideologia,
            "voto_retrospectivo_id": voto_retro,
            "vinculadas": datos_arbol["vinculadas"]  
        }
        
    guardar_json(json_final, output_file)
        
    return json_final

# =============================================================================
# PROCESAMIENTO DE VOTOS
# =============================================================================

def leer_votos_municipales(file_path: str) -> pd.DataFrame:
    """
    Lee y filtra el archivo de resultados electorales municipales.
    Args:
        file_path (str): Ruta al archivo .DAT de votos municipales.
    Returns:
        pd.DataFrame: DataFrame con los votos procesados a nivel de municipio.
    """
    col_specs = [(2, 6), (6, 8), (9, 11), (11, 14), (14, 16), (16, 22), (22, 30)]
    col_names = ['ANYO', 'MES', 'PROVINCIA', 'MUNICIPIO', 'DISTRITO', 'ID_CANDIDATURA', 'VOTOS']
    df = pd.read_fwf(file_path, colspecs=col_specs, names=col_names, dtype=str, encoding='latin-1')
    df = df[df['DISTRITO'] == '99'].copy()
    df['ID_MUNICIPIO'] = df['PROVINCIA'].str.zfill(2) + df['MUNICIPIO'].str.zfill(3)
    df['FECHA_ELECCION'] = df['ANYO'] + "-" + df['MES'].str.zfill(2)
    df['ID_CANDIDATURA'] = df['ID_CANDIDATURA'].str.strip()
    df['VOTOS'] = pd.to_numeric(df['VOTOS'], errors='coerce').fillna(0).astype(int)
    
    return df[['ID_MUNICIPIO', 'FECHA_ELECCION', 'ID_CANDIDATURA', 'VOTOS']]


def extraer_mapa_siglas_del_json(json_path: str) -> Dict[str, str]:
    """
    Extrae un diccionario plano que relaciona códigos de candidaturas con sus siglas.
    Args:
        json_path (str): Ruta al archivo JSON maestro.
    Returns:
        Dict[str, str]: Diccionario con el mapeo de código a siglas.
    """
    datos_json = leer_json(json_path)
        
    mapa_siglas: Dict[str, str] = {}
    for _, info in datos_json.items():
        for vinculada in info.get("vinculadas", []):
            mapa_siglas[vinculada["codigo"]] = vinculada["sigla"]
    return mapa_siglas


def _pivotar_votos(df: pd.DataFrame, col_partido: str, year_suffix: str, output_file: str) -> pd.DataFrame:
    """Pivota votos por municipio, prefija columnas con V_ y guarda el CSV."""
    df_pivot = df.pivot_table(
        index=['ID_MUNICIPIO', 'FECHA_ELECCION'],
        columns=col_partido,
        values='VOTOS',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    df_pivot.columns = [
        col if col in ('ID_MUNICIPIO', 'FECHA_ELECCION') else f"V_{col}_{year_suffix}"
        for col in df_pivot.columns
    ]
    guardar_dataframe_csv(df_pivot, output_file)
    return df_pivot


def generar_csv_alta_granularidad(df_votos: pd.DataFrame, mapa_siglas: Dict[str, str], year_suffix: str, output_file: str) -> pd.DataFrame:
    """Pivota el DataFrame de votos manteniendo todas las filiales como columnas separadas."""
    df = df_votos.copy()
    df['SIGLAS'] = df['ID_CANDIDATURA'].map(mapa_siglas).fillna(df['ID_CANDIDATURA'])
    df['SIGLAS'] = df['SIGLAS'].str.replace(' ', '_').str.replace('.', '')
    return _pivotar_votos(df, 'SIGLAS', year_suffix, output_file)


def obtener_diccionario_padres(json_path: str) -> Dict[str, str]:
    """
    Genera un diccionario que asocia los códigos de filiales con el nombre de su partido matriz.
    Args:
        json_path (str): Ruta al archivo JSON maestro.
    Returns:
        Dict[str, str]: Diccionario con la relación de códigos filiales y nombres unificados.
    """
    datos_json = leer_json(json_path)
        
    mapa_unificacion: Dict[str, str] = {}
    for _, info in datos_json.items():
        nombre_padre = info.get("nombre_unificado", "DESCONOCIDO")
        for vinculada in info.get("vinculadas", []):
            mapa_unificacion[vinculada["codigo"]] = nombre_padre
    return mapa_unificacion


def procesar_votos_agrupados_por_padre(file_path_06: str, json_path: str, year_suffix: str, output_file: str) -> pd.DataFrame:
    """Pivota el DataFrame agrupando los votos de filiales bajo la columna de su partido padre."""
    mapa_padres = obtener_diccionario_padres(json_path)
    df = leer_votos_municipales(file_path_06)
    df['PARTIDO_UNIFICADO'] = df['ID_CANDIDATURA'].map(mapa_padres).fillna('OTROS')
    df['PARTIDO_UNIFICADO'] = df['PARTIDO_UNIFICADO'].str.replace(' ', '_').str.replace('.', '')
    return _pivotar_votos(df, 'PARTIDO_UNIFICADO', year_suffix, output_file)


def procesar_participacion_municipios(file_path_05: str, anio: str, output_file: str) -> pd.DataFrame:
    """
    Lee el fichero de totales municipales (05), extrae votos en blanco y calcula la participación.
    Args:
        file_path_05 (str): Ruta al archivo .DAT de totales municipales.
        anio (str): Año de las elecciones para sufijos.
        output_file (str): Ruta donde guardar el CSV.
    Returns:
        pd.DataFrame: DataFrame con ID_MUNICIPIO, VOTOS_BLANCO y PARTICIPACION.
    """
    # Especificaciones según formato MIR para el fichero tipo 05 (Totales de Municipio)
    col_specs = [
        (11, 13),   # Código I.N.E. de la provincia (Pos 12-13)
        (13, 16),   # Código I.N.E. del municipio (Pos 14-16)
        (16, 18),   # Número de distrito municipal (Pos 17-18)
        (149, 157), # Censo de escrutinio (Pos 150-157)
        (189, 197), # Votos en blanco (Pos 190-197)
        (197, 205), # Votos nulos (Pos 198-205)
        (205, 213)  # Votos a candidaturas (Pos 206-213)
    ]
    col_names = [
        'PROVINCIA', 'MUNICIPIO', 'DISTRITO', 'CENSO', 
        'VOTOS_BLANCO', 'VOTOS_NULOS', 'VOTOS_CANDIDATURAS'
    ]

    df = pd.read_fwf(file_path_05, colspecs=col_specs, names=col_names, dtype=str, encoding='latin-1')

    # Filtrar por total municipal
    df = df[df['DISTRITO'] == '99'].copy()

    # Crear ID_MUNICIPIO
    df['ID_MUNICIPIO'] = df['PROVINCIA'].str.zfill(2) + df['MUNICIPIO'].str.zfill(3)

    # Convertir a numérico
    for col in ['CENSO', 'VOTOS_BLANCO', 'VOTOS_NULOS', 'VOTOS_CANDIDATURAS']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Calcular participación
    # Votos emitidos = Blancos + Nulos + Candidaturas
    df['VOTOS_EMITIDOS'] = df['VOTOS_BLANCO'] + df['VOTOS_NULOS'] + df['VOTOS_CANDIDATURAS']

    # Evitar división por cero
    df['PARTICIPACION'] = 0.0
    mask = df['CENSO'] > 0
    df.loc[mask, 'PARTICIPACION'] = (df.loc[mask, 'VOTOS_EMITIDOS'] / df.loc[mask, 'CENSO']) * 100

    # Renombrar columnas con el año para consistencia con el resto del pipeline
    df = df.rename(columns={
        'VOTOS_BLANCO': f'V_BLANCOS_{anio}',
        'PARTICIPACION': f'PARTICIPACION_{anio}'
    })

    df_result = df[['ID_MUNICIPIO', f'V_BLANCOS_{anio}', f'PARTICIPACION_{anio}']]

    guardar_dataframe_csv(df_result, output_file)

    return df_result


# =============================================================================
# 3. FUNCIÓN MAESTRA
# =============================================================================

def limpieza_votos_partidos(fichero_cis: str, fichero_03: str, fichero_05: str, fichero_06: str, ruta_guardado: str, anio: str = "2023") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Función orquestadora que ejecuta el pipeline completo de limpieza, agrupación y exportación.
    Args:
        fichero_cis (str): Ruta al archivo Excel del CIS.
        fichero_03 (str): Ruta al archivo de candidaturas (.DAT).
        fichero_05 (str): Ruta al archivo de totales municipales (.DAT).
        fichero_06 (str): Ruta al archivo de votos municipales (.DAT).
        ruta_guardado (str): Directorio destino para guardar todos los artefactos.
        anio (str, optional): Año electoral para sufijos. Por defecto es "2023".
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames granular, agrupado y de participación.
    """

    archivo_maestro_ideologia = os.path.join(ruta_guardado, f"maestro_ideologia_{anio}.json")
    archivo_arbol = os.path.join(ruta_guardado, f"arbol_candidaturas_{anio}.json")
    archivo_fusionado = os.path.join(ruta_guardado, f"candidaturas_ideologia_{anio}.json")
    archivo_csv_granular = os.path.join(ruta_guardado, f"Votos_Granularidad_Total_{anio}.csv")
    archivo_csv_padres = os.path.join(ruta_guardado, f"Votos_Agrupados_Padres_{anio}.csv")
    archivo_csv_participacion = os.path.join(ruta_guardado, f"participación_electoral_{anio}.csv")

    generar_json_desde_cis(fichero_cis, output_json=archivo_maestro_ideologia)
    exportar_a_json(fichero_03, output_file=archivo_arbol)

    fusionar_ideologia_y_arbol(archivo_maestro_ideologia, archivo_arbol, output_file=archivo_fusionado)

    df_votos_base = leer_votos_municipales(fichero_06)
    mapa_diccionario = extraer_mapa_siglas_del_json(archivo_fusionado)
    df_granular = generar_csv_alta_granularidad(df_votos_base, mapa_diccionario, anio, output_file=archivo_csv_granular)

    df_padres = procesar_votos_agrupados_por_padre(fichero_06, archivo_fusionado, anio, output_file=archivo_csv_padres)

    df_participacion = procesar_participacion_municipios(fichero_05, anio, output_file=archivo_csv_participacion)

    return df_granular, df_padres, df_participacion