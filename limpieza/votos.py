import os
import re
import json
import pandas as pd
from .funciones_genericas_limpieza import guardar_dataframe_csv, guardar_json, leer_json


def generar_json_desde_cis(fichero_cis, output_json):
    """Lee el Excel del CIS y genera un JSON con la ideología de cada partido."""
    df_cis = pd.read_excel(fichero_cis, header=None)
    fila_nombres = df_cis[df_cis.eq('PSOE').any(axis=1)].iloc[0].tolist()
    fila_media = df_cis[df_cis[0] == 'Media'].iloc[0].tolist()

    notas_cis = {}
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
        "000023": {"cis_name": "Teruel Existe", "voto_retrospectivo_id": "bloque_nacionalista_Aragones"}
    }

    json_maestro = {}
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


def leer_fichero_candidaturas(fichero_03):
    """Lee el fichero de candidaturas en formato de ancho fijo."""
    colspecs = [(8, 14), (14, 64), (226, 232)]
    names = ['COD_CANDIDATURA', 'SIGLAS', 'COD_NACIONAL']

    df = pd.read_fwf(fichero_03, colspecs=colspecs, names=names, dtype=str, encoding='latin-1')
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def construir_relaciones_jerarquicas(fichero_03):
    """Construye las relaciones jerárquicas entre candidaturas nacionales y sus filiales."""
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


def exportar_a_json(fichero_03, output_file):
    """Exporta las relaciones jerárquicas de candidaturas a un archivo JSON."""
    lista_jerarquia = construir_relaciones_jerarquicas(fichero_03)
    json_output = {}
    for item in lista_jerarquia:
        json_output[item[0]] = {
            "siglas_generales": item[1],
            "vinculadas": [{"codigo": c, "sigla": s} for c, s in zip(item[2], item[3])]
        }
    guardar_json(json_output, output_file)


def fusionar_ideologia_y_arbol(fichero_maestro_ideologia, fichero_arbol, output_file):
    """Fusiona el árbol de candidaturas con el diccionario maestro de ideología."""
    try:
        dic_ideologia = leer_json(fichero_maestro_ideologia)
        dic_arbol = leer_json(fichero_arbol)
    except FileNotFoundError as e:
        print(f"Error crítico de lectura de archivos: {e}")
        return None

    json_final = {}
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


def leer_votos_municipales(file_path):
    """Lee y filtra el archivo de resultados electorales municipales (fichero 06)."""
    col_specs = [(2, 6), (6, 8), (9, 11), (11, 14), (14, 16), (16, 22), (22, 30)]
    col_names = ['ANYO', 'MES', 'PROVINCIA', 'MUNICIPIO', 'DISTRITO', 'ID_CANDIDATURA', 'VOTOS']
    df = pd.read_fwf(file_path, colspecs=col_specs, names=col_names, dtype=str, encoding='latin-1')
    df = df[df['DISTRITO'] == '99'].copy()
    df['ID_MUNICIPIO'] = df['PROVINCIA'].str.zfill(2) + df['MUNICIPIO'].str.zfill(3)
    df['FECHA_ELECCION'] = df['ANYO'] + "-" + df['MES'].str.zfill(2)
    df['ID_CANDIDATURA'] = df['ID_CANDIDATURA'].str.strip()
    df['VOTOS'] = pd.to_numeric(df['VOTOS'], errors='coerce').fillna(0).astype(int)

    return df[['ID_MUNICIPIO', 'FECHA_ELECCION', 'ID_CANDIDATURA', 'VOTOS']]


def extraer_mapa_siglas_del_json(json_path):
    """Extrae un dict código → siglas a partir del JSON maestro."""
    datos_json = leer_json(json_path)

    mapa_siglas = {}
    for _, info in datos_json.items():
        for vinculada in info.get("vinculadas", []):
            mapa_siglas[vinculada["codigo"]] = vinculada["sigla"]
    return mapa_siglas


def _pivotar_votos(df, col_partido, year_suffix, output_file):
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


def generar_csv_alta_granularidad(df_votos, mapa_siglas, year_suffix, output_file):
    """Pivota el DataFrame de votos manteniendo todas las filiales como columnas separadas."""
    df = df_votos.copy()
    df['SIGLAS'] = df['ID_CANDIDATURA'].map(mapa_siglas).fillna(df['ID_CANDIDATURA'])
    df['SIGLAS'] = df['SIGLAS'].str.replace(' ', '_').str.replace('.', '')
    return _pivotar_votos(df, 'SIGLAS', year_suffix, output_file)


def obtener_diccionario_padres(json_path):
    """Genera un dict código filial → nombre partido matriz."""
    datos_json = leer_json(json_path)

    mapa_unificacion = {}
    for _, info in datos_json.items():
        nombre_padre = info.get("nombre_unificado", "DESCONOCIDO")
        for vinculada in info.get("vinculadas", []):
            mapa_unificacion[vinculada["codigo"]] = nombre_padre
    return mapa_unificacion


def procesar_votos_agrupados_por_padre(file_path_06, json_path, year_suffix, output_file):
    """Pivota el DataFrame agrupando los votos de filiales bajo la columna de su partido padre."""
    mapa_padres = obtener_diccionario_padres(json_path)
    df = leer_votos_municipales(file_path_06)
    df['PARTIDO_UNIFICADO'] = df['ID_CANDIDATURA'].map(mapa_padres).fillna('OTROS')
    df['PARTIDO_UNIFICADO'] = df['PARTIDO_UNIFICADO'].str.replace(' ', '_').str.replace('.', '')
    return _pivotar_votos(df, 'PARTIDO_UNIFICADO', year_suffix, output_file)


def procesar_participacion_municipios(file_path_05, anio, output_file):
    """Extrae votos en blanco y participación del fichero de totales municipales (05)."""
    col_specs = [
        (11, 13),
        (13, 16),
        (16, 18),
        (149, 157),
        (189, 197),
        (197, 205),
        (205, 213)
    ]
    col_names = [
        'PROVINCIA', 'MUNICIPIO', 'DISTRITO', 'CENSO',
        'VOTOS_BLANCO', 'VOTOS_NULOS', 'VOTOS_CANDIDATURAS'
    ]

    df = pd.read_fwf(file_path_05, colspecs=col_specs, names=col_names, dtype=str, encoding='latin-1')

    df = df[df['DISTRITO'] == '99'].copy()

    df['ID_MUNICIPIO'] = df['PROVINCIA'].str.zfill(2) + df['MUNICIPIO'].str.zfill(3)

    for col in ['CENSO', 'VOTOS_BLANCO', 'VOTOS_NULOS', 'VOTOS_CANDIDATURAS']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['VOTOS_EMITIDOS'] = df['VOTOS_BLANCO'] + df['VOTOS_NULOS'] + df['VOTOS_CANDIDATURAS']

    df['PARTICIPACION'] = 0.0
    mask = df['CENSO'] > 0
    df.loc[mask, 'PARTICIPACION'] = (df.loc[mask, 'VOTOS_EMITIDOS'] / df.loc[mask, 'CENSO']) * 100

    df = df.rename(columns={
        'VOTOS_BLANCO': f'V_BLANCOS_{anio}',
        'PARTICIPACION': f'PARTICIPACION_{anio}'
    })

    df_result = df[['ID_MUNICIPIO', f'V_BLANCOS_{anio}', f'PARTICIPACION_{anio}']]

    guardar_dataframe_csv(df_result, output_file)

    return df_result


SLOT_MAPPING_DEFAULT = {
    'pct_psoe':     ['PSOE', 'PSC-PSOE', 'PSdeG-PSOE', 'PSE-EE_(PSOE)'],
    'pct_pp':       ['PP', 'PP-FORO', 'PPSO'],
    'pct_vox':      ['VOX'],
    'pct_cs':       ['Cs'],
    'pct_up_sumar': [
        'PODEMOS-IU', 'PODEMOS-IU_LV_CA', 'ECP-GUANYEM',
        'PODEMOS-EUPV', 'PODEMOS-EU', 'PODEMOS-EUIB',
        'PODEMOS-IX', 'PODEMOS-IU-BATZARRE', 'PODEMOS-IU-',
        'MÁS_PAÍS-EQ', 'MÁS_PAÍS-AN', 'MÁS_PAÍS-CA', 'MÁS_PAÍS',
        'M_PAÍS-CHA-', 'M_PAÍS', 'MÉS_COMPROM', 'MÉS-ESQUERR',
    ],
    'pct_erc':      ['ERC-SOBIRANISTES'],
    'pct_jxcat':    ['JxCAT-JUNTS'],
    'pct_cup':      ['CUP-PR'],
    'pct_pnv':      ['EAJ-PNV'],
    'pct_ehbildu':  ['EH_Bildu'],
    'pct_bng':      ['BNG'],
    'pct_cc':       ['CCa-PNC-NC', 'NC-CCa-PNC'],
    'pct_prc':      ['PRC'],
    'pct_naplus':   ['NA+'],
    'pct_teruel':   ['¡TERUEL_EXI'],
}

SLOT_MAPPING_2023 = {
    'pct_psoe':     ['PSOE', 'PSC', 'PSdeG-PSOE', 'PSE-EE_(PSOE)', 'PSIB-PSOE', 'PSN-PSOE'],
    'pct_pp':       ['PP'],
    'pct_vox':      ['VOX'],
    'pct_cs':       [],
    'pct_up_sumar': [
        'SUMAR', 'SUMAR_-_ECP',
        'SUMAR_-_COMPROMÍS',
        'SUMAR_ARAGÓN',
        'MÉS_PER_MALLORCA-MÉS_PER_MENORCA-SUMAR',
    ],
    'pct_erc':      ['ERC'],
    'pct_jxcat':    ['JxCAT_-_JUNTS'],
    'pct_cup':      ['CUP-PR'],
    'pct_pnv':      ['EAJ-PNV'],
    'pct_ehbildu':  ['EH_Bildu'],
    'pct_bng':      ['BNG'],
    'pct_cc':       ['CCa'],
    'pct_prc':      [],
    'pct_naplus':   ['UPN'],
    'pct_teruel':   ['EXISTE', 'ASTURIAS_EXISTE_EV', 'ESPAÑA_VACIADA'],
}


def generar_columnas_pct_votos(csv_votos_granular, anio, output_file, slot_mapping=None):
    """Calcula el porcentaje de voto por slot de partido para cada municipio."""
    if slot_mapping is None:
        slot_mapping = SLOT_MAPPING_DEFAULT

    df = pd.read_csv(csv_votos_granular, sep=';', encoding='utf-8-sig')

    vote_cols = [c for c in df.columns if c.startswith('V_')]
    votos_total = df[vote_cols].sum(axis=1)

    def _norm(s):
        return re.sub(r'[^\x00-\x7F]+', '?', s)

    col_norm_lookup = {}
    for col in vote_cols:
        if col.endswith(f'_{anio}'):
            raw_name = col[2:-(len(anio) + 1)]
            col_norm_lookup[_norm(raw_name)] = col

    result = df[['ID_MUNICIPIO']].copy()
    asignadas = set()

    for slot, candidaturas in slot_mapping.items():
        cols_presentes = []
        for cand in candidaturas:
            exact = f'V_{cand}_{anio}'
            if exact in df.columns:
                cols_presentes.append(exact)
            else:
                matched = col_norm_lookup.get(_norm(cand))
                if matched:
                    cols_presentes.append(matched)
        votos_slot = df[cols_presentes].sum(axis=1) if cols_presentes else pd.Series(0, index=df.index)
        result[slot] = (votos_slot / votos_total).fillna(0).round(6)
        asignadas.update(cols_presentes)

    cols_otras = [c for c in vote_cols if c not in asignadas]
    votos_otros = df[cols_otras].sum(axis=1) if cols_otras else pd.Series(0, index=df.index)
    result['pct_otros'] = (votos_otros / votos_total).fillna(0).round(6)

    pct_cols = [c for c in result.columns if c.startswith('pct_')]
    suma = result[pct_cols].sum(axis=1)
    discrepancias = (suma - 1.0).abs() > 0.01
    if discrepancias.any():
        print(f"[AVISO] {discrepancias.sum()} municipios con suma pct != 1.0 (diff > 1%)")

    guardar_dataframe_csv(result, output_file)
    print(f"[OK] {len(result)} municipios, {len(pct_cols)} slots guardados en {output_file}")
    return result


def limpieza_votos_partidos(fichero_cis, fichero_03, fichero_05, fichero_06, ruta_guardado, anio="2023"):
    """Pipeline completo de limpieza, agrupación y exportación de votos."""

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
