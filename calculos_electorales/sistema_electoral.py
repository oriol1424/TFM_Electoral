import pandas as pd
import json


def calcular_y_guardar_escanos_json(ruta_json_poblacion):
    """
    Lee el JSON maestro de población, calcula los escaños por provincia,
    añade el campo "escaños" a cada provincia y guarda un nuevo JSON.
    """
    with open(ruta_json_poblacion, 'r', encoding='utf-8') as f:
        datos_json = json.load(f)
        
    provincias = datos_json.get("provincias", {})
    
    escaños_a_repartir = 248
    poblacion_total_50_prov = 0
    
    for id_prov, datos in provincias.items():
        if id_prov in ['51', '52']: 
            datos['escaños'] = 1
        else:
            datos['escaños'] = 2
            poblacion_total_50_prov += datos['total_provincial']
            
    cuota = poblacion_total_50_prov / escaños_a_repartir
    restos = {}
    escaños_proporcionales = {}
    
    for id_prov, datos in provincias.items():
        if id_prov in ['51', '52']:
            continue
            
        pob = datos['total_provincial']
        asignacion_entera = int(pob // cuota)
        escaños_proporcionales[id_prov] = asignacion_entera
        restos[id_prov] = pob - (asignacion_entera * cuota)
        
    escaños_sobrantes = escaños_a_repartir - sum(escaños_proporcionales.values())
    
    provincias_por_resto = sorted(restos.keys(), key=lambda x: restos[x], reverse=True)
    for i in range(int(escaños_sobrantes)):
        provincia_ganadora = provincias_por_resto[i]
        escaños_proporcionales[provincia_ganadora] += 1
        
    for id_prov, escaños_prop in escaños_proporcionales.items():
        provincias[id_prov]['escaños'] += escaños_prop
        
    ruta_salida_json = ruta_json_poblacion.replace('.json', '_con_escanos.json')
    with open(ruta_salida_json, 'w', encoding='utf-8') as f:
        json.dump(datos_json, f, indent=4, ensure_ascii=False)
        
    return {id_prov: datos['escaños'] for id_prov, datos in provincias.items()}


def agrupar_votos_csv(ruta_csv_votos):
    """
    Lee el CSV de votos agrupados, extrae la provincia y suma los votos.
    IMPORTANTE: Usa sep=';' para leer correctamente tus datos.
    """
    df = pd.read_csv(ruta_csv_votos, sep=';', dtype={'ID_MUNICIPIO': str})
    df['ID_MUNICIPIO'] = df['ID_MUNICIPIO'].str.zfill(5)
    df['id_provincia'] = df['ID_MUNICIPIO'].str[:2]
    cols_a_borrar = ['ID_MUNICIPIO', 'FECHA_ELECCION']
    cols_presentes = [col for col in cols_a_borrar if col in df.columns]
    df = df.drop(columns=cols_presentes)
    df_largo = pd.melt(df, id_vars=['id_provincia'], var_name='partido', value_name='votos')
    df_largo['votos'] = pd.to_numeric(df_largo['votos'], errors='coerce').fillna(0)
    votos_provinciales = df_largo.groupby(['id_provincia', 'partido'])['votos'].sum().reset_index()
    
    return votos_provinciales


def aplicar_dhondt(dict_votos, num_escaños):
    """
    Aplica la Ley D'Hondt con la barrera electoral del 3% provincial.
    """
    if num_escaños == 0:
        return {}

    total_votos_validos = sum(dict_votos.values())
    if total_votos_validos == 0:
        return {}
        
    umbral = total_votos_validos * 0.03
    partidos_validos = {p: v for p, v in dict_votos.items() if v >= umbral}
    
    if not partidos_validos:
        return {}

    cocientes = []
    for partido, votos in partidos_validos.items():
        for divisor in range(1, num_escaños + 1):
            cocientes.append((votos / divisor, partido))
            
    cocientes.sort(key=lambda x: x[0], reverse=True)
    escaños_ganados = cocientes[:num_escaños]
    
    resultados_partidos = {p: 0 for p in partidos_validos.keys()}
    for _, partido in escaños_ganados:
        resultados_partidos[partido] += 1
        
    return {p: esc for p, esc in resultados_partidos.items() if esc > 0}


def sistema_electoral(ruta_json_poblacion, ruta_csv_votos, ruta_salida_csv):
    """
    Coordina todo el TFM: Calcula escaños poblacionales, agrupa votos y aplica D'Hondt.
    """
    diccionario_escanos = calcular_y_guardar_escanos_json(ruta_json_poblacion)
    
    total_escanos = sum(diccionario_escanos.values())
    assert total_escanos == 350, f"Error Crítico: El reparto poblacional sumó {total_escanos} escaños en vez de 350."
    
    df_votos_prov = agrupar_votos_csv(ruta_csv_votos)
    resultados_finales = []
    
    for prov, num_escaños in diccionario_escanos.items():
        df_prov_votos = df_votos_prov[df_votos_prov['id_provincia'] == prov]
        dict_votos_prov = dict(zip(df_prov_votos['partido'], df_prov_votos['votos']))
        
        reparto_provincia = aplicar_dhondt(dict_votos_prov, num_escaños)
        
        for partido, votos in dict_votos_prov.items():
            if votos > 0:
                escaños_obtenidos = reparto_provincia.get(partido, 0)
                resultados_finales.append({
                    'id_provincia': prov,
                    'escaños_en_juego': num_escaños,
                    'partido': partido,
                    'votos': votos,
                    'escaños_obtenidos': escaños_obtenidos
                })
                
    df_resultados = pd.DataFrame(resultados_finales)
    df_resultados.to_csv(ruta_salida_csv, index=False, encoding='utf-8', sep=';')
    print(f"SIMULACIÓN COMPLETADA\nResultados exportados en: {ruta_salida_csv}")
    
    return df_resultados