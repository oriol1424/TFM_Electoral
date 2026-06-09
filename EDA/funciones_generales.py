import os
import json
import pandas as pd
import numpy as np

RANGOS_MUNICIPIO = [
    "<100", "101-500", "501-1000", "1001-2000", "2001-5000",
    "5001-10000", "10001-20000", "20001-50000", "50000-100000",
    "100001-500000", ">500000"
]

def categorizar_municipios_tfm(pob: int) -> str:
    """Categoriza un municipio según su población en los rangos estándar del TFM."""
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

def resolver_col_id(df):
    """Detecta la columna de código de municipio de forma robusta."""
    for nombre in ['Cod_Muni', 'cod_Muni', 'Código', 'Codigo', 'id_municipio', 'index', 'CPROCMUN', 'id']:
        if nombre in df.columns:
            return nombre
    for c in df.columns:
        if any(key in c.lower() for key in ['cod', 'muni', 'id']):
            return c
    return None

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
        print("Nota: Si quieres regenerar (nueva columna o auditoría), borra este archivo CSV.")
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

    df_edad = pd.read_csv(config["edad_media_municipios"], sep=';')
    df_edad['id_municipio'] = df_edad['id_municipio'].astype(str).str.zfill(5)
    auditar_municipios_extintos(df_edad.rename(columns={'id_municipio': 'Cod_Muni'}), df_pob, "Edad Media", 'Cod_Muni')

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

    participacion_path = os.path.join(config["carpeta_votos"], f"participación_electoral_{anyo_str}.csv")
    if os.path.exists(participacion_path):
        df_part = pd.read_csv(participacion_path, sep=';')
        df_part['ID_MUNICIPIO'] = df_part['ID_MUNICIPIO'].astype(str).str.zfill(5)
    else:
        df_part = pd.DataFrame(columns=['ID_MUNICIPIO', f'V_BLANCOS_{anyo_str}', f'PARTICIPACION_{anyo_str}'])

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

    df_final = pd.merge(df_final, df_edad[['id_municipio', 'edad_media_ambos']], on='id_municipio', how='left')

    df_final = pd.merge(df_final, df_votos[['id_municipio', 'votos totales']], on='id_municipio', how='left')
    
    df_final = pd.merge(df_final, df_part, left_on='id_municipio', right_on='ID_MUNICIPIO', how='left')

    votos_pct_path = os.path.join(folder_path, f"votos_pct_{anyo_str}.csv")
    if os.path.exists(votos_pct_path):
        df_pct = pd.read_csv(votos_pct_path, sep=';')
        df_pct = df_pct.rename(columns={'ID_MUNICIPIO': 'id_municipio'})
        df_pct['id_municipio'] = df_pct['id_municipio'].astype(str).str.zfill(5)
        df_final = pd.merge(df_final, df_pct, on='id_municipio', how='left')

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
        f'prestaciones por desempleo {anyo_str}': 'desempleo',
        f'V_BLANCOS_{anyo_str}': 'votos blancos',
        f'PARTICIPACION_{anyo_str}': 'participacion',
        'edad_media_ambos': 'edad media',
    })
    
    pct_cols = [c for c in df_final.columns if c.startswith('pct_')]
    columnas_ordenadas = [
        "municipio", "nombre", "provincia", "superficie", "latitud", "longitud", "altitud",
        "rango tamaño población", "poblacion", "poblacion hombres", "poblacion mujeres",
        "densidad poblacional", "indice gini", "P80P20", "Renta media hogar",
        "renta media unidad consumo", "renta media persona", "salarios", "pensiones",
        "otros ingresos", "otras prestaciones", "desempleo", "edad media",
        "votos totales", "votos blancos", "participacion"
    ] + pct_cols
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

    print(" -> Imputando edad media mediante ponderación por género...")
    mask_edad = df_imputado['edad media'].isna()
    if mask_edad.any():
        df_edad_prov = pd.read_csv(
            f"data_processed/edad_media/{anyo_str}/edad_media_provincias_{anyo_str}.csv", sep=';'
        )
        df_edad_prov['cod_str'] = df_edad_prov['cod_provincia'].astype(str).str.zfill(2)
        edad_dict = df_edad_prov.set_index('cod_str')[['edad_media_hombres', 'edad_media_mujeres']].to_dict('index')

        for idx in df_imputado.index[mask_edad]:
            cod_prov = idx[:2]
            if cod_prov not in edad_dict:
                continue
            edad_h = edad_dict[cod_prov]['edad_media_hombres']
            edad_m = edad_dict[cod_prov]['edad_media_mujeres']
            pob_h = df_imputado.at[idx, 'poblacion hombres']
            pob_m = df_imputado.at[idx, 'poblacion mujeres']
            pob_total = pob_h + pob_m
            if pob_total > 0 and pd.notna(edad_h) and pd.notna(edad_m):
                df_imputado.at[idx, 'edad media'] = (edad_h * pob_h + edad_m * pob_m) / pob_total
                df_imputado.at[idx, 'imputado'] = True

        n_imputados_edad = mask_edad.sum()
        print(f"\n   Municipios con edad media imputada: {n_imputados_edad}")
        print(f"   Fórmula aplicada:")
        print(f"     edad_media = (edad_media_hombres_prov × pob_hombres")
        print(f"                  + edad_media_mujeres_prov × pob_mujeres)")
        print(f"                  / (pob_hombres + pob_mujeres)")
        print(f"   Fuente provincial: edad_media_provincias_{anyo_str}.csv")

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
    Genera dos análisis de correlación orientados a selección de features ML:
    1. Correlación features × features (detectar multicolinealidad)
    2. Correlación features × targets pct_* (poder predictivo por partido)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    cols_meta = ['municipio', 'nombre', 'provincia', 'latitud', 'longitud', 'altitud', 'imputado',
                 'rango tamaño población', 'votos blancos']
    cols_target = [c for c in df.columns if c.startswith('pct_')]
    cols_feature = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in cols_meta and c not in cols_target]

    df_all = df[cols_feature + cols_target]

    if df_all.empty:
        print(f"Aviso: No hay datos para calcular correlaciones en {anyo}.")
        return

    print(f"\nCORRELACIONES ML ({anyo})")
    print(f"Features: {cols_feature}")
    print(f"Targets:  {cols_target}")
    print(f"Total municipios: {len(df_all)}")

    os.makedirs("documentation/imagenes_EDA", exist_ok=True)

    df_features_clean = df_all[cols_feature].dropna()
    corr_ff = df_features_clean.corr()
    mask = np.triu(np.ones_like(corr_ff, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_ff, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmax=1, vmin=-1, center=0, square=True, linewidths=.3,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Correlación entre Features — {anyo}", fontsize=13)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(f"documentation/imagenes_EDA/correlacion_features_{anyo}.png", dpi=150)
    plt.show()

    corr_ft = df_all.corr(min_periods=30).loc[cols_feature, cols_target]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr_ft, annot=True, fmt=".2f", cmap="coolwarm",
                vmax=1, vmin=-1, center=0, linewidths=.3,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Correlación Features → Targets (pct partido) — {anyo}", fontsize=13)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(f"documentation/imagenes_EDA/correlacion_features_targets_{anyo}.png", dpi=150)
    plt.show()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables derivadas a partir de las features socioeconómicas existentes.

    Transformaciones aplicadas:
      - log_poblacion, log_densidad_poblacional: corrigen skew extremo (47 y 13 respectivamente)
      - ratio_envejecimiento: pensiones / salarios — proxy de envejecimiento demográfico
      - dependencia_publica: suma de transferencias del Estado (pensiones + desempleo + otras_prestaciones)
      - precariedad_laboral: componente volátil del ingreso (desempleo + otras_prestaciones)
      - renta_ajustada_desigualdad: renta_media_persona × (1 - gini/100) — riqueza efectiva del ciudadano mediano
      - ratio_sexo: hombres / mujeres — proxy de zonas rurales masculinizadas o con emigración femenina

    Solo crea cada variable si las columnas fuente existen. Imprime un resumen de lo creado.
    """
    df = df.copy()
    creadas = []

    if "poblacion" in df.columns:
        df["log_poblacion"] = np.log1p(df["poblacion"])
        creadas.append("log_poblacion")

    if "densidad poblacional" in df.columns:
        df["log_densidad_poblacional"] = np.log1p(df["densidad poblacional"])
        creadas.append("log_densidad_poblacional")

    if {"pensiones", "salarios"}.issubset(df.columns):
        df["ratio_envejecimiento"] = df["pensiones"] / (df["salarios"] + 0.01)
        creadas.append("ratio_envejecimiento")

    cols_dep = ["pensiones", "desempleo", "otras prestaciones"]
    if all(c in df.columns for c in cols_dep):
        df["dependencia_publica"] = df[cols_dep].sum(axis=1)
        creadas.append("dependencia_publica")

    cols_prec = ["desempleo", "otras prestaciones"]
    if all(c in df.columns for c in cols_prec):
        df["precariedad_laboral"] = df[cols_prec].sum(axis=1)
        creadas.append("precariedad_laboral")

    if {"renta media persona", "indice gini"}.issubset(df.columns):
        df["renta_ajustada_desigualdad"] = df["renta media persona"] * (1 - df["indice gini"] / 100)
        creadas.append("renta_ajustada_desigualdad")

    if {"poblacion hombres", "poblacion mujeres"}.issubset(df.columns):
        df["ratio_sexo"] = df["poblacion hombres"] / (df["poblacion mujeres"] + 1)
        creadas.append("ratio_sexo")

    print(f"Feature engineering completado: {len(creadas)} variables creadas")
    for v in creadas:
        n_nulos = df[v].isna().sum()
        print(f"  + {v:<35} nulos={n_nulos}")

    return df
