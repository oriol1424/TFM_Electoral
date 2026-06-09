import os
import numpy as np
import pandas as pd

from .funciones_genericas_limpieza import (
    leer_json, guardar_dataframe_csv, limpiar_valor_numerico,
    leer_archivo_csv, leer_datos_mixtos, formatear_serie_codigo,
)
from .votos import limpieza_votos_partidos, generar_columnas_pct_votos


# Junio 2016: Unidos Podemos (PODEMOS + IU) corrieron juntos.
# JxCAT no existía aún (era DL/CDC). VOX marginal.
SLOT_MAPPING_2016 = {
    'pct_psoe':     ['PSOE', 'PSC', 'PSdeG-PSOE', 'PSE-EE_(PSOE)', 'PSIB-PSOE', 'PSN-PSOE'],
    'pct_pp':       ['PP', 'PP-FORO', 'PPSO'],
    'pct_vox':      ['VOX'],
    'pct_cs':       ['Cs'],
    'pct_up_sumar': [
        'PODEMOS-IU', 'PODEMOS-IU_LV_CA', 'ECP-GUANYEM',
        'PODEMOS-EUPV', 'PODEMOS-EU', 'PODEMOS-EUIB',
        'PODEMOS-IX', 'PODEMOS-IU-BATZARRE', 'EN_MAREA',
    ],
    'pct_erc':      ['ERC-SOBIRANISTES', 'ERC'],
    'pct_jxcat':    ['DL', 'CDC', 'DiL'],
    'pct_cup':      ['CUP', 'CUP-PR'],
    'pct_pnv':      ['EAJ-PNV'],
    'pct_ehbildu':  ['EH_Bildu'],
    'pct_bng':      ['BNG'],
    'pct_cc':       ['CCa-PNC-NC', 'NC-CCa-PNC', 'CCa'],
    'pct_prc':      ['PRC'],
    'pct_naplus':   ['UPN'],
    'pct_teruel':   [],
}

# Diciembre 2015: Podemos e IU corrieron por separado.
# JxCAT no existía (era DL = Democràcia i Llibertat, fusión CiU/CDC).
SLOT_MAPPING_2015 = {
    'pct_psoe':     ['PSOE', 'PSC', 'PSdeG-PSOE', 'PSE-EE_(PSOE)', 'PSIB-PSOE', 'PSN-PSOE'],
    'pct_pp':       ['PP', 'PP-FORO', 'PPSO'],
    'pct_vox':      ['VOX'],
    'pct_cs':       ['Cs'],
    'pct_up_sumar': [
        'PODEMOS', 'PODEMOS-EQUO', 'PODEMOS-EUIB',
        'PODEMOS-EU', 'PODEMOS-IX', 'PODEMOS-IU-BATZARRE',
        'EN_COMÚ_PODEM', 'EN_MAREA', 'PODEMOS-COMPROMÍS',
        'IU', 'IU-UP',
    ],
    'pct_erc':      ['SOBIRANISTES', 'ERC-SOBIRANISTES', 'ERC-CATSÍ', 'ERC'],
    'pct_jxcat':    ['DL', 'CDC', 'DiL', 'CiU'],
    'pct_cup':      ['CUP', 'CUP-PR'],
    'pct_pnv':      ['EAJ-PNV'],
    'pct_ehbildu':  ['EH_Bildu'],
    'pct_bng':      ['BNG'],
    'pct_cc':       ['CCa-PNC-NC', 'NC-CCa-PNC', 'CCa'],
    'pct_prc':      ['PRC'],
    'pct_naplus':   ['UPN'],
    'pct_teruel':   [],
}

_SLOT_MAPPING_HISTORICO = {
    '2015': SLOT_MAPPING_2015,
    '2016': SLOT_MAPPING_2016,
}

ANOS_ELECCIONES = [2015, 2016, 2019, 2023]

_RENTA_MAP = {
    "Renta neta media por persona":               "renta_neta_persona",
    "Renta bruta media por persona":              "renta_bruta_persona",
    "Renta neta media por hogar":                 "renta_neta_hogar",
    "Mediana de la renta por unidad de consumo":  "mediana_renta_uc",
}

_FUENTE_MAP = {
    "Fuente de ingreso: salario":                      "salarios",
    "Fuente de ingreso: pensiones":                    "pensiones",
    "Fuente de ingreso: prestaciones por desempleo":   "desempleo",
    "Fuente de ingreso: otras prestaciones":           "otras_prestaciones",
    "Fuente de ingreso: otros ingresos":               "otros_ingresos",
}

_GINI_MAP = {
    "Índice de Gini":                    "gini",
    "Distribución de la renta P80/P20":  "p80p20",
}

_PCT_COLS = [
    "pct_psoe", "pct_pp", "pct_vox", "pct_cs", "pct_up_sumar",
    "pct_erc", "pct_jxcat", "pct_cup", "pct_pnv", "pct_ehbildu",
    "pct_bng", "pct_cc", "pct_prc", "pct_naplus", "pct_teruel", "pct_otros",
]

_FIXED_COLS = {"Municipios", "Distritos", "Secciones", "Periodo", "Total"}


def _etl_votos_anio_historico(config, anio):
    """Procesa los ficheros de votos de un año histórico (2015 ó 2016) si no existen."""
    raw  = config[anio]['raw']
    proc = config[anio]['processed']

    carpeta_votos = proc['carpeta_votos']
    votos_pct     = proc['votos_pct']
    csv_granular  = os.path.join(carpeta_votos, f'Votos_Granularidad_Total_{anio}.csv')

    ya_procesado = (
        os.path.exists(csv_granular) and os.path.exists(votos_pct)
    )
    if ya_procesado:
        print(f"  [OK] Votos {anio} ya procesados — omitiendo ETL.")
        return

    print(f"  [ETL] Procesando votos históricos {anio}...")
    os.makedirs(carpeta_votos, exist_ok=True)
    os.makedirs(os.path.dirname(votos_pct), exist_ok=True)

    limpieza_votos_partidos(
        fichero_cis=raw['ideologias'],
        fichero_03=raw['votos_candidaturas'],
        fichero_05=raw['votos_municipios_totales'],
        fichero_06=raw['Votos_candidaturas_agrupados'],
        ruta_guardado=carpeta_votos,
        anio=anio,
    )

    generar_columnas_pct_votos(
        csv_votos_granular=csv_granular,
        anio=anio,
        output_file=votos_pct,
        slot_mapping=_SLOT_MAPPING_HISTORICO[anio],
    )
    print(f"  [OK] Votos {anio} generados.")


def _etl_votos_historicos(config):
    """Asegura que los ficheros de votos de 2015 y 2016 existen."""
    for anio in ('2015', '2016'):
        if anio in config:
            _etl_votos_anio_historico(config, anio)
        else:
            print(f"  [AVISO] Año {anio} no encontrado en config — omitido.")


def _leer_ine_municipios(path):
    """Lee un CSV del INE filtrando a nivel municipio. Añade columna cod_ine."""
    df = leer_archivo_csv(path)
    df = df[df["Distritos"].isna() | (df["Distritos"].astype(str).str.strip() == "")].copy()
    df["cod_ine"] = formatear_serie_codigo(df["Municipios"], 5)
    df = df[df["cod_ine"].notna() & df["cod_ine"].str.match(r"^\d{5}$")].copy()
    return df


def _municipios_completos_ine(path):
    """Devuelve los cod_ine sin ningún valor '.' ni nulo en la columna Total."""
    df = _leer_ine_municipios(path)
    bad = df["Total"].isna() | df["Total"].astype(str).str.strip().isin(
        [".", "..", '""', "", "-"]
    )
    bad_munis = set(df.loc[bad, "cod_ine"].unique())
    return set(df["cod_ine"].unique()) - bad_munis


def _ine_muni_wide(path, col_map):
    """Carga un CSV del INE, pivota en ancho (cod_ine × año) y renombra con col_map."""
    df = _leer_ine_municipios(path)

    indicator_cols = [c for c in df.columns if c not in _FIXED_COLS and c != "cod_ine"]
    col_indicador = indicator_cols[0]

    df = df[df[col_indicador].isin(col_map.keys())].copy()
    df["ind_short"] = df[col_indicador].map(col_map)
    df["val"] = df["Total"].apply(lambda x: limpiar_valor_numerico(x, to_nan=True))

    df_pivot = df.pivot_table(
        index="cod_ine",
        columns=["ind_short", "Periodo"],
        values="val",
        aggfunc="first",
    )
    df_pivot.columns = [f"{ind}_{int(yr)}" for ind, yr in df_pivot.columns]
    return df_pivot.reset_index()


def _cargar_poblacion_anio(path_xlsx, anio):
    """Carga el Padrón Municipal de un año. Devuelve cod_ine + poblacion/hombres/mujeres."""
    if not path_xlsx or not os.path.exists(path_xlsx):
        return None

    # leer_datos_mixtos detecta Excel vs CSV y usa el engine correcto (xlrd/openpyxl)
    df_raw = leer_datos_mixtos(path_xlsx)
    df = df_raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = range(df.shape[1])

    # CPRO(0) PROVINCIA(1) CMUN(2) NOMBRE(3) POB(4) HOMBRES(5) MUJERES(6)
    cpro = formatear_serie_codigo(df[0], 2)
    cmun = formatear_serie_codigo(df[2], 3)
    df["cod_ine"] = cpro.fillna("") + cmun.fillna("")
    df = df[df["cod_ine"].str.match(r"^\d{5}$")].copy()

    for idx in [4, 5, 6]:
        df[idx] = (
            pd.to_numeric(
                df[idx].astype(str).str.replace(".", "", regex=False),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    return df[["cod_ine", 4, 5, 6]].rename(
        columns={
            4: f"poblacion_{anio}",
            5: f"hombres_{anio}",
            6: f"mujeres_{anio}",
        }
    )


def _cargar_participacion_anio(path_csv, anio):
    """Carga el CSV de participación electoral para un año."""
    if not path_csv or not os.path.exists(path_csv):
        return None

    df = leer_archivo_csv(path_csv, decimal='.')
    df["cod_ine"] = formatear_serie_codigo(df["ID_MUNICIPIO"], 5)
    return df.rename(
        columns={
            f"PARTICIPACION_{anio}": f"participacion_{anio}",
            f"V_BLANCOS_{anio}": f"votos_blancos_{anio}",
        }
    )[["cod_ine", f"participacion_{anio}", f"votos_blancos_{anio}"]].copy()


def _cargar_targets_anio(path_csv, anio):
    """Carga el CSV de porcentajes de voto y añade sufijo _anio a columnas pct_*."""
    if not path_csv or not os.path.exists(path_csv):
        return None

    df = leer_archivo_csv(path_csv, decimal='.')
    df["cod_ine"] = formatear_serie_codigo(df["ID_MUNICIPIO"], 5)
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    df = df.rename(columns={c: f"{c}_{anio}" for c in pct_cols})
    return df[["cod_ine"] + [f"{c}_{anio}" for c in pct_cols]].copy()


def calcular_filtro_demo(config):
    """Calcula los municipios con datos INE completos (sin '.' en 30824, 30825 y 37677/37731)."""
    raw19 = config["2019"]["raw"]

    ok_30824 = _municipios_completos_ine(raw19["renta_disponible"])
    ok_30825 = _municipios_completos_ine(raw19["fuente_ingresos"])

    # 37677 (nacional) + 37731 (suplemento Navarra con datos propios)
    ok_37677 = _municipios_completos_ine(raw19["GINI_P80P20"])
    ok_37731 = _municipios_completos_ine(
        config["demo"]["raw"]["GINI_P80P20_navarra_multianual"]
    )
    ok_gini = ok_37677 | ok_37731

    resultado = ok_30824 & ok_30825 & ok_gini
    print(f"  [Filtro] Sin '.' en 30824+30825+37677/37731: {len(resultado)} municipios")
    return resultado


def ETL_historico(config_path="config_path.json"):
    """Construye y guarda panel_historico.csv con todos los años electorales."""
    config = leer_json(config_path)
    output_path = config["demo"]["processed"]["panel_demo"]

    if os.path.exists(output_path):
        print(f"[OK] panel_historico.csv ya existe — cargando desde '{output_path}'")
        panel = pd.read_csv(output_path, sep=";", encoding="utf-8-sig", dtype={"cod_ine": str})
        print(f"     {len(panel)} municipios × {panel.shape[1]} columnas")
        return panel

    _etl_votos_historicos(config)

    raw19 = config["2019"]["raw"]
    print("=== CONSTRUYENDO PANEL HISTÓRICO ===")

    munis_demo = calcular_filtro_demo(config)

    # 2. Base geográfica
    df_sup = pd.read_csv(
        config["2019"]["processed"]["geografia"], sep=";", encoding="utf-8-sig"
    )
    df_sup["cod_ine"] = df_sup["id_municipio"].astype(str).str.zfill(5)
    df_sup = df_sup[df_sup["cod_ine"].isin(munis_demo)].copy()
    df_sup["superficie_km2"] = pd.to_numeric(
        df_sup["superficie_km2"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    panel = df_sup[["cod_ine", "superficie_km2", "latitud", "longitud", "altitud"]].copy()
    panel["provincia_enc"] = panel["cod_ine"].str[:2].astype(int)
    print(f"  [Base] {len(panel)} municipios con datos geográficos")

    pop_paths = {
        anio: config.get(str(anio), {}).get("raw", {}).get("poblacion")
        for anio in ANOS_ELECCIONES
    }

    for anio, path in pop_paths.items():
        df_pop = _cargar_poblacion_anio(path, anio)
        if df_pop is None:
            print(f"  [AVISO] Población {anio}: archivo no disponible — columnas omitidas.")
            continue
        df_pop = df_pop[df_pop["cod_ine"].isin(munis_demo)].copy()
        df_pop[f"log_poblacion_{anio}"] = np.log1p(df_pop[f"poblacion_{anio}"])
        df_pop[f"ratio_sexo_{anio}"] = (
            df_pop[f"hombres_{anio}"] / (df_pop[f"mujeres_{anio}"] + 1)
        )
        cols = [
            "cod_ine",
            f"poblacion_{anio}",
            f"log_poblacion_{anio}",
            f"ratio_sexo_{anio}",
        ]
        panel = panel.merge(df_pop[cols], on="cod_ine", how="left")

        panel[f"densidad_{anio}"] = panel[f"poblacion_{anio}"] / panel["superficie_km2"]
        panel[f"log_densidad_{anio}"] = np.log1p(panel[f"densidad_{anio}"])

    df_renta = _ine_muni_wide(raw19["renta_disponible"], _RENTA_MAP)
    df_renta = df_renta[df_renta["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_renta, on="cod_ine", how="left")
    print(f"  [Renta] {len(df_renta.columns) - 1} columnas añadidas")

    df_fuente = _ine_muni_wide(raw19["fuente_ingresos"], _FUENTE_MAP)
    df_fuente = df_fuente[df_fuente["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_fuente, on="cod_ine", how="left")
    print(f"  [Fuente ingresos] {len(df_fuente.columns) - 1} columnas añadidas")

    df_gini_nac = _ine_muni_wide(raw19["GINI_P80P20"], _GINI_MAP)
    df_gini_nav = _ine_muni_wide(
        config["demo"]["raw"]["GINI_P80P20_navarra_multianual"], _GINI_MAP
    )
    df_gini_all = (
        df_gini_nac.set_index("cod_ine")
        .combine_first(df_gini_nav.set_index("cod_ine"))
        .reset_index()
    )
    df_gini_all = df_gini_all[df_gini_all["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_gini_all, on="cod_ine", how="left")
    print(f"  [Gini/P80P20] {len(df_gini_all.columns) - 1} columnas añadidas")

    for anio in ANOS_ELECCIONES:
        carpeta = config[str(anio)]["processed"]["carpeta_votos"]
        path_part = os.path.join(carpeta, f"participación_electoral_{anio}.csv")
        df_part = _cargar_participacion_anio(path_part, anio)
        if df_part is None:
            print(f"  [AVISO] Participación {anio}: archivo no encontrado.")
            continue
        df_part = df_part[df_part["cod_ine"].isin(munis_demo)]
        panel = panel.merge(df_part, on="cod_ine", how="left")

    for anio in ANOS_ELECCIONES:
        path_pct = config[str(anio)]["processed"]["votos_pct"]
        df_pct = _cargar_targets_anio(path_pct, anio)
        if df_pct is None:
            print(f"  [AVISO] Votos pct {anio}: archivo no encontrado.")
            continue
        df_pct = df_pct[df_pct["cod_ine"].isin(munis_demo)]
        panel = panel.merge(df_pct, on="cod_ine", how="left")

    _IZQ  = ["psoe", "up_sumar", "erc", "cup", "ehbildu", "bng", "naplus"]
    _DER  = ["pp", "vox", "cs"]
    _NAC  = ["pnv", "jxcat", "cc", "prc", "teruel"]

    for anio in ANOS_ELECCIONES:
        def _sum_slots(slots):
            cols = [f"pct_{s}_{anio}" for s in slots if f"pct_{s}_{anio}" in panel.columns]
            return panel[cols].sum(axis=1, min_count=1) if cols else None

        izq = _sum_slots(_IZQ)
        der = _sum_slots(_DER)
        nac = _sum_slots(_NAC)

        if izq is not None:
            panel[f"pct_izquierda_{anio}"] = izq
        if der is not None:
            panel[f"pct_derecha_{anio}"] = der
        if nac is not None:
            panel[f"pct_nacionalistas_{anio}"] = nac
        if izq is not None and der is not None:
            panel[f"indice_ideologico_{anio}"] = der - izq

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    guardar_dataframe_csv(panel, output_path)

    print(f"\n[OK] panel_historico.csv guardado en '{output_path}'")
    print(f"     {len(panel)} municipios × {panel.shape[1]} columnas")
    return panel
