"""
limpieza/panel.py
Construye el panel multianual demo: una fila por municipio, columnas con sufijo _YYYY.

Subconjunto (~1.241 municipios) que cumple:
  1. POB_2019 >= 5.000 hab  (pobmun19.xlsx)
  2. Sin '.' ni nulos en 30824, 30825 y 37677/37731 a nivel municipio
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set

from .funciones_genericas_limpieza import (
    leer_json, guardar_dataframe_csv, limpiar_valor_numerico,
    leer_archivo_csv, leer_datos_mixtos, formatear_serie_codigo,
)


# Años con datos electorales (participación + votos pct)
ANOS_ELECCIONES: List[int] = [2015, 2016, 2019, 2023]

# ── Mapeos: nombre completo INE → nombre corto para columnas del panel ────────

_RENTA_MAP: Dict[str, str] = {
    "Renta neta media por persona":               "renta_neta_persona",
    "Renta bruta media por persona":              "renta_bruta_persona",
    "Renta neta media por hogar":                 "renta_neta_hogar",
    "Mediana de la renta por unidad de consumo":  "mediana_renta_uc",
}

_FUENTE_MAP: Dict[str, str] = {
    "Fuente de ingreso: salario":                      "salarios",
    "Fuente de ingreso: pensiones":                    "pensiones",
    "Fuente de ingreso: prestaciones por desempleo":   "desempleo",
    "Fuente de ingreso: otras prestaciones":           "otras_prestaciones",
    "Fuente de ingreso: otros ingresos":               "otros_ingresos",
}

_GINI_MAP: Dict[str, str] = {
    "Índice de Gini":                    "gini",
    "Distribución de la renta P80/P20":  "p80p20",
}

_PCT_COLS: List[str] = [
    "pct_psoe", "pct_pp", "pct_vox", "pct_cs", "pct_up_sumar",
    "pct_erc", "pct_jxcat", "pct_cup", "pct_pnv", "pct_ehbildu",
    "pct_bng", "pct_cc", "pct_prc", "pct_naplus", "pct_teruel", "pct_otros",
]

_FIXED_COLS = {"Municipios", "Distritos", "Secciones", "Periodo", "Total"}


# ── Helpers internos ─────────────────────────────────────────────────────────

def _leer_ine_municipios(path: str) -> pd.DataFrame:
    """
    Lee un CSV del INE filtrando a nivel de municipio (Distritos NaN/vacío).
    Añade columna cod_ine (5 dígitos).
    """
    df = leer_archivo_csv(path)
    df = df[df["Distritos"].isna() | (df["Distritos"].astype(str).str.strip() == "")].copy()
    df["cod_ine"] = formatear_serie_codigo(df["Municipios"], 5)
    df = df[df["cod_ine"].notna() & df["cod_ine"].str.match(r"^\d{5}$")].copy()
    return df


def _municipios_completos_ine(path: str) -> Set[str]:
    """
    Devuelve los cod_ine sin ningún valor '.' ni nulo en la columna Total
    para cualquier año/indicador a nivel municipio.
    """
    df = _leer_ine_municipios(path)
    bad = df["Total"].isna() | df["Total"].astype(str).str.strip().isin(
        [".", "..", '""', "", "-"]
    )
    bad_munis = set(df.loc[bad, "cod_ine"].unique())
    return set(df["cod_ine"].unique()) - bad_munis


def _ine_muni_wide(path: str, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Carga un CSV del INE a nivel municipio, pivota en ancho (cod_ine × año)
    y renombra columnas con col_map.
    Auto-detecta la columna de indicadores (4ª columna no estándar).
    Devuelve df con cod_ine + columnas tipo nombre_corto_YYYY.
    """
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


def _cargar_poblacion_anio(path_xlsx: str, anio: int) -> Optional[pd.DataFrame]:
    """
    Carga el Padrón Municipal de un año.
    CPRO y CMUN pueden venir como float o str; se normalizan correctamente.
    Devuelve df con: cod_ine, poblacion_YYYY, hombres_YYYY, mujeres_YYYY.
    Retorna None si el archivo no existe.
    """
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


def _cargar_participacion_anio(path_csv: str, anio: int) -> Optional[pd.DataFrame]:
    """
    Carga el CSV de participación electoral para un año.
    Devuelve df con: cod_ine, participacion_YYYY, votos_blancos_YYYY.
    """
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


def _cargar_targets_anio(path_csv: str, anio: int) -> Optional[pd.DataFrame]:
    """
    Carga el CSV de porcentajes de voto (votos_pct_YYYY.csv).
    Añade sufijo _YYYY a todas las columnas pct_*.
    Devuelve df con: cod_ine + columnas pct_*_YYYY.
    """
    if not path_csv or not os.path.exists(path_csv):
        return None

    df = leer_archivo_csv(path_csv, decimal='.')
    df["cod_ine"] = formatear_serie_codigo(df["ID_MUNICIPIO"], 5)
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    df = df.rename(columns={c: f"{c}_{anio}" for c in pct_cols})
    return df[["cod_ine"] + [f"{c}_{anio}" for c in pct_cols]].copy()


# ── Filtro ───────────────────────────────────────────────────────────────────

def calcular_filtro_demo(config: dict) -> Set[str]:
    """
    Calcula el subconjunto demo (~1.241 municipios) que cumple:
      1. POB_2019 >= 5.000 hab
      2. Sin '.' ni nulos en 30824, 30825 y 37677/37731 a nivel municipio

    Returns:
        Set de cod_ine (strings de 5 dígitos) que pasan ambos criterios.
    """
    raw19 = config["2019"]["raw"]

    # ── Filtro 1: población ────────────────────────────────────────────────
    df_pop = _cargar_poblacion_anio(raw19["poblacion"], 2019)
    if df_pop is None:
        raise FileNotFoundError("No se encuentra pobmun19.xlsx")
    munis_pop = set(df_pop.loc[df_pop["poblacion_2019"] >= 5000, "cod_ine"])
    print(f"  [Filtro 1] POB_2019 >= 5.000 hab: {len(munis_pop)} municipios")

    # ── Filtro 2: completitud INE ──────────────────────────────────────────
    ok_30824 = _municipios_completos_ine(raw19["renta_disponible"])
    ok_30825 = _municipios_completos_ine(raw19["fuente_ingresos"])

    # 37677 (nacional) + 37731 (suplemento Navarra con datos propios)
    ok_37677 = _municipios_completos_ine(raw19["GINI_P80P20"])
    ok_37731 = _municipios_completos_ine(
        config["demo"]["raw"]["GINI_P80P20_navarra_multianual"]
    )
    ok_gini = ok_37677 | ok_37731

    munis_ok = ok_30824 & ok_30825 & ok_gini
    print(f"  [Filtro 2] Sin '.' en 30824+30825+37677/37731: {len(munis_ok)} municipios")

    resultado = munis_pop & munis_ok
    print(f"  [Filtro combinado] Municipios demo: {len(resultado)}")
    return resultado


# ── Orquestador ──────────────────────────────────────────────────────────────

def construir_panel_demo(config_path: str = "config_path.json") -> pd.DataFrame:
    """
    Construye y guarda panel_demo.csv.

    Columnas resultantes:
      - Identificador: cod_ine
      - Geografía: superficie_km2, latitud, longitud, altitud, provincia_enc
      - Población (años electorales): poblacion_YYYY, log_poblacion_YYYY, ratio_sexo_YYYY
      - Densidad: densidad_YYYY, log_densidad_YYYY
      - Renta media (2015–2023): renta_neta_persona_YYYY, renta_bruta_persona_YYYY,
                                  renta_neta_hogar_YYYY, mediana_renta_uc_YYYY
      - Fuente ingresos (2015–2023): salarios_YYYY, pensiones_YYYY, desempleo_YYYY,
                                      otras_prestaciones_YYYY, otros_ingresos_YYYY
      - Desigualdad (2015–2023): gini_YYYY, p80p20_YYYY
      - Participación (años electorales): participacion_YYYY, votos_blancos_YYYY
      - Targets electorales: pct_*_YYYY
      - Bloques ideológicos: pct_izquierda_YYYY, pct_derecha_YYYY,
                              pct_nacionalistas_YYYY, indice_ideologico_YYYY

    Returns:
        DataFrame con el panel construido.
    """
    config = leer_json(config_path)
    output_path = config["demo"]["processed"]["panel_demo"]

    if os.path.exists(output_path):
        print(f"[OK] panel_demo.csv ya existe — cargando desde '{output_path}'")
        panel = pd.read_csv(output_path, sep=";", encoding="utf-8-sig", dtype={"cod_ine": str})
        print(f"     {len(panel)} municipios × {panel.shape[1]} columnas")
        return panel

    raw19 = config["2019"]["raw"]
    print("=== CONSTRUYENDO PANEL DEMO ===")

    # ── 1. Filtro de municipios ────────────────────────────────────────────
    munis_demo = calcular_filtro_demo(config)

    # ── 2. Base geográfica ─────────────────────────────────────────────────
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

    # ── 3. Población y densidad (años electorales) ─────────────────────────
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

        # Densidad derivada inmediatamente
        panel[f"densidad_{anio}"] = panel[f"poblacion_{anio}"] / panel["superficie_km2"]
        panel[f"log_densidad_{anio}"] = np.log1p(panel[f"densidad_{anio}"])

    # ── 4. Renta media — 30824.csv (todos los años disponibles) ───────────
    df_renta = _ine_muni_wide(raw19["renta_disponible"], _RENTA_MAP)
    df_renta = df_renta[df_renta["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_renta, on="cod_ine", how="left")
    print(f"  [Renta] {len(df_renta.columns) - 1} columnas añadidas")

    # ── 5. Fuente de ingresos — 30825.csv ─────────────────────────────────
    df_fuente = _ine_muni_wide(raw19["fuente_ingresos"], _FUENTE_MAP)
    df_fuente = df_fuente[df_fuente["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_fuente, on="cod_ine", how="left")
    print(f"  [Fuente ingresos] {len(df_fuente.columns) - 1} columnas añadidas")

    # ── 6. Gini / P80P20 — 37677.csv + suplemento 37731.csv ──────────────
    df_gini_nac = _ine_muni_wide(raw19["GINI_P80P20"], _GINI_MAP)
    df_gini_nav = _ine_muni_wide(
        config["demo"]["raw"]["GINI_P80P20_navarra_multianual"], _GINI_MAP
    )
    # Prioriza datos de Navarra (37731) sobre los nacionales (37677) donde haya NaN
    df_gini_all = (
        df_gini_nac.set_index("cod_ine")
        .combine_first(df_gini_nav.set_index("cod_ine"))
        .reset_index()
    )
    df_gini_all = df_gini_all[df_gini_all["cod_ine"].isin(munis_demo)]
    panel = panel.merge(df_gini_all, on="cod_ine", how="left")
    print(f"  [Gini/P80P20] {len(df_gini_all.columns) - 1} columnas añadidas")

    # ── 7. Participación electoral ─────────────────────────────────────────
    for anio in ANOS_ELECCIONES:
        carpeta = config[str(anio)]["processed"]["carpeta_votos"]
        path_part = os.path.join(carpeta, f"participación_electoral_{anio}.csv")
        df_part = _cargar_participacion_anio(path_part, anio)
        if df_part is None:
            print(f"  [AVISO] Participación {anio}: archivo no encontrado.")
            continue
        df_part = df_part[df_part["cod_ine"].isin(munis_demo)]
        panel = panel.merge(df_part, on="cod_ine", how="left")

    # ── 8. Targets electorales ─────────────────────────────────────────────
    for anio in ANOS_ELECCIONES:
        path_pct = config[str(anio)]["processed"]["votos_pct"]
        df_pct = _cargar_targets_anio(path_pct, anio)
        if df_pct is None:
            print(f"  [AVISO] Votos pct {anio}: archivo no encontrado.")
            continue
        df_pct = df_pct[df_pct["cod_ine"].isin(munis_demo)]
        panel = panel.merge(df_pct, on="cod_ine", how="left")

    # ── 9. Bloques ideológicos (columnas derivadas) ───────────────────────
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

    # ── 10. Guardar ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    guardar_dataframe_csv(panel, output_path)

    print(f"\n[OK] panel_demo.csv guardado en '{output_path}'")
    print(f"     {len(panel)} municipios × {panel.shape[1]} columnas")
    return panel
