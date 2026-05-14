import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA.visuals import (
    mapear_nombres_provincias, plot_histogram, plot_boxplot,
    plot_votos_apilados_provincial, plot_votos_individuales_por_provincia
)
from EDA.funciones_generales import RANGOS_MUNICIPIO

_COLS_META = {'id_provincia', 'nombre_provincia', 'id_municipio', 'nombre_muni', 'fecha_eleccion'}

def _cols_votos(df: pd.DataFrame) -> list:
    """Devuelve las columnas de votos filtrando las columnas de metadatos."""
    return [c for c in df.columns if c.lower() not in _COLS_META]


def limpieza_columnas_votos(df: pd.DataFrame, anyo: str = "2019") -> pd.DataFrame:
    """Elimina el prefijo 'V_' y el sufijo del año de los nombres de columna."""
    sufijo = f'_{anyo}'
    def limpiar(col):
        if col.upper().startswith('V_'):
            col = col[2:]
        if col.upper().endswith(sufijo.upper()):
            col = col[:-len(sufijo)]
        return col
    return df.rename(columns={col: limpiar(col) for col in df.columns})


def participacion_electoral(df: pd.DataFrame, anyo: str = ""):
    """
    Analiza la participación y los votos en blanco.
    Muestra histogramas y boxplots por rango de municipio.
    """
    title_suffix = f" ({anyo})" if anyo else ""
    print(f"\n--- Análisis de Participación y Votos en Blanco{title_suffix} ---")

    plot_histogram(
        df, 'participacion',
        title=f"Distribución de la Participación Electoral{title_suffix}",
        xlabel="Participación (%)", ylabel="Número de Municipios",
        color="mediumseagreen"
    )

    col_rango = 'rango tamaño población'
    if col_rango in df.columns:
        df_plot = df.copy()
        present_rangos = [r for r in RANGOS_MUNICIPIO if r in df_plot[col_rango].unique()]
        df_plot[col_rango] = pd.Categorical(df_plot[col_rango], categories=present_rangos, ordered=True)

        plot_boxplot(
            df_plot, col_rango, 'participacion',
            title=f"Participación por Tamaño de Municipio{title_suffix}",
            xlabel="Rango de Población", ylabel="Participación (%)", palette="Greens"
        )

    if 'votos blancos' in df.columns and 'votos totales' in df.columns:
        df_temp = df.copy()
        df_temp['votos blancos %'] = (df_temp['votos blancos'] / df_temp['votos totales']) * 100

        plot_histogram(
            df_temp, 'votos blancos %',
            title=f"Distribución de Votos en Blanco (%){title_suffix}",
            xlabel="Votos en Blanco (%)", ylabel="Número de Municipios", color="lightgray"
        )

        if col_rango in df_temp.columns:
            df_temp[col_rango] = pd.Categorical(df_temp[col_rango], categories=present_rangos, ordered=True)
            plot_boxplot(
                df_temp, col_rango, 'votos blancos %',
                title=f"Votos en Blanco por Tamaño de Municipio{title_suffix}",
                xlabel="Rango de Población", ylabel="Votos en Blanco (%)", palette="Greys"
            )
    else:
        print("Aviso: Faltan columnas 'votos blancos' o 'votos totales' para calcular porcentajes.")


def participacion_electoral_100(df: pd.DataFrame, anyo: str = ""):
    """Analiza los votos en blanco específicamente para municipios de menos de 100 habitantes."""
    title_suffix = f" ({anyo})" if anyo else ""
    col_rango = 'rango tamaño población'

    if col_rango not in df.columns:
        print(f"Error: No se encontró la columna '{col_rango}'")
        return

    df_100 = df[df[col_rango] == "<100"].copy()
    if df_100.empty:
        print(f"Aviso: No hay municipios con menos de 100 habitantes{title_suffix}.")
        return

    if 'votos blancos' not in df_100.columns or 'votos totales' not in df_100.columns:
        print(f"Error: Faltan columnas 'votos blancos' o 'votos totales'{title_suffix}.")
        return

    df_100['votos blancos %'] = (df_100['votos blancos'] / df_100['votos totales']) * 100

    print(f"\n--- Análisis de Votos en Blanco: Municipios < 100 hab.{title_suffix} ---")
    print(f"Total de municipios analizados: {len(df_100)}")

    plot_histogram(
        df_100, 'votos blancos %',
        title=f"Distribución de Votos en Blanco en Municipios < 100 hab.{title_suffix}",
        xlabel="Votos en Blanco (%)", ylabel="Número de Municipios", color="lightgray"
    )
    plot_boxplot(
        df_100, col_rango, 'votos blancos %',
        title=f"Dispersión de Votos en Blanco en Municipios < 100 hab.{title_suffix}",
        xlabel="Rango Poblacional (< 100 hab.)", ylabel="Votos en Blanco (%)", palette="Greys"
    )


def agrupar_minorias_provincial(df_prov: pd.DataFrame, umbral: float = 0.03) -> pd.DataFrame:
    """
    Para cada provincia, agrupa los partidos que no alcancen el umbral bajo 'Otros'.
    """
    cols_votos = _cols_votos(df_prov)
    df_resultado = []

    for _, row in df_prov.iterrows():
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

        nueva_fila['Otros'] = nueva_fila.get('Otros', 0) + votos_otros
        df_resultado.append(nueva_fila)

    return pd.DataFrame(df_resultado)


def resumen_estadistico_votos(df_prov: pd.DataFrame):
    """Muestra un resumen estadístico global del recuento de votos nacional."""
    cols_votos = _cols_votos(df_prov)
    total_por_partido = df_prov[cols_votos].sum().sort_values(ascending=False)
    votos_totales_nacionales = total_por_partido.sum()

    print("RESUMEN ESTADÍSTICO NACIONAL DE CANDIDATURAS")
    print(f"Total de candidaturas analizadas: {len(cols_votos)}")

    partidos_cero = total_por_partido[total_por_partido == 0].index.tolist()
    if partidos_cero:
        print(f"\nPartidos con 0 votos ({len(partidos_cero)}): {', '.join(partidos_cero)}")
    else:
        print("\nNo hay partidos con 0 votos en el recuento.")

    print("\nTOP 5 - CANDIDATURAS MÁS VOTADAS:")
    for partido, votos in total_por_partido.head(5).items():
        print(f" - {partido:<25} | {int(votos):>10,} votos | {(votos/votos_totales_nacionales)*100:>6.2f}%")

    print("\nTOP 5 - CANDIDATURAS MENOS VOTADAS (con al menos 1 voto):")
    for partido, votos in total_por_partido[total_por_partido > 0].tail(5).sort_values().items():
        print(f" - {partido:<25} | {int(votos):>10,} votos | {(votos/votos_totales_nacionales)*100:>8.5f}%")


def analizar_umbral_votos_nacional(df_votos: pd.DataFrame, umbral: float = 0.03, anyo: str = "2019"):
    """
    Calcula cuántos partidos superan el umbral del % por provincia y resume
    cuáles son descartados globalmente por no alcanzarlo en ninguna.
    """
    df = df_votos.copy()
    col_muni = next((c for c in df.columns if c.upper() == 'ID_MUNICIPIO'), None)
    if col_muni is None:
        print("Error: No se encontró la columna ID_MUNICIPIO")
        return

    df = limpieza_columnas_votos(df, anyo)
    df[col_muni] = df[col_muni].astype(str).str.zfill(5)
    df['id_provincia_temp'] = df[col_muni].str[:2]

    cols_votos = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]
    df_prov = df.groupby('id_provincia_temp')[cols_votos].sum()
    votos_totales_prov = df_prov.sum(axis=1).replace(0, np.nan)
    df_pct_prov = df_prov.div(votos_totales_prov, axis=0)
    superan_mask = df_pct_prov >= umbral
    superan_en_alguna = superan_mask.any()
    partidos_superan = superan_en_alguna[superan_en_alguna].index.tolist()
    partidos_no_superan = superan_en_alguna[~superan_en_alguna].index.tolist()

    print(f"ANÁLISIS DE RELEVANCIA PROVINCIAL (Umbral: {umbral*100}%)")
    for id_prov in sorted(df_pct_prov.index):
        serie_prov = superan_mask.loc[id_prov]
        print(f"Provincia {id_prov}: {serie_prov.sum():>2} partidos superan umbral | {(~serie_prov).sum():>2} no lo alcanzan.")

    print("RESUMEN GLOBAL DE CANDIDATURAS")
    print(f"Total candidaturas analizadas: {len(cols_votos)}")
    print(f"Partidos RELEVANTES (pasan el {umbral*100}% en al menos UNA provincia): {len(partidos_superan)}")
    print(f"Partidos DESCARTADOS (no pasan el {umbral*100}% en NINGUNA provincia): {len(partidos_no_superan)}")

    if partidos_superan:
        print(f"\nCandidaturas que se mantienen ({len(partidos_superan)}):")
        print(f" - {', '.join(sorted(partidos_superan))}")
    if partidos_no_superan:
        muestra = sorted(partidos_no_superan)[:20]
        resto = f" ... (y {len(partidos_no_superan)-20} más)" if len(partidos_no_superan) > 20 else ""
        print(f"\nCandidaturas descartadas (ejemplos):\n - {', '.join(muestra)}{resto}")


def eda_votos_granularidad_total(df_votos_total: pd.DataFrame, anyo: str = "2019", individual: bool = True):
    """Orquestador del EDA de votos con granularidad total."""
    print(f"Iniciando EDA de Resultados Electorales ({anyo})")

    df = df_votos_total.copy()
    df.columns = [c.lower() for c in df.columns]
    df = limpieza_columnas_votos(df, anyo)

    col_muni = 'id_municipio' if 'id_municipio' in df.columns else 'cod_muni'
    if col_muni not in df.columns:
        raise KeyError("No se encontró la columna de municipio.")

    df['id_provincia'] = df[col_muni].astype(str).str.zfill(5).str[:2]

    cols_id = [col_muni, 'nombre_muni', 'nombre_provincia', 'fecha_eleccion', 'id_provincia']
    cols_votos = [c for c in df.columns if c not in cols_id]
    df_prov = df.groupby('id_provincia')[cols_votos].sum().reset_index()
    df_prov = mapear_nombres_provincias(df_prov, anyo)

    resumen_estadistico_votos(df_prov)

    df_final = agrupar_minorias_provincial(df_prov, umbral=0.03)
    plot_votos_apilados_provincial(df_final, title=f"Distribución del Peso del Voto por Provincia (Nov {anyo})")

    if individual:
        plot_votos_individuales_por_provincia(df_final)

    return df_final
