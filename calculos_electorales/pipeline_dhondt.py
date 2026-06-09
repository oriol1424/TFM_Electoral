import pandas as pd
from typing import Dict

from modelos.prediccion import pipeline_prediccion
from calculos_electorales.dhondt import (
    escanos_por_provincia,
    dhondt_todas_provincias,
)
from calculos_electorales.resultados import (
    votos_reales_por_provincia,
    votos_predichos_por_provincia,
    tabla_escanos,
    grafico_barras_escanos,
    grafico_error_escanos,
)


def pipeline_dhondt_predicho(
    modelos: Dict,
    df_2023: pd.DataFrame,
    ruta_json_poblacion: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Pipeline completo sobre votos PREDICHOS:
      1. Predice porcentajes y convierte a votos absolutos por municipio.
      2. Agrega votos de municipal a provincial.
      3. Aplica D'Hondt provincia a provincia.
    Devuelve {slot: escanos_nacionales}.
    """
    if verbose: print("  [Predicho] Generando predicciones por municipio...")
    df_pred = pipeline_prediccion(modelos, df_2023)

    if verbose: print("  [Predicho] Agregando a nivel provincial...")
    df_prov = votos_predichos_por_provincia(df_pred)

    if verbose: print("  [Predicho] Aplicando Ley D'Hondt...")
    dict_escanos = escanos_por_provincia(ruta_json_poblacion)
    escanos = dhondt_todas_provincias(df_prov, dict_escanos)

    if verbose: print(f"  [Predicho] Total escaños repartidos: {sum(escanos.values())}")
    return escanos


def pipeline_dhondt_real(
    df_2023_completo: pd.DataFrame,
    ruta_json_poblacion: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Pipeline completo sobre votos REALES (benchmark):
      1. Calcula votos reales por slot (pct_* × votos totales) por municipio.
      2. Agrega a nivel provincial.
      3. Aplica D'Hondt provincia a provincia.
    Devuelve {slot: escanos_nacionales}.

    Nota: usa los mismos slots que el modelo, por lo que el resultado puede diferir
    ligeramente del reparto oficial (que opera con candidaturas individuales).
    """
    if verbose: print("  [Real] Calculando votos reales por slot...")
    df_prov = votos_reales_por_provincia(df_2023_completo)

    if verbose: print("  [Real] Aplicando Ley D'Hondt...")
    dict_escanos = escanos_por_provincia(ruta_json_poblacion)
    escanos = dhondt_todas_provincias(df_prov, dict_escanos)

    if verbose: print(f"  [Real] Total escaños repartidos: {sum(escanos.values())}")
    return escanos


def pipeline_comparacion_dhondt(
    modelos: Dict,
    df_2023_completo: pd.DataFrame,
    ruta_json_poblacion: str,
    etiqueta: str = '2023',
    guardar_graficos: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Orquestador completo: ejecuta D'Hondt sobre votos predichos y reales,
    imprime la tabla comparativa y genera los gráficos de escaños y error.
    Usa verbose=False para suprimir toda la salida por pantalla.

    Uso en main.ipynb:
        from calculos_electorales.pipeline_dhondt import pipeline_comparacion_dhondt
        df_escanos = pipeline_comparacion_dhondt(modelos, df_2023_completo, ruta_json_pob)

    Devuelve el DataFrame comparativo (partido, escanos_pred, escanos_real, error).
    """
    if verbose:
        print("=" * 55)
        print("  SIMULACIÓN LEY D'HONDT")
        print("=" * 55)
        print("\nPaso 1 — Escaños predichos:")

    escanos_pred = pipeline_dhondt_predicho(modelos, df_2023_completo, ruta_json_poblacion, verbose=verbose)

    if verbose: print("\nPaso 2 — Escaños reales (benchmark slots):")
    escanos_real = pipeline_dhondt_real(df_2023_completo, ruta_json_poblacion, verbose=verbose)

    df_comp = tabla_escanos(escanos_pred, escanos_real)

    if verbose:
        print("\nPaso 3 — Tabla comparativa:")
        print(df_comp[['partido', 'escanos_pred', 'escanos_real', 'error']].to_string(index=False))
        mae_escanos = df_comp['error_abs'].mean()
        print(f"\nMAE escaños (media error absoluto por partido): {mae_escanos:.2f}")
        print(f"Error total absoluto (suma): {df_comp['error_abs'].sum()} escaños")
        print("\nPaso 4 — Gráficos:")

    if verbose:
        grafico_barras_escanos(df_comp, etiqueta, guardar_graficos)
        grafico_error_escanos(df_comp, etiqueta, guardar_graficos)

    return df_comp
