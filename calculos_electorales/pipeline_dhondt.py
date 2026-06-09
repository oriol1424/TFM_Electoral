import pandas as pd

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


def pipeline_dhondt_predicho(modelos, df_2023, ruta_json_poblacion, verbose=True, w=None):
    """Pipeline sobre votos PREDICHOS: predice → agrega a provincial → D'Hondt. Devuelve {slot: escanos}."""
    if verbose: print("  [Predicho] Generando predicciones por municipio...")
    if w is not None:
        from modelos.alternativos.espacial import pipeline_prediccion_espacial
        df_pred = pipeline_prediccion_espacial(modelos, df_2023, w)
    else:
        df_pred = pipeline_prediccion(modelos, df_2023)

    if verbose: print("  [Predicho] Agregando a nivel provincial...")
    df_prov = votos_predichos_por_provincia(df_pred)

    if verbose: print("  [Predicho] Aplicando Ley D'Hondt...")
    dict_escanos = escanos_por_provincia(ruta_json_poblacion)
    escanos = dhondt_todas_provincias(df_prov, dict_escanos)

    if verbose: print(f"  [Predicho] Total escaños repartidos: {sum(escanos.values())}")
    return escanos


def pipeline_dhondt_real(df_2023_completo, ruta_json_poblacion, verbose=True):
    """Pipeline sobre votos REALES (benchmark): votos reales por slot → agrega → D'Hondt."""
    if verbose: print("  [Real] Calculando votos reales por slot...")
    df_prov = votos_reales_por_provincia(df_2023_completo)

    if verbose: print("  [Real] Aplicando Ley D'Hondt...")
    dict_escanos = escanos_por_provincia(ruta_json_poblacion)
    escanos = dhondt_todas_provincias(df_prov, dict_escanos)

    if verbose: print(f"  [Real] Total escaños repartidos: {sum(escanos.values())}")
    return escanos


def pipeline_comparacion_dhondt(
    modelos,
    df_2023_completo,
    ruta_json_poblacion,
    etiqueta='2023',
    guardar_graficos=True,
    verbose=True,
    w=None,
):
    """Orquestador: D'Hondt predicho y real, tabla comparativa y gráficos. Devuelve DataFrame comparativo."""
    if verbose:
        print("-" * 50)
        print("  SIMULACIÓN LEY D'HONDT")
        print("-" * 50)
        print("\nPaso 1 — Escaños predichos:")

    escanos_pred = pipeline_dhondt_predicho(modelos, df_2023_completo, ruta_json_poblacion, verbose=verbose, w=w)

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
