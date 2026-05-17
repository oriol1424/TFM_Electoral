import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional

from calculos_electorales.dhondt import agregar_votos_a_provincia, SLOTS_DHONDT

CARPETA_IMAGENES = 'documentation/imagenes_EDA'


# ── Preparación de votos ──────────────────────────────────────────────────────

def votos_reales_por_provincia(df_datos_unificados: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los votos reales por slot a nivel provincial desde datos_unificados.
    Multiplica pct_* (reales) por votos totales (reales) y agrega por provincia.
    La columna pct_otros se incluye como votos_otros (umbral D'Hondt) pero no compite.
    """
    df = df_datos_unificados.copy()
    cols_pct = [c for c in df.columns if c.startswith('pct_')]

    for col_pct in cols_pct:
        col_votos = col_pct.replace('pct_', 'votos_')
        df[col_votos] = (df[col_pct] * df['votos totales']).round().astype(int)

    cols_votos = [c.replace('pct_', 'votos_') for c in cols_pct]
    return agregar_votos_a_provincia(df, cols_votos)


def votos_predichos_por_provincia(df_prediccion: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los votos predichos (columnas votos_* del pipeline_prediccion) a nivel provincial.
    """
    cols_votos = [c for c in df_prediccion.columns if c.startswith('votos_')]
    return agregar_votos_a_provincia(df_prediccion, cols_votos)


# ── Tabla comparativa ─────────────────────────────────────────────────────────

def tabla_escanos(
    escanos_pred: Dict[str, int],
    escanos_real: Dict[str, int],
) -> pd.DataFrame:
    """
    Construye la tabla comparativa predichos vs reales.
    Incluye todos los slots modelados aunque tengan 0 escaños.
    """
    slots = SLOTS_DHONDT
    filas = []
    for slot in slots:
        nombre = slot.replace('votos_', '')
        pred = escanos_pred.get(slot, 0)
        real = escanos_real.get(slot, 0)
        filas.append({
            'partido': nombre,
            'escanos_pred': pred,
            'escanos_real': real,
            'error': pred - real,
        })

    df = pd.DataFrame(filas).sort_values('escanos_real', ascending=False)
    df['error_abs'] = df['error'].abs()
    return df.reset_index(drop=True)


# ── Gráficos ──────────────────────────────────────────────────────────────────

def grafico_barras_escanos(
    df_comp: pd.DataFrame,
    etiqueta: str = '2023',
    guardar: bool = True,
) -> None:
    """
    Barras horizontales dobles: escaños predichos vs reales por partido.
    Ordenado de mayor a menor escaños reales.
    """
    df = df_comp.sort_values('escanos_real', ascending=True)
    y = np.arange(len(df))
    h = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + h / 2, df['escanos_real'], h, label='Real',     color='#2E86AB', alpha=0.9)
    ax.barh(y - h / 2, df['escanos_pred'], h, label='Predicho', color='#F0B27A', alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df['partido'].str.upper())
    ax.set_xlabel('Escaños')
    ax.set_title(f'Predicción de escaños vs resultado real — {etiqueta}\n(Ley D\'Hondt aplicada a slots del modelo)')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/dhondt_escanos_{etiqueta}.png', dpi=150, bbox_inches='tight')
    plt.show()


def grafico_error_escanos(
    df_comp: pd.DataFrame,
    etiqueta: str = '2023',
    guardar: bool = True,
) -> None:
    """
    Barras del error por partido (predicho - real).
    Verde = sobreestimación, Rojo = subestimación.
    """
    df = df_comp[df_comp['escanos_real'] + df_comp['escanos_pred'] > 0].copy()
    df = df.sort_values('error')

    colors = ['#1E8449' if e >= 0 else '#C0392B' for e in df['error']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df['error'], color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=1)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['partido'].str.upper())
    ax.set_xlabel('Error (predicho − real) en escaños')
    ax.set_title(f'Error de predicción de escaños por partido — {etiqueta}')

    verde = mpatches.Patch(color='#1E8449', label='Sobreestimación')
    rojo  = mpatches.Patch(color='#C0392B', label='Subestimación')
    ax.legend(handles=[verde, rojo])

    plt.tight_layout()
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/dhondt_error_{etiqueta}.png', dpi=150, bbox_inches='tight')
    plt.show()
