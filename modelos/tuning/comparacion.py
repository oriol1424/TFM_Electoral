import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

CARPETA_IMAGENES = 'documentation/imagenes_EDA'



def tabla_comparacion_mae(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cruza MAE de baseline y tuned.
    mejora > 0 significa que tuned es mejor (MAE más bajo).
    """
    df = metricas_base[['partido', 'MAE']].merge(
        metricas_tuned[['partido', 'MAE']],
        on='partido', suffixes=('_base', '_tuned')
    )
    df['mejora_pp'] = (df['MAE_base'] - df['MAE_tuned']).round(4)
    df['mejora_pct'] = np.where(
        df['MAE_base'] > 0,
        ((df['MAE_base'] - df['MAE_tuned']) / df['MAE_base'] * 100).round(1),
        0.0
    )
    return df.sort_values('MAE_base', ascending=False).reset_index(drop=True)


def tabla_comparacion_r2(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cruza R2 de baseline y tuned.
    mejora > 0 significa que tuned es mejor (R2 más alto).
    """
    df = metricas_base[['partido', 'R2']].merge(
        metricas_tuned[['partido', 'R2']],
        on='partido', suffixes=('_base', '_tuned')
    )
    df['mejora_pp'] = (df['R2_tuned'] - df['R2_base']).round(4)
    df['mejora_pct'] = np.where(
        df['R2_base'] != 0,
        ((df['R2_tuned'] - df['R2_base']) / df['R2_base'].abs() * 100).round(1),
        0.0
    )
    return df.sort_values('R2_tuned', ascending=False).reset_index(drop=True)



def grafico_comparacion_mae(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
    etiqueta: str = '2023 (test)',
    guardar: bool = True,
) -> None:
    """
    Barras horizontales dobles: MAE baseline vs tuned por partido.
    Ordenado de mayor a menor error baseline (los peores arriba).
    """
    df = tabla_comparacion_mae(metricas_base, metricas_tuned)

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(df))
    h = 0.35

    ax.barh(y + h / 2, df['MAE_base'],  h, label='Baseline', color='#7FB3D3', alpha=0.9)
    ax.barh(y - h / 2, df['MAE_tuned'], h, label='Tuned',    color='#1A5276', alpha=0.9)

    for i, row in df.iterrows():
        delta = row['mejora_pct']
        color = '#1E8449' if delta > 0 else '#C0392B'
        simbolo = '↓' if delta > 0 else '↑'
        x_pos = max(row['MAE_base'], row['MAE_tuned']) + 0.001
        ax.text(x_pos, i, f"{simbolo}{abs(delta):.1f}%", va='center',
                fontsize=8.5, color=color, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(df['partido'])
    ax.set_xlabel('MAE (puntos porcentuales)')
    ax.set_title(f'Comparación MAE — Baseline vs Tuned\n{etiqueta}')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/comparacion_mae.png', dpi=150, bbox_inches='tight')
    plt.show()


def grafico_comparacion_r2(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
    etiqueta: str = '2023 (test)',
    guardar: bool = True,
) -> None:
    """
    Barras horizontales dobles: R2 baseline vs tuned por partido.
    Ordenado de mayor a menor R2 tuned.
    """
    df = tabla_comparacion_r2(metricas_base, metricas_tuned)

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(df))
    h = 0.35

    ax.barh(y + h / 2, df['R2_base'],  h, label='Baseline', color='#F0B27A', alpha=0.9)
    ax.barh(y - h / 2, df['R2_tuned'], h, label='Tuned',    color='#B7770D', alpha=0.9)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df['partido'])
    ax.set_xlabel('R²')
    ax.set_title(f'Comparación R² — Baseline vs Tuned\n{etiqueta}')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/comparacion_r2.png', dpi=150, bbox_inches='tight')
    plt.show()


def grafico_scatter_mejora(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
    etiqueta: str = '2023 (test)',
    guardar: bool = True,
) -> None:
    """
    Scatter MAE baseline (eje X) vs MAE tuned (eje Y).
    Puntos por debajo de la diagonal = mejora.
    """
    df = tabla_comparacion_mae(metricas_base, metricas_tuned)
    max_val = max(df['MAE_base'].max(), df['MAE_tuned'].max()) * 1.08

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.4, label='Sin cambio')

    for _, row in df.iterrows():
        color = '#1E8449' if row['mejora_pp'] > 0 else '#C0392B'
        ax.scatter(row['MAE_base'], row['MAE_tuned'], color=color, s=70, zorder=5)
        ax.annotate(row['partido'], (row['MAE_base'], row['MAE_tuned']),
                    textcoords='offset points', xytext=(5, 3), fontsize=8.5)

    verde = mpatches.Patch(color='#1E8449', label='Mejora (tuned < baseline)')
    rojo  = mpatches.Patch(color='#C0392B', label='Empeora (tuned > baseline)')
    ax.legend(handles=[verde, rojo], fontsize=9)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('MAE Baseline')
    ax.set_ylabel('MAE Tuned')
    ax.set_title(f'Baseline vs Tuned — MAE por partido\n{etiqueta} | puntos bajo la diagonal = mejora')
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/comparacion_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()



def pipeline_comparacion(
    metricas_base: pd.DataFrame,
    metricas_tuned: pd.DataFrame,
    etiqueta: str = '2023 (test)',
    guardar: bool = True,
) -> None:
    """
    Orquestador: imprime resumen de mejoras y genera los tres gráficos de comparación.
    Llamar desde main.ipynb tras evaluar_modelos() sobre ambos conjuntos de modelos.
    """
    df_mae = tabla_comparacion_mae(metricas_base, metricas_tuned)
    df_r2  = tabla_comparacion_r2(metricas_base,  metricas_tuned)

    print("-" * 50)
    print(f"  COMPARACIÓN BASELINE vs TUNED — {etiqueta}")
    print("-" * 50)

    print("\nMAE (↓ mejor):")
    print(df_mae[['partido', 'MAE_base', 'MAE_tuned', 'mejora_pp', 'mejora_pct']].to_string(index=False))

    print("\nR² (↑ mejor):")
    print(df_r2[['partido', 'R2_base', 'R2_tuned', 'mejora_pp', 'mejora_pct']].to_string(index=False))

    n_mejoran = (df_mae['mejora_pp'] > 0).sum()
    print(f"\nPartidos que mejoran MAE en test: {n_mejoran} / {len(df_mae)}")

    grafico_comparacion_mae(metricas_base, metricas_tuned, etiqueta, guardar)
    grafico_comparacion_r2(metricas_base,  metricas_tuned, etiqueta, guardar)
    grafico_scatter_mejora(metricas_base,  metricas_tuned, etiqueta, guardar)
