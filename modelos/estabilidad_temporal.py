import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from modelos.entrenamiento import FEATURES

FEATURES_NUMERICAS = [f for f in FEATURES if f != 'provincia_enc']

LABELS = {
    'log_poblacion':            'Población (log)',
    'log_densidad_poblacional': 'Densidad (log)',
    'superficie':               'Superficie',
    'indice gini':              'Índice Gini',
    'renta media persona':      'Renta media/persona',
    'ratio_sexo':               'Ratio sexo (H/M)',
    'salarios':                 'Salarios (%)',
    'pensiones':                'Pensiones (%)',
    'otros ingresos':           'Otros ingresos (%)',
    'otras prestaciones':       'Otras prestaciones (%)',
    'desempleo':                'Desempleo (%)',
}


def analizar_estabilidad_features(df_2019, df_2023):
    """Correlación de Pearson y drift de la media entre 2019 y 2023 para cada feature del modelo."""
    sufijo_19 = {f: f + '_2019' for f in FEATURES_NUMERICAS}
    sufijo_23 = {f: f + '_2023' for f in FEATURES_NUMERICAS}

    merged = pd.merge(
        df_2019[['municipio'] + FEATURES_NUMERICAS].rename(columns=sufijo_19),
        df_2023[['municipio'] + FEATURES_NUMERICAS].rename(columns=sufijo_23),
        on='municipio',
        how='inner',
    )
    print(f"Municipios en común 2019-2023: {len(merged)}")

    filas = []
    for feat in FEATURES_NUMERICAS:
        col_19 = feat + '_2019'
        col_23 = feat + '_2023'

        mask = merged[[col_19, col_23]].notna().all(axis=1)
        x = merged.loc[mask, col_19].values
        y = merged.loc[mask, col_23].values

        r, _ = stats.pearsonr(x, y)
        media_19 = float(np.mean(x))
        media_23 = float(np.mean(y))
        drift = abs(media_23 - media_19) / (abs(media_19) + 1e-10)

        filas.append({
            'feature':       feat,
            'label':         LABELS.get(feat, feat),
            'correlacion':   round(r, 4),
            'media_2019':    round(media_19, 4),
            'media_2023':    round(media_23, 4),
            'std_2019':      round(float(np.std(x)), 4),
            'std_2023':      round(float(np.std(y)), 4),
            'drift_relativo': round(drift, 4),
        })

    return pd.DataFrame(filas).sort_values('correlacion').reset_index(drop=True)


def visualizar_estabilidad_features(
    df_est,
    guardar=False,
    ruta='documentation/imagenes_EDA/estabilidad_features.png',
):
    """
    Dos paneles:
    - Izquierda: correlación de Pearson por feature (verde ≥0.97, naranja ≥0.90, rojo <0.90)
    - Derecha:   drift relativo de la media entre años (%)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def _color_corr(r):
        if r >= 0.97:
            return '#2ca02c'
        if r >= 0.90:
            return '#ff7f0e'
        return '#d62728'

    def _color_drift(d):
        if d <= 0.02:
            return '#2ca02c'
        if d <= 0.05:
            return '#ff7f0e'
        return '#d62728'

    # Panel izquierdo — correlación
    ax = axes[0]
    colores = [_color_corr(r) for r in df_est['correlacion']]
    bars = ax.barh(df_est['label'], df_est['correlacion'], color=colores)
    ax.axvline(x=0.95, color='gray', linestyle='--', linewidth=0.8, label='ρ = 0.95')
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel('Correlación de Pearson (2019 vs 2023)')
    ax.set_title('Estabilidad temporal de las features del modelo')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, df_est['correlacion']):
        ax.text(
            val - 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='right',
            fontsize=8, color='white', fontweight='bold',
        )

    # Panel derecho — drift
    ax2 = axes[1]
    colores_d = [_color_drift(d) for d in df_est['drift_relativo']]
    bars2 = ax2.barh(df_est['label'], df_est['drift_relativo'] * 100, color=colores_d)
    ax2.axvline(x=2, color='gray', linestyle='--', linewidth=0.8, label='2 % drift')
    ax2.set_xlabel('Drift relativo de la media (%)')
    ax2.set_title('Cambio de la media entre 2019 y 2023')
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, df_est['drift_relativo'] * 100):
        ax2.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', ha='left', fontsize=8,
        )

    plt.tight_layout()

    if guardar:
        import os
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {ruta}")

    plt.show()


def pipeline_estabilidad(df_2019, df_2023, guardar=False):
    """
    Orquestador: calcula y visualiza la estabilidad temporal de las features.
    Devuelve el DataFrame de resultados.
    """
    df_est = analizar_estabilidad_features(df_2019, df_2023)

    print("\nRESUMEN DE ESTABILIDAD TEMPORAL DE FEATURES")
    print("-" * 50)
    print(df_est[['label', 'correlacion', 'media_2019', 'media_2023', 'drift_relativo']].to_string(index=False))
    print()

    muy_estables = df_est[df_est['correlacion'] >= 0.97]['label'].tolist()
    moderadas    = df_est[(df_est['correlacion'] >= 0.90) & (df_est['correlacion'] < 0.97)]['label'].tolist()
    cambiadas    = df_est[df_est['correlacion'] < 0.90]['label'].tolist()

    print(f"Muy estables  (ρ ≥ 0.97): {len(muy_estables)}  → {', '.join(muy_estables) or '—'}")
    print(f"Moderadas (0.90 ≤ ρ < 0.97): {len(moderadas)} → {', '.join(moderadas) or '—'}")
    print(f"Con cambio    (ρ < 0.90):  {len(cambiadas)}  → {', '.join(cambiadas) or '—'}")

    visualizar_estabilidad_features(df_est, guardar=guardar)

    return df_est
