import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from modelos.entrenamiento import preparar_features, TARGETS


def analizar_shap_partido(
    modelos: Dict,
    df: pd.DataFrame,
    partido: str,
    max_display: int = 15,
    guardar: bool = True
):
    """
    SHAP summary plot for one party.
    Shows which features push the vote share up or down.
    """
    try:
        import shap
    except ImportError:
        print("Instala shap: pip install shap")
        return

    if partido not in modelos:
        print(f"Modelo '{partido}' no encontrado.")
        return

    X = preparar_features(df)
    explainer = shap.TreeExplainer(modelos[partido])
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False
    )
    plt.title(f'SHAP — {partido}')
    plt.tight_layout()

    if guardar:
        os.makedirs('documentation/imagenes_EDA', exist_ok=True)
        plt.savefig(f'documentation/imagenes_EDA/shap_{partido}.png', dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


def analizar_shap_todos(
    modelos: Dict,
    df: pd.DataFrame,
    max_display: int = 15,
    guardar: bool = True
):
    """
    Generates SHAP summary plots for all trained parties.
    """
    for partido in modelos:
        print(f"Calculando SHAP para {partido}...")
        analizar_shap_partido(modelos, df, partido, max_display=max_display, guardar=guardar)



def heatmap_direccion_shap(
    modelos: Dict,
    df: pd.DataFrame,
    guardar: bool = True,
    annot: bool = True,
) -> pd.DataFrame:
    """
    Heatmap de dirección SHAP: mean(SHAP con signo) por feature × partido.
    Rojo  = la feature aumenta el voto del partido cuando su valor es alto.
    Azul  = la feature reduce el voto del partido cuando su valor es alto.
    Normalizado por partido para que partidos pequeños (PNV, BNG) sean comparables
    con partidos grandes (PP, PSOE) en la misma escala visual.
    Devuelve el DataFrame con los valores brutos (sin normalizar).
    """
    try:
        import shap
    except ImportError:
        print("Instala shap: pip install shap")
        return pd.DataFrame()

    import seaborn as sns

    X = preparar_features(df)
    direcciones = {}

    for partido, modelo in modelos.items():
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X)
        direcciones[partido.replace('pct_', '')] = shap_values.mean(axis=0)

    df_dir = pd.DataFrame(direcciones, index=X.columns)

    df_norm = df_dir.div(df_dir.abs().max())

    df_norm = df_norm.loc[df_dir.abs().mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        df_norm,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        linewidths=0.3,
        annot=annot,
        fmt='.2f',
        annot_kws={'size': 7},
        cbar_kws={'label': 'Efecto relativo (rojo = más voto  |  azul = menos voto)'}
    )
    ax.set_title(
        'Dirección del efecto SHAP por variable y partido\n'
        'Rojo = la variable aumenta el voto · Azul = la variable lo reduce'
    )
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()

    if guardar:
        os.makedirs('documentation/imagenes_EDA', exist_ok=True)
        plt.savefig('documentation/imagenes_EDA/shap_direccion.png', dpi=150, bbox_inches='tight')
    plt.show()

    return df_dir



    os.makedirs('documentation/imagenes_EDA', exist_ok=True)
    plt.savefig('documentation/imagenes_EDA/shap_importancia_global.png', dpi=150)
    plt.show()
