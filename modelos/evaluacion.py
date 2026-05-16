import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from modelos.entrenamiento import preparar_features, TARGETS


def evaluar_modelos(
    modelos: Dict,
    df_test: pd.DataFrame,
    etiqueta: str = "test"
) -> pd.DataFrame:
    """
    Evaluates each model on df_test.
    Returns DataFrame with MAE, RMSE, R² and mean real value per party.
    """
    X_test = preparar_features(df_test)
    resultados = []

    for partido, modelo in modelos.items():
        if partido not in df_test.columns:
            continue
        y_true = df_test[partido].dropna().values
        y_pred = modelo.predict(X_test.loc[df_test[partido].notna()])

        mae  = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        resultados.append({
            'partido': partido.replace('pct_', ''),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'media_real': round(y_true.mean(), 4)
        })

    df_res = pd.DataFrame(resultados).sort_values('MAE')
    print(f"\nMETRICAS — {etiqueta}")
    print(df_res.to_string(index=False))
    return df_res


def visualizar_metricas(df_metricas: pd.DataFrame, etiqueta: str = "2023"):
    """
    Horizontal bar plots of MAE and R² per party.
    Saved to documentation/imagenes_EDA/.
    """
    os.makedirs('documentation/imagenes_EDA', exist_ok=True)
    df_s = df_metricas.sort_values('MAE', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df_s, x='MAE', y='partido', ax=ax1, palette='Blues_r')
    ax1.set_title(f'MAE por partido — {etiqueta}')
    ax1.set_xlabel('Error Absoluto Medio (puntos porcentuales)')

    sns.barplot(data=df_s, x='R2', y='partido', ax=ax2, palette='Greens_r')
    ax2.set_title(f'R2 por partido — {etiqueta}')
    ax2.set_xlabel('R2 (varianza explicada)')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'documentation/imagenes_EDA/metricas_ml_{etiqueta}.png', dpi=150)
    plt.show()


def comparar_real_predicho(modelos: Dict, df_test: pd.DataFrame, partido: str):
    """
    Scatter plot real vs predicted for one party.
    """
    X_test = preparar_features(df_test)
    y_true = df_test[partido].values
    y_pred = modelos[partido].predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', label='prediccion perfecta')
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicho')
    ax.set_title(f'Real vs Predicho — {partido}')
    ax.legend()
    plt.tight_layout()
    plt.show()
