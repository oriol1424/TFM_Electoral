import numpy as np
import pandas as pd
from typing import Dict, List
from modelos.entrenamiento import preparar_features, TARGETS


def predecir_porcentajes(
    modelos: Dict,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 1: Predicts pct for each of the 15 modeled parties.
    Negative predictions are clamped to 0.
    Returns DataFrame with municipio + provincia + 15 pct columns.
    """
    X = preparar_features(df)

    cols_id = [c for c in ['municipio', 'provincia'] if c in df.columns]
    df_pred = df[cols_id].copy()

    for partido, modelo in modelos.items():
        df_pred[partido] = np.clip(modelo.predict(X), 0, None)

    return df_pred


def normalizar_predicciones(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Calculates pct_otros = 1 - sum(15 predictions), clamped to 0.
    Normalizes so all 16 slots sum exactly to 1.0 per municipality.
    """
    df_norm = df_pred.copy()
    cols_pred = [c for c in TARGETS if c in df_norm.columns]

    suma_15 = df_norm[cols_pred].sum(axis=1)
    df_norm['pct_otros'] = (1 - suma_15).clip(lower=0)

    # Renormalize to guarantee sum = 1.0
    total = df_norm[cols_pred + ['pct_otros']].sum(axis=1)
    for col in cols_pred + ['pct_otros']:
        df_norm[col] = df_norm[col] / total

    desviacion_media = (df_norm[cols_pred + ['pct_otros']].sum(axis=1) - 1).abs().mean()
    print(f"  Desviacion media de la suma respecto a 1.0: {desviacion_media:.2e}")

    return df_norm


def predicciones_a_votos(
    df_pred_norm: pd.DataFrame,
    df_votos_reales: pd.DataFrame,
    col_votos: str = 'votos totales'
) -> pd.DataFrame:
    """
    Step 3: Multiplies predicted pct by real total votes per municipality.
    Uses real 2023 total votes — the model predicts vote SHARE, not turnout.
    Returns DataFrame with absolute vote counts per party per municipality.
    """
    votos_totales = df_votos_reales[['municipio', col_votos]].copy()
    df_votos = df_pred_norm.merge(votos_totales, on='municipio', how='left')

    cols_pct = [c for c in df_votos.columns if c.startswith('pct_')]
    for col in cols_pct:
        col_votos_partido = col.replace('pct_', 'votos_')
        df_votos[col_votos_partido] = (df_votos[col] * df_votos[col_votos]).round().astype(int)

    return df_votos


def pipeline_prediccion(
    modelos: Dict,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Full prediction pipeline for one year:
      1. Predict 15 pct per municipality (clip negatives)
      2. Calculate pct_otros as residual, normalize to sum = 1.0
      3. Convert to absolute votes using real total votes from df
    Returns DataFrame with pct_* and votos_* columns per municipality.
    """
    print("Paso 1: Prediciendo porcentajes por municipio...")
    df_pred = predecir_porcentajes(modelos, df)

    print("Paso 2: Normalizando (calculando pct_otros como residuo)...")
    df_norm = normalizar_predicciones(df_pred)

    print("Paso 3: Convirtiendo a votos absolutos con votos totales reales...")
    df_votos = predicciones_a_votos(df_norm, df)

    print(f"Prediccion completada: {len(df_votos)} municipios")
    return df_votos
