import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional

from modelos.entrenamiento import TARGETS, PARAMS_DEFAULT
from modelos.prediccion import normalizar_predicciones, predicciones_a_votos

RANGOS_RURAL      = ['<100', '101-500', '501-1000']
RANGOS_SEMIURBANO = ['1001-2000', '2001-5000', '5001-10000', '10001-20000', '20001-50000']
RANGOS_URBANO     = ['50000-100000', '100001-500000', '>500000']

FEATURES_V2 = [
    'superficie', 'poblacion', 'densidad poblacional', 'indice gini',
    'renta media persona', 'salarios', 'pensiones', 'otros ingresos',
    'otras prestaciones', 'desempleo', 'edad media', 'participacion',
    'provincia_enc', 'grupo_tamano',
]


def _col_rango(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if 'rango' in c.lower():
            return c
    return None


def asignar_grupo_numerico(df: pd.DataFrame) -> pd.Series:
    """0 = rural (<1000 hab)  |  1 = semiurbano (1000-50000)  |  2 = urbano (>50000)."""
    col = _col_rango(df)
    grupo = pd.Series(1, index=df.index, dtype=int)
    if col is not None:
        grupo[df[col].isin(RANGOS_RURAL)]  = 0
        grupo[df[col].isin(RANGOS_URBANO)] = 2
    return grupo


def preparar_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    df_prep = df.copy()
    if 'provincia' in df_prep.columns:
        df_prep['provincia_enc'] = df_prep['provincia'].astype('category').cat.codes
    df_prep['grupo_tamano'] = asignar_grupo_numerico(df_prep)
    cols = [c for c in FEATURES_V2 if c in df_prep.columns]
    missing = [c for c in FEATURES_V2 if c not in df_prep.columns]
    if missing:
        print(f"  Aviso: features no encontradas: {missing}")
    return df_prep[cols]


def entrenar_modelos_v2(
    df_train: pd.DataFrame,
    params: Optional[dict] = None,
    guardar: bool = True,
    carpeta: str = 'modelos/modelos_v2',
) -> Dict[str, xgb.XGBRegressor]:
    """
    Entrena 15 XGBoost (uno por partido) con las mismas features del baseline
    más 'grupo_tamano' (rural=0 / semiurbano=1 / urbano=2).
    Guarda en carpeta/. Devuelve {partido: modelo}.
    """
    if os.path.isdir(carpeta) and all(
        os.path.exists(os.path.join(carpeta, f"{p}.json")) for p in TARGETS
    ):
        print(f"Modelos V2 ya existentes en '{carpeta}/' — cargando sin reentrenar.")
        return cargar_modelos_v2(carpeta)

    X_train = preparar_features_v2(df_train)
    params_uso = params or PARAMS_DEFAULT

    print("ENTRENAMIENTO ML V2  (baseline + grupo_tamano)")
    print(f"Municipios train : {len(X_train)}")
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}\n")

    modelos = {}
    for partido in TARGETS:
        if partido not in df_train.columns:
            print(f"  [SKIP] {partido}")
            continue
        y_train = df_train[partido]
        modelo = xgb.XGBRegressor(**params_uso)
        modelo.fit(X_train, y_train)
        modelos[partido] = modelo
        mae = np.mean(np.abs(modelo.predict(X_train) - y_train))
        print(f"  {partido:<20} MAE train = {mae:.4f}")

    if guardar:
        os.makedirs(carpeta, exist_ok=True)
        for partido, modelo in modelos.items():
            modelo.save_model(os.path.join(carpeta, f"{partido}.json"))
        with open(os.path.join(carpeta, 'features.json'), 'w') as f:
            json.dump(list(X_train.columns), f)
        print(f"\nModelos v2 guardados en '{carpeta}/'")

    return modelos


def evaluar_modelos_v2(
    modelos: Dict[str, xgb.XGBRegressor],
    df_test: pd.DataFrame,
    etiqueta: str = "test v2",
) -> pd.DataFrame:
    X_test = preparar_features_v2(df_test)
    resultados = []

    for partido, modelo in modelos.items():
        if partido not in df_test.columns:
            continue
        mask = df_test[partido].notna()
        y_true = df_test.loc[mask, partido].values
        y_pred = modelo.predict(X_test.loc[mask])
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
            'media_real': round(y_true.mean(), 4),
        })

    df_res = pd.DataFrame(resultados).sort_values('MAE')
    print(f"\nMETRICAS — {etiqueta}")
    print(df_res.to_string(index=False))
    return df_res


def pipeline_prediccion_v2(
    modelos: Dict[str, xgb.XGBRegressor],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prediction pipeline para modelos v2. Mismo formato de salida que
    pipeline_prediccion() — compatible con D'Hondt y visualización.
    """
    print("Paso 1: Prediciendo porcentajes (V2)...")
    X = preparar_features_v2(df)
    cols_id = [c for c in ['municipio', 'provincia'] if c in df.columns]
    df_pred = df[cols_id].copy()
    for partido, modelo in modelos.items():
        df_pred[partido] = np.clip(modelo.predict(X), 0, None)

    print("Paso 2: Normalizando...")
    df_norm = normalizar_predicciones(df_pred)

    print("Paso 3: Convirtiendo a votos absolutos...")
    df_votos = predicciones_a_votos(df_norm, df)
    print(f"Prediccion V2 completada: {len(df_votos)} municipios")
    return df_votos


def cargar_modelos_v2(
    carpeta: str = 'modelos/modelos_v2',
) -> Dict[str, xgb.XGBRegressor]:
    modelos = {}
    for partido in TARGETS:
        path = os.path.join(carpeta, f"{partido}.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            modelos[partido] = m
    print(f"Cargados {len(modelos)} modelos v2 desde '{carpeta}/'")
    return modelos
