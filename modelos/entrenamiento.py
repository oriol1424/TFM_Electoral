import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple

TARGETS = [
    'pct_psoe', 'pct_pp', 'pct_vox', 'pct_cs', 'pct_up_sumar',
    'pct_erc', 'pct_jxcat', 'pct_cup', 'pct_pnv', 'pct_ehbildu',
    'pct_bng', 'pct_cc', 'pct_prc', 'pct_naplus', 'pct_teruel'
]

FEATURES_LEGACY = [
    'superficie', 'poblacion', 'densidad poblacional', 'indice gini',
    'renta media persona', 'salarios', 'pensiones', 'otros ingresos',
    'otras prestaciones', 'desempleo', 'edad media', 'participacion',
    'provincia_enc'
]

FEATURES = [
    'log_poblacion', 'log_densidad_poblacional', 'superficie',
    'indice gini', 'renta media persona', 'ratio_sexo',
    'salarios', 'pensiones', 'otros ingresos', 'otras prestaciones', 'desempleo',
    'provincia_enc'
]

FEATURES_PERCENTIL = ['renta media persona']

PARAMS_DEFAULT = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1
}


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes 'provincia' and returns a DataFrame with exactly FEATURES columns.
    Consistent between train and predict.
    """
    df_prep = df.copy()

    if 'provincia' in df_prep.columns:
        df_prep['provincia_enc'] = df_prep['provincia'].astype('category').cat.codes

    for col in FEATURES_PERCENTIL:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].rank(pct=True)

    cols_presentes = [c for c in FEATURES if c in df_prep.columns]
    cols_faltantes = [c for c in FEATURES if c not in df_prep.columns]
    if cols_faltantes:
        print(f"  Aviso: features no encontradas en el dataset: {cols_faltantes}")

    X = df_prep[cols_presentes]
    nulos = X.isnull().sum().sum()
    if nulos > 0:
        X = X.fillna(X.median())
    return X


def entrenar_modelos(
    df_train: pd.DataFrame,
    params: Optional[dict] = None,
    guardar: bool = True,
    carpeta: str = 'modelos/modelos_guardados'
) -> Dict[str, xgb.XGBRegressor]:
    """
    Trains one XGBoostRegressor per party (15 models).
    Saves models and feature list to carpeta if guardar=True.
    Returns dict {partido: modelo}.
    """
    X_train = preparar_features(df_train)
    features_usadas = list(X_train.columns)
    params_uso = params or PARAMS_DEFAULT

    print(f"ENTRENAMIENTO ML")
    print(f"Municipios train : {len(X_train)}")
    print(f"Las variables seleccionadas son: {features_usadas}")
    print(f"Modelos a entrenar: {len(TARGETS)}\n")

    modelos = {}
    for partido in TARGETS:
        if partido not in df_train.columns:
            print(f"  [SKIP] {partido} no encontrado en el dataset")
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
            json.dump(features_usadas, f)
        print(f"\nModelos guardados en '{carpeta}/'")

    return modelos


def cargar_modelos(
    carpeta: str = 'modelos/modelos_guardados'
) -> Tuple[Dict[str, xgb.XGBRegressor], List[str]]:
    """
    Loads saved models and feature list from carpeta.
    Returns (modelos_dict, features_list).
    """
    modelos = {}
    for partido in TARGETS:
        path = os.path.join(carpeta, f"{partido}.json")
        if os.path.exists(path):
            modelo = xgb.XGBRegressor()
            modelo.load_model(path)
            modelos[partido] = modelo

    with open(os.path.join(carpeta, 'features.json')) as f:
        features = json.load(f)

    print(f"Cargados {len(modelos)} modelos desde '{carpeta}/'")
    return modelos, features
