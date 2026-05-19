import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Tuple, Optional

from modelos.entrenamiento import preparar_features, TARGETS, cargar_modelos

PARAM_DIST = {
    'max_depth':        [4, 5, 6, 7, 8],
    'learning_rate':    [0.01, 0.03, 0.05, 0.08, 0.1],
    'n_estimators':     [300, 400, 500, 600, 700],
    'subsample':        [0.7, 0.75, 0.8, 0.85, 0.9],
    'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9],
    'min_child_weight': [3, 5, 7, 10],
    'gamma':            [0, 0.05, 0.1, 0.2],
}


def tunear_partido(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    partido: str,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[xgb.XGBRegressor, dict, float]:
    """
    Ajusta hiperparámetros para un partido con RandomizedSearchCV.
    Devuelve (mejor_modelo, mejores_params, cv_mae).
    """
    base = xgb.XGBRegressor(random_state=random_state, n_jobs=1, verbosity=0)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_mae = -search.best_score_
    print(f"  {partido:<20} CV MAE = {cv_mae:.4f}  | {search.best_params_}")
    return search.best_estimator_, search.best_params_, cv_mae


def tunear_todos_modelos(
    df_train: pd.DataFrame,
    n_iter: int = 30,
    cv: int = 5,
    guardar: bool = True,
    carpeta: str = 'modelos/modelos_tuned',
    forzar: bool = False,
) -> Dict[str, xgb.XGBRegressor]:
    """
    Orquestador: ajusta un XGBoost por partido con RandomizedSearchCV.
    Si los modelos ya existen en carpeta y forzar=False, los carga directamente.
    Guarda modelos, features y mejores_params en carpeta si guardar=True.
    """
    features_path = os.path.join(carpeta, 'features.json')
    if not forzar and os.path.exists(features_path):
        print(f"Modelos tuned encontrados en '{carpeta}/' — cargando sin re-tunear.")
        print("  (Usa forzar=True para volver a ejecutar el tuning.)")
        modelos, _ = cargar_modelos(carpeta)
        return modelos

    X_train = preparar_features(df_train)
    features_usadas = list(X_train.columns)

    print("TUNING HIPERPARÁMETROS (RandomizedSearchCV)")
    print(f"Municipios train   : {len(X_train)}")
    print(f"Iteraciones/partido: {n_iter}  |  CV folds: {cv}")
    print(f"Partidos a tunear  : {len(TARGETS)}")
    print(f"Fits totales ~      {n_iter * cv * len(TARGETS)}\n")

    modelos = {}
    mejores_params = {}

    for partido in TARGETS:
        if partido not in df_train.columns:
            print(f"  [SKIP] {partido} no encontrado en el dataset")
            continue
        y_train = df_train[partido]
        modelo, params, _ = tunear_partido(X_train, y_train, partido, n_iter, cv)
        modelos[partido] = modelo
        mejores_params[partido] = params

    if guardar:
        os.makedirs(carpeta, exist_ok=True)
        for partido, modelo in modelos.items():
            modelo.save_model(os.path.join(carpeta, f"{partido}.json"))
        with open(os.path.join(carpeta, 'mejores_params.json'), 'w') as f:
            json.dump(mejores_params, f, indent=2)
        with open(os.path.join(carpeta, 'features.json'), 'w') as f:
            json.dump(features_usadas, f)
        print(f"\nModelos tuned guardados en '{carpeta}/'")

    return modelos


def cargar_mejores_params(carpeta: str = 'modelos/modelos_tuned') -> dict:
    """Carga el JSON con los mejores hiperparámetros encontrados durante el tuning."""
    with open(os.path.join(carpeta, 'mejores_params.json')) as f:
        return json.load(f)
