import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV

from modelos.entrenamiento import preparar_features, TARGETS

PARAMS_DEFAULT_BAYES = {
    'max_iter': 300,
    'tol': 1e-3,
    'alpha_1': 1e-6,
    'alpha_2': 1e-6,
    'lambda_1': 1e-6,
    'lambda_2': 1e-6,
}

# Grid log-escala sobre los priors más influyentes
PARAM_GRID_BAYES = {
    'alpha_1':  [1e-8, 1e-6, 1e-4, 1e-2],
    'lambda_1': [1e-8, 1e-6, 1e-4, 1e-2],
}


def _buscar_mejores_params(X_train, y_train) -> dict:
    """GridSearchCV sobre alpha_1 y lambda_1 con CV=5, scoring neg_MAE."""
    base = BayesianRidge(max_iter=300, tol=1e-3)
    gs = GridSearchCV(
        base,
        PARAM_GRID_BAYES,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        refit=False,
    )
    gs.fit(X_train, y_train)
    mejores = gs.best_params_
    # Completar con los priors no buscados (alpha_2, lambda_2 = misma escala)
    mejores.setdefault('alpha_2', mejores['alpha_1'])
    mejores.setdefault('lambda_2', mejores['lambda_1'])
    mejores['max_iter'] = 300
    mejores['tol'] = 1e-3
    return mejores


def entrenar_modelos_bayesiano(
    df_train,
    params=None,
    guardar=True,
    carpeta='modelos/modelos_bayesiano',
    buscar_params=True,
):
    if os.path.isdir(carpeta) and all(
        os.path.exists(os.path.join(carpeta, f"{p}.pkl")) for p in TARGETS
    ):
        print(f"Modelos Bayesianos ya existentes en '{carpeta}/' — cargando sin reentrenar.")
        return cargar_modelos_bayesiano(carpeta)

    X_train = preparar_features(df_train)

    print("ENTRENAMIENTO BAYESIANO (BayesianRidge)")
    print(f"Municipios train : {len(X_train)}")
    print(f"Features          : {list(X_train.columns)}")
    print(f"Modelos a entrenar: {len(TARGETS)}\n")

    modelos = {}
    for partido in TARGETS:
        if partido not in df_train.columns:
            print(f"  [SKIP] {partido} no encontrado en el dataset")
            continue
        y_train = df_train[partido]

        if params:
            params_uso = params
        elif buscar_params:
            params_uso = _buscar_mejores_params(X_train, y_train)
            print(f"  {partido:<20} mejores priors: alpha_1={params_uso['alpha_1']:.0e}  lambda_1={params_uso['lambda_1']:.0e}")
        else:
            params_uso = PARAMS_DEFAULT_BAYES

        modelo = BayesianRidge(**params_uso)
        modelo.fit(X_train, y_train)
        modelos[partido] = modelo
        mae = np.mean(np.abs(modelo.predict(X_train) - y_train))
        print(f"  {partido:<20} MAE train = {mae:.4f}")

    if guardar:
        os.makedirs(carpeta, exist_ok=True)
        for partido, modelo in modelos.items():
            joblib.dump(modelo, os.path.join(carpeta, f"{partido}.pkl"))
        print(f"\nModelos guardados en '{carpeta}/'")

    return modelos


def cargar_modelos_bayesiano(
    carpeta='modelos/modelos_bayesiano',
):
    modelos = {}
    for partido in TARGETS:
        path = os.path.join(carpeta, f"{partido}.pkl")
        if os.path.exists(path):
            modelos[partido] = joblib.load(path)

    print(f"Cargados {len(modelos)} modelos Bayesianos desde '{carpeta}/'")
    return modelos
