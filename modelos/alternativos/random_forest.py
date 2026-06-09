import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modelos.entrenamiento import preparar_features, TARGETS

PARAMS_DEFAULT_RF = {
    'n_estimators': 300,
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': 42,
}


def entrenar_modelos_rf(
    df_train,
    params=None,
    guardar=True,
    carpeta='modelos/modelos_rf',
):
    X_train = preparar_features(df_train)
    params_uso = params or PARAMS_DEFAULT_RF

    print("ENTRENAMIENTO RANDOM FOREST")
    print(f"Municipios train : {len(X_train)}")
    print(f"Features          : {list(X_train.columns)}")
    print(f"Modelos a entrenar: {len(TARGETS)}\n")

    modelos = {}
    for partido in TARGETS:
        if partido not in df_train.columns:
            print(f"  [SKIP] {partido} no encontrado en el dataset")
            continue
        y_train = df_train[partido]
        modelo = RandomForestRegressor(**params_uso)
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


def cargar_modelos_rf(
    carpeta='modelos/modelos_rf',
):
    modelos = {}
    for partido in TARGETS:
        path = os.path.join(carpeta, f"{partido}.pkl")
        if os.path.exists(path):
            modelos[partido] = joblib.load(path)

    print(f"Cargados {len(modelos)} modelos RF desde '{carpeta}/'")
    return modelos
