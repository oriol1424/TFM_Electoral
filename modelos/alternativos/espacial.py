import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from modelos.entrenamiento import preparar_features, TARGETS, PARAMS_DEFAULT

RUTA_PESOS_DEFAULT = 'data_processed/geografia/mapa_adyacencia.pickle'


def cargar_pesos_espaciales(ruta: str = RUTA_PESOS_DEFAULT):
    with open(ruta, 'rb') as f:
        w = pickle.load(f)
    print(f"Pesos espaciales cargados: {len(w.neighbors)} municipios")
    return w


def preparar_features_espacial(df, w):
    """Añade columnas de lag espacial (_lag) calculando la media de vecinos geográficos."""
    X = preparar_features(df)

    if 'municipio' not in df.columns:
        return X

    mun_values = df['municipio'].values
    mun_to_pos = {m: pos for pos, m in enumerate(mun_values)}

    lag_cols = {}
    for col in X.columns:
        if col == 'provincia_enc':
            continue
        vals = X[col].values.astype(float)
        lag = np.empty(len(vals))
        for pos, mun in enumerate(mun_values):
            nbs = [mun_to_pos[n] for n in w.neighbors.get(mun, []) if n in mun_to_pos]
            lag[pos] = vals[nbs].mean() if nbs else vals[pos]
        lag_cols[f'{col}_lag'] = lag

    lag_df = pd.DataFrame(lag_cols, index=X.index)
    return pd.concat([X, lag_df], axis=1)


def entrenar_modelos_espacial(
    df_train,
    w=None,
    params=None,
    guardar=True,
    carpeta='modelos/modelos_espacial',
):
    if os.path.isdir(carpeta) and all(
        os.path.exists(os.path.join(carpeta, f"{p}.json")) for p in TARGETS
    ):
        print(f"Modelos espaciales ya existentes en '{carpeta}/' — cargando sin reentrenar.")
        return cargar_modelos_espacial(carpeta)

    if w is None:
        w = cargar_pesos_espaciales()

    X_train = preparar_features_espacial(df_train, w)
    params_uso = params or PARAMS_DEFAULT

    print("ENTRENAMIENTO ESPACIAL (XGBoost + lag espacial)")
    print(f"Municipios train  : {len(X_train)}")
    print(f"Features totales  : {X_train.shape[1]} ({len(X_train.columns) // 2} base + {len(X_train.columns) // 2} lag)")
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
        print(f"\nModelos guardados en '{carpeta}/'")

    return modelos


def cargar_modelos_espacial(
    carpeta='modelos/modelos_espacial',
):
    modelos = {}
    for partido in TARGETS:
        path = os.path.join(carpeta, f"{partido}.json")
        if os.path.exists(path):
            modelo = xgb.XGBRegressor()
            modelo.load_model(path)
            modelos[partido] = modelo

    print(f"Cargados {len(modelos)} modelos espaciales desde '{carpeta}/'")
    return modelos


def pipeline_prediccion_espacial(modelos, df, w):
    """
    Versión espacial de pipeline_prediccion: usa preparar_features_espacial
    en lugar de preparar_features. Reutiliza normalizar y conversión a votos.
    """
    from modelos.prediccion import normalizar_predicciones, predicciones_a_votos

    X = preparar_features_espacial(df, w)
    cols_id = [c for c in ['municipio', 'provincia'] if c in df.columns]
    df_pred = df[cols_id].copy()
    for partido, modelo in modelos.items():
        df_pred[partido] = np.clip(modelo.predict(X), 0, None)

    df_norm = normalizar_predicciones(df_pred)
    return predicciones_a_votos(df_norm, df)
