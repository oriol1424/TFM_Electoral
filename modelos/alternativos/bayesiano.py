import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from modelos.entrenamiento import TARGETS

PARAMS_DEFAULT_BAYES = {
    'max_iter': 300,
    'tol': 1e-3,
    'alpha_1': 1e-6,
    'alpha_2': 1e-6,
    'lambda_1': 1e-6,
    'lambda_2': 1e-6,
}

PARAM_GRID_BAYES = {
    'alpha_1':  [1e-8, 1e-6, 1e-4, 1e-2],
    'lambda_1': [1e-8, 1e-6, 1e-4, 1e-2],
}

# Sin 'otros ingresos': rompe la dependencia composicional (salarios+pensiones+otros≈100%)
FEAT_CONT_BAYES = [
    'log_poblacion', 'log_densidad_poblacional', 'superficie',
    'indice gini', 'renta media persona', 'ratio_sexo',
    'salarios', 'pensiones', 'otras prestaciones', 'desempleo'
]


def preparar_features_bayesiano(df, carpeta=None, scaler=None, ohe_cols=None):
    """
    Preprocessing para BayesianRidge: OHE de provincia + StandardScaler.

    Modos:
      Fit   (scaler=None, sin transformers.pkl en carpeta): ajusta y guarda en carpeta.
      Transform (scaler dado O transformers.pkl existe en carpeta): aplica sin reajustar.

    Retorna: (X_DataFrame, scaler, ohe_cols)
    """
    trans_path = os.path.join(carpeta, 'transformers.pkl') if carpeta else None

    if scaler is None and trans_path and os.path.exists(trans_path):
        scaler, ohe_cols = joblib.load(trans_path)

    df_p = df.copy()

    if 'renta media persona' in df_p.columns:
        df_p['renta media persona'] = df_p['renta media persona'].rank(pct=True)

    cols_disp = [c for c in FEAT_CONT_BAYES if c in df_p.columns]
    cols_falt = [c for c in FEAT_CONT_BAYES if c not in df_p.columns]
    if cols_falt:
        print(f'  Aviso: features no encontradas en Bayesiano: {cols_falt}')

    X_cont = df_p[cols_disp].fillna(df_p[cols_disp].median())

    prov_dummies = pd.get_dummies(df_p['provincia'], prefix='prov').astype(float)
    if ohe_cols is not None:
        prov_dummies = prov_dummies.reindex(columns=ohe_cols, fill_value=0.0)
    else:
        ohe_cols = list(prov_dummies.columns)

    X = pd.concat([X_cont, prov_dummies], axis=1)

    if scaler is None:
        scaler = StandardScaler()
        X_vals = scaler.fit_transform(X)
        if trans_path:
            os.makedirs(carpeta, exist_ok=True)
            joblib.dump((scaler, ohe_cols), trans_path)
    else:
        X_vals = scaler.transform(X)

    return pd.DataFrame(X_vals, columns=X.columns, index=df_p.index), scaler, ohe_cols


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
    trans_path = os.path.join(carpeta, 'transformers.pkl')

    if os.path.isdir(carpeta) and all(
        os.path.exists(os.path.join(carpeta, f"{p}.pkl")) for p in TARGETS
    ) and os.path.exists(trans_path):
        print(f"Modelos Bayesianos ya existentes en '{carpeta}/' — cargando sin reentrenar.")
        return cargar_modelos_bayesiano(carpeta)

    X_train, _, _ = preparar_features_bayesiano(
        df_train, carpeta=carpeta if guardar else None
    )

    print("ENTRENAMIENTO BAYESIANO (BayesianRidge + OHE provincia + StandardScaler)")
    print(f"Municipios train : {len(X_train)}")
    print(f"Features          : {X_train.shape[1]} ({len(FEAT_CONT_BAYES)} continuas + OHE provincia)")
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


def pipeline_prediccion_bayesiano(modelos, df, carpeta='modelos/modelos_bayesiano'):
    """Pipeline de predicción para BayesianRidge con preprocessing v2 (OHE + StandardScaler)."""
    from modelos.prediccion import normalizar_predicciones, predicciones_a_votos

    X, _, _ = preparar_features_bayesiano(df, carpeta=carpeta)

    cols_id = [c for c in ['municipio', 'provincia'] if c in df.columns]
    df_pred = df[cols_id].copy()
    for partido, modelo in modelos.items():
        df_pred[partido] = np.clip(modelo.predict(X), 0, None)

    df_norm = normalizar_predicciones(df_pred)
    return predicciones_a_votos(df_norm, df)
