import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from modelos.entrenamiento import TARGETS, PARAMS_DEFAULT, preparar_features
from modelos.prediccion import normalizar_predicciones, predicciones_a_votos

GRUPOS = {
    'rural':      ['<100', '101-500', '501-1000'],
    'semiurbano': ['1001-2000', '2001-5000', '5001-10000', '10001-20000', '20001-50000'],
    'urbano':     ['50000-100000', '100001-500000', '>500000'],
}
ETIQUETAS_GRUPOS = {
    'rural':      '<1000 hab',
    'semiurbano': '1000-50000 hab',
    'urbano':     '>50000 hab',
}


def _col_rango(df):
    for c in df.columns:
        if 'rango' in c.lower():
            return c
    return None


def asignar_grupo(df):
    """Devuelve Series con 'rural' / 'semiurbano' / 'urbano' por municipio."""
    col = _col_rango(df)
    grupo = pd.Series('semiurbano', index=df.index)
    if col is not None:
        for nombre, rangos in GRUPOS.items():
            grupo[df[col].isin(rangos)] = nombre
    return grupo


def entrenar_modelos_subgrupos(
    df_train,
    params=None,
    guardar=True,
    carpeta='modelos/modelos_subgrupos',
    min_muestras=20,
):
    """
    Entrena un juego de 15 modelos por cada grupo de tamaño.
    Devuelve {grupo: {partido: modelo}}.

    min_muestras: mínimo de municipios con voto>0 para entrenar un partido
                  en un grupo (evita modelos con datos insuficientes).
    Partidos que no alcanzan el umbral en un grupo quedan sin modelo —
    el predictor usará el modelo baseline como fallback.
    """
    df_train = df_train.copy()
    df_train['_grupo'] = asignar_grupo(df_train)
    params_uso = params or PARAMS_DEFAULT
    modelos_sub = {}

    for nombre in GRUPOS:
        df_g = df_train[df_train['_grupo'] == nombre].drop(columns='_grupo')
        print(f"\n== Grupo '{nombre}'  ({ETIQUETAS_GRUPOS[nombre]})  —  {len(df_g)} municipios ==")
        X_g = preparar_features(df_g)
        modelos_g = {}

        for partido in TARGETS:
            if partido not in df_g.columns:
                continue
            y_g = df_g[partido].dropna()
            n_nonzero = (y_g > 0).sum()
            if n_nonzero < min_muestras:
                print(f"  [SKIP] {partido:<20} — solo {n_nonzero} municipios con voto>0")
                continue
            X_p = X_g.loc[y_g.index]
            modelo = xgb.XGBRegressor(**params_uso)
            modelo.fit(X_p, y_g)
            modelos_g[partido] = modelo
            mae = np.mean(np.abs(modelo.predict(X_p) - y_g))
            print(f"  {partido:<20} MAE train = {mae:.4f}  (n={len(y_g)})")

        modelos_sub[nombre] = modelos_g

        if guardar:
            carpeta_g = os.path.join(carpeta, nombre)
            os.makedirs(carpeta_g, exist_ok=True)
            for partido, modelo in modelos_g.items():
                modelo.save_model(os.path.join(carpeta_g, f"{partido}.json"))
            with open(os.path.join(carpeta_g, 'features.json'), 'w') as f:
                json.dump(list(X_g.columns), f)

    print(f"\nModelos subgrupos guardados en '{carpeta}/'")
    return modelos_sub


def cargar_modelos_subgrupos(
    carpeta='modelos/modelos_subgrupos',
):
    modelos_sub = {}
    for nombre in GRUPOS:
        carpeta_g = os.path.join(carpeta, nombre)
        if not os.path.exists(carpeta_g):
            continue
        modelos_g = {}
        for partido in TARGETS:
            path = os.path.join(carpeta_g, f"{partido}.json")
            if os.path.exists(path):
                m = xgb.XGBRegressor()
                m.load_model(path)
                modelos_g[partido] = m
        modelos_sub[nombre] = modelos_g
        print(f"  {nombre}: {len(modelos_g)} modelos cargados")
    return modelos_sub


def pipeline_prediccion_subgrupos(
    modelos_sub,
    df,
    modelos_fallback=None,
):
    """
    Predice enrutando cada municipio a su modelo de subgrupo.
    Para partidos sin modelo en un subgrupo usa modelos_fallback (baseline).
    Formato de salida idéntico a pipeline_prediccion() — compatible con D'Hondt.
    """
    df_work = df.copy()
    df_work['_grupo'] = asignar_grupo(df_work)
    resultados = []

    for nombre in GRUPOS:
        df_g = df_work[df_work['_grupo'] == nombre].drop(columns='_grupo')
        if df_g.empty:
            continue
        modelos_g = modelos_sub.get(nombre, {})
        X_g = preparar_features(df_g)
        cols_id = [c for c in ['municipio', 'provincia'] if c in df_g.columns]
        df_pred = df_g[cols_id].copy()

        for partido in TARGETS:
            if partido in modelos_g:
                df_pred[partido] = np.clip(modelos_g[partido].predict(X_g), 0, None)
            elif modelos_fallback and partido in modelos_fallback:
                X_fb = preparar_features(df_g)
                df_pred[partido] = np.clip(modelos_fallback[partido].predict(X_fb), 0, None)
            else:
                df_pred[partido] = 0.0

        resultados.append(df_pred)

    df_pred_all = pd.concat(resultados).sort_index()
    print("Normalizando predicciones subgrupos...")
    df_norm = normalizar_predicciones(df_pred_all)
    df_votos = predicciones_a_votos(df_norm, df_work)
    print(f"Prediccion subgrupos completada: {len(df_votos)} municipios")
    return df_votos


def evaluar_modelos_subgrupos(
    modelos_sub,
    df_test,
    modelos_fallback=None,
    etiqueta="test subgrupos",
):
    """
    Evalúa los modelos de subgrupo enrutando cada municipio a su grupo.
    Devuelve DataFrame en el mismo formato que evaluar_modelos().
    """
    df_test = df_test.copy()
    df_test['_grupo'] = asignar_grupo(df_test)

    y_true_all = {p: [] for p in TARGETS}
    y_pred_all = {p: [] for p in TARGETS}

    for nombre in GRUPOS:
        df_g = df_test[df_test['_grupo'] == nombre].drop(columns='_grupo')
        if df_g.empty:
            continue
        modelos_g = modelos_sub.get(nombre, {})
        X_g = preparar_features(df_g)

        for partido in TARGETS:
            if partido not in df_g.columns:
                continue
            mask = df_g[partido].notna()
            y_true = df_g.loc[mask, partido].values
            if len(y_true) == 0:
                continue
            X_p = X_g.loc[mask]
            if partido in modelos_g:
                y_pred = np.clip(modelos_g[partido].predict(X_p), 0, None)
            elif modelos_fallback and partido in modelos_fallback:
                y_pred = np.clip(modelos_fallback[partido].predict(X_p), 0, None)
            else:
                continue
            y_true_all[partido].extend(y_true.tolist())
            y_pred_all[partido].extend(y_pred.tolist())

    resultados = []
    for partido in TARGETS:
        yt = np.array(y_true_all[partido])
        yp = np.array(y_pred_all[partido])
        if len(yt) == 0:
            continue
        mae  = np.mean(np.abs(yp - yt))
        rmse = np.sqrt(np.mean((yp - yt) ** 2))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        resultados.append({
            'partido': partido.replace('pct_', ''),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'media_real': round(yt.mean(), 4),
        })

    df_res = pd.DataFrame(resultados).sort_values('MAE')
    print(f"\nMETRICAS — {etiqueta}")
    print(df_res.to_string(index=False))
    return df_res
