import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from modelos.entrenamiento import TARGETS, PARAMS_DEFAULT, preparar_features
from modelos.prediccion import pipeline_prediccion
from calculos_electorales.resultados import votos_predichos_por_provincia
from calculos_electorales.dhondt import dhondt_todas_provincias


def preparar_dataset_sin_cs(df_2019):
    """
    Devuelve copia del dataset 2019 con Cs absorbida por PP y PRC por PSOE.
    Simula que ambos partidos no se presentan en 2023.
    """
    df = df_2019.copy()
    df["pct_pp"]   = df["pct_pp"]   + df["pct_cs"]
    df["pct_psoe"] = df["pct_psoe"] + df["pct_prc"]
    df["pct_cs"]   = 0.0
    df["pct_prc"]  = 0.0
    return df


def entrenar_modelos_sin_cs(
    df_2019,
    params=None,
    guardar=True,
    carpeta='modelos/modelos_sin_cs',
):
    """
    Entrena modelos sobre el dataset 2019 con conocimiento previo de candidaturas:
      - Cs fusionada en PP   (colapso previsto)
      - PRC fusionada en PSOE (no se presenta en 2023)
    Excluye pct_cs y pct_prc de los targets.
    """
    targets_sin_cs = [t for t in TARGETS if t not in ("pct_cs", "pct_prc")]

    if os.path.isdir(carpeta) and all(
        os.path.exists(os.path.join(carpeta, f"{p}.json")) for p in targets_sin_cs
    ):
        print(f"Modelos sin_cs ya existentes en '{carpeta}/' — cargando sin reentrenar.")
        return cargar_modelos_sin_cs(carpeta)

    df_train = preparar_dataset_sin_cs(df_2019)
    X_train = preparar_features(df_train)
    params_uso = params or PARAMS_DEFAULT

    print("ENTRENAMIENTO — modelo con conocimiento previo de candidaturas")
    print("  Cs  fusionada en PP   (Cs -> PP)")
    print("  PRC fusionada en PSOE (PRC no se presenta -> PSOE)")
    print(f"Municipios train: {len(X_train)}")
    print()

    modelos = {}
    for partido in targets_sin_cs:
        if partido not in df_train.columns:
            continue
        y = df_train[partido]
        m = xgb.XGBRegressor(**params_uso)
        m.fit(X_train, y)
        modelos[partido] = m
        mae = np.mean(np.abs(m.predict(X_train) - y))
        print("  %-20s MAE train = %.4f" % (partido, mae))

    if guardar:
        os.makedirs(carpeta, exist_ok=True)
        for partido, modelo in modelos.items():
            modelo.save_model(os.path.join(carpeta, f"{partido}.json"))
        with open(os.path.join(carpeta, 'features.json'), 'w') as f:
            json.dump(list(X_train.columns), f)
        print(f"\nModelos sin_cs guardados en '{carpeta}/'")

    return modelos


def cargar_modelos_sin_cs(carpeta='modelos/modelos_sin_cs'):
    targets_sin_cs = [t for t in TARGETS if t not in ("pct_cs", "pct_prc")]
    modelos = {}
    for partido in targets_sin_cs:
        path = os.path.join(carpeta, f"{partido}.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            modelos[partido] = m
    print(f"Cargados {len(modelos)} modelos sin_cs desde '{carpeta}/'")
    return modelos


def pipeline_comparacion_contrafactual(
    modelos_sin_cs,
    df_2023,
    dict_esc,
    esc_base,
    esc_real,
):
    """
    Calcula D'Hondt para el escenario contrafactual (sin Cs+PRC), muestra la tabla
    comparativa y descompone el error del PP y la correccion del PSOE.
    Devuelve DataFrame con columnas: partido, real, baseline, sin_cs_prc.
    """
    df_pred_sin_cs = pipeline_prediccion(modelos_sin_cs, df_2023)
    esc_sin_cs_raw = dhondt_todas_provincias(
        votos_predichos_por_provincia(df_pred_sin_cs), dict_esc
    )
    esc_sin_cs = {k.replace("votos_", ""): v for k, v in esc_sin_cs_raw.items()}

    todos = sorted(set(esc_real) | set(esc_base) | set(esc_sin_cs))
    rows = [
        {
            "partido":    p,
            "real":       esc_real.get(p, 0),
            "baseline":   esc_base.get(p, 0),
            "sin_cs_prc": esc_sin_cs.get(p, 0),
        }
        for p in todos
    ]
    df_cmp = pd.DataFrame(rows).sort_values("real", ascending=False)

    print("ESCANOS: real vs baseline vs sin Cs+PRC")
    print(df_cmp.to_string(index=False))
    print()
    for nombre, col in [("Baseline", "baseline"), ("Sin Cs+PRC", "sin_cs_prc")]:
        mae = (df_cmp["real"] - df_cmp[col]).abs().mean()
        print("MAE escanos %s: %.2f" % (nombre, mae))

    def _get(partido, col):
        row = df_cmp.loc[df_cmp.partido == partido, col]
        return int(row.values[0]) if len(row) else 0

    pp_base     = _get("pp",   "baseline")
    pp_sin_cs   = _get("pp",   "sin_cs_prc")
    pp_real     = _get("pp",   "real")
    psoe_base   = _get("psoe", "baseline")
    psoe_sin_cs = _get("psoe", "sin_cs_prc")
    psoe_real   = _get("psoe", "real")

    print()
    print("Descomposicion error PP:")
    print("  Error baseline:            %+d escanos" % (pp_base   - pp_real))
    print("  Recuperado (Cs -> PP):     %+d escanos" % (pp_sin_cs - pp_base))
    print("  Error residual (Feijoo):   %+d escanos" % (pp_sin_cs - pp_real))
    print()
    print("Correccion PSOE (PRC -> PSOE):")
    print("  Baseline PSOE:    %d (real %d, error %+d)" % (psoe_base,   psoe_real, psoe_base   - psoe_real))
    print("  Sin Cs+PRC PSOE:  %d (real %d, error %+d)" % (psoe_sin_cs, psoe_real, psoe_sin_cs - psoe_real))

    return df_cmp
