import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

CARPETA_IMAGENES = 'documentation/imagenes_EDA'

BLOQUES = ['derecha', 'izquierda', 'nacionalistas']

IZQ_PANEL = ['psoe', 'up_sumar']
DER_PANEL = ['pp', 'vox', 'cs']
NAC_PANEL = ['pnv', 'jxcat', 'cc', 'prc', 'teruel',
             'erc', 'cup', 'ehbildu', 'bng', 'naplus']
BLOQUE_MAP_PANEL = {'izquierda': IZQ_PANEL, 'derecha': DER_PANEL, 'nacionalistas': NAC_PANEL}

IZQ = ['pct_psoe', 'pct_up_sumar']
DER = ['pct_pp', 'pct_vox', 'pct_cs']
NAC = ['pct_pnv', 'pct_jxcat', 'pct_cc', 'pct_prc', 'pct_teruel',
       'pct_erc', 'pct_cup', 'pct_ehbildu', 'pct_bng', 'pct_naplus']
BLOQUE_MAP = {'izquierda': IZQ, 'derecha': DER, 'nacionalistas': NAC}

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

RONDAS_WF = [
    {'train': [2015],             'test': 2016},
    {'train': [2015, 2016],       'test': 2019},
    {'train': [2015, 2016, 2019], 'test': 2023},
]

FEAT_PANEL_BASE = [
    'renta_neta_persona', 'gini', 'p80p20', 'salarios', 'pensiones',
    'desempleo', 'otras_prestaciones', 'otros_ingresos',
    'log_poblacion', 'log_densidad',
]
FEAT_FIXED = ['provincia_enc']

FEAT_TOT_CAND = [
    'log_poblacion', 'log_densidad_poblacional', 'superficie',
    'indice gini', 'renta media persona', 'ratio_sexo',
    'salarios', 'pensiones', 'otros ingresos', 'otras prestaciones',
    'desempleo', 'provincia_enc',
]


def _get_X_panel(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    cols = [f'{f}_{anio}' for f in FEAT_PANEL_BASE if f'{f}_{anio}' in df.columns]
    rename = {f'{f}_{anio}': f for f in FEAT_PANEL_BASE if f'{f}_{anio}' in cols}
    return df[FEAT_FIXED + cols].copy().rename(columns=rename)


def _get_y_bloque_panel(df: pd.DataFrame, anio: int, partidos: list) -> pd.Series:
    cols = [f'pct_{p}_{anio}' for p in partidos if f'pct_{p}_{anio}' in df.columns]
    return df[cols].fillna(0).sum(axis=1)


def _suma(df: pd.DataFrame, cols: list) -> pd.Series:
    return df[[c for c in cols if c in df.columns]].fillna(0).sum(axis=1)


def walk_forward_bloques(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward sobre el panel de 1.207 municipios (sin imputación KNN).
    Tres rondas: test 2016, 2019 y 2023.
    Devuelve DataFrame con R²/MAE por bloque y año de test.
    """
    res_panel = {b: {} for b in BLOQUES}

    for bloque in BLOQUES:
        partidos = BLOQUE_MAP_PANEL[bloque]
        for ronda in RONDAS_WF:
            train_anos, test_ano = ronda['train'], ronda['test']
            frames = []
            for a in train_anos:
                Xa = _get_X_panel(panel, a)
                ya = _get_y_bloque_panel(panel, a, partidos)
                dfa = Xa.copy()
                dfa['__y__'] = ya.values
                frames.append(dfa)
            df_tr = pd.concat(frames, ignore_index=True).dropna(subset=['__y__'])
            X_te = _get_X_panel(panel, test_ano)
            y_te = _get_y_bloque_panel(panel, test_ano, partidos)
            mask = X_te.notna().all(axis=1) & y_te.notna()
            X_te, y_te = X_te[mask], y_te[mask]
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(df_tr.drop('__y__', axis=1), df_tr['__y__'])
            y_pred = model.predict(X_te)
            res_panel[bloque][test_ano] = {
                'r2':  r2_score(y_te, y_pred),
                'mae': mean_absolute_error(y_te, y_pred),
            }

    print('PASO 1 — Walk-forward panel_demo (1.207 municipios, sin imputacion KNN)')
    print('  izquierda    = PSOE + UP/Sumar')
    print('  derecha      = PP + VOX + Cs')
    print('  nacionalistas= PNV+JxCAT+CC+PRC+Teruel+ERC+CUP+EHBildu+BNG+Na+')
    print(f'\n{"Bloque":<18}', end='')
    for r in RONDAS_WF:
        print(f'  R2_{r["test"]}  MAE_{r["test"]}', end='')
    print('\n' + '-' * 72)
    for b in BLOQUES:
        print(f'{b:<18}', end='')
        for r in RONDAS_WF:
            t = r['test']
            v = res_panel[b].get(t, {})
            print(f'  {v.get("r2", float("nan")):>6.3f}  {v.get("mae", float("nan")):>7.4f}', end='')
        print()

    rows = []
    for b in BLOQUES:
        row = {'bloque': b}
        for r in RONDAS_WF:
            t = r['test']
            v = res_panel[b].get(t, {})
            row[f'r2_{t}'] = v.get('r2', float('nan'))
            row[f'mae_{t}'] = v.get('mae', float('nan'))
        rows.append(row)
    return pd.DataFrame(rows)


def muestra_total_bloques(
    df_2019: pd.DataFrame,
    df_2023: pd.DataFrame,
) -> tuple:
    """
    Entrena un modelo XGBoost por bloque sobre df_2019 y evalúa en df_2023.
    Añade columnas pct_izquierda/pct_derecha/pct_nacionalistas a ambos DataFrames.
    Devuelve (res_total, models_bloq, feat_tot).
    """
    for df_ in [df_2019, df_2023]:
        df_['pct_izquierda']     = _suma(df_, IZQ)
        df_['pct_derecha']       = _suma(df_, DER)
        df_['pct_nacionalistas'] = _suma(df_, NAC)

    cats = sorted(set(df_2019['provincia'].dropna()) | set(df_2023['provincia'].dropna()))
    cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
    df_2019['provincia_enc'] = df_2019['provincia'].astype(cat_type).cat.codes
    df_2023['provincia_enc'] = df_2023['provincia'].astype(cat_type).cat.codes
    print(f'provincia_enc: {df_2019["provincia_enc"].nunique()} provincias codificadas')

    feat_tot = [c for c in FEAT_TOT_CAND
                if c in df_2019.columns and c in df_2023.columns]
    print(f'Features disponibles ({len(feat_tot)}): {feat_tot}')

    res_total = {}
    models_bloq = {}

    for bloque in BLOQUES:
        target = f'pct_{bloque}'
        df_tr = df_2019[feat_tot + [target]].dropna()
        df_te = df_2023[feat_tot + [target]].dropna()
        X_tr, y_tr = df_tr[feat_tot], df_tr[target]
        X_te, y_te = df_te[feat_tot], df_te[target]
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        models_bloq[bloque] = {'model': model, 'X_te': X_te, 'y_te': y_te, 'y_pred': y_pred}
        res_total[bloque] = {
            'r2':  r2_score(y_te, y_pred),
            'mae': mean_absolute_error(y_te, y_pred),
            'n':   len(y_te),
        }
        print(f'{bloque:<18}  R2={res_total[bloque]["r2"]:>6.3f}  '
              f'MAE={res_total[bloque]["mae"]:.4f}  n={res_total[bloque]["n"]}')

    return res_total, models_bloq, feat_tot


def hallazgo_central(
    df_2019: pd.DataFrame,
    df_2023: pd.DataFrame,
    res_total: dict,
    feat_tot: list,
    guardar: bool = True,
) -> tuple:
    """
    Compara R² de partidos individuales vs bloques (test 2023, muestra total).
    Genera gráfico de barras comparativo.
    Devuelve (df_comparativa, r2_part) donde r2_part es {partido: r2_total}.
    """
    PARTIDOS_CMP = ['pp', 'vox', 'cs', 'psoe', 'up_sumar']
    r2_part = {}
    for p in PARTIDOS_CMP:
        target = f'pct_{p}'
        if target not in df_2019.columns or target not in df_2023.columns:
            continue
        df_tr = df_2019[feat_tot + [target]].dropna()
        df_te = df_2023[feat_tot + [target]].dropna()
        if len(df_tr) < 50 or len(df_te) < 50:
            continue
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(df_tr[feat_tot], df_tr[target])
        r2_part[p] = r2_score(df_te[target], m.predict(df_te[feat_tot]))

    comparativa = [
        ('pct_pp',            r2_part.get('pp',       float('nan')), 'partido'),
        ('pct_vox',           r2_part.get('vox',      float('nan')), 'partido'),
        ('pct_cs',            r2_part.get('cs',       float('nan')), 'partido'),
        ('pct_derecha',       res_total['derecha']['r2'],            '-> BLOQUE ***'),
        ('pct_psoe',          r2_part.get('psoe',     float('nan')), 'partido'),
        ('pct_up_sumar',      r2_part.get('up_sumar', float('nan')), 'partido'),
        ('pct_izquierda',     res_total['izquierda']['r2'],          '-> BLOQUE'),
        ('pct_nacionalistas', res_total['nacionalistas']['r2'],      '-> BLOQUE ***'),
    ]

    print('HALLAZGO CENTRAL: partido individual vs bloque total (test 2023, muestra total)')
    print('=' * 65)
    print(f'  {"Target":<22}  {"R2":>8}  Tipo')
    print(f'  {"-"*22}  {"-"*8}  {"-"*18}')
    for nombre, r2, tipo in comparativa:
        marca = '***' if 'BLOQUE' in tipo else '   '
        print(f'{marca}  {nombre:<22}  {r2:>8.3f}  {tipo}')

    fig, ax = plt.subplots(figsize=(10, 5))
    nombres = [c[0].replace('pct_', '') for c in comparativa]
    valores = [c[1] for c in comparativa]
    colores = ['steelblue' if 'BLOQUE' in c[2] else 'lightcoral' for c in comparativa]
    bars = ax.bar(range(len(nombres)), valores, color=colores, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=25, ha='right')
    ax.set_ylabel('$R^2$ test 2023')
    ax.set_title('Predicción partido individual (rojo) vs bloque total (azul)\n'
                 'train 2019 → test 2023, muestra total')
    ax.legend(handles=[
        mpatches.Patch(color='steelblue',  label='Bloque total'),
        mpatches.Patch(color='lightcoral', label='Partido individual'),
    ], fontsize=9)
    for bar, val in zip(bars, valores):
        if val == val:  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.02 if val >= 0 else val - 0.05,
                    f'{val:.2f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/comparativa_partido_vs_bloque.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

    df_cmp = pd.DataFrame(comparativa, columns=['target', 'r2', 'tipo'])
    return df_cmp, r2_part
