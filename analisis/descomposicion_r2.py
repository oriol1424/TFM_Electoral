import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

CARPETA_IMAGENES = 'documentation/imagenes_EDA'

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

IZQ = ['pct_psoe', 'pct_up_sumar']
DER = ['pct_pp', 'pct_vox', 'pct_cs']
NAC = ['pct_pnv', 'pct_jxcat', 'pct_cc', 'pct_prc', 'pct_teruel',
       'pct_erc', 'pct_cup', 'pct_ehbildu', 'pct_bng', 'pct_naplus']
BLOQUES_COLS = {'derecha': DER, 'izquierda': IZQ, 'nacionalistas': NAC}


def calcular_descomposicion(df_2019, df_2023, res_total, feat_tot, r2_part):
    """R² económico puro vs económico+provincial para partidos y bloques."""
    feat_eco = [c for c in feat_tot if c != 'provincia_enc']

    r2_eco_part = {}
    for p in ['pp', 'vox', 'cs', 'psoe', 'up_sumar']:
        target = f'pct_{p}'
        if target not in df_2019.columns:
            continue
        mask_tr = df_2019[feat_eco + [target]].notna().all(axis=1)
        mask_te = df_2023[feat_eco + [target]].notna().all(axis=1)
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(df_2019.loc[mask_tr, feat_eco], df_2019.loc[mask_tr, target])
        r2_eco_part[p] = r2_score(df_2023.loc[mask_te, target],
                                   m.predict(df_2023.loc[mask_te, feat_eco]))

    print('PARTIDOS — R² descomposición (train 2019 → test 2023, n≈7.880)')
    print(f'{"Partido":<14} {"R²_eco":>8} {"R²_total":>10} {"Δterritorial":>14} {"Residual":>10}')
    for p in ['pp', 'vox', 'cs', 'psoe', 'up_sumar']:
        r_eco = r2_eco_part.get(p, float('nan'))
        r_tot = r2_part.get(p, float('nan'))
        delta = max(0.0, r_tot - r_eco)
        resid = 1.0 - max(r_tot, 0.0)
        print(f'  pct_{p:<10} {r_eco:>8.3f} {r_tot:>10.3f} {delta:>14.3f} {resid:>10.3f}')

    r2_eco_bloque = {}
    for bloque, cols in BLOQUES_COLS.items():
        cols_ok = [c for c in cols if c in df_2019.columns]
        y_tr = df_2019[cols_ok].fillna(0).sum(axis=1)
        y_te = df_2023[cols_ok].fillna(0).sum(axis=1)
        mask_tr = df_2019[feat_eco].notna().all(axis=1) & y_tr.notna()
        mask_te = df_2023[feat_eco].notna().all(axis=1) & y_te.notna()
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(df_2019.loc[mask_tr, feat_eco], y_tr[mask_tr])
        r2_eco_bloque[bloque] = r2_score(y_te[mask_te],
                                          m.predict(df_2023.loc[mask_te, feat_eco]))

    print()
    print('BLOQUES — R² descomposición (train 2019 → test 2023, n≈7.880)')
    print(f'{"Bloque":<16} {"R²_eco":>8} {"R²_total":>10} {"Δterritorial":>14} {"Residual":>10}')
    for bloque in ['derecha', 'izquierda', 'nacionalistas']:
        r_eco = r2_eco_bloque.get(bloque, float('nan'))
        r_tot = res_total.get(bloque, {}).get('r2', float('nan'))
        delta = max(0.0, r_tot - r_eco)
        resid = 1.0 - max(r_tot, 0.0)
        print(f'  {bloque:<16} {r_eco:>8.3f} {r_tot:>10.3f} {delta:>14.3f} {resid:>10.3f}')

    return r2_eco_part, r2_eco_bloque


def _componentes(r_eco: float, r_tot: float) -> tuple:
    # Cuando r_eco < 0 las variables económicas predicen al revés;
    # todo el poder predictivo se atribuye a la provincia (terr = r_tot).
    # Así eco + terr + resid = 1 siempre y las barras no se salen de 1.
    r_tot_clamped = max(r_tot, 0.0)
    eco   = max(r_eco, 0.0)
    terr  = r_tot_clamped - eco
    resid = 1.0 - r_tot_clamped
    return eco, terr, resid


def visualizar_descomposicion(r2_eco_part, r2_part, r2_eco_bloque, res_total, guardar=True):
    """Barras apiladas: componente económico, territorial y no explicado por partido y bloque."""
    etiquetas_part = ['PP', 'VOX', 'CS', 'PSOE', 'UP/Sumar']
    keys_part      = ['pp', 'vox', 'cs', 'psoe', 'up_sumar']
    etiquetas_blq  = ['Derecha', 'Izquierda', 'Nacionalistas']
    keys_blq       = ['derecha', 'izquierda', 'nacionalistas']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, etiquetas, keys, get_eco, get_tot, titulo in [
        (axes[0], etiquetas_part, keys_part,
         lambda k: r2_eco_part.get(k, 0),
         lambda k: r2_part.get(k, 0),
         'Partidos individuales'),
        (axes[1], etiquetas_blq, keys_blq,
         lambda k: r2_eco_bloque.get(k, 0),
         lambda k: res_total.get(k, {}).get('r2', 0),
         'Bloques ideológicos'),
    ]:
        eco_vals   = [_componentes(get_eco(k), get_tot(k))[0] for k in keys]
        terr_vals  = [_componentes(get_eco(k), get_tot(k))[1] for k in keys]
        resid_vals = [_componentes(get_eco(k), get_tot(k))[2] for k in keys]

        x = np.arange(len(etiquetas))
        ax.bar(x, eco_vals,  label='Económico',    color='#2196F3', alpha=0.85)
        ax.bar(x, terr_vals, bottom=eco_vals,       label='Territorial', color='#FF9800', alpha=0.85)
        ax.bar(x, resid_vals,
               bottom=[e + t for e, t in zip(eco_vals, terr_vals)],
               label='No explicado', color='#BDBDBD', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Proporción de la varianza (R² = 1)')
        ax.set_title(titulo, fontsize=12)
        ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
        ax.legend(fontsize=9, loc='upper right')

        for i, (e, t, r) in enumerate(zip(eco_vals, terr_vals, resid_vals)):
            if e > 0.03:
                ax.text(x[i], e / 2,         f'{e:.2f}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            if t > 0.03:
                ax.text(x[i], e + t / 2,     f'{t:.2f}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            if r > 0.03:
                ax.text(x[i], e + t + r / 2, f'{r:.2f}', ha='center', va='center',
                        fontsize=8, color='#333')

    plt.suptitle('Descomposición de la varianza del voto\nEconómico | Territorial | No explicado',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/capacidad_explicativa.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
