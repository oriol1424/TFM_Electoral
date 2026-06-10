import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

CARPETA_IMAGENES = 'documentation/imagenes_EDA'

FEAT_ECO_BASE = [
    'renta_neta_persona', 'gini', 'p80p20', 'salarios', 'pensiones',
    'desempleo', 'otras_prestaciones', 'otros_ingresos',
    'log_poblacion', 'log_densidad',
]


def correlacion_geografica(panel, guardar=True):
    """Heatmap de correlación entre % de voto por partido en 2019."""
    voto_map = {
        'PP': 'pct_pp_2019', 'VOX': 'pct_vox_2019', 'CS': 'pct_cs_2019',
        'PSOE': 'pct_psoe_2019', 'UP/Sumar': 'pct_up_sumar_2019',
    }
    voto_map = {k: v for k, v in voto_map.items() if v in panel.columns}

    df_geo = panel[list(voto_map.values())].rename(columns={v: k for k, v in voto_map.items()})
    corr_geo = df_geo.corr()

    mask_diag = np.zeros_like(corr_geo, dtype=bool)
    np.fill_diagonal(mask_diag, True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_geo, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, mask=mask_diag, annot_kws={'size': 11})
    ax.set_title('Correlación del % voto entre partidos por municipio (2019)\n'
                 '1.207 municipios con datos completos', fontsize=12)
    plt.tight_layout()
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/correlacion_perfiles_voto.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

    print('\nMatriz de correlación geográfica (2019):')
    print(corr_geo.round(3).to_string())
    return corr_geo


def correlacion_transferencias(panel, guardar=True):
    """Correlación entre cambios de voto 2019→2023; confirma transferencia Cs→PP."""
    delta_data = {}
    for party, label in [('pp', 'PP'), ('cs', 'CS'), ('vox', 'VOX'),
                         ('psoe', 'PSOE'), ('up_sumar', 'UP/Sumar')]:
        c19, c23 = f'pct_{party}_2019', f'pct_{party}_2023'
        if c19 in panel.columns and c23 in panel.columns:
            delta_data[label] = panel[c23].values - panel[c19].values

    df_delta = pd.DataFrame(delta_data, index=panel.index)
    corr_delta = df_delta.corr()

    print('Correlación entre cambios de voto 2019→2023 (Pearson, por municipio):')
    print(corr_delta.round(3).to_string())
    if 'CS' in corr_delta.columns and 'PP' in corr_delta.columns:
        print(f'\n  → Δcs  vs Δpp:        r = {corr_delta.loc["CS","PP"]:.3f}  (cuando Cs baja, PP sube)')
    if 'PSOE' in corr_delta.columns and 'UP/Sumar' in corr_delta.columns:
        print(f'  → Δpsoe vs ΔUP/Sumar: r = {corr_delta.loc["PSOE","UP/Sumar"]:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if 'CS' in df_delta.columns and 'PP' in df_delta.columns:
        mask_cs = df_delta[['CS', 'PP']].notna().all(axis=1)
        x_cs, y_pp = df_delta.loc[mask_cs, 'CS'], df_delta.loc[mask_cs, 'PP']
        ax.scatter(x_cs, y_pp, alpha=0.3, s=10, color='steelblue')
        m, b = np.polyfit(x_cs, y_pp, 1)
        xl = np.linspace(x_cs.min(), x_cs.max(), 100)
        ax.plot(xl, m * xl + b, 'r-', linewidth=2)
        ax.set_xlabel('Δ% voto CS (2019→2023)', fontsize=11)
        ax.set_ylabel('Δ% voto PP (2019→2023)', fontsize=11)
        ax.set_title(f'Transferencia Cs → PP\nr = {corr_delta.loc["CS","PP"]:.3f}', fontsize=12)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

    ax = axes[1]
    if 'PSOE' in df_delta.columns and 'UP/Sumar' in df_delta.columns:
        mask_iz = df_delta[['PSOE', 'UP/Sumar']].notna().all(axis=1)
        x_ps, y_up = df_delta.loc[mask_iz, 'PSOE'], df_delta.loc[mask_iz, 'UP/Sumar']
        ax.scatter(x_ps, y_up, alpha=0.3, s=10, color='tomato')
        m2, b2 = np.polyfit(x_ps, y_up, 1)
        xl2 = np.linspace(x_ps.min(), x_ps.max(), 100)
        ax.plot(xl2, m2 * xl2 + b2, 'r-', linewidth=2)
        ax.set_xlabel('Δ% voto PSOE (2019→2023)', fontsize=11)
        ax.set_ylabel('Δ% voto UP/Sumar (2019→2023)', fontsize=11)
        ax.set_title(f'PSOE vs UP/Sumar — movimiento conjunto\n'
                     f'r = {corr_delta.loc["PSOE","UP/Sumar"]:.3f}', fontsize=12)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/transferencia_votos.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

    return corr_delta


def perfil_socioeconomico(panel, anio=2019, guardar=True):
    """Heatmap de correlación entre variables socioeconómicas y % de voto por partido."""
    feat_eco_cols = [f'{f}_{anio}' for f in FEAT_ECO_BASE if f'{f}_{anio}' in panel.columns]
    feat_eco_labels = [f.replace(f'_{anio}', '').replace('_', ' ') for f in feat_eco_cols]

    voto_eco = {
        'PP': f'pct_pp_{anio}', 'VOX': f'pct_vox_{anio}', 'CS': f'pct_cs_{anio}',
        'PSOE': f'pct_psoe_{anio}', 'UP/Sumar': f'pct_up_sumar_{anio}',
    }
    voto_eco = {k: v for k, v in voto_eco.items() if v in panel.columns}

    cols_all = feat_eco_cols + list(voto_eco.values())
    corr_eco = panel[cols_all].corr().loc[list(voto_eco.values()), feat_eco_cols]
    corr_eco.index = list(voto_eco.keys())
    corr_eco.columns = feat_eco_labels

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(corr_eco, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.7, vmax=0.7,
                linewidths=0.5, ax=ax, annot_kws={'size': 9})
    ax.set_title(f'Perfil socioeconómico de cada partido — correlación con % voto ({anio})\n'
                 '1.207 municipios con datos completos', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/perfil_socioeconomico_partidos.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

    print('\nTop 3 correlaciones por partido:')
    for partido in corr_eco.index:
        top3 = corr_eco.loc[partido].abs().nlargest(3)
        vals = [f'{feat}={corr_eco.loc[partido, feat]:+.3f}' for feat in top3.index]
        print(f'  {partido}: {", ".join(vals)}')

    return corr_eco
