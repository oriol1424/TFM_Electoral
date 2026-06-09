import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

from modelos.prediccion import pipeline_prediccion
from calculos_electorales.dhondt import (
    escanos_por_provincia,
    dhondt_todas_provincias,
)
from calculos_electorales.resultados import (
    votos_reales_por_provincia,
    votos_predichos_por_provincia,
)

CARPETA_IMAGENES = 'documentation/imagenes_EDA'

COLORES_PARTIDOS = {
    'psoe':     '#E30613',
    'pp':       '#0056A2',
    'vox':      '#63BE21',
    'cs':       '#EB6109',
    'up_sumar': '#C4188B',
    'erc':      '#F2A900',
    'jxcat':    '#0F3766',
    'cup':      '#FCDB00',
    'pnv':      '#009A44',
    'ehbildu':  '#B5CF18',
    'bng':      '#5E9EC8',
    'cc':       '#FFCB00',
    'prc':      '#0064A4',
    'naplus':   '#FF6B00',
    'teruel':   '#999999',
}

NOMBRES_DISPLAY = {
    'psoe':     'PSOE',
    'pp':       'PP',
    'vox':      'VOX',
    'cs':       'Cs',
    'up_sumar': 'Sumar',
    'erc':      'ERC',
    'jxcat':    'JxCat',
    'cup':      'CUP',
    'pnv':      'PNV',
    'ehbildu':  'EH Bildu',
    'bng':      'BNG',
    'cc':       'CC',
    'prc':      'PRC',
    'naplus':   'NA+/UPN',
    'teruel':   'Teruel Existe',
}

ORDEN_POLITICO = [
    'cup', 'ehbildu', 'bng', 'erc', 'up_sumar', 'jxcat',
    'psoe', 'pnv', 'prc', 'cc', 'naplus', 'teruel', 'cs', 'pp', 'vox'
]


def _posiciones_hemiciclo(n_total, n_filas=10):
    """Devuelve array (n_total, 2) con posiciones (x, y) en arcos concéntricos."""
    radios = np.linspace(1.5, 3.0, n_filas)
    raw = radios / radios.sum() * n_total
    asientos = np.round(raw).astype(int)

    diff = n_total - asientos.sum()
    if diff != 0:
        ajuste = int(np.sign(diff))
        for i in np.argsort(np.abs(asientos - raw))[:abs(diff)]:
            asientos[i] += ajuste

    posiciones = []
    for r, n in zip(radios, asientos):
        for ang in np.linspace(np.pi, 0, n + 2)[1:-1]:
            posiciones.append((r * np.cos(ang), r * np.sin(ang)))

    return np.array(posiciones)


def _dibujar_hemiciclo(ax, escanos_dict, titulo):
    """Dibuja un único hemiciclo en el eje ax."""
    total = sum(escanos_dict.values())
    if total == 0:
        return

    pos = _posiciones_hemiciclo(total)

    angulos = np.arctan2(pos[:, 1], pos[:, 0])
    pos = pos[np.argsort(angulos)[::-1]]

    colores = []
    for partido in ORDEN_POLITICO:
        n = escanos_dict.get(partido, 0)
        colores.extend([COLORES_PARTIDOS.get(partido, '#CCCCCC')] * n)
    colores = (colores + ['#CCCCCC'] * total)[:total]

    ax.scatter(pos[:, 0], pos[:, 1],
               c=colores, s=55, zorder=3,
               linewidths=0.3, edgecolors='white')
    ax.set_aspect('equal')
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-0.4, 3.5)
    ax.axis('off')
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=8)
    ax.text(0, -0.3, f'{total} escaños', ha='center', fontsize=9, color='#555555')


def hemiciclo_escanos(escanos_pred, escanos_real, etiqueta='2023', guardar=True):
    """Dos hemiciclos lado a lado: real (izquierda) y predicho (derecha) con leyenda compartida."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    _dibujar_hemiciclo(ax1, escanos_real,  f'Congreso real — {etiqueta}')
    _dibujar_hemiciclo(ax2, escanos_pred, f'Congreso predicho — {etiqueta}')

    todos = set(escanos_pred) | set(escanos_real)
    parches = [
        mpatches.Patch(
            color=COLORES_PARTIDOS[p],
            label=(
                f"{NOMBRES_DISPLAY.get(p, p)}  "
                f"{escanos_real.get(p, 0)}r / {escanos_pred.get(p, 0)}p"
            )
        )
        for p in ORDEN_POLITICO
        if p in todos and (escanos_real.get(p, 0) + escanos_pred.get(p, 0)) > 0
    ]
    fig.legend(handles=parches, loc='lower center', ncol=5, fontsize=8,
               title='Partido  (r = real | p = predicho)', title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f'Distribución de escaños — Elecciones generales {etiqueta}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/hemiciclo_{etiqueta}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def ganador_por_provincia(df_votos_prov):
    """Devuelve {cod_provincia: slot_ganador} según el partido con más votos en cada provincia."""
    cols = [c for c in df_votos_prov.columns
            if c.startswith('votos_') and c != 'votos_otros']
    df = df_votos_prov.set_index('cod_provincia')[cols]
    return df.idxmax(axis=1).str.replace('votos_', '', regex=False).to_dict()


def mapa_ganadores_provincia(
    ganadores_pred,
    ganadores_real,
    ruta_parquet_municipios,
    etiqueta='2023',
    guardar=True,
):
    """Mapa de España coloreado por partido más votado en cada provincia: real vs predicho."""
    gdf = gpd.read_parquet(ruta_parquet_municipios)
    gdf['cod_provincia'] = gdf['municipio'].astype(str).str.zfill(5).str[:2]

    gdf['geometry'] = gdf['geometry'].buffer(0)
    gdf_prov = (
        gdf.dissolve(by='cod_provincia')
        .reset_index()[['cod_provincia', 'geometry']]
    )

    gdf_prov['ganador_real'] = gdf_prov['cod_provincia'].map(ganadores_real).fillna('otros')
    gdf_prov['ganador_pred'] = gdf_prov['cod_provincia'].map(ganadores_pred).fillna('otros')
    gdf_prov['color_real'] = gdf_prov['ganador_real'].map(COLORES_PARTIDOS).fillna('#CCCCCC')
    gdf_prov['color_pred'] = gdf_prov['ganador_pred'].map(COLORES_PARTIDOS).fillna('#CCCCCC')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    for ax, col_color, titulo in [
        (ax1, 'color_real', f'Partido más votado — Real {etiqueta}'),
        (ax2, 'color_pred', f'Partido más votado — Predicho {etiqueta}'),
    ]:
        gdf_prov.plot(color=gdf_prov[col_color], ax=ax,
                      linewidth=0.5, edgecolor='white')
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.axis('off')

    partidos_en_mapa = set(ganadores_real.values()) | set(ganadores_pred.values())
    parches = [
        mpatches.Patch(color=COLORES_PARTIDOS[p], label=NOMBRES_DISPLAY.get(p, p))
        for p in ORDEN_POLITICO if p in partidos_en_mapa
    ]
    fig.legend(handles=parches, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Partido más votado por provincia — {etiqueta}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f'{CARPETA_IMAGENES}/mapa_provincias_{etiqueta}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def mapa_error_partido(
    pred_list,
    df_real,
    ruta_parquet,
    partido='pp',
    guardar=True,
    lim=0.18,
):
    """Mapa de error medio por provincia para un partido. Azul = subestima | Rojo = sobreestima."""
    col_partido = f"pct_{partido}"

    def _error_provincia(df_pred_mun):
        df_p = df_pred_mun[["municipio", col_partido]].copy()
        df_r = df_real[["municipio", col_partido]].copy()
        df_p["cod_prov"] = df_p["municipio"].astype(str).str.zfill(5).str[:2]
        df_r["cod_prov"] = df_r["municipio"].astype(str).str.zfill(5).str[:2]
        pred_prov = df_p.groupby("cod_prov")[col_partido].mean()
        real_prov = df_r.groupby("cod_prov")[col_partido].mean()
        return (pred_prov - real_prov).rename("error")

    gdf = gpd.read_parquet(ruta_parquet)
    gdf["cod_provincia"] = gdf["municipio"].astype(str).str.zfill(5).str[:2]
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf_prov = gdf.dissolve(by="cod_provincia").reset_index()[["cod_provincia", "geometry"]]

    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (label, df_pred) in zip(axes, pred_list):
        err = _error_provincia(df_pred)
        col_err = f"err_{label}"
        gdf_prov[col_err] = gdf_prov["cod_provincia"].map(err)
        gdf_prov.plot(
            column=col_err, ax=ax,
            cmap="RdBu", vmin=-lim, vmax=lim,
            linewidth=0.5, edgecolor="white",
            legend=True,
            legend_kwds={
                "label": "Error (predicho - real)",
                "orientation": "horizontal",
                "shrink": 0.6, "pad": 0.02,
            },
        )
        ax.set_title(
            f"Error {partido.upper()} - {label}\nAzul = subestima  |  Rojo = sobreestima",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"Error de prediccion del voto {partido.upper()} por provincia - 2023",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if guardar:
        os.makedirs(CARPETA_IMAGENES, exist_ok=True)
        plt.savefig(f"{CARPETA_IMAGENES}/mapa_error_{partido}.png", dpi=150, bbox_inches="tight")
    plt.show()

    first_label, first_df = pred_list[0]
    err_0 = _error_provincia(first_df).reset_index()
    err_0.columns = ["cod_prov", "error"]
    err_0["error_abs"] = err_0["error"].abs()
    err_0 = err_0.sort_values("error_abs", ascending=False)
    print(f"TOP 10 PROVINCIAS con mayor error en {partido.upper()} ({first_label}):")
    print(err_0.head(10).round(3).to_string(index=False))


def pipeline_visualizacion(
    modelos,
    df_2023_completo,
    ruta_json_poblacion,
    ruta_parquet_municipios,
    etiqueta='2023',
    guardar=True,
    w=None,
):
    """Orquestador: genera hemiciclo y mapa provincial real vs predicho."""
    print("-" * 50)
    print("  VISUALIZACIÓN DE RESULTADOS ELECTORALES")
    print("-" * 50)

    print("\n[1/4] Prediciendo votos por municipio...")
    if w is not None:
        from modelos.alternativos.espacial import pipeline_prediccion_espacial
        df_pred = pipeline_prediccion_espacial(modelos, df_2023_completo, w)
    else:
        df_pred = pipeline_prediccion(modelos, df_2023_completo)

    print("[2/4] Agregando a nivel provincial...")
    df_prov_pred = votos_predichos_por_provincia(df_pred)
    df_prov_real = votos_reales_por_provincia(df_2023_completo)

    print("[3/4] Calculando D'Hondt y ganadores por provincia...")
    dict_escanos  = escanos_por_provincia(ruta_json_poblacion)
    escanos_pred  = dhondt_todas_provincias(df_prov_pred, dict_escanos)
    escanos_real  = dhondt_todas_provincias(df_prov_real, dict_escanos)
    ganadores_pred = ganador_por_provincia(df_prov_pred)
    ganadores_real = ganador_por_provincia(df_prov_real)

    print("[4/4] Generando gráficos...")
    esc_pred_viz = {k.replace('votos_', ''): v for k, v in escanos_pred.items()}
    esc_real_viz = {k.replace('votos_', ''): v for k, v in escanos_real.items()}
    hemiciclo_escanos(esc_pred_viz, esc_real_viz, etiqueta, guardar)
    mapa_ganadores_provincia(ganadores_pred, ganadores_real,
                             ruta_parquet_municipios, etiqueta, guardar)

    print("\nListo. Imágenes en", CARPETA_IMAGENES)
