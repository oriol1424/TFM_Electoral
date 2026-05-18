import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os

GRUPO_A = [
    "indice gini",
    "P80P20",
    "Renta media hogar",
    "renta media unidad consumo",
    "renta media persona",
]
GRUPO_B = [
    "salarios",
    "pensiones",
    "otros ingresos",
    "otras prestaciones",
    "desempleo",
]
FEATS_DISTANCIA = ["latitud", "longitud", "poblacion", "densidad poblacional"]

KS_DEFAULT = [3, 5, 7, 9, 11, 15, 20, 30]


def _predicciones_knn(ref_scaled, ref_vals, test_scaled, k):
    n_test = test_scaled.shape[0]
    k_real = min(k, ref_scaled.shape[0])
    batch  = 500
    preds  = np.empty((n_test, ref_vals.shape[1]))

    for bs in range(0, n_test, batch):
        be   = min(bs + batch, n_test)
        diff = test_scaled[bs:be, np.newaxis, :] - ref_scaled[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        idx  = np.argpartition(dist, k_real, axis=1)[:, :k_real]
        for b_i in range(be - bs):
            top = idx[b_i]
            d   = dist[b_i, top]
            w   = 1.0 / (d + 1e-10)
            preds[bs + b_i] = np.average(ref_vals[top], axis=0, weights=w)

    return preds


def calcular_rmse_por_k(
    df: pd.DataFrame,
    ks: list = KS_DEFAULT,
    pct_mascara: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enmascara pct_mascara de los municipios con datos completos y mide
    el RMSE normalizado por rango para cada k en ks.
    """
    cols_todas = GRUPO_A + GRUPO_B
    df_known   = df[df[cols_todas].notna().all(axis=1)].copy().reset_index(drop=True)
    rangos     = {col: df_known[col].max() - df_known[col].min() for col in cols_todas}

    rng     = np.random.default_rng(seed)
    n_mask  = int(len(df_known) * pct_mascara)
    idx_tst = rng.choice(len(df_known), n_mask, replace=False)
    mask    = np.zeros(len(df_known), dtype=bool)
    mask[idx_tst] = True

    df_test = df_known[mask].copy()
    df_ref  = df_known[~mask].copy()

    mu    = df_ref[FEATS_DISTANCIA].mean()
    sigma = df_ref[FEATS_DISTANCIA].std().replace(0, 1)
    ref_s = ((df_ref[FEATS_DISTANCIA] - mu) / sigma).values
    tst_s = ((df_test[FEATS_DISTANCIA] - mu) / sigma).values
    ref_v = df_ref[cols_todas].values
    tru_v = df_test[cols_todas].values

    resultados = {}
    for k in ks:
        preds = _predicciones_knn(ref_s, ref_v, tst_s, k)
        fila  = {}
        for j, col in enumerate(cols_todas):
            rmse     = np.sqrt(np.mean((tru_v[:, j] - preds[:, j]) ** 2))
            fila[col] = rmse / rangos[col]
        fila["mean_A"]     = np.mean([fila[c] for c in GRUPO_A])
        fila["mean_B"]     = np.mean([fila[c] for c in GRUPO_B])
        fila["mean_total"] = np.mean([fila[c] for c in cols_todas])
        resultados[k] = fila

    df_res = pd.DataFrame(resultados).T
    df_res.index.name = "k"
    return df_res


def detectar_codo_geometrico(
    df_rmse: pd.DataFrame,
    col: str = "mean_total",
) -> tuple:
    """
    Método geométrico del codo (Kneedle).

    Procedimiento:
      1. Normaliza los valores de k y RMSE a [0, 1] de forma independiente.
      2. Traza la recta que une el primer punto (k mínimo) con el último (k máximo).
      3. Calcula la distancia perpendicular de cada punto a esa recta.
      4. El k con máxima distancia perpendicular es el codo matemático.

    No requiere ningún umbral arbitrario.

    Returns
    -------
    k_optimo   : int  — k con mayor distancia perpendicular
    distancias : pd.Series — distancia perpendicular de cada k a la recta
    k_norm     : np.ndarray — valores k normalizados (para graficar)
    v_norm     : np.ndarray — valores RMSE normalizados (para graficar)
    """
    ks   = np.array(df_rmse.index.tolist(), dtype=float)
    vals = df_rmse[col].values.astype(float)

    k_norm = (ks   - ks[0])        / (ks[-1]       - ks[0])
    v_norm = (vals - vals.min())    / (vals.max()   - vals.min())

    x1, y1 = k_norm[0],  v_norm[0]
    x2, y2 = k_norm[-1], v_norm[-1]

    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1

    dists = np.abs(A * k_norm + B * v_norm + C) / np.sqrt(A ** 2 + B ** 2)

    k_optimo  = int(ks[np.argmax(dists)])
    distancias = pd.Series(dists, index=df_rmse.index.astype(int), name="dist_perpendicular")

    return k_optimo, distancias, k_norm, v_norm


def imprimir_tabla_k(
    df_rmse: pd.DataFrame,
    k_optimo: int,
    distancias: pd.Series,
) -> None:
    ks   = df_rmse.index.tolist()
    vals = df_rmse["mean_total"].values

    print()
    print("RMSE NORMALIZADO POR K — validacion cruzada (20% enmascarado, seed=42)")
    print("%4s | %16s | %20s | %s" % ("k", "RMSE_norm_medio", "Dist. perpendicular", ""))
    max_dist = distancias.max()
    for i, k in enumerate(ks):
        dist = distancias[k]
        marca = " <- CODO GEOMETRICO" if dist == max_dist else ""
        print("%4d | %16.5f | %20.4f |%s" % (k, vals[i], dist, marca))

    print()
    print("RMSE NORMALIZADO POR VARIABLE (* = minimo por variable):")
    cols = GRUPO_A + GRUPO_B
    header = "%-32s" % "Variable"
    for k in ks:
        header += "  k=%-3d" % k
    print(header)
    print("-" * (32 + 7 * len(ks)))
    for col in cols:
        grupo    = "A" if col in GRUPO_A else "B"
        vals_col = df_rmse[col].values
        min_val  = vals_col.min()
        fila     = "[%s] %-29s" % (grupo, col[:29])
        for v in vals_col:
            fila += "  %4.3f%s" % (v, "*" if v == min_val else " ")
        print(fila)

    print()
    print("K RECOMENDADO: %d" % k_optimo)
    print("Metodo: distancia perpendicular maxima al segmento que une")
    print("        el primer y ultimo punto de la curva RMSE normalizada.")
    print("Ningún parametro arbitrario. La geometría de los datos elige k.")


def plot_codo_k(
    df_rmse: pd.DataFrame,
    k_optimo: int,
    distancias: pd.Series,
    k_norm: np.ndarray,
    v_norm: np.ndarray,
    anyo: int,
    output_dir: str = "documentation/imagenes_EDA",
) -> None:
    """
    Tres paneles:
      1. Curvas RMSE por variable + medias de grupo + marcador del codo.
      2. Curva normalizada con la recta extremo-a-extremo y la distancia perpendicular máxima.
      3. Distancias perpendiculares por k (barra resaltada = codo).
    """
    ks     = df_rmse.index.tolist()
    ks_arr = np.array(ks, dtype=float)

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    colores_a = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"]
    colores_b = ["#FF7043", "#FF8A65", "#FFAB91", "#EF9A9A", "#FFCCBC"]

    ax1 = fig.add_subplot(gs[0])

    for col, c in zip(GRUPO_A, colores_a):
        ax1.plot(ks, df_rmse[col].values, color=c, lw=1.2, ls="--", alpha=0.65)
    for col, c in zip(GRUPO_B, colores_b):
        ax1.plot(ks, df_rmse[col].values, color=c, lw=1.2, ls="-.", alpha=0.65)

    ax1.plot(ks, df_rmse["mean_A"].values,     color="#1565C0", lw=2.5,
             label="Media Grupo A (renta/desigualdad)")
    ax1.plot(ks, df_rmse["mean_B"].values,     color="#BF360C", lw=2.5,
             label="Media Grupo B (fuentes ingreso)")
    ax1.plot(ks, df_rmse["mean_total"].values, color="#212121", lw=3,
             label="Media total")

    ax1.axvline(x=k_optimo, color="#388E3C", lw=2, ls="--",
                label=f"k = {k_optimo} (codo geométrico)")

    ax1.set_xlabel("k (número de vecinos)", fontsize=11)
    ax1.set_ylabel("RMSE normalizado", fontsize=11)
    ax1.set_title("Curvas RMSE por variable", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])

    ax2.plot([k_norm[0], k_norm[-1]], [v_norm[0], v_norm[-1]],
             color="#9E9E9E", lw=1.5, ls="--", zorder=1, label="Recta extremo-extremo")

    ax2.plot(k_norm, v_norm, color="#212121", lw=2.5, marker="o",
             markersize=6, zorder=2, label="RMSE normalizado")

    idx_opt = list(df_rmse.index).index(k_optimo)
    xo, yo  = k_norm[idx_opt], v_norm[idx_opt]

    x1n, y1n = k_norm[0],  v_norm[0]
    x2n, y2n = k_norm[-1], v_norm[-1]
    dx, dy   = x2n - x1n, y2n - y1n
    t        = ((xo - x1n) * dx + (yo - y1n) * dy) / (dx**2 + dy**2)
    xp, yp   = x1n + t * dx, y1n + t * dy

    ax2.annotate(
        "", xy=(xp, yp), xytext=(xo, yo),
        arrowprops=dict(arrowstyle="<->", color="#E53935", lw=2),
    )
    ax2.plot(xo, yo, "o", color="#388E3C", markersize=10, zorder=5,
             label=f"Codo k={k_optimo}\n(dist={distancias[k_optimo]:.4f})")
    ax2.plot(xp, yp, "x", color="#E53935", markersize=8, zorder=5)

    ax2.set_xlabel("k normalizado [0, 1]", fontsize=11)
    ax2.set_ylabel("RMSE normalizado [0, 1]", fontsize=11)
    ax2.set_title("Método geométrico del codo\n(espacio normalizado)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])

    colores_barras = ["#388E3C" if k == k_optimo else "#90A4AE" for k in ks]
    bars = ax3.bar(range(len(ks)), distancias.values,
                   color=colores_barras, edgecolor="white", width=0.6)

    for bar, k, val in zip(bars, ks, distancias.values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold" if k == k_optimo else "normal")

    ax3.set_xticks(range(len(ks)))
    ax3.set_xticklabels([f"k={k}" for k in ks], fontsize=10)
    ax3.set_xlabel("k", fontsize=11)
    ax3.set_ylabel("Distancia perpendicular", fontsize=11)
    ax3.set_title("Distancia perpendicular por k\n(máximo = codo)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Selección de k — KNN Espacial ({anyo})   |   "
        f"Validación cruzada 20% de municipios completos",
        fontsize=12, fontweight="bold", y=1.02,
    )

    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, f"knn_seleccion_k_{anyo}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafico guardado en: {ruta}")

_LABELS_A = {
    "indice gini":              "Gini",
    "P80P20":                   "P80/P20",
    "Renta media hogar":        "R. Hogar",
    "renta media unidad consumo": "R. UC",
    "renta media persona":      "R. Persona",
}
_LABELS_B = {
    "salarios":           "Salarios",
    "pensiones":          "Pensiones",
    "otros ingresos":     "Otros ing.",
    "otras prestaciones": "Otras prest.",
    "desempleo":          "Desempleo",
}


def _corr_annotada(ax, corr_df, labels, titulo, umbral_negrita=0.6):
    """
    Heatmap de correlación con matplotlib puro (sin seaborn).
    Triángulo inferior + diagonal. Valores fuertes en negrita.
    """
    n      = len(corr_df)
    data   = corr_df.values.astype(float)
    cmap   = plt.cm.RdBu_r
    norm   = mcolors.Normalize(vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            if j > i:
                ax.add_patch(plt.Rectangle(
                    (j, n - 1 - i), 1, 1,
                    facecolor="white", edgecolor="#cccccc", lw=0.5,
                ))
                continue
            val   = data[i, j]
            color = cmap(norm(val))
            ax.add_patch(plt.Rectangle(
                (j, n - 1 - i), 1, 1,
                facecolor=color, edgecolor="white", lw=0.8,
            ))
            lum   = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            tc    = "white" if lum < 0.45 else "black"
            fw    = "bold" if (i != j and abs(val) >= umbral_negrita) else "normal"
            fs    = 11 if fw == "bold" else 9
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=fs, fontweight=fw, color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([x + 0.5 for x in range(n)])
    ax.set_yticks([y + 0.5 for y in range(n)])
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels[::-1], rotation=0, fontsize=10)
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Pearson r", fontsize=9)


def plot_correlaciones_grupos(
    df: pd.DataFrame,
    anyo: int,
    output_dir: str = "documentation/imagenes_EDA",
) -> None:
    """
    Genera dos matrices de correlación de Pearson:
      - Izquierda : Grupo A (Gini, P80P20, rentas)
      - Derecha   : Grupo B (fuentes de ingresos — datos composicionales)

    Solo usa municipios con datos completos en ambos grupos.
    Las correlaciones fuertes (|r| >= 0.6) aparecen en negrita.

    Justificación metodológica que aportan:
      · Grupo A: subgrupo renta (r > 0.84) y subgrupo desigualdad (r = 0.79) →
        KNN multivariante aprovecha estas correlaciones al calcular distancias.
      · Grupo B: complementariedad salarios-pensiones (r = -0.84) → la
        renormalización composicional preserva esta restricción estructural.
    """
    cols_todas = GRUPO_A + GRUPO_B
    df_comp    = df[df[cols_todas].notna().all(axis=1)].copy()
    n_comp     = len(df_comp)

    corr_a = df_comp[GRUPO_A].corr()
    corr_b = df_comp[GRUPO_B].corr()

    labels_a = [_LABELS_A[c] for c in GRUPO_A]
    labels_b = [_LABELS_B[c] for c in GRUPO_B]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.45)

    _corr_annotada(
        axes[0], corr_a, labels_a,
        titulo=f"Grupo A — Renta y Desigualdad ({anyo})",
    )
    _corr_annotada(
        axes[1], corr_b, labels_b,
        titulo=f"Grupo B — Fuentes de Ingresos ({anyo})\n(datos composicionales)",
    )

    leyenda = (
        f"Municipios con datos completos: {n_comp:,}  |  "
        "Triángulo inferior  |  Negrita = |r| ≥ 0.6"
    )
    fig.text(0.5, -0.02, leyenda, ha="center", fontsize=10, color="#555555")

    plt.suptitle(
        f"Estructura de correlaciones — Justificación del KNN multivariante ({anyo})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, f"correlaciones_grupos_imputacion_{anyo}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafico guardado en: {ruta}")

    print()
    print("CORRELACIONES CLAVE — Grupo A")
    pares_a = [
        ("Gini", "P80P20",      "indice gini",              "P80P20"),
        ("R.Hogar", "R.UC",     "Renta media hogar",        "renta media unidad consumo"),
        ("R.Hogar", "R.Persona","Renta media hogar",        "renta media persona"),
        ("Gini",  "R.Persona",  "indice gini",              "renta media persona"),
    ]
    for etq1, etq2, c1, c2 in pares_a:
        r = df_comp[c1].corr(df_comp[c2])
        intensidad = "muy alta" if abs(r) > 0.8 else "alta" if abs(r) > 0.5 else "baja"
        print(f"  {etq1:<12} vs {etq2:<12}  r = {r:+.3f}  ({intensidad})")

    print()
    print("CORRELACIONES CLAVE — Grupo B")
    pares_b = [
        ("Salarios",  "Pensiones",    "salarios",     "pensiones"),
        ("Desempleo", "Otras prest.", "desempleo",    "otras prestaciones"),
        ("Salarios",  "Desempleo",    "salarios",     "desempleo"),
        ("Otros ing.","Salarios",     "otros ingresos","salarios"),
    ]
    for etq1, etq2, c1, c2 in pares_b:
        r = df_comp[c1].corr(df_comp[c2])
        intensidad = "muy alta" if abs(r) > 0.8 else "alta" if abs(r) > 0.5 else "baja"
        print(f"  {etq1:<14} vs {etq2:<14}  r = {r:+.3f}  ({intensidad})")

    print()
    print("INTERPRETACION:")
    print("  Grupo A: las 3 variables de renta forman un subgrupo muy correlado (r>0.84).")
    print("    Gini y P80P20 tienen r=0.79 entre si pero baja correlacion con las rentas.")
    print("    El KNN multivariante explota estas correlaciones internas al calcular")
    print("    la distancia entre municipios.")
    print()
    print("  Grupo B: salarios y pensiones son complementarios (r=-0.84).")
    print("    Son las dos caras del perfil laboral del municipio.")
    print("    Imputarlos de forma independiente romperia esta restriccion estructural;")
    print("    la renormalizacion composicional la preserva.")


def pipeline_busqueda_k(
    df: pd.DataFrame,
    anyo: int,
    ks: list = KS_DEFAULT,
    pct_mascara: float = 0.20,
    seed: int = 42,
    output_dir: str = "documentation/imagenes_EDA",
) -> int:
    """
    Determina k mediante el método geométrico del codo (Kneedle),
    imprime la justificación numérica y genera los gráficos.

    Parámetros
    ----------
    df          : dataset unificado con nulos (datos_unificados_{anyo}.csv)
    anyo        : año electoral (2019 o 2023)
    ks          : valores de k a evaluar
    pct_mascara : fracción de municipios completos a enmascarar para CV
    seed        : semilla de reproducibilidad (fija el experimento)
    output_dir  : carpeta donde guardar el gráfico

    Retorna
    -------
    int : k óptimo según el codo geométrico
    """
    print(f"SELECCION DE K — KNN ESPACIAL ({anyo})")
    print("Metodo: codo geometrico (Kneedle), sin umbrales arbitrarios")
    print(f"Valores de k evaluados : {ks}")
    print(f"Fraccion enmascarada   : {pct_mascara*100:.0f}%  (seed={seed})")

    df_rmse = calcular_rmse_por_k(df, ks=ks, pct_mascara=pct_mascara, seed=seed)
    k_optimo, distancias, k_norm, v_norm = detectar_codo_geometrico(df_rmse)

    imprimir_tabla_k(df_rmse, k_optimo, distancias)
    plot_codo_k(df_rmse, k_optimo, distancias, k_norm, v_norm, anyo, output_dir=output_dir)

    return k_optimo


def pipeline_justificacion_imputacion(
    df: pd.DataFrame,
    anyo: int,
    ks: list = KS_DEFAULT,
    pct_mascara: float = 0.20,
    seed: int = 42,
    output_dir: str = "documentation/imagenes_EDA",
) -> int:
    """
    Orquestador completo de justificación metodológica de la imputación.

    Ejecuta en orden:
      1. Matrices de correlación por grupo (justifica KNN multivariante
         y la separación en dos grupos independientes).
      2. Búsqueda del k óptimo por codo geométrico (Kneedle).

    Parámetros — idénticos a pipeline_busqueda_k.

    Retorna
    -------
    int : k óptimo
    """
    print(f"JUSTIFICACION METODOLOGICA — IMPUTACION KNN ({anyo})")

    print("\n[1/2] Estructura de correlaciones por grupo de variables")
    plot_correlaciones_grupos(df, anyo, output_dir=output_dir)

    print("\n[2/2] Seleccion del hiperparametro k")
    k_optimo = pipeline_busqueda_k(
        df=df, anyo=anyo, ks=ks,
        pct_mascara=pct_mascara, seed=seed, output_dir=output_dir,
    )

    return k_optimo
