import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COLS_EXCLUIR = {
    "municipio", "nombre", "provincia", "rango tamaño población",
    "pct_psoe", "pct_pp", "pct_vox", "pct_cs", "pct_up_sumar",
    "pct_erc", "pct_jxcat", "pct_cup", "pct_pnv", "pct_ehbildu",
    "pct_bng", "pct_cc", "pct_prc", "pct_naplus", "pct_teruel", "pct_otros",
    "imputado", "imputado_grupo_a", "imputado_grupo_b", "calidad_datos",
    "poblacion", "densidad poblacional",
    "poblacion hombres", "poblacion mujeres",
    "edad media",
}

def analizar_pca(
    df: pd.DataFrame,
    anyo: int,
    output_dir: str = "documentation/imagenes_EDA",
    n_components: int = 10,
    cols_override: list = None,
) -> pd.DataFrame:
    """
    Realiza un PCA sobre las features socioeconómicas del dataset municipal.

    Pasos:
      1. Selecciona features: si se pasa cols_override, usa esa lista exacta;
         si no, toma todas las numéricas excluyendo identificadores, targets y flags.
      2. Elimina filas con NaN residuales (no debería haber tras imputación).
      3. Estandariza (media=0, std=1) con StandardScaler.
      4. Aplica PCA y genera:
         a. Scree plot (varianza explicada acumulada).
         b. Heatmap de loadings de los primeros 4 componentes.
         c. Scatter plot municipios en PC1 vs PC2, coloreado por log_poblacion.
      5. Imprime tabla de loadings de PC1 y PC2.

    Devuelve el dataframe original con las columnas PC1..PCn añadidas.
    """
    os.makedirs(output_dir, exist_ok=True)

    if cols_override is not None:
        cols_features = [c for c in cols_override if c in df.columns]
        missing = [c for c in cols_override if c not in df.columns]
        if missing:
            print(f"AVISO: columnas no encontradas en el dataframe y omitidas: {missing}")
    else:
        cols_features = [
            c for c in df.columns
            if c not in COLS_EXCLUIR and pd.api.types.is_numeric_dtype(df[c])
        ]
    print(f"PCA {anyo} — Features seleccionadas ({len(cols_features)}):")
    for c in cols_features:
        print(f"  · {c}")

    df_feat = df[cols_features].copy()

    n_antes = len(df_feat)
    df_feat = df_feat.dropna()
    n_despues = len(df_feat)
    if n_antes != n_despues:
        print(f"\nAVISO: {n_antes - n_despues} filas eliminadas por NaN residuales.")
    print(f"\nMunicipios en PCA: {n_despues}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)

    n_comp = min(n_components, len(cols_features))
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X)

    var_ratio = pca.explained_variance_ratio_
    var_acum  = np.cumsum(var_ratio)

    print(f"\nVarianza explicada por componente:")
    for i, (v, va) in enumerate(zip(var_ratio, var_acum), 1):
        print(f"  PC{i:02d}: {v*100:5.1f}%  (acumulada: {va*100:5.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, n_comp + 1), var_ratio * 100, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Varianza explicada (%)")
    axes[0].set_title(f"Varianza por componente — {anyo}")
    axes[0].set_xticks(range(1, n_comp + 1))

    axes[1].plot(range(1, n_comp + 1), var_acum * 100, marker="o", color="#4C72B0", linewidth=2)
    axes[1].axhline(80, color="red",    linestyle="--", linewidth=1, label="80%")
    axes[1].axhline(90, color="orange", linestyle="--", linewidth=1, label="90%")
    axes[1].set_xlabel("Número de componentes")
    axes[1].set_ylabel("Varianza acumulada (%)")
    axes[1].set_title(f"Varianza acumulada — {anyo}")
    axes[1].set_xticks(range(1, n_comp + 1))
    axes[1].legend()
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_scree_{anyo}.png"), dpi=150)
    plt.show()

    n_plot = min(4, n_comp)
    loadings = pd.DataFrame(
        pca.components_[:n_plot].T,
        index=cols_features,
        columns=[f"PC{i+1}" for i in range(n_plot)],
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(cols_features) * 0.4)))
    data = loadings.values
    vmax = 0.6

    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(n_plot))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_plot)], fontsize=10)
    ax.set_yticks(range(len(cols_features)))
    ax.set_yticklabels(cols_features, fontsize=8)
    ax.set_title(f"Loadings PCA — {anyo}", fontsize=12)

    for i in range(len(cols_features)):
        for j in range(n_plot):
            val = data[i, j]
            peso = "bold" if abs(val) >= 0.3 else "normal"
            color = "white" if abs(val) >= 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, fontweight=peso, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_loadings_{anyo}.png"), dpi=150)
    plt.show()

    color_col = "log_poblacion" if "log_poblacion" in df_feat.columns else cols_features[0]
    color_vals = df_feat[color_col].values

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(scores[:, 0], scores[:, 1],
                    c=color_vals, cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(f"Municipios en espacio PC1-PC2 — {anyo}")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_scatter_{anyo}.png"), dpi=150)
    plt.show()

    print(f"\nTop loadings PC1 (varianza {var_ratio[0]*100:.1f}%):")
    pc1_sorted = loadings["PC1"].abs().sort_values(ascending=False)
    for feat in pc1_sorted.index[:8]:
        print(f"  {feat:<35} {loadings.loc[feat,'PC1']:+.3f}")

    print(f"\nTop loadings PC2 (varianza {var_ratio[1]*100:.1f}%):")
    pc2_sorted = loadings["PC2"].abs().sort_values(ascending=False)
    for feat in pc2_sorted.index[:8]:
        print(f"  {feat:<35} {loadings.loc[feat,'PC2']:+.3f}")

    idx_validos = df_feat.index
    pc_cols = [f"PC{i+1}" for i in range(n_comp)]
    df_scores = pd.DataFrame(scores, index=idx_validos, columns=pc_cols)
    df_out = df.copy()
    for col in pc_cols:
        df_out[col] = np.nan
    df_out.loc[idx_validos, pc_cols] = df_scores

    n_80 = int(np.searchsorted(var_acum, 0.80)) + 1
    n_90 = int(np.searchsorted(var_acum, 0.90)) + 1
    print(f"\nComponentes para explicar 80% varianza: {n_80}")
    print(f"Componentes para explicar 90% varianza: {n_90}")

    return df_out
