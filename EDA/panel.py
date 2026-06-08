"""
EDA/panel.py
EDA comparativo del panel multianual demo (panel_demo.csv).
Todas las funciones son llamadas desde DEMO/demo.ipynb — TAREA 3.
Reutiliza plot_* de EDA/visuals.py siempre que sea posible.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from EDA.visuals import (
    plot_histogram_with_reference,
    plot_histogram,
    plot_boxplot,
)

# ── Constantes del panel ──────────────────────────────────────────────────────

ANOS_ELECCIONES: List[int] = [2015, 2016, 2019, 2023]

# Variables para análisis de variabilidad y distribuciones
VARS_PANEL = ["renta_neta_persona", "desempleo", "gini", "participacion"]

# 11 features ML: 8 socioeconómicas + log_poblacion + log_densidad + superficie_km2
FEATURES_ML = [
    "renta_neta_persona", "gini", "p80p20",
    "salarios", "pensiones", "desempleo", "otras_prestaciones", "otros_ingresos",
    "log_poblacion", "log_densidad",
]
# superficie_km2 es fija (sin sufijo de año)

# 4 targets de interés para correlaciones
TARGETS_CORR = ["pct_psoe", "pct_pp", "pct_vox", "pct_up_sumar"]

# Todos los targets del panel
_ALL_TARGETS = [
    "pct_psoe", "pct_pp", "pct_vox", "pct_cs", "pct_up_sumar",
    "pct_erc", "pct_jxcat", "pct_cup", "pct_pnv", "pct_ehbildu",
    "pct_bng", "pct_cc", "pct_prc", "pct_naplus", "pct_teruel", "pct_otros",
]


# ── Utilidades internas ───────────────────────────────────────────────────────

def _col(var: str, anio: int) -> str:
    return f"{var}_{anio}"


def _panel_a_long(panel: pd.DataFrame, var: str, anos: List[int]) -> pd.DataFrame:
    """Extrae columnas `var_YYYY` y devuelve formato largo (cod_ine, anyo, valor)."""
    rows = []
    for anio in anos:
        col = _col(var, anio)
        if col in panel.columns:
            sub = panel[["cod_ine", col]].copy()
            sub.columns = ["cod_ine", "valor"]
            sub["anyo"] = str(anio)
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["cod_ine", "anyo", "valor"])
    return pd.concat(rows, ignore_index=True)


def _detectar_anos_var(panel: pd.DataFrame, var: str) -> List[int]:
    """Detecta todos los años disponibles para `var` en el panel (columnas `var_YYYY`)."""
    anos = []
    prefix = f"{var}_"
    for col in panel.columns:
        if col.startswith(prefix):
            sufijo = col[len(prefix):]
            if sufijo.isdigit() and len(sufijo) == 4:
                anos.append(int(sufijo))
    return sorted(anos)


# ── Sección 1: Cobertura y nulos ──────────────────────────────────────────────

def eda_panel_nulos(
    panel: pd.DataFrame,
    anos_elec: List[int] = ANOS_ELECCIONES,
) -> None:
    """
    Sección 1: Cobertura y nulos del panel demo.
    - Heatmap % nulos por grupo de variables × año electoral
    - Tabla: % municipios con dato completo por variable
    - Municipios con nulos sistemáticos en algún año electoral
    """
    n = len(panel)
    print(f"=== SECCIÓN 1: COBERTURA Y NULOS ===")
    print(f"Panel: {n} municipios × {panel.shape[1]} columnas\n")

    grupos = {
        "renta_neta_persona": "renta_neta_persona",
        "desempleo":          "desempleo",
        "gini":               "gini",
        "p80p20":             "p80p20",
        "participacion":      "participacion",
        "pct_psoe":           "pct_psoe",
        "pct_pp":             "pct_pp",
        "pct_vox":            "pct_vox",
        "pct_up_sumar":       "pct_up_sumar",
    }

    # ── 1a. Heatmap % nulos ──
    rows_heat = {}
    for var, label in grupos.items():
        row = {}
        for anio in anos_elec:
            col = _col(var, anio)
            row[str(anio)] = round(panel[col].isna().mean() * 100, 1) if col in panel.columns else np.nan
        rows_heat[label] = row

    df_heat = pd.DataFrame(rows_heat).T
    df_heat.columns = [str(a) for a in anos_elec]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df_heat.astype(float), annot=True, fmt=".1f",
        cmap="Reds", vmin=0, vmax=30,
        linewidths=0.4, linecolor="#cccccc",
        cbar_kws={"label": "% nulos"},
        ax=ax,
    )
    ax.set_title("% municipios sin dato — panel demo (por año electoral)", pad=12)
    ax.set_xlabel("Año electoral")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.show()

    # ── 1b. Tabla % cobertura ──
    print("COBERTURA: % de municipios CON dato en cada año electoral")
    header = f"{'Variable':<25}" + "".join(f"  {a:>6}" for a in anos_elec)
    print(header)
    print("-" * len(header))
    for var, label in grupos.items():
        fila = f"{label:<25}"
        for anio in anos_elec:
            col = _col(var, anio)
            if col in panel.columns:
                pct = (1 - panel[col].isna().mean()) * 100
                fila += f"  {pct:>5.1f}%"
            else:
                fila += f"  {'N/A':>6}"
        print(fila)

    # ── 1c. Municipios con nulos sistemáticos ──
    clave_cols = [_col("renta_neta_persona", a) for a in anos_elec
                  if _col("renta_neta_persona", a) in panel.columns]
    if clave_cols:
        mask = panel[clave_cols].isna().any(axis=1)
        n_sis = mask.sum()
        print(f"\nMUNICIPIOS CON NULO EN renta_neta_persona EN ≥1 AÑO ELECTORAL: {n_sis}")
        if n_sis > 0:
            print(panel.loc[mask, ["cod_ine"] + clave_cols].head(10).to_string(index=False))


# ── Sección 2: Variabilidad temporal ─────────────────────────────────────────

def eda_variabilidad_temporal(
    panel: pd.DataFrame,
    variables: List[str] = VARS_PANEL,
) -> pd.DataFrame:
    """
    Sección 2: CV intra-municipio (std/mean×100) a lo largo de todos los años disponibles.
    - Histograma del CV con línea de mediana (reutiliza plot_histogram_with_reference)
    - Tabla: mediana CV, P25, P75, % municipios con CV > 5%
    - Devuelve DataFrame de CV por municipio × variable
    """
    print("=== SECCIÓN 2: VARIABILIDAD TEMPORAL (CV INTRA-MUNICIPIO) ===")
    resultados = {}

    for var in variables:
        anos_var = _detectar_anos_var(panel, var)
        cols = [_col(var, a) for a in anos_var if _col(var, a) in panel.columns]
        if len(cols) < 2:
            print(f"  [{var}] < 2 años disponibles — omitido")
            continue
        sub = panel[cols]
        cv = (sub.std(axis=1) / sub.mean(axis=1).abs()) * 100
        resultados[var] = cv.dropna()

    if not resultados:
        print("No hay variables con datos suficientes.")
        return pd.DataFrame()

    for var, cv_s in resultados.items():
        anos_var = _detectar_anos_var(panel, var)
        mediana = cv_s.median()
        plot_histogram_with_reference(
            df=cv_s.rename("cv").reset_index(drop=True).to_frame(),
            num_col="cv",
            ref_value=mediana,
            ref_label=f"Mediana CV = {mediana:.1f}%",
            title=f"CV intra-municipio: {var}  ({anos_var[0]}–{anos_var[-1]})",
            xlabel="Coeficiente de Variación (%)",
            ylabel="Nº municipios",
            bins=40,
            color="steelblue",
        )

    print("\nRESUMEN DE VARIABILIDAD TEMPORAL")
    print(f"{'Variable':<25} {'Med CV%':>8} {'P25':>8} {'P75':>8} {'%CV>5%':>8} {'N':>6}")
    print("-" * 68)
    for var, cv_s in resultados.items():
        print(
            f"{var:<25} {cv_s.median():>7.2f}%"
            f" {cv_s.quantile(.25):>7.2f}%"
            f" {cv_s.quantile(.75):>7.2f}%"
            f" {(cv_s > 5).mean()*100:>7.1f}%"
            f" {len(cv_s):>6}"
        )

    return pd.DataFrame(resultados)


# ── Sección 3: Distribuciones por año electoral ───────────────────────────────

def eda_distribuciones_por_anyo(
    panel: pd.DataFrame,
    variables: List[str] = VARS_PANEL,
    anos_elec: List[int] = ANOS_ELECCIONES,
) -> None:
    """
    Sección 3: Boxplots comparativos de variables para los 4 años electorales.
    Reutiliza plot_boxplot de visuals.py tras transformar a formato largo.
    Detecta cambios de distribución: crisis (2015/2016) → recuperación (2019) → post-covid (2023).
    """
    print("=== SECCIÓN 3: DISTRIBUCIONES POR AÑO ELECTORAL ===")

    for var in variables:
        df_long = _panel_a_long(panel, var, anos_elec).dropna(subset=["valor"])
        if df_long.empty:
            print(f"  [{var}] sin datos — omitido")
            continue

        plot_boxplot(
            df=df_long,
            cat_col="anyo",
            num_col="valor",
            title=f"Distribución de {var} — comparativa de olas electorales",
            xlabel="Año electoral",
            ylabel=var,
            palette=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"],
            rotation=0,
        )

        resumen = df_long.groupby("anyo")["valor"].agg(["median", "mean", "std"]).round(3)
        print(f"  {var}:")
        print(resumen.to_string())
        print()


# ── Sección 4: Correlación features–targets ───────────────────────────────────

def eda_correlacion_features_targets(
    panel: pd.DataFrame,
    features: List[str] = FEATURES_ML,
    targets: List[str] = TARGETS_CORR,
    anos_elec: List[int] = ANOS_ELECCIONES,
) -> None:
    """
    Sección 4: Heatmap correlación features × targets para cada año electoral.
    - Un heatmap por año + detección de cambios entre años
    - Scatter multi-año: renta vs pct_pp y desempleo vs pct_psoe
    """
    print("=== SECCIÓN 4: CORRELACIÓN FEATURES–TARGETS ===")

    # ── 4a. Heatmap por año ──
    for anio in anos_elec:
        feat_cols = [_col(f, anio) for f in features if _col(f, anio) in panel.columns]
        if "superficie_km2" in panel.columns:
            feat_cols = ["superficie_km2"] + feat_cols
        tgt_cols = [_col(t, anio) for t in targets if _col(t, anio) in panel.columns]

        if not feat_cols or not tgt_cols:
            print(f"  [{anio}] columnas insuficientes — omitido")
            continue

        df_sub = panel[feat_cols + tgt_cols].dropna()
        if len(df_sub) < 30:
            continue

        corr = df_sub.corr(min_periods=30).loc[feat_cols, tgt_cols]
        corr.index = [c.replace(f"_{anio}", "") for c in feat_cols]
        corr.columns = [c.replace(f"_{anio}", "") for c in tgt_cols]

        fig, ax = plt.subplots(figsize=(7, max(4, len(feat_cols) * 0.45 + 1)))
        sns.heatmap(
            corr.astype(float), annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, center=0,
            linewidths=0.3, annot_kws={"size": 8}, ax=ax,
        )
        ax.set_title(f"Correlación features → targets — {anio}", pad=12)
        plt.tight_layout()
        plt.show()

    # ── 4b. Scatter multi-año: renta vs pct_pp y desempleo vs pct_psoe ──
    colores = {2015: "#3498db", 2016: "#2ecc71", 2019: "#e67e22", 2023: "#e74c3c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for anio in anos_elec:
        c = colores.get(anio, "gray")
        r_col, pp_col = _col("renta_neta_persona", anio), _col("pct_pp", anio)
        d_col, ps_col = _col("desempleo", anio), _col("pct_psoe", anio)

        if r_col in panel.columns and pp_col in panel.columns:
            sub = panel[[r_col, pp_col]].dropna()
            axes[0].scatter(sub[r_col], sub[pp_col], alpha=0.15, s=8, color=c, label=str(anio))

        if d_col in panel.columns and ps_col in panel.columns:
            sub2 = panel[[d_col, ps_col]].dropna()
            axes[1].scatter(sub2[d_col], sub2[ps_col], alpha=0.15, s=8, color=c, label=str(anio))

    axes[0].set_xlabel("renta_neta_persona (€)")
    axes[0].set_ylabel("pct_pp")
    axes[0].set_title("Renta vs % PP — coloreado por año")
    axes[0].legend(markerscale=3, fontsize=9)

    axes[1].set_xlabel("desempleo (%)")
    axes[1].set_ylabel("pct_psoe")
    axes[1].set_title("Desempleo vs % PSOE — coloreado por año")
    axes[1].legend(markerscale=3, fontsize=9)

    plt.tight_layout()
    plt.show()


# ── Sección 5: Bloques ideológicos ────────────────────────────────────────────

def eda_bloques_ideologicos(
    panel: pd.DataFrame,
    anos_elec: List[int] = ANOS_ELECCIONES,
) -> None:
    """
    Sección 5: Evolución y estructura de los bloques ideológicos.
    - Serie temporal de la media del índice ideológico (global)
    - Heatmap provincia × año con el índice medio
    - Scatter renta_neta_persona_2019 vs indice_ideologico_2019, coloreado por log_poblacion
    """
    print("=== SECCIÓN 5: BLOQUES IDEOLÓGICOS ===")

    # ── 5a. Evolución temporal de la media ──
    medias = {a: panel[_col("indice_ideologico", a)].mean()
              for a in anos_elec if _col("indice_ideologico", a) in panel.columns}

    if medias:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(medias.keys()), list(medias.values()),
                marker="o", linewidth=2, color="#e74c3c")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(list(medias.keys()))
        ax.set_xlabel("Año electoral")
        ax.set_ylabel("Índice ideológico medio (derecha − izquierda)")
        ax.set_title("Evolución del índice ideológico medio\n(+) más derecha  |  (−) más izquierda")
        plt.tight_layout()
        plt.show()

        print("Índice ideológico medio por año:")
        for anio, val in medias.items():
            print(f"  {anio}: {val:+.4f}  {'→ derecha' if val > 0 else '→ izquierda'}")

    # ── 5b. Heatmap provincia × año ──
    if "provincia_enc" in panel.columns:
        idx_cols = [_col("indice_ideologico", a) for a in anos_elec
                    if _col("indice_ideologico", a) in panel.columns]
        if idx_cols:
            df_prov = panel.groupby("provincia_enc")[idx_cols].mean()
            df_prov.columns = [str(a) for a in anos_elec
                               if _col("indice_ideologico", a) in panel.columns]

            fig, ax = plt.subplots(figsize=(7, max(6, len(df_prov) * 0.28)))
            sns.heatmap(
                df_prov.astype(float), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0,
                linewidths=0.3, linecolor="#cccccc",
                annot_kws={"size": 7},
                cbar_kws={"label": "índice ideológico medio"},
                ax=ax,
            )
            ax.set_title("Índice ideológico medio — provincia × año electoral", pad=12)
            ax.set_xlabel("Año")
            ax.set_ylabel("Provincia (código)")
            plt.tight_layout()
            plt.show()

    # ── 5c. Scatter renta vs índice ideológico 2019, coloreado por log_poblacion ──
    r_col = "renta_neta_persona_2019"
    i_col = "indice_ideologico_2019"
    p_col = "log_poblacion_2019"

    if r_col in panel.columns and i_col in panel.columns:
        cols_use = [r_col, i_col] + ([p_col] if p_col in panel.columns else [])
        sub = panel[cols_use].dropna()

        fig, ax = plt.subplots(figsize=(9, 6))
        if p_col in sub.columns:
            sc = ax.scatter(sub[r_col], sub[i_col],
                            c=sub[p_col], cmap="viridis", alpha=0.5, s=12)
            plt.colorbar(sc, ax=ax, label="log(población 2019)")
        else:
            ax.scatter(sub[r_col], sub[i_col], alpha=0.4, s=10)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Renta neta media por persona (2019)")
        ax.set_ylabel("Índice ideológico 2019 (der − izq)")
        ax.set_title("Renta vs Ideología (2019) — coloreado por tamaño de municipio")
        plt.tight_layout()
        plt.show()


# ── Sección 6: Leakage check y consistencia ───────────────────────────────────

def eda_leakage_check(
    panel: pd.DataFrame,
    anos_elec: List[int] = ANOS_ELECCIONES,
) -> None:
    """
    Sección 6: Verificaciones de consistencia para el ML.
    - Distribución de la suma de pct_* por municipio y año (debe ≈ 1.0)
    - pct_izquierda + pct_derecha + pct_nacionalistas + pct_otros ≈ 1
    - Distribución de pct_otros por año (via plot_boxplot)
    - Rango de participación: alerta si hay valores fuera de [10, 100]
    """
    print("=== SECCIÓN 6: LEAKAGE CHECK Y CONSISTENCIA ===")

    # ── 6a. Suma pct_* ≈ 1 ──
    anos_con_data = [a for a in anos_elec
                     if any(_col(t, a) in panel.columns for t in _ALL_TARGETS)]
    n_plots = len(anos_con_data)
    if n_plots:
        ncols = min(n_plots, 2)
        nrows = (n_plots + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, anio in enumerate(anos_con_data):
            tgt_cols = [_col(t, anio) for t in _ALL_TARGETS if _col(t, anio) in panel.columns]
            suma = panel[tgt_cols].sum(axis=1)
            desv = (suma - 1.0).abs()

            axes_flat[idx].hist(suma.dropna(), bins=50,
                                color="#3498db", edgecolor="white", linewidth=0.3)
            axes_flat[idx].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="1.0")
            axes_flat[idx].set_title(f"Suma pct_* — {anio}")
            axes_flat[idx].set_xlabel("Suma de pct_*")
            axes_flat[idx].legend()

            print(f"\n[{anio}] Suma pct_*: media={suma.mean():.4f}  std={suma.std():.6f}  "
                  f"municipios con |suma−1|>0.01: {(desv > 0.01).sum()}")

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Distribución de la suma de targets pct_*", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()

    # ── 6b. Bloques + otros ≈ 1 ──
    print("\nVERIFICACIÓN: pct_izq + pct_der + pct_nac + pct_otros ≈ 1")
    for anio in anos_elec:
        bl_cols = [_col(b, anio) for b in
                   ["pct_izquierda", "pct_derecha", "pct_nacionalistas", "pct_otros"]
                   if _col(b, anio) in panel.columns]
        if len(bl_cols) < 2:
            continue
        suma_bl = panel[bl_cols].sum(axis=1)
        desv_bl = (suma_bl - 1.0).abs()
        print(f"  {anio}: cols_usadas={len(bl_cols)}  "
              f"media_suma={suma_bl.mean():.4f}  "
              f"max_desv={desv_bl.max():.4f}  "
              f"municipios con |suma-1|>0.01: {(desv_bl > 0.01).sum()}")

    # ── 6c. Distribución pct_otros por año ──
    df_otros = _panel_a_long(panel, "pct_otros", anos_elec).dropna(subset=["valor"])
    if not df_otros.empty:
        plot_boxplot(
            df=df_otros,
            cat_col="anyo",
            num_col="valor",
            title="Distribución de pct_otros por año — ¿voto residual anómalo?",
            xlabel="Año electoral",
            ylabel="pct_otros (fracción de voto)",
            palette=["#95a5a6", "#7f8c8d", "#636e72", "#2d3436"],
            rotation=0,
        )
        print("\npct_otros por año:")
        for anio_str, grp in df_otros.groupby("anyo"):
            v = grp["valor"]
            print(f"  {anio_str}: median={v.median():.3f}  "
                  f"P90={v.quantile(.9):.3f}  "
                  f"%>15%: {(v > 0.15).mean()*100:.1f}%")

    # ── 6d. Rango participación ──
    print("\nRANGO DE PARTICIPACIÓN:")
    for anio in anos_elec:
        col = _col("participacion", anio)
        if col not in panel.columns:
            continue
        s = panel[col].dropna()
        fuera = ((s < 10) | (s > 100)).sum()
        flag = f"  ⚠ fuera de [10, 100]: {fuera} municipios" if fuera > 0 else ""
        print(f"  {anio}: min={s.min():.1f}  max={s.max():.1f}  "
              f"media={s.mean():.1f}{flag}")


# ── Sección 7: Señal del panel temporal ──────────────────────────────────────

def eda_senyal_panel_temporal(
    panel: pd.DataFrame,
    pares_olas: List[Tuple[int, int]] = [(2015, 2016), (2016, 2019), (2019, 2023)],
) -> None:
    """
    Sección 7: Análisis de cambios entre olas electorales consecutivas.
    Para cada par (a1, a2):
    - Calcula Δrenta, Δdesempleo, Δgini, Δpct_pp, Δpct_psoe por municipio
    - Scatter: Δdesempleo vs Δpct_pp y Δrenta vs Δpct_psoe con coef. Pearson
    """
    print("=== SECCIÓN 7: SEÑAL DEL PANEL TEMPORAL ===")

    _VARS_DELTA = [
        "renta_neta_persona", "desempleo", "gini", "pct_pp", "pct_psoe",
    ]

    for a1, a2 in pares_olas:
        print(f"\n--- Par: {a1} → {a2} ---")

        delta_dict = {}
        for var in _VARS_DELTA:
            c1, c2 = _col(var, a1), _col(var, a2)
            if c1 in panel.columns and c2 in panel.columns:
                delta_dict[f"delta_{var}"] = panel[c2] - panel[c1]

        if len(delta_dict) < 2:
            print(f"  Insuficientes columnas — omitido")
            continue

        df_d = pd.DataFrame(delta_dict).dropna()
        print(f"  Municipios con datos completos: {len(df_d)}")

        for var in ["renta_neta_persona", "desempleo", "gini"]:
            dk = f"delta_{var}"
            if dk in df_d.columns:
                s = df_d[dk]
                print(f"  Δ{var}: media={s.mean():.2f}  std={s.std():.2f}  "
                      f"%aumento={(s > 0).mean()*100:.1f}%")

        # Scatter Δdesempleo vs Δpct_pp y Δrenta vs Δpct_psoe
        has_dep = "delta_desempleo" in df_d.columns and "delta_pct_pp" in df_d.columns
        has_ren = "delta_renta_neta_persona" in df_d.columns and "delta_pct_psoe" in df_d.columns

        if has_dep or has_ren:
            n_ax = sum([has_dep, has_ren])
            fig, axes = plt.subplots(1, n_ax, figsize=(7 * n_ax, 5))
            if n_ax == 1:
                axes = [axes]
            ax_i = 0

            if has_dep:
                r = df_d[["delta_desempleo", "delta_pct_pp"]].corr().iloc[0, 1]
                axes[ax_i].scatter(df_d["delta_desempleo"], df_d["delta_pct_pp"],
                                   alpha=0.3, s=8, color="#e74c3c")
                axes[ax_i].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
                axes[ax_i].axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
                axes[ax_i].set_xlabel(f"Δdesempleo ({a1}→{a2})")
                axes[ax_i].set_ylabel(f"Δpct_pp ({a1}→{a2})")
                axes[ax_i].set_title(f"Δdesempleo vs Δpct_pp  r={r:.3f}")
                ax_i += 1

            if has_ren:
                r2 = df_d[["delta_renta_neta_persona", "delta_pct_psoe"]].corr().iloc[0, 1]
                axes[ax_i].scatter(df_d["delta_renta_neta_persona"], df_d["delta_pct_psoe"],
                                   alpha=0.3, s=8, color="#3498db")
                axes[ax_i].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
                axes[ax_i].axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
                axes[ax_i].set_xlabel(f"Δrenta_neta_persona ({a1}→{a2})")
                axes[ax_i].set_ylabel(f"Δpct_psoe ({a1}→{a2})")
                axes[ax_i].set_title(f"Δrenta vs Δpct_psoe  r={r2:.3f}")

            plt.suptitle(f"Señal del panel temporal: {a1} → {a2}", fontsize=12, y=1.02)
            plt.tight_layout()
            plt.show()
