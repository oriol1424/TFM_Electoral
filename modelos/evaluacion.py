import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from modelos.entrenamiento import preparar_features, TARGETS


def evaluar_modelos(
    modelos: Dict,
    df_test: pd.DataFrame,
    etiqueta: str = "test",
    X_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Evaluates each model on df_test.
    Pass X_test explicitly when features differ from preparar_features output (e.g. spatial model).
    Returns DataFrame with MAE, RMSE, R² and mean real value per party.
    """
    if X_test is None:
        X_test = preparar_features(df_test)
    resultados = []

    for partido, modelo in modelos.items():
        if partido not in df_test.columns:
            continue
        y_true = df_test[partido].dropna().values
        y_pred = modelo.predict(X_test.loc[df_test[partido].notna()])

        mae  = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        resultados.append({
            'partido': partido.replace('pct_', ''),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'media_real': round(y_true.mean(), 4)
        })

    df_res = pd.DataFrame(resultados).sort_values('MAE')
    print(f"\nMETRICAS — {etiqueta}")
    print(df_res.to_string(index=False))
    return df_res


def visualizar_metricas(df_metricas: pd.DataFrame, etiqueta: str = "2023"):
    """
    Horizontal bar plots of MAE and R² per party.
    Saved to documentation/imagenes_EDA/.
    """
    os.makedirs('documentation/imagenes_EDA', exist_ok=True)
    df_s = df_metricas.sort_values('MAE', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df_s, x='MAE', y='partido', ax=ax1, palette='Blues_r')
    ax1.set_title(f'MAE por partido — {etiqueta}')
    ax1.set_xlabel('Error Absoluto Medio (puntos porcentuales)')

    sns.barplot(data=df_s, x='R2', y='partido', ax=ax2, palette='Greens_r')
    ax2.set_title(f'R2 por partido — {etiqueta}')
    ax2.set_xlabel('R2 (varianza explicada)')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'documentation/imagenes_EDA/metricas_ml_{etiqueta}.png', dpi=150)
    plt.show()


def comparar_dhondt_modelos(
    esc_real: Dict[str, int],
    modelos_esc: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    Muestra tabla comparativa de escanos real vs varios modelos y calcula MAE.
    modelos_esc: {"base": {pp:96,...}, "v2": {...}, ...}
    """
    todos_keys: set = set(esc_real)
    for esc in modelos_esc.values():
        todos_keys |= set(esc)
    todos = sorted(todos_keys)

    rows = []
    for p in todos:
        row = {"partido": p, "real": esc_real.get(p, 0)}
        for col, esc_dict in modelos_esc.items():
            row[col] = esc_dict.get(p, 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("real", ascending=False)

    cols_modelo = list(modelos_esc.keys())
    print("ESCANOS: real vs " + " vs ".join(cols_modelo))
    print(df.to_string(index=False))
    print()
    for col in cols_modelo:
        mae = (df["real"] - df[col]).abs().mean()
        print(f"MAE escanos {col}: {mae:.2f}")

    return df


def evaluar_accuracy_ganador(
    df_real: pd.DataFrame,
    pred_dict: Dict[str, pd.DataFrame],
    guardar: bool = True,
) -> pd.DataFrame:
    """
    Calcula accuracy y recall del partido ganador por municipio para varios modelos.
    pred_dict: {"Baseline": df_base, "V2 (+grupo)": df_v2, ...}
    Cada DataFrame debe tener columna 'municipio' y columnas pct_*.
    """
    targets_pct = [t for t in df_real.columns if t.startswith("pct_") and t != "pct_otros"]
    for df_p in pred_dict.values():
        targets_pct = [t for t in targets_pct if t in df_p.columns]

    df_real_m = df_real.set_index("municipio")
    preds_m = {label: df_p.set_index("municipio") for label, df_p in pred_dict.items()}

    idx = df_real_m.index
    for df_p in preds_m.values():
        idx = idx.intersection(df_p.index)

    ganador_real = (
        df_real_m.loc[idx, targets_pct]
        .idxmax(axis=1)
        .str.replace("pct_", "", regex=False)
    )
    ganadores = {
        label: df_p.loc[idx, targets_pct].idxmax(axis=1).str.replace("pct_", "", regex=False)
        for label, df_p in preds_m.items()
    }

    print("ACCURACY DEL PARTIDO GANADOR POR MUNICIPIO (test 2023)")
    print("-" * 52)
    for label, g in ganadores.items():
        n_ok = (ganador_real == g).sum()
        acc = n_ok / len(ganador_real)
        print("  %-16s %.1f%%  (%d / %d municipios)" % (label, acc * 100, n_ok, len(ganador_real)))

    print()
    print("RECALL POR PARTIDO (% municipios que gana en realidad y el modelo acierta)")
    print("-" * 60)

    conteo_real = ganador_real.value_counts()
    rows_acc = []
    for partido in conteo_real.index:
        mask = ganador_real == partido
        n_real = int(mask.sum())
        row: Dict = {"partido": partido, "n_real": n_real}
        for label, g in ganadores.items():
            col_key = "recall_" + label.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
            row[col_key] = round(float((g[mask] == partido).sum()) / n_real, 3)
        rows_acc.append(row)

    df_acc = pd.DataFrame(rows_acc).sort_values("n_real", ascending=False)
    print(df_acc.to_string(index=False))

    recall_cols = [c for c in df_acc.columns if c.startswith("recall_")]
    labels_plot = [c.replace("recall_", "").replace("_", " ") for c in recall_cols]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(recall_cols)]

    fig, ax = plt.subplots(figsize=(12, 5))
    partidos_plot = df_acc["partido"].tolist()
    x = list(range(len(partidos_plot)))
    w = 0.8 / max(len(recall_cols), 1)
    offset = -(len(recall_cols) - 1) * w / 2
    for i, (col, lbl, color) in enumerate(zip(recall_cols, labels_plot, colors)):
        ax.bar([xi + offset + i * w for xi in x], df_acc[col], width=w, label=lbl, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(partidos_plot, rotation=45, ha="right")
    ax.set_ylabel("Recall (% municipios ganadores correctamente identificados)")
    ax.set_title("Recall del partido ganador por municipio - test 2023")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0.67, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if guardar:
        os.makedirs("documentation/imagenes_EDA", exist_ok=True)
        plt.savefig("documentation/imagenes_EDA/accuracy_ganador.png", dpi=150, bbox_inches="tight")
    plt.show()

    return df_acc


def comparar_real_predicho(modelos: Dict, df_test: pd.DataFrame, partido: str):
    """
    Scatter plot real vs predicted for one party.
    """
    X_test = preparar_features(df_test)
    y_true = df_test[partido].values
    y_pred = modelos[partido].predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', label='prediccion perfecta')
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicho')
    ax.set_title(f'Real vs Predicho — {partido}')
    ax.legend()
    plt.tight_layout()
    plt.show()


def comparar_metricas_modelos(
    metricas_dict: Dict[str, pd.DataFrame],
    guardar: bool = True,
) -> pd.DataFrame:
    """
    Tabla y grafico de barras comparando MAE y R2 de varios modelos.
    metricas_dict: {"base": df_metricas_base, "v2": df_v2, "sub": df_sub, ...}
    Claves del dict se usan como sufijos de columna (MAE_base, R2_v2, ...).
    """
    df_cmp = None
    for sufijo, df_m in metricas_dict.items():
        df_m2 = df_m[["partido", "MAE", "R2"]].rename(
            columns={"MAE": f"MAE_{sufijo}", "R2": f"R2_{sufijo}"}
        )
        df_cmp = df_m2 if df_cmp is None else df_cmp.merge(df_m2, on="partido", how="outer")

    primera_mae = [c for c in df_cmp.columns if c.startswith("MAE_")][0]
    df_cmp = df_cmp.fillna(0).sort_values(primera_mae, ascending=False)

    mae_cols = [c for c in df_cmp.columns if c.startswith("MAE_")]
    r2_cols  = [c for c in df_cmp.columns if c.startswith("R2_")]
    labels = list(metricas_dict.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)]

    partidos = df_cmp["partido"].tolist()
    x = list(range(len(partidos)))
    w = 0.8 / max(len(labels), 1)
    offset = -(len(labels) - 1) * w / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    titulo_base = " vs ".join(labels)
    for ax, cols, titulo, ylabel in [
        (ax1, mae_cols, f"MAE test 2023 — {titulo_base}", "MAE (pp)"),
        (ax2, r2_cols,  f"R2 test 2023 — {titulo_base}",  "R2"),
    ]:
        for i, (col, lbl, color) in enumerate(zip(cols, labels, colors)):
            ax.bar([xi + offset + i * w for xi in x], df_cmp[col], width=w, label=lbl, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(partidos, rotation=45, ha="right")
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.legend()
        if ylabel == "R2":
            ax.axhline(0, color="red", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if guardar:
        os.makedirs("documentation/imagenes_EDA", exist_ok=True)
        plt.savefig("documentation/imagenes_EDA/comparativa_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nTABLA COMPARATIVA MAE TEST 2023")
    print(df_cmp[["partido"] + mae_cols].to_string(index=False))

    return df_cmp


def pipeline_comparacion_dhondt_modelos(
    modelos_base: Dict,
    modelos_v2_dict: Dict,
    modelos_sub_dict: Dict,
    df_2023: pd.DataFrame,
    ruta_json_pob: str,
    esc_base: Dict[str, int],
    esc_real: Dict[str, int],
) -> tuple:
    """
    Ejecuta las pipelines de prediccion para V2 y subgrupos, aplica D'Hondt
    y llama a comparar_dhondt_modelos. Devuelve (df_pred_v2, df_pred_sub, df_dhondt_cmp).
    """
    from modelos.v2 import pipeline_prediccion_v2
    from modelos.subgrupos import pipeline_prediccion_subgrupos
    from calculos_electorales.resultados import votos_predichos_por_provincia
    from calculos_electorales.dhondt import dhondt_todas_provincias, escanos_por_provincia

    dict_esc = escanos_por_provincia(ruta_json_pob)

    df_pred_v2 = pipeline_prediccion_v2(modelos_v2_dict, df_2023)
    esc_v2 = {
        k.replace("votos_", ""): v
        for k, v in dhondt_todas_provincias(
            votos_predichos_por_provincia(df_pred_v2), dict_esc
        ).items()
    }

    df_pred_sub = pipeline_prediccion_subgrupos(
        modelos_sub_dict, df_2023, modelos_fallback=modelos_base
    )
    esc_sub = {
        k.replace("votos_", ""): v
        for k, v in dhondt_todas_provincias(
            votos_predichos_por_provincia(df_pred_sub), dict_esc
        ).items()
    }

    df_dhondt_cmp = comparar_dhondt_modelos(
        esc_real, {"base": esc_base, "v2": esc_v2, "sub": esc_sub}
    )

    return df_pred_v2, df_pred_sub, df_dhondt_cmp
