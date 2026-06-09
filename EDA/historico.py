import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from EDA.funciones_generales import categorizar_municipios_tfm, RANGOS_MUNICIPIO
from EDA.visuals import obtener_mapeo_provincias, plot_smart_bar


def eda_panel_historico_cobertura(panel: pd.DataFrame) -> None:
    """Cobertura del panel histórico: resumen numérico, barras por provincia y distribución por tamaño."""
    n_munis    = len(panel)
    n_provincias = panel["provincia_enc"].nunique()
    anos_elec  = [2015, 2016, 2019, 2023]

    mapeo = obtener_mapeo_provincias("2019")

    todas_prov = set(range(1, 53))
    prov_panel = set(panel["provincia_enc"].unique())
    prov_faltantes = sorted(todas_prov - prov_panel)
    n_menos100 = int((panel["poblacion_2019"] < 100).sum()) if "poblacion_2019" in panel.columns else "N/A"

    print("PANEL HISTÓRICO — COBERTURA")
    print(f"  Municipios           : {n_munis}")
    print(f"  Provincias           : {n_provincias} / 52")
    print(f"  Años                 : {', '.join(str(a) for a in anos_elec)}")
    print(f"  Municipios < 100 hab : {n_menos100}")
    if prov_faltantes:
        nombres_falt = [mapeo.get(str(p), str(p).zfill(2)) for p in prov_faltantes]
        print(f"  Provincias sin datos : {', '.join(nombres_falt)}")
    print()

    df_prov = (
        panel["provincia_enc"]
        .value_counts()
        .reset_index()
    )
    df_prov.columns = ["provincia_enc", "n_municipios"]
    df_prov["cod_str"] = df_prov["provincia_enc"].astype(str).str.zfill(2)
    df_prov["provincia"] = df_prov["cod_str"].map(mapeo).fillna(df_prov["cod_str"])
    df_prov = df_prov.sort_values("n_municipios", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(8, len(df_prov) * 0.22)))
    sns.set_theme(style="whitegrid")
    ax.barh(df_prov["provincia"], df_prov["n_municipios"], color="steelblue")
    ax.set_xlabel("Número de municipios")
    ax.set_title("Municipios del panel histórico por provincia", pad=12)
    ax.tick_params(axis="y", labelsize=8)
    for bar, val in zip(ax.patches, df_prov["n_municipios"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)
    plt.tight_layout()
    plt.show()

    if "poblacion_2019" in panel.columns:
        df_tam = panel.copy()
        df_tam["rango"] = df_tam["poblacion_2019"].apply(categorizar_municipios_tfm)

        conteo = (
            df_tam["rango"]
            .value_counts()
            .reindex(RANGOS_MUNICIPIO, fill_value=0)
            .reset_index()
        )
        conteo.columns = ["rango", "n"]

        plot_smart_bar(
            df=conteo,
            cat_col="rango",
            val_col="n",
            orientation="v",
            title="Distribución del panel histórico por tamaño de municipio (población 2019)",
            xlabel="Rango de población",
            ylabel="Número de municipios",
            palette="Blues_d",
            rotation=45,
        )
