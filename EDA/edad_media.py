import pandas as pd
from EDA.visuals import plot_histogram_with_reference, plot_barh_with_reference


def _normalizar_cod_provincia(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que cod_provincia sea string con cero a la izquierda."""
    df = df.copy()
    df['cod_provincia'] = df['cod_provincia'].astype(str).str.zfill(2)
    return df


def _plot_provincias_ambos(df_provincias: pd.DataFrame, anio: int) -> None:
    """Barras horizontales de edad media provincial (Ambos sexos) con linea Total Nacional."""
    df_provincias = _normalizar_cod_provincia(df_provincias)
    total_nac = df_provincias.loc[df_provincias['cod_provincia'] == '00', 'edad_media_ambos']
    df_prov = df_provincias[df_provincias['cod_provincia'] != '00'].dropna(subset=['edad_media_ambos']).copy()

    if total_nac.empty or df_prov.empty:
        print("Aviso: datos insuficientes para el grafico provincial (Ambos sexos).")
        return

    val_nac = float(total_nac.iloc[0])
    print(f"  Edad media Total Nacional ({anio}): {val_nac:.2f} anos")
    print(f"  Rango provincial: {df_prov['edad_media_ambos'].min():.2f} - {df_prov['edad_media_ambos'].max():.2f} anos")

    plot_barh_with_reference(
        df=df_prov,
        val_col='edad_media_ambos',
        label_col='nombre_provincia',
        ref_value=val_nac,
        ref_label='Total Nacional',
        title=f'Edad Media Provincial - Ambos sexos ({anio})',
        xlabel='Edad Media (anos)',
        color_above='steelblue',
        color_below='salmon',
    )


def _plot_provincias_genero(df_provincias: pd.DataFrame, anio: int) -> None:
    """Barras horizontales por genero (Hombres / Mujeres) con linea de Total Nacional."""
    colores_above = {'edad_media_hombres': 'cornflowerblue', 'edad_media_mujeres': 'mediumorchid'}
    colores_below = {'edad_media_hombres': 'lightskyblue', 'edad_media_mujeres': 'plum'}
    etiquetas = {'edad_media_hombres': 'Hombres', 'edad_media_mujeres': 'Mujeres'}

    df_provincias = _normalizar_cod_provincia(df_provincias)
    total_nac = df_provincias[df_provincias['cod_provincia'] == '00']
    df_prov = df_provincias[df_provincias['cod_provincia'] != '00'].copy()

    for col, genero in etiquetas.items():
        if col not in df_provincias.columns:
            print(f"Aviso: columna '{col}' no encontrada en df_provincias.")
            continue
        if total_nac.empty or total_nac[col].isna().all():
            print(f"Aviso: sin Total Nacional para {genero}.")
            continue

        val_nac = float(total_nac[col].iloc[0])
        print(f"  {genero} - Total Nacional ({anio}): {val_nac:.2f} anos")

        plot_barh_with_reference(
            df=df_prov.dropna(subset=[col]),
            val_col=col,
            label_col='nombre_provincia',
            ref_value=val_nac,
            ref_label=f'Total Nacional',
            title=f'Edad Media Provincial - {genero} ({anio})',
            xlabel='Edad Media (anos)',
            color_above=colores_above[col],
            color_below=colores_below[col],
        )


def _plot_municipios(df_municipios: pd.DataFrame, anio: int) -> None:
    """Tres histogramas de edad media municipal (Ambos sexos, Hombres, Mujeres)."""
    colores = {
        'edad_media_ambos': 'mediumseagreen',
        'edad_media_hombres': 'cornflowerblue',
        'edad_media_mujeres': 'salmon',
    }
    etiquetas = {
        'edad_media_ambos': 'Ambos sexos',
        'edad_media_hombres': 'Hombres',
        'edad_media_mujeres': 'Mujeres',
    }

    for col, genero in etiquetas.items():
        if col not in df_municipios.columns:
            print(f"  Aviso: columna '{col}' no encontrada en df_municipios.")
            continue

        df_clean = df_municipios.dropna(subset=[col])
        if df_clean.empty:
            print(f"  Aviso: sin datos municipales para {genero}.")
            continue

        media_nac = df_clean[col].mean()
        print(f"  {genero} — municipios con dato: {len(df_clean):,} | "
              f"media: {media_nac:.2f} | "
              f"rango: {df_clean[col].min():.2f} - {df_clean[col].max():.2f} anos")

        plot_histogram_with_reference(
            df=df_clean,
            num_col=col,
            ref_value=media_nac,
            ref_label=f'Media de municipios ({genero})',
            title=f'Distribucion de la Edad Media Municipal - {genero} ({anio})',
            xlabel='Edad Media (anos)',
            ylabel='Numero de Municipios',
            bins=35,
            color=colores[col],
        )



def eda_edad_media(
    df_provincias: pd.DataFrame,
    df_municipios: pd.DataFrame,
    anio: int
) -> None:
    """
    EDA completo de la edad media provincial y municipal.

    Genera tres bloques de histogramas:
      1. Provincias - Ambos sexos, con linea Total Nacional.
      2. Provincias - Hombres y Mujeres por separado, con linea Total Nacional.
      3. Municipios - distribucion con linea de media nacional calculada.

    Args:
        df_provincias: DataFrame del ETL con columnas cod_provincia, nombre_provincia,
                       edad_media_ambos, edad_media_hombres, edad_media_mujeres.
        df_municipios: DataFrame del ETL con columnas id_municipio, nombre_municipio,
                       edad_media_ambos, edad_media_hombres, edad_media_mujeres.
        anio: Anno a analizar.
    """
    print(f"\n{'='*50}")
    print(f"  EDA EDAD MEDIA - {anio}")
    print(f"{'='*50}")

    print("\n[1/3] Provincias - Ambos sexos")
    _plot_provincias_ambos(df_provincias, anio)

    print("\n[2/3] Provincias - Por genero")
    _plot_provincias_genero(df_provincias, anio)

    print("\n[3/3] Municipios")
    _plot_municipios(df_municipios, anio)
