import pandas as pd
from .funciones_genericas_limpieza import (
    leer_archivo_csv,
    guardar_dataframe_csv,
    limpiar_serie_numerica,
)


def procesar_edad_media_municipios(path_csv, anio, path_salida):
    """Extrae la edad media municipal del INE para un año y guarda CSV en formato ancho."""
    df = leer_archivo_csv(path_csv)
    df = df[df["Periodo"].astype(str) == str(anio)].copy()

    df["id_municipio"] = df["Municipios"].str[:5]
    df["nombre_municipio"] = df["Municipios"].str[6:].str.strip()
    df["edad_media"] = limpiar_serie_numerica(df["Total"])

    df_wide = df.pivot_table(
        index=["id_municipio", "nombre_municipio"],
        columns="Sexo",
        values="edad_media"
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={
        "Ambos sexos": "edad_media_ambos",
        "Hombres": "edad_media_hombres",
        "Mujeres": "edad_media_mujeres",
    })

    guardar_dataframe_csv(df_wide, path_salida)
    print(f"Edad media municipios {anio}: {len(df_wide)} registros → '{path_salida}'.")


def procesar_edad_media_provincias(path_csv, anio, path_salida):
    """Extrae la edad media provincial del INE para un año y guarda CSV en formato ancho."""
    df = leer_archivo_csv(path_csv)
    df = df[df["Periodo"].astype(str) == str(anio)].copy()

    def _parse_provincia(val: str):
        val = str(val).strip()
        if val == "Total Nacional":
            return "00", "Total Nacional"
        parts = val.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[0].zfill(2), parts[1].strip()
        return pd.NA, val

    df[["cod_provincia", "nombre_provincia"]] = df["Provincias"].apply(
        lambda x: pd.Series(_parse_provincia(x), index=["cod_provincia", "nombre_provincia"])
    )
    df["edad_media"] = limpiar_serie_numerica(df["Total"])

    df_wide = df.pivot_table(
        index=["cod_provincia", "nombre_provincia"],
        columns="Sexo",
        values="edad_media"
    ).reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={
        "Ambos sexos": "edad_media_ambos",
        "Hombres": "edad_media_hombres",
        "Mujeres": "edad_media_mujeres",
    })

    guardar_dataframe_csv(df_wide, path_salida)
    print(f"Edad media provincias {anio}: {len(df_wide)} registros → '{path_salida}'.")
