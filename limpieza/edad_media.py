import pandas as pd
from .funciones_genericas_limpieza import (
    leer_archivo_csv,
    guardar_dataframe_csv,
    limpiar_serie_numerica,
)


def procesar_edad_media_municipios(path_csv: str, anio: int, path_salida: str) -> None:
    """
    Extrae la edad media municipal del INE para un año dado y la guarda en CSV.
    Filtra 'Ambos sexos', separa el código de municipio (5 dígitos) del nombre
    y convierte el valor decimal al formato float.

    Args:
        path_csv: Ruta al fichero raw del INE (30699.csv, municipios).
        anio: Año objetivo.
        path_salida: Ruta del CSV de salida.
    """
    df = leer_archivo_csv(path_csv)
    df = df[
        (df["Sexo"] == "Ambos sexos") &
        (df["Periodo"].astype(str) == str(anio))
    ].copy()

    df["id_municipio"] = df["Municipios"].str[:5]
    df["nombre_municipio"] = df["Municipios"].str[6:].str.strip()
    df["edad_media"] = limpiar_serie_numerica(df["Total"])

    df_out = df[["id_municipio", "nombre_municipio", "edad_media"]].reset_index(drop=True)
    guardar_dataframe_csv(df_out, path_salida)
    print(f"Edad media municipios {anio}: {len(df_out)} registros → '{path_salida}'.")


def procesar_edad_media_provincias(path_csv: str, anio: int, path_salida: str) -> None:
    """
    Extrae la edad media provincial del INE para un año dado y la guarda en CSV.
    Separa el código de provincia (2 dígitos con cero a la izquierda) del nombre.
    'Total Nacional' recibe el código '00'.

    Args:
        path_csv: Ruta al fichero raw del INE (3199.csv, provincias).
        anio: Año objetivo.
        path_salida: Ruta del CSV de salida.
    """
    df = leer_archivo_csv(path_csv)
    df = df[
        (df["Sexo"] == "Ambos sexos") &
        (df["Periodo"].astype(str) == str(anio))
    ].copy()

    def _parse_provincia(val: str):
        val = str(val).strip()
        if val == "Total Nacional":
            return "00", "Total Nacional"
        parts = val.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[0].zfill(2), parts[1].strip()
        return pd.NA, val

    parsed = df["Provincias"].apply(
        lambda x: pd.Series(_parse_provincia(x), index=["cod_provincia", "nombre_provincia"])
    )
    df = pd.concat([df.reset_index(drop=True), parsed], axis=1)
    df["edad_media"] = limpiar_serie_numerica(df["Total"])

    df_out = df[["cod_provincia", "nombre_provincia", "edad_media"]].reset_index(drop=True)
    guardar_dataframe_csv(df_out, path_salida)
    print(f"Edad media provincias {anio}: {len(df_out)} registros → '{path_salida}'.")
