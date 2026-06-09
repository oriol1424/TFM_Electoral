import pandas as pd
import os
from limpieza.funciones_genericas_limpieza import (
    normalizar_nombres_columnas,
    guardar_dataframe_csv,
    leer_archivo_csv,
    limpiar_valor_numerico
)

CONFIG_NOMBRES_NAVARRA = [
    "Distribución S80/S20=Distribución de la renta P80/P20:37677.csv",
    "Índice de Gini=Índice de Gini:37677.csv",
    "Renta neta media por unidad de consumo=Media de la renta por unidad de consumo:30824.csv",
    "Renta neta media por persona=Renta neta media por persona:30824.csv",
    "Renta neta media por hogar=Renta neta media por hogar:30824.csv"
]


def obtener_mapeo_nombres(csv_destino, lista_config=CONFIG_NOMBRES_NAVARRA):
    """Genera un dict para renombrar columnas a partir de la configuración de Navarra."""
    mapeo = {}
    for item in lista_config:
        try:
            if ":" in item:
                config_parte, csv_name = item.rsplit(":", 1)
                if csv_name.strip() == csv_destino:
                    if "=" in config_parte:
                        xlsx_name, correct_name = config_parte.split("=")
                        mapeo[xlsx_name.strip()] = correct_name.strip()
        except ValueError:
            continue
    return mapeo


def renombrar_columnas_navarra(df, csv_destino, lista_config=CONFIG_NOMBRES_NAVARRA):
    """Renombra columnas del DataFrame usando la configuración de Navarra."""
    mapeo = obtener_mapeo_nombres(csv_destino, lista_config)
    if mapeo:
        return df.rename(columns=mapeo)
    return df


def extraer_datos_navarra(path_xlsx):
    """Lee el Excel de Navarra e identifica dinámicamente las filas de cabecera."""
    df_raw = pd.read_excel(path_xlsx, header=None)

    try:
        header_idx = df_raw[df_raw[0].astype(str).str.contains("Código", case=False, na=False)].index[0]
        atributo_row_idx = df_raw[df_raw[0].astype(str).str.contains("Atributo", case=False, na=False)].index[0]
    except (IndexError, AttributeError):
        header_idx = 3
        atributo_row_idx = 1

    sex_row_idx = atributo_row_idx + 1

    indicators = df_raw.iloc[atributo_row_idx].ffill()
    sex = df_raw.iloc[sex_row_idx]
    df = df_raw.iloc[header_idx+1:].copy()

    relevant_cols = [0, 1, 2]
    for i in range(3, len(sex)):
        if str(sex[i]).strip().lower() == "ambos sexos":
            relevant_cols.append(i)

    df = df[relevant_cols]

    col_names = ["Código", "Municipio", "Subárea ETN"]
    for i in relevant_cols[3:]:
        col_names.append(indicators[i])

    df.columns = col_names

    def formatear_codigo(cod):
        cod_str = str(cod).split('.')[0].strip()
        if cod_str.isdigit():
            return "31" + cod_str.zfill(3)
        return cod_str

    df["Código"] = df["Código"].apply(formatear_codigo)

    df = df[df["Código"].astype(str).str.startswith("31")]
    df = df.dropna(subset=["Código"])

    for col in df.columns[3:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def generar_renta_con_navarra(año, output_dir="data_processed/renta/"):
    """Integra datos del INE con los de Navarra para un año y guarda el CSV."""
    print(f"Iniciando proceso de integración para el año {año}...")

    path_37677 = "data_raw/renta/37677.csv"
    path_30824 = "data_raw/renta/30824.csv"
    path_navarra = f"data_raw/renta/data_navarra_{año}.xlsx"

    if not os.path.exists(path_navarra):
        print(f"Error: No se encontró el archivo de Navarra: {path_navarra}")
        return

    def cargar_ine(path, year):
        df = leer_archivo_csv(path)
        df = normalizar_nombres_columnas(df)

        df['Código'] = df['Municipios'].str.extract(r'^(\d{5})')
        mask = (df['Periodo'] == year) & (df['Distritos'].isna()) & (df['Secciones'].isna())
        df_filtered = df[mask].copy()
        df_pivot = df_filtered.pivot(index='Código', columns='Indicadores de renta media', values='Total')

        for col in df_pivot.columns:
            df_pivot[col] = df_pivot[col].apply(limpiar_valor_numerico, to_nan=True)

        return df_pivot

    try:
        df_ine_gini = cargar_ine(path_37677, año)
        df_ine_renta = cargar_ine(path_30824, año)
        df_navarra_raw = extraer_datos_navarra(path_navarra)

        mapeo_gini = obtener_mapeo_nombres("37677.csv")
        mapeo_renta = obtener_mapeo_nombres("30824.csv")

        cols_nav_gini = [c for c in df_navarra_raw.columns if c in mapeo_gini]
        df_nav_gini = df_navarra_raw[['Código'] + cols_nav_gini].rename(columns=mapeo_gini).set_index('Código')

        cols_nav_renta = [c for c in df_navarra_raw.columns if c in mapeo_renta]
        df_nav_renta = df_navarra_raw[['Código'] + cols_nav_renta].rename(columns=mapeo_renta).set_index('Código')

        df_final_gini = pd.concat([df_ine_gini, df_nav_gini])
        df_final_gini = df_final_gini[~df_final_gini.index.duplicated(keep='last')]

        df_final_renta = pd.concat([df_ine_renta, df_nav_renta])
        df_final_renta = df_final_renta[~df_final_renta.index.duplicated(keep='last')]

        df_resultado = pd.merge(df_final_gini, df_final_renta, left_index=True, right_index=True, how='outer')

        output_path = os.path.join(output_dir, f"renta_con_navarra_{año}.csv")
        guardar_dataframe_csv(df_resultado.reset_index(), output_path)

        print(f"Éxito: Archivo '{os.path.basename(output_path)}' generado correctamente.")
        return df_resultado

    except Exception as e:
        print(f"Error durante el procesamiento: {str(e)}")
        return None


def integrar_datos_navarra(path_xlsx, csv_destino, lista_config=CONFIG_NOMBRES_NAVARRA):
    """Wrapper de compatibilidad: extrae y renombra datos de Navarra."""
    df = extraer_datos_navarra(path_xlsx)
    df = renombrar_columnas_navarra(df, csv_destino, lista_config)
    return df
