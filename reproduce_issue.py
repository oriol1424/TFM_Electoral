import sys
import os
import pandas as pd

# Añadir el directorio actual al path para poder importar limpieza
sys.path.append(os.getcwd())

from limpieza.paro_contratos import limpiar_y_exportar_sepe

raw_path = 'data_raw/mercado_laboral/mercado_laboral_noviembre_2019.xls'
output_path = 'data_processed/mercado_laboral/mercado_laboral_2019_TEST.csv'

print(f"Procesando {raw_path}...")
df = limpiar_y_exportar_sepe(raw_path, output_path)

if df is not None:
    print("Éxito!")
    print(df.head())
else:
    print("Error: El DataFrame devuelto es None.")
