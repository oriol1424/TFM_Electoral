import os
import json
import geopandas as gpd
import pandas as pd
import libpysal
from libpysal.weights import Queen
import pickle
import warnings

# Silenciar warnings de geometría si es necesario
warnings.filterwarnings('ignore')

def generar_grafo_adyacencia():
    """
    Carga, unifica y calcula la adyacencia física de los municipios españoles.
    Si los archivos ya existen en data_processed, los carga directamente.
    Al finalizar, llama a la visualización en EDA/visuals.py.
    """
    # Rutas de salida
    output_dir = os.path.join('data_processed', 'geografia')
    os.makedirs(output_dir, exist_ok=True)
    
    out_parquet = os.path.join(output_dir, 'mapa_municipios.parquet')
    out_weights = os.path.join(output_dir, 'mapa_adyacencia.pickle')
    out_gal = os.path.join(output_dir, 'mapa_adyacencia.gal')

    # 1. Comprobar si ya existen los archivos procesados
    if os.path.exists(out_parquet) and os.path.exists(out_weights):
        print("Cargando grafo de adyacencia existente desde caché...")
        gdf_clean = gpd.read_parquet(out_parquet)
        with open(out_weights, 'rb') as f:
            w = pickle.load(f)
        
        print(f"Grafo cargado con {len(gdf_clean)} municipios.")
    else:
        # 2. Proceso de creación si no existen
        path_peninbal = r'data_raw/lineas_limite/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp'
        path_canarias = r'data_raw/lineas_limite/recintos_municipales_inspire_canarias_regcan95/recintos_municipales_inspire_canarias_regcan95.shp'
        
        if not os.path.exists(path_peninbal) or not os.path.exists(path_canarias):
            print("Error: No se encuentran los archivos Shapefile originales en data_raw/lineas_limite/")
            return

        print("1. Cargando y unificando capas geográficas (EPSG:4326)...")
        gdf_peninbal = gpd.read_file(path_peninbal).to_crs(epsg=4326)
        gdf_canarias = gpd.read_file(path_canarias).to_crs(epsg=4326)
        
        gdf = pd.concat([gdf_peninbal, gdf_canarias], ignore_index=True)
        
        print("2. Limpiando códigos municipales (ID INE 5 dígitos)...")
        # Aseguramos que es string y rellenamos a 5 dígitos
        gdf['municipio'] = gdf['NATCODE'].str[-5:].str.zfill(5)
        
        print("3. Disolviendo geometrías multiparte...")
        gdf_clean = gdf.dissolve(by='municipio').reset_index()
        gdf_clean = gdf_clean[['municipio', 'NAMEUNIT', 'geometry']]
        gdf_clean.rename(columns={'NAMEUNIT': 'nombre_oficial'}, inplace=True)

        print("4. Calculando matriz de contigüidad Queen...")
        # Queen considera adyacencia por bordes y vértices
        w = Queen.from_dataframe(gdf_clean, idVariable='municipio')
        
        # Guardado
        print(f"5. Guardando resultados en {output_dir}...")
        gdf_clean.to_parquet(out_parquet)
        
        # Guardar como Pickle (para Python)
        with open(out_weights, 'wb') as f:
            pickle.dump(w, f)
            
        # Guardar como GAL (Formato estándar usando el IO de libpysal)
        try:
            libpysal.io.open(out_gal, 'w').write(w)
        except Exception as e:
            print(f"Nota: No se pudo guardar el archivo GAL (error no crítico): {e}")
            
        print("Proceso de creación completado.")

    # 3. Llamada automática a la visualización
    print("6. Generando visualización del grafo...")
    from EDA.visuals import plot_adjacency_graph
    plot_adjacency_graph(gdf_clean, w)

    return gdf_clean, w

if __name__ == "__main__":
    generar_grafo_adyacencia()
