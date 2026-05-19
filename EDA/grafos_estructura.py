import os
import json
import geopandas as gpd
import pandas as pd
import libpysal
from libpysal.weights import Queen
import pickle
import warnings

warnings.filterwarnings('ignore')

def visualizar_grafo_maestro(gdf, w):
    """
    Función independiente para visualizar el grafo de adyacencia.
    Requiere el GeoDataFrame (gdf) y el objeto de pesos (w).
    """
    from EDA.visuals import plot_adjacency_graph
    print("Iniciando visualización del grafo...")
    plot_adjacency_graph(gdf, w)

def generar_grafo_adyacencia(anyo: int = 2019, force: bool = False):
    """
    Carga, unifica y calcula la adyacencia física de los municipios españoles.
    Filtra los resultados para que coincidan exactamente con el padrón de población del año indicado.
    Si los archivos ya existen en data_processed, los carga directamente (a menos que force=True).
    """
    output_dir = os.path.join('data_processed', 'geografia')
    os.makedirs(output_dir, exist_ok=True)
    
    out_parquet = os.path.join(output_dir, 'mapa_municipios.parquet')
    out_weights = os.path.join(output_dir, 'mapa_adyacencia.pickle')
    out_gal = os.path.join(output_dir, 'mapa_adyacencia.gal')

    if not force and os.path.exists(out_parquet) and os.path.exists(out_weights):
        print("Cargando grafo de adyacencia existente desde caché...")
        gdf_clean = gpd.read_parquet(out_parquet)
        with open(out_weights, 'rb') as f:
            w = pickle.load(f)
        
        print(f"Grafo cargado con {len(gdf_clean)} municipios.")
        return gdf_clean, w

    print(f"Iniciando creación del grafo filtrado por el padrón de {anyo}...")
    try:
        with open('config_path.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        path_pob = config[str(anyo)]["processed"]["poblacion_csv"]
        df_pob = pd.read_csv(path_pob, sep=';')
        
        ids_validos = set(df_pob['id_municipio'].astype(str).str.zfill(5).unique())
        print(f"-> Cargada lista maestra: {len(ids_validos)} municipios oficiales.")
    except Exception as e:
        print(f"Error al cargar el padrón para filtrar el grafo: {e}")
        return None, None

    path_peninbal = r'data_raw/lineas_limite/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp'
    path_canarias = r'data_raw/lineas_limite/recintos_municipales_inspire_canarias_regcan95/recintos_municipales_inspire_canarias_regcan95.shp'
    
    if not os.path.exists(path_peninbal) or not os.path.exists(path_canarias):
        print("Error: No se encuentran los archivos Shapefile originales en data_raw/lineas_limite/")
        return None, None

    print("1. Cargando y unificando capas geográficas (EPSG:4326)")
    gdf_peninbal = gpd.read_file(path_peninbal).to_crs(epsg=4326)
    gdf_canarias = gpd.read_file(path_canarias).to_crs(epsg=4326)
    
    gdf = pd.concat([gdf_peninbal, gdf_canarias], ignore_index=True)
    
    print("2. Limpiando códigos municipales (ID INE 5 dígitos)...")
    gdf['municipio'] = gdf['NATCODE'].str[-5:].str.zfill(5)
    
    print("3. Disolviendo geometrías (gestión de enclaves y multipartes)...")
    gdf_clean = gdf.dissolve(by='municipio').reset_index()
    gdf_clean = gdf_clean[['municipio', 'NAMEUNIT', 'geometry']]
    gdf_clean.rename(columns={'NAMEUNIT': 'nombre_oficial'}, inplace=True)

    print(f"4. Filtrando mapa: Eliminando recintos no oficiales o desaparecidos en {anyo}...")
    total_antes = len(gdf_clean)
    gdf_clean = gdf_clean[gdf_clean['municipio'].isin(ids_validos)].copy()
    print(f"   - Se han eliminado {total_antes - len(gdf_clean)} registros de la cartografía.")
    
    faltantes = ids_validos - set(gdf_clean['municipio'])
    if faltantes:
        print(f"   - Nota: {len(faltantes)} municipios del padrón no tienen geometría en los Shapefiles (ej. {list(faltantes)[:5]}...)")

    print("5. Calculando matriz de contigüidad Queen (vecindad física)...")
    w = Queen.from_dataframe(gdf_clean, idVariable='municipio')
    
    print(f"6. Guardando resultados en {output_dir}...")
    gdf_clean.to_parquet(out_parquet)
    with open(out_weights, 'wb') as f:
        pickle.dump(w, f)
            
    try:
        libpysal.io.open(out_gal, 'w').write(w)
    except Exception as e:
        print(f"Nota: No se pudo guardar el archivo GAL (error no crítico): {e}")
        
    print(f"Proceso de creación completado. Grafo final con {len(gdf_clean)} municipios.")

    return gdf_clean, w