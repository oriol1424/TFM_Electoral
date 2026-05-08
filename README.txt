# TFM: Análisis y Simulación del Sistema Electoral Español

Este proyecto de Trabajo de Fin de Máster (TFM) tiene como objetivo analizar la relación entre indicadores socioeconómicos (demografía, renta, mercado laboral) y los resultados electorales en España, además de permitir la simulación del reparto de escaños.

## Estructura del Proyecto

- **main.ipynb**: Cuaderno principal que orquesta todo el flujo de trabajo (Limpieza -> EDA -> Simulación).
- **config_path.json**: Archivo de configuración que gestiona las rutas de los archivos raw y procesados para diferentes años (ej. 2019, 2023).
- **limpieza/**: Módulos de procesamiento y limpieza de datos (ETL).
  - `votos.py`: Procesa resultados electorales y asigna ideologías basadas en datos del CIS.
  - `rentas.py` / `renta_navarra.py`: Limpieza de datos de renta del INE.
  - `poblacion.py`: Gestión de censos y padrones.
  - `paro_contratos.py`: Datos del mercado laboral (SEPE).
- **EDA/**: Scripts para el Análisis Exploratorio de Datos. Genera visualizaciones y estadísticas sobre demografía, desigualdad y comportamiento de voto.
- **calculos_electorales/**:
  - `sistema_electoral.py`: Implementación del sistema electoral español (reparto provincial de escaños y aplicación de la Ley D'Hondt con barrera del 3%).
- **data_raw/**: Datos originales (fuentes: INE, Ministerio del Interior, SEPE, CIS).
- **data_processed/**: Datos limpios y estructurados en formato CSV/JSON tras ejecutar la limpieza.
- **documentation/**: Archivos de documentación (LaTeX).

## Requisitos y Configuración

El proyecto utiliza Python y las siguientes librerías principales:
- `pandas`, `numpy`: Manipulación de datos.
- `seaborn`, `matplotlib`: Visualización.
- `openpyxl`, `xlrd`: Lectura de archivos Excel.

Para asegurar el funcionamiento:
1. Verifica que las rutas en `config_path.json` coincidan con tu estructura local.
2. Los datos de votos deben estar en la carpeta `data_raw/votos/` siguiendo la estructura de ficheros .DAT del Ministerio del Interior.

## Flujo de Trabajo

1. **ETL (Limpieza)**: Ejecuta las funciones de limpieza en `main.ipynb` para transformar los datos brutos en datasets procesados.
2. **EDA**: Genera insights sobre la relación entre variables (ej. Gini vs Voto).
3. **Simulación**: El módulo `sistema_electoral` calcula el reparto de los 350 escaños del Congreso basándose en la población provincial y los votos obtenidos.

---