# TFM Electoral — Contexto para Claude Code

## Objetivo
Predecir el reparto de los 350 escaños del Congreso de los Diputados usando indicadores socioeconómicos (sin encuestas). Hipótesis: el voto es una respuesta a las condiciones de vida. Entrenamiento con datos de 2019, predicción de 2023.

## Pipeline

```
Fuentes brutas → ETL (limpieza/) → EDA (EDA/) → ML → Simulación D'Hondt → 350 escaños
```

## Estructura del proyecto

```
TFM_Electoral/
├── main.ipynb                        # Orquestador principal
├── config_path.json                  # Rutas de datos para 2019 y 2023
├── limpieza/                         # ETL modularizado
│   ├── votos.py                      # Procesa .DAT del Ministerio + asigna ideología CIS
│   ├── rentas.py / renta_navarra.py  # INE renta (Navarra tiene fuente propia)
│   ├── poblacion.py                  # Padrón municipal
│   ├── paro_contratos.py             # SEPE mercado laboral
│   └── superficie.py                 # IGN geografía
├── EDA/                              # Análisis exploratorio
│   ├── votos.py                      # Distribución y comportamiento del voto
│   ├── desigualdad_renta.py          # Gini, P80P20
│   ├── indicadores_renta.py          # Renta media, neta, bruta
│   ├── mercado_laboral.py            # Paro, contratos, temporalidad
│   ├── demografia_superficie.py      # Población y densidad
│   ├── grafos_estructura.py          # Teoría de grafos (efecto comarca)
│   └── visuals.py / funciones_generales.py
├── calculos_electorales/
│   └── sistema_electoral.py          # Reparto provincial + Ley D'Hondt (barrera 3%)
├── data_raw/                         # Datos originales sin tocar
│   ├── votos/                        # Ficheros .DAT del Ministerio del Interior
│   ├── renta/                        # CSVs del INE
│   ├── mercado_laboral/              # XLS del SEPE
│   ├── demografia/                   # XLSX del INE (padrón)
│   └── superficie/                   # CSV del IGN
├── data_processed/                   # Datos limpios (CSV/JSON/parquet)
│   ├── votos_ideologia/              # Votos con etiqueta ideológica
│   ├── renta/                        # Gini, indicadores, fuente ingresos
│   ├── mercado_laboral/
│   ├── demografia/
│   ├── geografia/                    # Mapa adyacencia (grafo), parquet municipios
│   └── data_end/                     # Dataset final fusionado por año
├── documentation/                    # Memoria en LaTeX
│   ├── main.tex                      # Documento principal
│   ├── 01_resumen.tex ... 10_anexos.tex
│   └── imagenes_EDA/                 # Gráficos generados por EDA/
└── estado_del_arte/                  # PDFs de referencias bibliográficas
```

## Estado actual del proyecto

| Fase | Estado |
|------|--------|
| ETL (`limpieza/`) | Completado |
| EDA (`EDA/`) | Completado — gráficos en `documentation/imagenes_EDA/` |
| Grafos (efecto comarca) | Implementado, en exploración |
| ML (predicción de votos) | **Pendiente** — siguiente fase |
| Simulación D'Hondt | Implementado y validado (assert 350 escaños) |
| Memoria LaTeX | En progreso — completado hasta EDA |

## Fuentes de datos

| Fuente | Qué contiene | Formato |
|--------|-------------|---------|
| Ministerio del Interior | Resultados electorales 2019/2023 | `.DAT` binario |
| INE | Renta disponible, Gini, P80P20, fuente ingresos, padrón | CSV / XLSX |
| SEPE | Paro y contratos por municipio | XLS |
| IGN | Superficie y geografía municipal | CSV |
| CIS | Ideología de partidos (escala izquierda-derecha) | XLSX |

## Convenciones clave

- `config_path.json` centraliza todas las rutas — nunca hardcodear rutas en el código.
- Los IDs de municipio son strings de 5 dígitos con ceros a la izquierda (ej. `"01001"`).
- Navarra siempre tiene tratamiento especial (datos de renta propios).
- Ceuta (51) y Melilla (52) tienen 1 escaño fijo; el resto se reparte por cuota + resto mayor.
- La barrera D'Hondt es el 3% de votos válidos **a nivel provincial** (no nacional).

## Módulo electoral (`sistema_electoral.py`)

Tres funciones:
1. `calcular_y_guardar_escanos_json()` — reparto constitucional de escaños por provincia
2. `agrupar_votos_csv()` — agrega votos municipales a nivel provincial
3. `sistema_electoral()` — orquestadora: llama a las dos anteriores y aplica D'Hondt provincia a provincia

## ML — próxima fase

- **Modelos objetivo**: XGBoost / LightGBM
- **Target**: % de voto por partido a nivel municipal
- **Features**: renta, Gini, paro, temporalidad, densidad, edad media, fuente ingresos, efecto comarca (grafos)
- **Validación**: entrenar en 2019, evaluar contra resultados reales de 2023

## Entorno

- Python 3.11
- Librerías principales: `pandas`, `numpy`, `seaborn`, `matplotlib`, `openpyxl`, `xlrd`
- Pendiente instalar: `xgboost`, `lightgbm`, `scikit-learn`
- Jupyter Notebook como entorno de ejecución principal (`main.ipynb`)
