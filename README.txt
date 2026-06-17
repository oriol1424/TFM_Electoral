# TFM: Predicción Electoral Española mediante Indicadores Socioeconómicos

Trabajo de Fin de Máster que predice el reparto de los 350 escaños del Congreso de los
Diputados a partir de indicadores socioeconómicos municipales, sin recurrir a encuestas.
Entrenamiento con datos de las elecciones generales de 2019; validación contra los
resultados reales de 2023.

Hipótesis central: el voto es una respuesta a las condiciones materiales de vida.
La socioeconomía —renta, desigualdad, mercado laboral, demografía— deja una huella
estadística que permite inferir el comportamiento electoral a nivel municipal.

---

## Estructura del Proyecto

- **main.ipynb** — Orquestador principal. Ejecuta el flujo completo en orden:
  ETL → Imputación → EDA → Feature Engineering → ML → D'Hondt → Análisis.
- **config_path.json** — Centraliza todas las rutas de entrada y salida para cada
  año electoral. Nunca hay rutas hardcodeadas en el código fuente.
- **requirements.txt** — Dependencias del entorno (Python 3.11).

### Módulos

- **limpieza/** — ETL modularizado, un script por fuente de datos:
  - `votos.py`: resultados electorales del Ministerio del Interior (ficheros .DAT).
  - `rentas.py` / `renta_navarra.py`: indicadores de renta del INE y Nastat (Navarra).
  - `paro_contratos.py`: paro registrado y contratos del SEPE.
  - `poblacion.py`: padrón municipal del INE.
  - `superficie.py`: delimitaciones territoriales del IGN.
  - `edad_media.py`: edad media municipal del INE.
  - `historicos.py`: resultados de elecciones anteriores (2015, 2016).
  - `funciones_genericas_limpieza.py`: utilidades compartidas entre módulos.

- **imputacion/** — Imputación KNN espacial para el 23,2 % de municipios con datos
  ausentes por secreto estadístico del INE:
  - `orquestador.py`: punto de entrada del proceso de imputación.
  - `_knn.py`: algoritmo KNN con coordenadas geográficas, población y densidad.
  - `busqueda_valor_k.py`: selección de k óptimo por validación cruzada (método del codo).
  - `grupo_a.py`: imputación multivariante para variables de renta y desigualdad.
  - `grupo_b.py`: imputación composicional para fuentes de ingreso (salarios, pensiones…).
  - `calidad.py`: construcción de la variable `calidad_datos` (indicador ordinal 0–4).

- **EDA/** — Análisis exploratorio de datos y feature engineering:
  - Distribución demográfica y desequilibrio territorial.
  - Mercado laboral, renta, desigualdad y fuentes de ingresos.
  - Análisis del voto: candidaturas, participación, voto en blanco.
  - Feature engineering (variables derivadas) y selección final de 12 features.
  - Análisis de componentes principales (PCA) y correlaciones socioeconómicas.

- **modelos/** — Pipeline completo de Machine Learning:
  - `entrenamiento.py`: entrena 15 modelos XGBoost independientes (uno por partido).
  - `prediccion.py`: genera predicciones municipales y las convierte en votos absolutos.
  - `evaluacion.py`: métricas de rendimiento (MAE, R²) en train (2019) y test (2023).
  - `shap_analysis.py`: interpretabilidad mediante valores SHAP por partido.
  - `contrafactual.py`: experimento sin_cs (fusión Ciudadanos→PP para diagnóstico).
  - `v2.py`: modelo con variable `grupo_tamano` (rural/semiurbano/urbano).
  - `estabilidad_temporal.py`: análisis de data drift entre 2019 y 2023.
  - `comparacion_modelos.py`: evaluación comparativa de arquitecturas.
  - `alternativos/`: implementaciones de regresión bayesiana y modelo espacial (lag vecinal).
  - `tuning/`: búsqueda de hiperparámetros con RandomizedSearchCV.
  - `modelos_guardados/`, `modelos_espacial/`, `modelos_sin_cs/`: modelos persistidos en JSON.

- **analisis/** — Análisis post-ML sobre los resultados del modelo:
  - `bloques.py`: análisis de bloques ideológicos (derecha, izquierda, nacionalistas).
  - `descomposicion_r2.py`: descomposición de la varianza explicada (económica vs. territorial).
  - `validacion_perfiles.py`: validación de perfiles socioeconómicos por partido.

- **calculos_electorales/** — Motor electoral D'Hondt:
  - `dhondt.py`: algoritmo de reparto D'Hondt con barrera del 3 % provincial.
  - `sistema_electoral.py`: reparto constitucional de escaños por provincia.
  - `pipeline_dhondt.py`: orquestador (predicho vs. real, agrega a provincial y reparte).
  - `resultados.py`: generación de tablas comparativas predicho vs. real.
  - `visualizacion.py`: gráficos de escaños y error por partido.

### Datos

- **data_raw/** — Ficheros originales de cada fuente, sin modificar.
- **data_processed/** — Datasets limpios e integrados listos para análisis.
- **estado_del_arte/** — PDFs de referencias bibliográficas.

---

## Requisitos

Python 3.11. Instalar dependencias con:

    pip install -r requirements.txt

---

## Configuración

Antes de ejecutar, edita `config_path.json` para que las rutas apunten a la ubicación
local de los ficheros en `data_raw/`. La estructura del JSON diferencia rutas por año
electoral (claves "2019" y "2023").

Los IDs de municipio son strings de 5 dígitos con ceros a la izquierda (ej. "01001").
Ceuta (51) y Melilla (52) tienen tratamiento especial: 1 escaño fijo cada una.

---

## Flujo de Trabajo

1. **ETL** — `ETL_limpieza(year, config_path)` transforma los ficheros en bruto de cada
   fuente en un conjunto de DataFrames limpios e integrados a nivel municipal.
   Se ejecuta de forma independiente para 2019 y 2023.

2. **Imputación KNN** — Completa el 23,2 % de municipios con datos ausentes por secreto
   estadístico del INE. Selección de k=9 por validación cruzada. Tratamiento diferenciado
   por grupos de variables (renta/desigualdad y fuentes composicionales de ingreso).

3. **EDA y Feature Engineering** — Análisis exploratorio, construcción de variables
   derivadas y selección de las 12 features finales del modelo.

4. **ML — 15 modelos XGBoost** — Entrenamiento sobre los 8.131 municipios de 2019.
   Los modelos se persisten en disco para su reutilización sin reentrenar.

5. **Simulación D'Hondt** — Predicciones municipales → votos absolutos → agregación
   provincial → algoritmo D'Hondt → 350 escaños del Congreso.

6. **Análisis** — Bloques ideológicos, descomposición R² y validación de perfiles
   socioeconómicos por partido.

---

## Resultado

Modelo final: XGBoost Espacial. MAE = 6,53 escaños sobre los 350 del Congreso,
sin utilizar ningún dato de encuesta ni información de campaña.
