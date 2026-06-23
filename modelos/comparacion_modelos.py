import pandas as pd

from modelos.entrenamiento import entrenar_modelos, cargar_modelos
from modelos.evaluacion import evaluar_modelos, comparar_metricas_modelos, comparar_dhondt_modelos
from modelos.alternativos.bayesiano import (
    entrenar_modelos_bayesiano, cargar_modelos_bayesiano,
    preparar_features_bayesiano, pipeline_prediccion_bayesiano,
)
from modelos.alternativos.espacial import (
    entrenar_modelos_espacial, cargar_modelos_espacial,
    cargar_pesos_espaciales, preparar_features_espacial,
    pipeline_prediccion_espacial,
)


def pipeline_comparacion_modelos(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    w=None,
    guardar: bool = True,
    sin_cs: bool = False,
) -> pd.DataFrame:
    """
    Entrena XGBoost, Bayesian Ridge y XGBoost espacial sobre df_train.
    Evalúa todos contra df_test y devuelve tabla comparativa de MAE y R2 por partido.

    sin_cs=True: aplica la redistribución CS→PP y PRC→PSOE antes de entrenar.
    Los modelos se guardan en carpetas con sufijo _sin_cs para no sobreescribir los base.
    """
    from modelos.contrafactual import preparar_dataset_sin_cs

    if w is None:
        w = cargar_pesos_espaciales()

    sufijo = '_sin_cs' if sin_cs else ''
    etiqueta = ' (sin CS+PRC)' if sin_cs else ''
    df_tr = preparar_dataset_sin_cs(df_train) if sin_cs else df_train


    modelos_xgb = entrenar_modelos(df_tr, guardar=guardar,
                                   carpeta=f'modelos/modelos_guardados{sufijo}')

    modelos_bayes = entrenar_modelos_bayesiano(df_tr, guardar=guardar,
                                               carpeta=f'modelos/modelos_bayesiano{sufijo}')

    modelos_esp = entrenar_modelos_espacial(df_tr, w=w, guardar=guardar,
                                            carpeta=f'modelos/modelos_espacial{sufijo}')

    print(f"\nEVALUACIÓN TEST (2023){etiqueta}\n")
    met_xgb   = evaluar_modelos(modelos_xgb,   df_test, etiqueta=f"XGBoost{etiqueta}")
    X_test_bayes = preparar_features_bayesiano(df_test, carpeta=f'modelos/modelos_bayesiano{sufijo}')[0]
    met_bayes = evaluar_modelos(modelos_bayes,  df_test, etiqueta=f"Bayesiano{etiqueta}", X_test=X_test_bayes)

    X_test_esp = preparar_features_espacial(df_test, w)
    met_esp = evaluar_modelos(modelos_esp, df_test, etiqueta=f"Espacial{etiqueta}",
                              X_test=X_test_esp)

    df_cmp = comparar_metricas_modelos(
        {
            f'xgboost{sufijo}':   met_xgb,
            f'bayesiano{sufijo}': met_bayes,
            f'espacial{sufijo}':  met_esp,
        },
        guardar=guardar,
    )

    return df_cmp


def pipeline_escanos_todos_modelos(
    modelos_dict,
    df_2023,
    w,
    ruta_json_pob,
    esc_real=None,
):
    """
    Aplica D'Hondt a las predicciones de cada modelo y devuelve tabla comparativa.
    Filas = partidos, columnas = real + cada modelo.

    modelos_dict: {'xgboost': modelos_xgb, 'bayesiano': modelos_bayes, 'espacial': modelos_esp}
    esc_real: dict {partido: escanos} del resultado oficial (opcional).

    Uso en main.ipynb:
        df_escanos = pipeline_escanos_todos_modelos(
            {'xgboost': modelos, 'bayesiano': modelos_bayes, 'espacial': modelos_esp},
            df_2023_completo, w, ruta_json_pob, esc_real
        )
    """
    from modelos.prediccion import pipeline_prediccion
    from calculos_electorales.dhondt import escanos_por_provincia, dhondt_todas_provincias
    from calculos_electorales.resultados import votos_predichos_por_provincia

    dict_esc = escanos_por_provincia(ruta_json_pob)
    escanos_modelos = {}

    for nombre, modelos in modelos_dict.items():
        print(f"\nD'Hondt — {nombre}...")
        if 'espacial' in nombre:
            df_pred = pipeline_prediccion_espacial(modelos, df_2023, w)
        elif 'v2' in nombre:
            from modelos.v2 import pipeline_prediccion_v2
            df_pred = pipeline_prediccion_v2(modelos, df_2023)
        elif 'bayesiano' in nombre:
            carpeta_bayes = f'modelos/modelos_{nombre}'
            df_pred = pipeline_prediccion_bayesiano(modelos, df_2023, carpeta_bayes)
        else:
            df_pred = pipeline_prediccion(modelos, df_2023)

        esc_raw = dhondt_todas_provincias(votos_predichos_por_provincia(df_pred), dict_esc)
        escanos_modelos[nombre] = {k.replace('votos_', ''): v for k, v in esc_raw.items()}

    if esc_real is None:
        df_tabla = pd.DataFrame(escanos_modelos).fillna(0).astype(int)
        df_tabla.index.name = 'partido'
        return df_tabla.reset_index().sort_values(list(escanos_modelos.keys())[0], ascending=False)

    return comparar_dhondt_modelos(esc_real, escanos_modelos)
