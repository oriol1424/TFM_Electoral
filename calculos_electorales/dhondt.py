import json
import pandas as pd
from typing import Dict

from modelos.entrenamiento import TARGETS

# Slots que compiten en D'Hondt (excluye votos_otros — nunca gana escaños individualmente)
SLOTS_DHONDT = [t.replace('pct_', 'votos_') for t in TARGETS]


def aplicar_dhondt(votos_dict: Dict[str, float], n_escanos: int) -> Dict[str, int]:
    """
    Ley D'Hondt con barrera del 3% provincial.

    votos_dict : {slot: votos} — puede incluir votos_otros (se usa para calcular el umbral
                                 pero votos_otros no compite por escaños).
    n_escanos  : escaños a repartir en esta provincia.
    Devuelve   : {slot: escanos} solo para slots que ganan ≥ 1 escaño.
    """
    if n_escanos == 0 or not votos_dict:
        return {}

    total_votos = sum(votos_dict.values())
    if total_votos == 0:
        return {}

    umbral = total_votos * 0.03
    competidores = {
        slot: votos
        for slot, votos in votos_dict.items()
        if slot in SLOTS_DHONDT and votos >= umbral
    }

    if not competidores:
        return {}

    cocientes = [
        (votos / divisor, slot)
        for slot, votos in competidores.items()
        for divisor in range(1, n_escanos + 1)
    ]
    cocientes.sort(reverse=True)

    resultado = {slot: 0 for slot in competidores}
    for _, slot in cocientes[:n_escanos]:
        resultado[slot] += 1

    return {slot: esc for slot, esc in resultado.items() if esc > 0}


def escanos_por_provincia(ruta_json: str) -> Dict[str, int]:
    """
    Reparto constitucional de escaños (LOREG):
    - Ceuta (51) y Melilla (52): 1 escaño fijo cada una.
    - 50 provincias continentales: 2 mínimos + 248 proporcionales por cuota + resto mayor.
    - Total garantizado = 350.
    """
    with open(ruta_json, encoding='utf-8') as f:
        datos = json.load(f)

    provincias = datos['provincias']
    escanos: Dict[str, int] = {}
    pob_continental = 0

    for cod, info in provincias.items():
        if cod in ('51', '52'):
            escanos[cod] = 1
        else:
            escanos[cod] = 2
            pob_continental += info['total_provincial']

    cuota = pob_continental / 248
    proporcional: Dict[str, int] = {}
    restos: Dict[str, float] = {}

    for cod, info in provincias.items():
        if cod in ('51', '52'):
            continue
        pob = info['total_provincial']
        entero = int(pob // cuota)
        proporcional[cod] = entero
        restos[cod] = pob - entero * cuota

    sobrantes = 248 - sum(proporcional.values())
    for cod in sorted(restos, key=restos.get, reverse=True)[:sobrantes]:
        proporcional[cod] += 1

    for cod, esc in proporcional.items():
        escanos[cod] += esc

    total = sum(escanos.values())
    assert total == 350, f"Error crítico: el reparto suma {total} escaños en lugar de 350"
    return escanos


def agregar_votos_a_provincia(df: pd.DataFrame, cols_votos: list) -> pd.DataFrame:
    """
    Suma votos de nivel municipal a provincial.
    Deriva el código de provincia (2 dígitos) de la columna 'municipio'.
    """
    df_agg = df.copy()
    df_agg['cod_provincia'] = df_agg['municipio'].astype(str).str.zfill(5).str[:2]

    cols_presentes = [c for c in cols_votos if c in df_agg.columns]
    return (
        df_agg
        .groupby('cod_provincia')[cols_presentes]
        .sum()
        .reset_index()
    )


def dhondt_todas_provincias(
    df_votos_prov: pd.DataFrame,
    dict_escanos: Dict[str, int],
) -> Dict[str, int]:
    """
    Aplica D'Hondt en cada provincia y agrega los escaños a nivel nacional.

    df_votos_prov : DataFrame con cod_provincia + columnas votos_* (nivel provincial).
    dict_escanos  : {cod_provincia: n_escanos} — salida de escanos_por_provincia().
    Devuelve      : {slot: escanos_totales_nacionales}.
    """
    cols_votos = [c for c in df_votos_prov.columns if c.startswith('votos_')]
    escanos_nacionales: Dict[str, int] = {}

    for _, row in df_votos_prov.iterrows():
        cod = str(row['cod_provincia']).zfill(2)
        n_escanos = dict_escanos.get(cod, 0)
        if n_escanos == 0:
            continue

        votos_prov = {col: float(row[col]) for col in cols_votos}
        reparto = aplicar_dhondt(votos_prov, n_escanos)
        for slot, esc in reparto.items():
            escanos_nacionales[slot] = escanos_nacionales.get(slot, 0) + esc

    return escanos_nacionales
