import numpy as np
import pandas as pd


def knn_impute_group(
    df: pd.DataFrame,
    distance_cols: list,
    target_cols: list,
    n_neighbors: int = 7,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Imputa target_cols usando KNN espacial con ponderación por distancia inversa.

    Reference set: filas con todos los target_cols no nulos.
    Distance features: distance_cols, normalizados con media/std del reference set.

    Returns:
        df_out   : DataFrame con valores imputados
        null_mask: Serie booleana (True = municipio tenía al menos un nulo antes de imputar)
    """
    df_out = df.copy()

    ref_mask  = df[target_cols].notna().all(axis=1).values
    null_mask = df[target_cols].isna().any(axis=1).values

    if not null_mask.any():
        return df_out, pd.Series(null_mask, index=df.index)

    ref_dist  = df[distance_cols].values[ref_mask].astype(np.float64)
    null_dist = df[distance_cols].values[null_mask].astype(np.float64)
    ref_tgt   = df[target_cols].values[ref_mask].astype(np.float64)
    null_tgt  = df[target_cols].values[null_mask].astype(np.float64)  # contiene NaN

    mu    = ref_dist.mean(axis=0)
    sigma = ref_dist.std(axis=0)
    sigma[sigma == 0] = 1.0

    ref_dist_s  = (ref_dist  - mu) / sigma
    null_dist_s = (null_dist - mu) / sigma

    k = min(n_neighbors, len(ref_dist_s))
    n_null = null_dist_s.shape[0]
    null_tgt_out = null_tgt.copy()

    for bs in range(0, n_null, batch_size):
        be    = min(bs + batch_size, n_null)
        batch = null_dist_s[bs:be]

        diff  = batch[:, np.newaxis, :] - ref_dist_s[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]

        for b_i in range(be - bs):
            top_k = nn_idx[b_i]
            d     = dists[b_i, top_k]
            w     = 1.0 / (d + 1e-10)

            for j in range(len(target_cols)):
                if np.isnan(null_tgt[bs + b_i, j]):
                    null_tgt_out[bs + b_i, j] = np.average(ref_tgt[top_k, j], weights=w)

    null_df_idx = df.index[null_mask]
    for j, col in enumerate(target_cols):
        orig_null_in_subset = np.isnan(null_tgt[:, j])
        positions = null_df_idx[orig_null_in_subset]
        values    = null_tgt_out[orig_null_in_subset, j]
        df_out.loc[positions, col] = values

    return df_out, pd.Series(null_mask, index=df.index)
