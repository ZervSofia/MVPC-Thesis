"""
compute_weights_continuous.py

Faithful Python translation of R's compute.weights.continuous
used in gaussCItest.drw.

Implements:
    - indx_test_wise_deletion
    - get_ind_r_xys
    - kde_weights
    - compute_weights_continuous
"""

import numpy as np
from scipy.stats import gaussian_kde

from .mvpc_utils import get_prt_i


# ---------------------------------------------------------
# 1. Index version of test-wise deletion (R: indx_test_wise_deletion)
# ---------------------------------------------------------
def indx_test_wise_deletion(var_ind, data):
    mask = np.ones(data.shape[0], dtype=bool)
    for v in var_ind:
        mask &= ~np.isnan(data[:, v])
    return np.where(mask)[0]


# ---------------------------------------------------------
# 2. Identify missingness indicators among x,y,S,W (R: get_ind_r_xys)
# ---------------------------------------------------------
def get_ind_r_xys(ind, suffstat):
    data = suffstat["data"]
    ind_r = []
    for i in ind:
        if np.isnan(data[:, i]).any():
            ind_r.append(i)
    return ind_r


# ---------------------------------------------------------
# 3. KDE-based density ratio weights (R: kde.weights)
# ---------------------------------------------------------
# def kde_weights(x_del, x_full):
#     """
#     R version:
#         f_w  = density(x_full)
#         f_wr = density(x_del)
#         beta = f_w(x_del) / f_wr(x_del)
#     """
#     if len(x_full) < 5 or len(x_del) < 5:
#         return np.ones_like(x_del)

#     kde_full = gaussian_kde(x_full)
#     kde_del = gaussian_kde(x_del)

#     f_full = kde_full.evaluate(x_del)
#     f_del = kde_del.evaluate(x_del)

#     # Avoid division by zero
#     f_del = np.where(f_del == 0, 1e-12, f_del)

#     return f_full / f_del

#____________________________________________________________

def kde_weights(x_del, x_full):
    
    if len(x_full) < 5 or len(x_del) < 5:
        return np.ones_like(x_del)

    try:
        kde_full = gaussian_kde(x_full)
        kde_del = gaussian_kde(x_del)
    except Exception:
        # singular covariance, etc. → no correction
        return np.ones_like(x_del)

    f_full = kde_full.evaluate(x_del)
    f_del = kde_del.evaluate(x_del)

    # Guard against zeros and non-finite densities
    f_full = np.where(~np.isfinite(f_full), 0.0, f_full)
    f_del = np.where(~np.isfinite(f_del), 0.0, f_del)
    f_del = np.where(f_del <= 0, 1e-12, f_del)

    beta = f_full / f_del

    # Replace non-finite ratios with 1 (no correction)
    beta = np.where(~np.isfinite(beta), 1.0, beta)

    return beta

#_________________________________________________________


# ---------------------------------------------------------
# 4. Main function: compute continuous DRW weights
# ---------------------------------------------------------
# def compute_weights_continuous(corr_ind, suffstat):
#     """
#     Faithful translation of R's compute.weights.continuous.

#     Returns:
#         weights : np.ndarray of shape (n_tw,)
#         idx_tw  : np.ndarray of row indices used after test-wise deletion
#     """
#     data = suffstat["data"]

#     # Step 1: test-wise deletion indices for corr_ind
#     idx_tw = indx_test_wise_deletion(corr_ind, data)
#     n_tw = len(idx_tw)

#     weights = np.ones(n_tw)

#     # Step 2: identify missingness indicators among corr_ind
#     ind_r = get_ind_r_xys(corr_ind, suffstat)

#     # Step 3: for each missingness indicator, compute density ratio weights
#     for ind_ri in ind_r:
#         prt_i = get_prt_i(ind_ri, suffstat)

#         if len(prt_i) == 0:
#             continue

#         # R assumes a single parent here; keep it 1D
#         pa = data[:, prt_i]      # shape (n, 1) or (n,)
#         pa = pa.reshape(-1)      # flatten

#         pa_del = pa[idx_tw]
#         pa_full = pa[~np.isnan(pa)]

#         beta = kde_weights(pa_del, pa_full)
#         weights *= beta

#     return weights, idx_tw


#___________________________________________________________


def compute_weights_continuous(corr_ind, suffstat):
    data = suffstat["data"]

    idx_tw = indx_test_wise_deletion(corr_ind, data)
    n_tw = len(idx_tw)

    weights = np.ones(n_tw)

    ind_r = get_ind_r_xys(corr_ind, suffstat)

    for ind_ri in ind_r:
        prt_i = get_prt_i(ind_ri, suffstat)
        if len(prt_i) == 0:
            continue

        pa = data[:, prt_i].reshape(-1)

        pa_del = pa[idx_tw]
        pa_full = pa[~np.isnan(pa)]

        beta = kde_weights(pa_del, pa_full)
        weights *= beta

    # Final sanity check
    if (n_tw == 0 or
        not np.all(np.isfinite(weights)) or
        np.all(weights == 0)):
        weights = np.ones(n_tw)

    return weights, idx_tw




#____________________________________________________________