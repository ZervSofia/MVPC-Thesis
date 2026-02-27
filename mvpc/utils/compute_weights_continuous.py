"""
translation of compute.weights.continuous
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



# Index version of test-wise deletion
def indx_test_wise_deletion(var_ind, data):
    mask = np.ones(data.shape[0], dtype=bool)
    for v in var_ind:
        mask &= ~np.isnan(data[:, v])
    return np.where(mask)[0]



# 2. Identify missingness indicators for x,y,S,W 
def get_ind_r_xys(ind, suffstat):
    data = suffstat["data"]
    ind_r = []
    for i in ind:
        if np.isnan(data[:, i]).any():
            ind_r.append(i)
    return ind_r



def kde_weights(x_del, x_full):
    
    if len(x_full) < 5 or len(x_del) < 5:
        return np.ones_like(x_del)

    try:
        kde_full = gaussian_kde(x_full)
        kde_del = gaussian_kde(x_del)
    except Exception:
        # singular covariance (no correction)
        return np.ones_like(x_del)

    f_full = kde_full.evaluate(x_del)
    f_del = kde_del.evaluate(x_del)

    f_full = np.where(~np.isfinite(f_full), 0.0, f_full)
    f_del = np.where(~np.isfinite(f_del), 0.0, f_del)
    f_del = np.where(f_del <= 0, 1e-12, f_del)

    beta = f_full / f_del

    # non-finite ratios = 1 (no correction)
    beta = np.where(~np.isfinite(beta), 1.0, beta)

    return beta


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

    if (n_tw == 0 or
        not np.all(np.isfinite(weights)) or
        np.all(weights == 0)):
        weights = np.ones(n_tw)

    return weights, idx_tw

