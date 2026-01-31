# """
# gauss_drw.py

# Faithful Python translation of R's gaussCItest.drw.
# Implements the DRW (density ratio weighting) correction for Gaussian CI tests.
# """

# import numpy as np
# from numpy.linalg import inv
# from scipy.stats import norm

# from ..utils.mvpc_utils import (
#     test_wise_deletion,
#     cond_PermC,
#     get_prt_m_xys,
# )
# from ..utils.compute_weights_continuous import compute_weights_continuous
# from .gauss_permc import gauss_ci_td

# DRW_COUNTER = {"used": 0, "fallback": 0}

# # ---------------------------------------------------------
# # 1. Weighted covariance (faithful to R's wtd.cors)
# # ---------------------------------------------------------
# def weighted_cov(X, weights):
#     """
#     Compute weighted covariance matrix.
#     Equivalent to R's wtd.cors(X, X, weights).
#     """
#     w = np.asarray(weights)
#     w = w / np.sum(w)

#     mean_w = np.sum(X * w[:, None], axis=0)
#     X_centered = X - mean_w

#     cov_w = (X_centered * w[:, None]).T @ X_centered
    
#     #___________________________________________________
#     # --- Stabilization: ridge --- 
#     cov_w += 1e-6 * np.eye(cov_w.shape[0])
    
#     #______________________________________________________
#     return cov_w


# # ---------------------------------------------------------
# # 2. Weighted Gaussian CI test (R: gaussCItest with weighted C)
# # ---------------------------------------------------------
# def gauss_ci_weighted(x, y, S, C, n_eff):
#     """
#     Gaussian CI test using a weighted covariance matrix C.
#     Mirrors R's gaussCItest but uses weighted covariance.
#     """
    
#     #___________________________________________
#     if not np.all(np.isfinite(C)):
#         return 1.0
#     #______________________________________________
    
    
#     idx = [x, y] + list(S)
#     C_sub = C[np.ix_(idx, idx)]
    
#     #__________________________________________  
#     if not np.all(np.isfinite(C_sub)):
#         return 1.0
#     #______________________________________________

#     try:
#         prec = inv(C_sub)
#     except np.linalg.LinAlgError:
#         return 1.0

#     r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
#     r_xy_S = np.clip(r_xy_S, -0.999999, 0.999999)

#     z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
#     stat = np.sqrt(n_eff - len(S) - 3) * abs(z)

#     return 2 * (1 - norm.cdf(stat))


# # ---------------------------------------------------------
# # 3. DRW-corrected Gaussian CI test (R: gaussCItest.drw)
# # ---------------------------------------------------------
# def gauss_ci_drw(x, y, S, suffstat):
#     data = suffstat["data"]

#     # Step 1: check if correction is needed
#     if not cond_PermC(x, y, S, suffstat):
#         from .gauss_permc import gauss_ci_td
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 2: parents of missingness indicators of {x, y, S}
#     ind_test = [x, y] + list(S)
#     ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

#     if len(ind_W) == 0:
#         from .gauss_permc import gauss_ci_td
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 3: recursively add parents of W until closure
#     pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#     candi_W = list(set(pa_W) - set(ind_W))

#     while len(candi_W) > 0:
#         ind_W = list(set(ind_W) | set(candi_W))
#         pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#         candi_W = list(set(pa_W) - set(ind_W))

#     ind_W = list(set(ind_W))

#     # Step 4: indices for DRW: corr_ind = {x, y, S, W}
#     corr_ind = ind_test + ind_W

#     # Step 5: compute DRW weights and row indices after deletion
#     weights_tw, idx_tw = compute_weights_continuous(corr_ind, suffstat)
    
#     # Step 5: compute DRW weights and row indices after deletion
#     #____________________________________________________________________________


#     weights_tw = np.asarray(weights_tw, dtype=float)

#     if (weights_tw.size == 0 or
#         not np.all(np.isfinite(weights_tw)) or
#         np.all(weights_tw == 0)):
#         # DRW unreliable → fall back to TD
#         return gauss_ci_td(x, y, S, suffstat)

#     # Stabilization: clip and normalize
#     weights_tw = np.clip(weights_tw, 1e-6, 1e6)
#     weights_tw = weights_tw / np.mean(weights_tw)

#     #____________________________________________________________________________


#     # Step 6: test-wise deletion on corr_ind using the same rows
#     data_tw = data[idx_tw, :]
#     X_corr = data_tw[:, corr_ind]

#     # Step 7: weighted covariance
#     C_w = weighted_cov(X_corr, weights_tw)
#     n_eff = np.sum(weights_tw)

#     # Step 8: Gaussian CI test using weighted covariance
#     # In C_w, variables are ordered as [x, y, S, W] -> x=0, y=1, S=2..(1+|S|)
#     S_local = list(range(2, 2 + len(S)))
    
    
#     #_______________________________________________________
    
#     from .gauss_permc import gauss_ci_td
    
#     try: 
#         return gauss_ci_weighted(0, 1, S_local, C_w, n_eff)
      
#     except Exception:
    
    
#         DRW_COUNTER["used"] += 1
#     #___________________________________________________________
#         return gauss_ci_weighted(0, 1, S_local, C_w, n_eff)   ### added tab
    
    
    
    
"""
gauss_drw.py

Faithful Python translation of R's gaussCItest.drw.
Implements the DRW (density ratio weighting) correction for Gaussian CI tests.
"""

import numpy as np
from numpy.linalg import inv
from scipy.stats import norm

from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
)
from ..utils.compute_weights_continuous import compute_weights_continuous

# =========================================================
# GLOBAL COUNTER TO TRACK DRW USAGE
# =========================================================
DRW_COUNTER = {"used": 0, "fallback": 0}


# ---------------------------------------------------------
# 1. Weighted covariance (faithful to R's wtd.cors)
# ---------------------------------------------------------
def weighted_cov(X, weights):
    w = np.asarray(weights)
    w = w / np.sum(w)

    mean_w = np.sum(X * w[:, None], axis=0)
    X_centered = X - mean_w

    cov_w = (X_centered * w[:, None]).T @ X_centered
    cov_w += 1e-6 * np.eye(cov_w.shape[0])  # ridge stabilization

    return cov_w


# ---------------------------------------------------------
# 2. Weighted Gaussian CI test
# ---------------------------------------------------------
def gauss_ci_weighted(x, y, S, C, n_eff):

    # Reject non-finite covariance
    if not np.all(np.isfinite(C)):
        return 1.0

    idx = [x, y] + list(S)
    C_sub = C[np.ix_(idx, idx)]

    if not np.all(np.isfinite(C_sub)):
        return 1.0

    try:
        prec = inv(C_sub)
    except (np.linalg.LinAlgError, ValueError):
        return 1.0

    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    r_xy_S = np.clip(r_xy_S, -0.999999, 0.999999)

    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    stat = np.sqrt(n_eff - len(S) - 3) * abs(z)

    return 2 * (1 - norm.cdf(stat))


# ---------------------------------------------------------
# 3. DRW-corrected Gaussian CI test
# ---------------------------------------------------------
def gauss_ci_drw(x, y, S, suffstat):
    global DRW_COUNTER

    data = suffstat["data"]

    # Step 1: check if correction is needed
    if not cond_PermC(x, y, S, suffstat):
        DRW_COUNTER["fallback"] += 1
        from .gauss_permc import gauss_ci_td
        return gauss_ci_td(x, y, S, suffstat)

    # Step 2: parents of missingness indicators of {x, y, S}
    ind_test = [x, y] + list(S)
    ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

    if len(ind_W) == 0:
        DRW_COUNTER["fallback"] += 1
        from .gauss_permc import gauss_ci_td
        return gauss_ci_td(x, y, S, suffstat)

    # Step 3: recursively add parents of W until closure
    pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
    candi_W = list(set(pa_W) - set(ind_W))

    while len(candi_W) > 0:
        ind_W = list(set(ind_W) | set(candi_W))
        pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
        candi_W = list(set(pa_W) - set(ind_W))

    ind_W = list(set(ind_W))

    # Step 4: indices for DRW: corr_ind = {x, y, S, W}
    corr_ind = ind_test + ind_W

    # Step 5: compute DRW weights
    weights_tw, idx_tw = compute_weights_continuous(corr_ind, suffstat)
    weights_tw = np.asarray(weights_tw, dtype=float)

    # If weights are unusable → fallback
    if (weights_tw.size == 0 or
        not np.all(np.isfinite(weights_tw)) or
        np.all(weights_tw == 0)):
        DRW_COUNTER["fallback"] += 1
        from .gauss_permc import gauss_ci_td
        return gauss_ci_td(x, y, S, suffstat)

    # Stabilize weights
    weights_tw = np.clip(weights_tw, 1e-6, 1e6)
    weights_tw = weights_tw / np.mean(weights_tw)

    # Step 6: test-wise deletion
    data_tw = data[idx_tw, :]
    X_corr = data_tw[:, corr_ind]

    # Step 7: weighted covariance
    C_w = weighted_cov(X_corr, weights_tw)
    n_eff = np.sum(weights_tw)

    # Step 8: weighted CI test
    S_local = list(range(2, 2 + len(S)))

    # If we reach here → DRW was actually used
    DRW_COUNTER["used"] += 1

    return gauss_ci_weighted(0, 1, S_local, C_w, n_eff)
