# """
# gauss_permc.py

# Faithful Python translation of the R function gaussCItest.permc
# from the MVPC paper implementation.

# Implements:
#     - gauss_ci_td    : deletion-based Gaussian CI test
#     - gauss_ci_permc : permutation-corrected Gaussian CI test (PermC)
# """

# import numpy as np
# from numpy.linalg import inv
# from scipy.stats import norm

# from ..utils.mvpc_utils import (
#     test_wise_deletion,
#     cond_PermC,
#     get_prt_m_xys,
#     perm,
# )


# # ---------------------------------------------------------
# # 1. Deletion-based Gaussian CI test (R: gaussCItest.td)
# # ---------------------------------------------------------
# def gauss_ci_td(x, y, S, suffstat):
#     data = suffstat["data"]
#     idx = [x, y] + list(S)

#     # 1) row-wise deletion
#     sub = test_wise_deletion(idx, data)

#     # 2) subset and reorder columns to [x, y, S]
#     sub = sub[:, idx]   # <-- THIS LINE IS CRITICAL
#     # print(f"x={x}, y={y}, S={list(S)}, n_after_deletion={sub.shape[0]}")
#     if sub.shape[0] < 5:
#         return 1.0

#     cov = np.cov(sub, rowvar=False)
#     try:
#         prec = inv(cov)
#     except np.linalg.LinAlgError:
#         return 1.0

#     r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
#     r_xy_S = np.clip(r_xy_S, -0.999999, 0.999999)

#     z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
#     n = sub.shape[0]
#     stat = np.sqrt(n - len(S) - 3) * abs(z)

#     return 2 * (1 - norm.cdf(stat))



# # ---------------------------------------------------------
# # 2. Permutation-corrected Gaussian CI test (R: gaussCItest.permc)
# # ---------------------------------------------------------
# def gauss_ci_permc(x, y, S, suffstat):
#     """
#     Faithful Python translation of R's gaussCItest.permc.

#     suffstat must contain:
#         - "data": np.ndarray
#         - "prt_m": missingness-parent structure
#         - "skel": initial skeleton adjacency (for cond_PermC)
#     """
#     data = suffstat["data"]

#     # Step 0: check whether correction is needed
#     if not cond_PermC(x, y, S, suffstat):
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 1: parents of missingness indicators of {x, y, S}
#     ind_test = [x, y] + list(S)
#     ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

#     if len(ind_W) == 0:
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 2: recursively add parents of W until closure
#     pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#     candi_W = list(set(pa_W) - set(ind_W))

#     while len(candi_W) > 0:
#         ind_W = list(set(ind_W) | set(candi_W))
#         pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#         candi_W = list(set(pa_W) - set(ind_W))

#     ind_W = list(set(ind_W))

#     # Step 3: build index set {x, y, S, W} and test-wise deletion
#     ind_permc = ind_test + ind_W

#     # this already returns only columns in ind_permc
#     data_tw = test_wise_deletion(ind_permc, data)
#     n_tw = data_tw.shape[0]

#     if n_tw < 5:
#         return gauss_ci_td(x, y, S, suffstat)

#     # Split into {x, y, S} and W
#     n_test = len(ind_test)
#     X_W = data_tw[:, n_test:]  # columns for W

#     if X_W.shape[1] == 0:
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 4: regress each of x, y, S on W (no intercept, as in R)
#     betas = []
#     residuals = []

#     for i in range(n_test):
#         y_i = data_tw[:, i]

#         try:
#             beta_i, *_ = np.linalg.lstsq(X_W, y_i, rcond=None)
#         except np.linalg.LinAlgError:
#             return gauss_ci_td(x, y, S, suffstat)

#         y_hat_i = X_W @ beta_i
#         res_i = y_i - y_hat_i

#         betas.append(beta_i)
#         residuals.append(res_i)

#     # Step 5: shuffle W using perm(W, data)
#     data_W_perm = perm(ind_W, data)  # already only W columns, rows permuted
#     data_W_perm = np.asarray(data_W_perm)

#     n_use = min(n_tw, data_W_perm.shape[0])
#     X_W_perm = data_W_perm[:n_use, :]

#     residuals_perm = [r[:n_use] for r in residuals]

#     # Step 6: generate virtual data for x, y, S
#     vir_cols = []
#     for beta_i, res_i in zip(betas, residuals_perm):
#         y_vir_i = X_W_perm @ beta_i + res_i
#         vir_cols.append(y_vir_i)

#     data_perm = np.column_stack(vir_cols)

#     # Step 7: run standard Gaussian CI test on virtual data
#     suff_perm = {"data": data_perm}

#     if len(ind_test) > 2:
#         S_perm = list(range(2, len(ind_test)))
#     else:
#         S_perm = []

#     return gauss_ci_td(0, 1, S_perm, suff_perm)




"""
gauss_permc.py

Faithful Python translation of the R function gaussCItest.permc
from the MVPC paper implementation.

Implements:
    - gauss_ci_td    : deletion-based Gaussian CI test
    - gauss_ci_permc : permutation-corrected Gaussian CI test (PermC)
"""

import numpy as np
from numpy.linalg import inv
from scipy.stats import norm

from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
    perm,
)

PERMC_FAIL = {
    "no_W": 0,
    "tw_too_small": 0,
    "no_W_columns": 0,
    "regression_fail": 0,
    "success": 0,
}


# ============================================================
# GLOBAL COUNTER TO TRACK PERMC USAGE
# ============================================================
PERMC_COUNTER = {
    "total_calls": 0,
    "used": 0,       # PermC actually executed
    "fallback": 0,   # fell back to TD (any reason)
}



# ---------------------------------------------------------
# 1. Deletion-based Gaussian CI test (R: gaussCItest.td)
# ---------------------------------------------------------
def gauss_ci_td(x, y, S, suffstat):
    data = suffstat["data"]
    idx = [x, y] + list(S)

    # 1) row-wise deletion
    sub = test_wise_deletion(idx, data)

    # 2) subset and reorder columns to [x, y, S]
    sub = sub[:, idx]
    if sub.shape[0] < 5:
        return 1.0

    cov = np.cov(sub, rowvar=False)
    try:
        prec = inv(cov)
    except np.linalg.LinAlgError:
        return 1.0

    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    r_xy_S = np.clip(r_xy_S, -0.999999, 0.999999)

    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    n = sub.shape[0]
    stat = np.sqrt(n - len(S) - 3) * abs(z)

    return 2 * (1 - norm.cdf(stat))



# ---------------------------------------------------------
# 2. Permutation-corrected Gaussian CI test (R: gaussCItest.permc)
# ---------------------------------------------------------
# def gauss_ci_permc(x, y, S, suffstat):
    
#     """
#     Faithful Python translation of R's gaussCItest.permc.

#     suffstat must contain:
#         - "data": np.ndarray
#         - "prt_m": missingness-parent structure
#         - "skel": initial skeleton adjacency (for cond_PermC)
#     """
#     data = suffstat["data"]

#     # Step 0: check whether correction is needed
#     if not cond_PermC(x, y, S, suffstat):
#         PERMC_COUNTER["fallback"] += 1
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 1: parents of missingness indicators of {x, y, S}
#     ind_test = [x, y] + list(S)
#     ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

#     if len(ind_W) == 0:
#         PERMC_COUNTER["fallback"] += 1
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 2: recursively add parents of W until closure
#     pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#     candi_W = list(set(pa_W) - set(ind_W))

#     while len(candi_W) > 0:
#         ind_W = list(set(ind_W) | set(candi_W))
#         pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
#         candi_W = list(set(pa_W) - set(ind_W))

#     ind_W = list(set(ind_W))

#     # Step 3: build index set {x, y, S, W} and test-wise deletion
#     ind_permc = ind_test + ind_W


#     data_tw = test_wise_deletion(ind_permc, data)
    
#     n_tw = data_tw.shape[0]

#     if n_tw < 5:
#         PERMC_COUNTER["fallback"] += 1
#         return gauss_ci_td(x, y, S, suffstat)

#     # Split into {x, y, S} and W
#     n_test = len(ind_test)
#     X_W = data_tw[:, n_test:]

#     if X_W.shape[1] == 0:
#         PERMC_COUNTER["fallback"] += 1
#         return gauss_ci_td(x, y, S, suffstat)

#     # Step 4: regress each of x, y, S on W (no intercept)
#     betas = []
#     residuals = []

#     for i in range(n_test):
#         y_i = data_tw[:, i]

#         try:
#             beta_i, *_ = np.linalg.lstsq(X_W, y_i, rcond=None)
#         except np.linalg.LinAlgError:
#             PERMC_COUNTER["fallback"] += 1
#             return gauss_ci_td(x, y, S, suffstat)

#         y_hat_i = X_W @ beta_i
#         res_i = y_i - y_hat_i

#         betas.append(beta_i)
#         residuals.append(res_i)

#     # Step 5: shuffle W using perm(W, data)
#     data_W_perm = perm(ind_W, data)
#     data_W_perm = np.asarray(data_W_perm)

#     n_use = min(n_tw, data_W_perm.shape[0])
#     X_W_perm = data_W_perm[:n_use, :]

#     residuals_perm = [r[:n_use] for r in residuals]

#     # Step 6: generate virtual data for x, y, S
#     vir_cols = []
#     for beta_i, res_i in zip(betas, residuals_perm):
#         y_vir_i = X_W_perm @ beta_i + res_i
#         vir_cols.append(y_vir_i)

#     data_perm = np.column_stack(vir_cols)

#     # Step 7: run standard Gaussian CI test on virtual data
#     suff_perm = {"data": data_perm}

#     if len(ind_test) > 2:
#         S_perm = list(range(2, len(ind_test)))
#     else:
#         S_perm = []

#     # If we reach here, PermC was actually used
#     PERMC_COUNTER["used"] += 1

#     return gauss_ci_td(0, 1, S_perm, suff_perm)



# Add this at the top of gauss_permc.py
def gauss_ci_permc(x, y, S, suffstat):
    PERMC_COUNTER["total_calls"] += 1
    data_full = suffstat["data"]

    # Step 0: check whether correction is needed
    if not cond_PermC(x, y, S, suffstat):
        PERMC_COUNTER["fallback"] += 1
        return gauss_ci_td(x, y, S, suffstat)

    # Step 1: parents of missingness indicators of {x, y, S}
    ind_test = [x, y] + list(S)
    ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))
    # match R: ind_W <- setdiff(ind_W, ind_test)
    ind_W = list(set(ind_W) - set(ind_test))

    if len(ind_W) == 0:
        PERMC_FAIL["no_W"] += 1
        PERMC_COUNTER["fallback"] += 1
        return gauss_ci_td(x, y, S, suffstat)

    # Step 2: recursively add parents of W until closure
    pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
    candi_W = list(set(pa_W) - set(ind_W))

    while len(candi_W) > 0:
        ind_W = list(set(ind_W) | set(candi_W))
        pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
        candi_W = list(set(pa_W) - set(ind_W))

    ind_W = list(set(ind_W))

    # Step 3: build index set {x, y, S, W} and test-wise deletion
    ind_permc = ind_test + ind_W
    data_tw = test_wise_deletion(ind_permc, data_full)
    data_tw = data_tw[:, ind_permc]
    n_tw = data_tw.shape[0]

    if n_tw < 5:
        PERMC_FAIL["tw_too_small"] += 1
        PERMC_COUNTER["fallback"] += 1
        return gauss_ci_td(x, y, S, suffstat)

    n_test = len(ind_test)
    X_W = data_tw[:, n_test:]
    if X_W.shape[1] == 0:
        PERMC_FAIL["no_W_columns"] += 1
        PERMC_COUNTER["fallback"] += 1
        return gauss_ci_td(x, y, S, suffstat)

    # Step 4: regress each of x, y, S on W (ridge)
    betas = []
    residuals = []

    XT_X = X_W.T @ X_W
    ridge = 1e-6 * np.eye(XT_X.shape[0])
    XT_X_reg = XT_X + ridge
    try:
        XT_X_reg_inv = np.linalg.inv(XT_X_reg)
    except Exception:
        PERMC_FAIL["regression_fail"] += 1
        PERMC_COUNTER["fallback"] += 1
        return gauss_ci_td(x, y, S, suffstat)

    XT = X_W.T
    for i in range(n_test):
        y_i = data_tw[:, i]
        beta_i = XT_X_reg_inv @ (XT @ y_i)
        y_hat_i = X_W @ beta_i
        res_i = y_i - y_hat_i
        betas.append(beta_i)
        residuals.append(res_i)

    # Step 5: shuffle W using the same logic as R's perm()
    data_W_perm = perm(ind_W, data_full)  # test-wise deletion on W, then permute rows
    # truncate to match n_tw
    data_W_perm = data_W_perm[:n_tw, :]

    # overwrite W columns in data_tw
    data_tw_perm = data_tw.copy()
    data_tw_perm[:, n_test:] = data_W_perm

    # Step 6: generate virtual data
    vir_cols = []
    for beta_i, res_i in zip(betas, residuals):
        y_vir_i = data_tw_perm[:, n_test:] @ beta_i + res_i
        vir_cols.append(y_vir_i)

    data_perm = np.column_stack(vir_cols)
    suff_perm = {"data": data_perm}

    if len(ind_test) > 2:
        S_perm = list(range(2, len(ind_test)))
    else:
        S_perm = []

    PERMC_FAIL["success"] += 1
    PERMC_COUNTER["used"] += 1
    return gauss_ci_td(0, 1, S_perm, suff_perm)

