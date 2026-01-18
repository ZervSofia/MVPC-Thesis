"""
gauss_permc.py

Permutation-corrected Gaussian CI test for MVPC.
This implements the PermC method described in the MVPC paper.

The idea:
    - Compute the observed Gaussian CI test statistic.
    - Permute the missingness indicator of the involved variable(s).
    - Recompute the statistic for each permutation.
    - Corrected p-value = proportion of permuted statistics >= observed statistic.
"""

import numpy as np
from scipy.stats import norm
from numpy.linalg import inv


def _gaussian_ci_stat(x, y, S, data):
    """
    Compute the Fisher-z statistic for Gaussian CI test.
    """
    vars = [x, y] + list(S)
    sub = data[:, vars]

    # Remove rows with missing values
    sub = sub[~np.isnan(sub).any(axis=1)]
    if sub.shape[0] < 5:
        return 0.0  # not enough data

    cov = np.cov(sub, rowvar=False)
    try:
        prec = inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    # Partial correlation formula
    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])

    # Fisher z-transform
    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    return abs(z)


def gauss_ci_permc(x, y, S, data, prt_m, n_perm=200):
    """
    Permutation-corrected Gaussian CI test.

    Parameters
    ----------
    x, y : int
        Variable indices.
    S : list[int]
        Conditioning set.
    data : np.ndarray
        Data matrix (n x p).
    prt_m : dict
        Missingness-parent structure.
    n_perm : int
        Number of permutations.

    Returns
    -------
    float
        Corrected p-value.
    """

    # ---------------------------------------------------------
    # Step 1 — Compute observed statistic
    # ---------------------------------------------------------
    obs_stat = _gaussian_ci_stat(x, y, S, data)

    # ---------------------------------------------------------
    # Step 2 — Determine if correction is needed
    # ---------------------------------------------------------
    # If neither x nor y has missingness parents, no correction
    missing_inds = prt_m["m"]
    if x not in missing_inds and y not in missing_inds:
        # Standard Gaussian CI p-value
        p = 2 * (1 - norm.cdf(obs_stat))
        return p

    # ---------------------------------------------------------
    # Step 3 — Permutation correction
    # ---------------------------------------------------------
    perm_stats = []

    # Identify which variable's missingness indicator to permute
    target = x if x in missing_inds else y
    m_indicator = np.isnan(data[:, target]).astype(int)

    for _ in range(n_perm):
        # Permute missingness indicator
        perm = np.random.permutation(m_indicator)

        # Create modified dataset
        data_perm = data.copy()
        data_perm[:, target] = np.where(perm == 1, np.nan, data[:, target])

        # Compute statistic under permutation
        stat = _gaussian_ci_stat(x, y, S, data_perm)
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)

    # ---------------------------------------------------------
    # Step 4 — Corrected p-value
    # ---------------------------------------------------------
    p_corr = np.mean(perm_stats >= obs_stat)
    return p_corr
