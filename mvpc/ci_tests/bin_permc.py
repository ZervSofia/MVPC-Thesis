"""
bin_permc.py

Permutation-corrected CI test for binary variables (MVPC).

This is the binary analogue of gauss_permc:
    - We use a logistic regression likelihood-ratio statistic
      as the CI test statistic.
    - We then apply permutation correction on the missingness
      indicator to obtain a corrected p-value.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


def _logistic_lrt_stat(x, y, S, data):
    """
    Logistic regression likelihood-ratio test statistic for
    testing X ⟂ Y | S when Y is binary.

    We model:
        Y ~ X + S   (full model)
        Y ~ S       (reduced model)

    LRT = 2 * (logL_full - logL_reduced)
    """
    vars_idx = [x] + list(S)
    y_idx = y

    X_full = data[:, vars_idx]
    y_bin = data[:, y_idx]

    # Drop rows with missing values
    mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_bin)
    X_full = X_full[mask]
    y_bin = y_bin[mask]

    if X_full.shape[0] < 10 or len(np.unique(y_bin)) < 2:
        return 0.0

    # Reduced model: without X
    if len(S) > 0:
        X_reduced = X_full[:, 1:]
    else:
        # Only intercept
        X_reduced = np.zeros((X_full.shape[0], 1))

    # Fit logistic models
    clf_full = LogisticRegression(max_iter=1000)
    clf_reduced = LogisticRegression(max_iter=1000)

    clf_full.fit(X_full, y_bin)
    clf_reduced.fit(X_reduced, y_bin)

    # Log-likelihoods
    prob_full = clf_full.predict_proba(X_full)[:, 1]
    prob_reduced = clf_reduced.predict_proba(X_reduced)[:, 1]

    eps = 1e-9
    prob_full = np.clip(prob_full, eps, 1 - eps)
    prob_reduced = np.clip(prob_reduced, eps, 1 - eps)

    ll_full = np.sum(
        y_bin * np.log(prob_full) + (1 - y_bin) * np.log(1 - prob_full)
    )
    ll_reduced = np.sum(
        y_bin * np.log(prob_reduced) + (1 - y_bin) * np.log(1 - prob_reduced)
    )

    lrt = 2 * (ll_full - ll_reduced)
    return max(lrt, 0.0)


def bin_ci_permc(x, y, S, data, prt_m, n_perm=200):
    """
    Permutation-corrected CI test for binary variables.

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

    # Decide which variable is binary "response"
    # Here we assume y is binary; if not, swap.
    y_vals = data[:, y]
    if len(np.unique(y_vals[~np.isnan(y_vals)])) != 2:
        x, y = y, x
        y_vals = data[:, y]

    # ---------------------------------------------------------
    # Step 1 — Observed statistic
    # ---------------------------------------------------------
    obs_stat = _logistic_lrt_stat(x, y, S, data)

    # ---------------------------------------------------------
    # Step 2 — If no missingness parents involved, use chi-square p
    # ---------------------------------------------------------
    missing_inds = prt_m["m"]
    if x not in missing_inds and y not in missing_inds:
        df = 1  # difference: one parameter for X
        p = 1 - chi2.cdf(obs_stat, df=df)
        return p

    # ---------------------------------------------------------
    # Step 3 — Permutation correction
    # ---------------------------------------------------------
    perm_stats = []

    # Choose which variable's missingness indicator to permute
    target = x if x in missing_inds else y
    m_indicator = np.isnan(data[:, target]).astype(int)

    for _ in range(n_perm):
        perm = np.random.permutation(m_indicator)

        data_perm = data.copy()
        data_perm[:, target] = np.where(perm == 1, np.nan, data[:, target])

        stat = _logistic_lrt_stat(x, y, S, data_perm)
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)

    p_corr = np.mean(perm_stats >= obs_stat)
    return p_corr
