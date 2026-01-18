"""
bin_drw.py

Density Ratio Weighted CI test for binary variables (MVPC).

Binary analogue of gauss_drw:
    - Model missingness via logistic regression on parents.
    - Use inverse-probability-like weights.
    - Run a weighted logistic regression LRT for CI.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


def _logistic_lrt_stat_weighted(x, y, S, data, weights):
    """
    Weighted logistic regression likelihood-ratio statistic
    for testing X ⟂ Y | S when Y is binary.
    """
    vars_idx = [x] + list(S)
    y_idx = y

    X_full = data[:, vars_idx]
    y_bin = data[:, y_idx]
    w = weights.copy()

    # Drop rows with missing values
    mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_bin)
    X_full = X_full[mask]
    y_bin = y_bin[mask]
    w = w[mask]

    if X_full.shape[0] < 10 or len(np.unique(y_bin)) < 2:
        return 0.0

    # Reduced model: without X
    if len(S) > 0:
        X_reduced = X_full[:, 1:]
    else:
        X_reduced = np.zeros((X_full.shape[0], 1))

    clf_full = LogisticRegression(max_iter=1000)
    clf_reduced = LogisticRegression(max_iter=1000)

    clf_full.fit(X_full, y_bin, sample_weight=w)
    clf_reduced.fit(X_reduced, y_bin, sample_weight=w)

    prob_full = clf_full.predict_proba(X_full)[:, 1]
    prob_reduced = clf_reduced.predict_proba(X_reduced)[:, 1]

    eps = 1e-9
    prob_full = np.clip(prob_full, eps, 1 - eps)
    prob_reduced = np.clip(prob_reduced, eps, 1 - eps)

    ll_full = np.sum(
        w * (y_bin * np.log(prob_full) + (1 - y_bin) * np.log(1 - prob_full))
    )
    ll_reduced = np.sum(
        w * (y_bin * np.log(prob_reduced) + (1 - y_bin) * np.log(1 - prob_reduced))
    )

    lrt = 2 * (ll_full - ll_reduced)
    return max(lrt, 0.0)


def _logistic_lrt_stat_unweighted(x, y, S, data):
    """
    Unweighted logistic regression LRT (fallback).
    """
    vars_idx = [x] + list(S)
    y_idx = y

    X_full = data[:, vars_idx]
    y_bin = data[:, y_idx]

    mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_bin)
    X_full = X_full[mask]
    y_bin = y_bin[mask]

    if X_full.shape[0] < 10 or len(np.unique(y_bin)) < 2:
        return 0.0

    if len(S) > 0:
        X_reduced = X_full[:, 1:]
    else:
        X_reduced = np.zeros((X_full.shape[0], 1))

    clf_full = LogisticRegression(max_iter=1000)
    clf_reduced = LogisticRegression(max_iter=1000)

    clf_full.fit(X_full, y_bin)
    clf_reduced.fit(X_reduced, y_bin)

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


def _compute_weights_for_variable(data, target, prt_m):
    """
    Compute DRW weights for a variable with missingness (binary case).
    """
    n, p = data.shape
    m_indicator = np.isnan(data[:, target]).astype(int)

    parents = prt_m["prt"].get(target, [])
    if len(parents) == 0:
        return np.ones(n)

    Z = data[:, parents]
    mask = ~np.isnan(Z).any(axis=1)
    Z_obs = Z[mask]
    R_obs = m_indicator[mask]

    if Z_obs.shape[0] < 10 or len(np.unique(R_obs)) < 2:
        return np.ones(n)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_obs, R_obs)
    prob = clf.predict_proba(Z_obs)[:, 1]

    eps = 1e-6
    prob = np.clip(prob, eps, 1 - eps)

    w_obs = 1.0 / (1.0 - prob)
    weights = np.ones(n)
    weights[mask] = w_obs

    return weights


def bin_ci_drw(x, y, S, data, prt_m):
    """
    DRW-corrected CI test for binary variables.

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

    Returns
    -------
    float
        DRW-corrected p-value.
    """
    # Ensure y is binary response; swap if needed
    y_vals = data[:, y]
    if len(np.unique(y_vals[~np.isnan(y_vals)])) != 2:
        x, y = y, x
        y_vals = data[:, y]

    missing_inds = prt_m["m"]

    # No missingness parents → standard LRT
    if x not in missing_inds and y not in missing_inds:
        stat = _logistic_lrt_stat_unweighted(x, y, S, data)
        df = 1
        return 1 - chi2.cdf(stat, df=df)

    # Build combined weights
    n = data.shape[0]
    weights = np.ones(n)

    if x in missing_inds:
        w_x = _compute_weights_for_variable(data, x, prt_m)
        weights *= w_x

    if y in missing_inds:
        w_y = _compute_weights_for_variable(data, y, prt_m)
        weights *= w_y

    stat = _logistic_lrt_stat_weighted(x, y, S, data, weights)
    df = 1
    p = 1 - chi2.cdf(stat, df=df)
    return p
