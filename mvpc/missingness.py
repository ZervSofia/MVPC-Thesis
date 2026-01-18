

"""
missingness.py

Implements:
    - get_m_ind: detect which variables contain missing values
    - get_prt_R_ind: identify parents of each missingness indicator
    - detection_prt_m: orchestrate missingness-parent detection

This module corresponds to MVPC Step 1.
"""

import numpy as np
from itertools import combinations


def get_m_ind(data):
    """
    Identify variables that contain missing values.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).

    Returns
    -------
    list[int]
        Indices of variables with at least one missing value.
    """
    return [j for j in range(data.shape[1]) if np.isnan(data[:, j]).any()]


def get_prt_R_ind(data, indep_test, alpha, R_ind):
    """
    Identify parents of the missingness indicator for variable R_ind.

    This function:
        1. Creates a binary missingness indicator for column R_ind.
        2. Replaces the original column with this indicator.
        3. Runs a restricted PC-style skeleton search:
            - Only tests edges involving the missingness indicator.
            - Uses the provided CI test (indep_test).
        4. Returns variables that remain dependent on the indicator.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix (n x p).
    indep_test : callable
        Function (x, y, S, data) -> p-value.
    alpha : float
        Significance threshold.
    R_ind : int
        Index of the variable whose missingness indicator is analyzed.

    Returns
    -------
    list[int]
        Parent variables of the missingness indicator.
    """
    n, p = data.shape

    # Step 1: Create missingness indicator
    m_indicator = np.isnan(data[:, R_ind]).astype(int)

    # Step 2: Replace column with indicator
    data_mod = data.copy()
    data_mod[:, R_ind] = m_indicator

    parents = []

    # Step 3: Only test edges involving R_ind
    for y in range(p):
        if y == R_ind:
            continue

        # All other variables except R_ind and y
        neighbors = [k for k in range(p) if k not in (R_ind, y)]

        independent = False

        # PC-style search: conditioning sets of increasing size
        for ord_size in range(len(neighbors) + 1):
            for S in combinations(neighbors, ord_size):
                pval = indep_test(R_ind, y, S, data_mod)

                if pval >= alpha:
                    independent = True
                    break

            if independent:
                break

        if not independent:
            parents.append(y)

    return parents


def detection_prt_m(data, indep_test, alpha, p):
    """
    Detect parents of all missingness indicators.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n x p).
    indep_test : callable
        Base CI test used for missingness-parent detection.
    alpha : float
        Significance threshold.
    p : int
        Number of variables.

    Returns
    -------
    dict
        {
            'm': list of missingness indicator indices,
            'prt': {R_ind: [parent indices]}
        }
    """
    m_inds = get_m_ind(data)
    prt = {}

    for R_ind in m_inds:
        prt[R_ind] = get_prt_R_ind(data, indep_test, alpha, R_ind)

    return {"m": m_inds, "prt": prt}
