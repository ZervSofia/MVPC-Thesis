"""
missingness.py

Functions for:
    - detecting missingness indicators
    - identifying parents of missingness indicators
    - preparing prt_m structure for MVPC
"""

import numpy as np
from itertools import combinations


def get_m_ind(data):
    """
    Return indices of variables that contain missing values.
    """
    return [j for j in range(data.shape[1]) if np.isnan(data[:, j]).any()]


def get_prt_R_ind(data, indep_test, alpha, R_ind):
    """
    Identify parents of the missingness indicator for variable R_ind.
    """
    # Placeholder — full implementation will be added later
    raise NotImplementedError("get_prt_R_ind is not implemented yet.")


def detection_prt_m(data, indep_test, alpha, p):
    """
    Detect parents of all missingness indicators.
    Returns a dictionary:
        {
            'm': [list of missingness indicator indices],
            'prt': {R_ind: [parent indices]}
        }
    """
    m_inds = get_m_ind(data)
    prt = {}

    for R_ind in m_inds:
        prt[R_ind] = get_prt_R_ind(data, indep_test, alpha, R_ind)

    return {"m": m_inds, "prt": prt}
