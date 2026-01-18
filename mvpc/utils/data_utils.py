"""
data_utils.py

General data preprocessing utilities.
"""

import numpy as np

def replace_with_indicator(data, col):
    """
    Replace column 'col' with missingness indicator.
    """
    new_data = data.copy()
    new_data[:, col] = np.isnan(data[:, col]).astype(int)
    return new_data
