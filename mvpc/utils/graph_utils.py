"""
graph_utils.py

Graph-related helper functions.
"""

import numpy as np

def empty_graph(p):
    G = np.ones((p, p), dtype=bool)
    np.fill_diagonal(G, False)
    return G
