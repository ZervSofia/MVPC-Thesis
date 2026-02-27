"""
Corrected skeleton search for MVPC.
Python version of the R function skeleton2()

Steps:
    - Start from the initial PC skeleton (skel_pre)
    - Re-test edges using the corrected CI test (corr_test)
    - Remove edges that become independent after correction
    - Return corrected adjacency matrix, separation sets, and pmax matrix
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm



def skeleton2(data, corr_test, alpha, skel_pre, prt_m):
    """
    Perform corrected skeleton search using MVPC logic.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n x p).
    corr_test : callable
        Corrected CI test (DRW or PermC).
        Signature: corr_test(x, y, S, data, prt_m)
    alpha : float
        Significance threshold.
    skel_pre : object
        Output of causal-learn's PC algorithm.
        We use skel_pre.G to initialize the adjacency matrix.
    prt_m : dict
        Missingness-parent structure from detection_prt_m().

    Returns
    -------
    G : np.ndarray
        Corrected adjacency matrix (p x p).
    sepset : list[list]
        Separation sets for each pair (x, y).
    pmax : np.ndarray
        Matrix of maximum p-values observed for each pair.
    """

    
    # extract initial skeleton from causal-learn
    G = skel_pre.G.copy()
    p = G.shape[0]

    # Initialize sepset and pmax
    sepset = [[None for _ in range(p)] for _ in range(p)]
    pmax = np.full((p, p), -np.inf)


    # PC-style iterative search over conditioning set sizes
    ord_size = 0
    

    while True:
        done = True

        # List all current edges
        edges = np.argwhere(G)
        degrees = G.sum(axis=0) 
        max_deg = degrees.max()
        
        if ord_size > max_deg: 
            break

        for x, y in tqdm(edges, desc=f"Corrected skeleton, ord={ord_size}", leave=False):
            if x >= y:
                continue  

            # Neighbors of x excluding y
            neighbors = [k for k in range(p) if G[k, x] and k != y]

            if len(neighbors) < ord_size:
                continue

            if len(neighbors) > ord_size:
                done = False

            independent = False

            
            # Test all conditioning sets of size ord_size
            for S in combinations(neighbors, ord_size):
                suffstat = {
                    "data": data,
                    "prt_m": prt_m,
                    "skel": G,
                }
                pval = corr_test(x, y, S, suffstat)


                # Track maximum p-value
                pmax[x, y] = max(pmax[x, y], pval)
                pmax[y, x] = pmax[x, y]

                if pval >= alpha:
                    # Remove edge
                    G[x, y] = G[y, x] = False
                    sepset[x][y] = list(S)
                    sepset[y][x] = list(S)
                    independent = True
                    break

            if independent:
                continue

        ord_size += 1

        # Stop if no edges remain
        if not G.any():
            break

    return G, sepset, pmax
