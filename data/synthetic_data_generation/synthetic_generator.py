import numpy as np

from .dag_and_data import (
    random_dag,
    weighted_adj_matrix,
    true_covariance,
    sample_complete_data,
    detect_colliders,
    detect_collider_parents,
)

from .missingness_synthetic import (
    create_mar_ind,
    create_mnar_ind,
    generate_missing_values,
    generate_mcar_reference,
)



def gen_data(
    num_samples: int,
    mode: str = "mar",
    num_var: int = 20,
    num_extra_e: int = 3,
    num_m: int = 6,
    seed: int | None = None,
    p_missing_h: float = 0.9,
    p_missing_l: float = 0.01,
):
    """
    Synthetic data generator

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    mode : str
        'mar' or 'mnar'.
    num_var : int
        Number of variables in the DAG.
    num_extra_e : int
        Number of collider-based missingness indicators.
    num_m : int
        Total number of missingness indicators.
    seed : int
        Random seed.
    p_missing_h : float
        High missingness probability.
    p_missing_l : float
        Low missingness probability.

    Returns
    -------
    dict
        {
            "data_complete": ndarray,
            "data_m": ndarray,
            "data_ref": ndarray,
            "ground_truth": {
                "adj": adjacency matrix,
                "colliders": list,
                "m_ind": list,
                "parent_m_ind": list
            }
        }
    """

    if seed is not None:
        np.random.seed(seed)

    
    G, adj = random_dag(num_var, seed=seed)

    W = weighted_adj_matrix(adj, seed=seed)
    Sigma = true_covariance(W)


    # Sample complete Gaussian data

    X_complete = sample_complete_data(Sigma, num_samples, seed=seed)

    
    colliders = detect_colliders(adj)
    collider_parents = detect_collider_parents(adj, colliders)

    # missingness indicators MAR or MNAR

    if mode == "mar":
        ms, prt_ms = create_mar_ind(
            colliders,
            collider_parents,
            num_var,
            num_extra_e,
            num_m,
            seed,
        )
    elif mode == "mnar":
        ms, prt_ms = create_mnar_ind(
            colliders,
            collider_parents,
            num_var,
            num_extra_e,
            num_m,
            seed,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


    # Inject missingness 
    X_m = generate_missing_values(
        X_complete,
        ms,
        prt_ms,
        p_missing_h,
        p_missing_l,
        seed,
    )


    # MCAR reference dataset

    X_ref = generate_mcar_reference(X_complete, X_m, ms, seed)

 

    ground_truth = {
        "adj": adj,
        "colliders": colliders,
        "m_ind": ms,
        "parent_m_ind": prt_ms,
    }

    return {
        "data_complete": X_complete,
        "data_m": X_m,
        "data_ref": X_ref,
        "ground_truth": ground_truth,
    }
