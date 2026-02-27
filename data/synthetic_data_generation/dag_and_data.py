import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal



def random_dag(n_nodes: int,
               p_edge: float | None = None,
               seed: int | None = None):
    """
    Generate a random DAG.

    Steps:
    1. Sample an undirected Erdős–Rényi graph with edge prob p_edge.
    2. Sample a random permutation (topological order) of nodes.
    3. Orient each undirected edge from earlier -> later in that order.
    """
    if seed is not None:
        np.random.seed(seed)

    if p_edge is None:
        p_edge = 2 / (n_nodes - 1)

  
    # Undirected Erdős–Rényi adjacency (symmetric, no self-loops)
    undirected = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < p_edge:
                undirected[i, j] = 1
                undirected[j, i] = 1


    # Random topological order (permutation of nodes)
    order = np.random.permutation(n_nodes)
    pos = {node: k for k, node in enumerate(order)}


    # Orient edges according to order
    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if undirected[i, j] == 1 and i < j:
                # Decide direction based on order positions
                if pos[i] < pos[j]:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1

    G = nx.DiGraph(adj)
    return G, adj



# Assign random edge weights (linear Gaussian SEM)

def weighted_adj_matrix(adj: np.ndarray,
                        weight_low: float = -1.0,
                        weight_high: float = 1.0,
                        seed: int | None = None):
    """
    Assign random weights to each directed edge in the DAG.
    """
    if seed is not None:
        np.random.seed(seed)

    W = adj.astype(float)
    edge_indices = np.where(adj == 1)

    W[edge_indices] = np.random.uniform(
        weight_low,
        weight_high,
        size=len(edge_indices[0])
    )

    return W



# Compute true covariance

def true_covariance(W: np.ndarray,
                    noise_var: float = 1.0):
    """
    Compute sigma = (I - B) ^(-1) omega (I - B)^(-T)
    where B is the weighted adjacency matrix.
    """
    B = W.T
    n = B.shape[0]

    I = np.eye(n)
    Omega = noise_var * np.eye(n)

    Sigma = np.linalg.inv(I - B) @ Omega @ np.linalg.inv(I - B).T
    return Sigma



# Sample complete data (joint Gaussian)

def sample_complete_data(Sigma: np.ndarray,
                          n_samples: int,
                          seed: int | None = None):
    """
    Sample X ~ N(0, sigma), matching rmvnorm in R.
    """
    if seed is not None:
        np.random.seed(seed)

    p = Sigma.shape[0]
    X = multivariate_normal.rvs(
        mean=np.zeros(p),
        cov=Sigma,
        size=n_samples
    )

    return X


# Collider detection (used for MAR / MNAR)

def detect_colliders(adj: np.ndarray):
    """
    Return indices of collider nodes (>=2 parents).
    """
    return [j for j in range(adj.shape[1]) if np.sum(adj[:, j]) > 1]


def detect_collider_parents(adj: np.ndarray, colliders: list[int]):
    """
    Return parents of each collider.
    """
    parents = []
    for c in colliders:
        parents.append(list(np.where(adj[:, c] == 1)[0]))
    return parents



