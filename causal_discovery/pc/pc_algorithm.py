import itertools
import networkx as nx

class PC:
    def __init__(self, ci_test, alpha=0.05, max_cond_set=3):
        self.ci_test = ci_test
        self.alpha = alpha
        self.max_cond_set = max_cond_set

    def learn_skeleton(self, X):
        p = X.shape[1]
        G = nx.complete_graph(p)
        sep_sets = {}

        for l in range(self.max_cond_set + 1):
            edges = list(G.edges())
            for (i, j) in edges:
                neighbors = list(set(G.neighbors(i)) - {j})
                if len(neighbors) < l:
                    continue

                for S in itertools.combinations(neighbors, l):
                    if self.ci_test.is_independent(X, i, j, list(S), self.alpha):
                        G.remove_edge(i, j)
                        sep_sets[(i, j)] = S
                        sep_sets[(j, i)] = S
                        break

        return G, sep_sets
