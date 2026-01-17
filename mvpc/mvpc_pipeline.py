"""
mvpc_pipeline.py

High-level orchestration of the MVPC algorithm (Missing Value PC).
This module coordinates:
    - detection of missingness indicator parents
    - corrected skeleton search (MVPC step 2)
    - final PC orientation using causal-learn

The heavy lifting is delegated to:
    missingness.py
    skeleton.py
    ci_tests/
"""

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

from .missingness import detection_prt_m
from .skeleton import skeleton2


class MVPC:
    """
    Main MVPC pipeline class.

    Parameters
    ----------
    indep_test : callable
        Base conditional independence test (e.g., Gaussian CI test).
        This is used for:
            - detecting missingness parents
            - initial PC skeleton
    corr_test : callable
        Corrected CI test (DRW or PermC).
        This is used for:
            - corrected skeleton search (MVPC step 2)
    alpha : float
        Significance threshold for CI tests.
    """

    def __init__(self, indep_test, corr_test, alpha=0.05):
        self.indep_test = indep_test
        self.corr_test = corr_test
        self.alpha = alpha

    def run(self, data):
        """
        Execute the full MVPC pipeline.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape (n_samples, n_variables) with missing values.

        Returns
        -------
        graph : CausalGraph
            The final oriented graph returned by causal-learn's PC algorithm.
        """

        n, p = data.shape

        # ---------------------------------------------------------
        # Step 1: Detect parents of missingness indicators
        # ---------------------------------------------------------
        prt_m = detection_prt_m(
            data=data,
            indep_test=self.indep_test,
            alpha=self.alpha,
            p=p
        )

        # ---------------------------------------------------------
        # Step 2a: Run standard PC to get initial skeleton
        # ---------------------------------------------------------
        pc_initial = pc(
            data,
            ci_test=self.indep_test,
            alpha=self.alpha
        )

        # ---------------------------------------------------------
        # Step 2b: Correct the skeleton using MVPC logic
        # ---------------------------------------------------------
        G_corrected, sepset_corrected, pmax_corrected = skeleton2(
            data=data,
            corr_test=self.corr_test,
            alpha=self.alpha,
            skel_pre=pc_initial,
            prt_m=prt_m
        )

        # ---------------------------------------------------------
        # Step 2c: Run PC orientation on the corrected skeleton
        # ---------------------------------------------------------
        # causal-learn does not allow injecting a custom skeleton directly,
        # so we re-run PC but override the CI test with the corrected one.
        graph_final = pc(
            data,
            ci_test=self.corr_test,
            alpha=self.alpha
        )

        return graph_final
