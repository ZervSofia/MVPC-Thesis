# Causal Discovery with Missing Data – MVPC for Synthetic & MIMIC Data

This repository provides code and notebooks for **Missingness-aware PC (MVPC)**, a causal discovery algorithm that handles datasets with missing values. It is applied to **synthetic data** and **MIMIC clinical data**, following Tu et al. (2020).


## Overview of MVPC 

MVPC is a **missingness-aware causal discovery algorithm** that extends the classical PC algorithm to handle datasets with missing values. It can use three types of CI tests: 

1. **TD (Test-wise Deletion):** Ignores missing values in the variables involved in each test. 


2. **PermC (Permutation Correction):** Corrects for missingness by permuting observed data to approximate the full data distribution. 


3. **DRW (Double Robust Weighting):** Uses weighting to adjust for missingness under MAR or MNAR assumptions. 



The pipeline follows the general steps of the PC algorithm: 
- Generate / load a dataset. 
- Compute conditional independence tests for pairs of variables. 
- Build a skeleton graph based on CI test results.
---

## Folder Structure 

This repository has three main folders:

- **data/** – Contains synthetic and MIMIC datasets. Synthetic data includes complete and missing  datasets with different types of missingness. MIMIC data is preprocessed and aggregated.

- **mvpc/** – Contains the implementation of the MVPC algorithm, including CI tests, skeleton construction, orientation, and utility functions.

- **notebooks/** – Jupyter notebooks demonstrating CI test validation, MVPC usage on synthetic and MIMIC data,  data generation, and result analysis.


---

## Data

### Synthetic Data
- Random DAGs with linear Gaussian SEMs.
- Missingness injected as:
  - **MAR:** depends on parents of collider nodes.
  - **MNAR:** may depend on other missing variables.
- Outputs:
  - **data_complete**: fully observed data  
  - **data_m**: missing data (MAR or MNAR)   
  - **ground_truth**: DAG structure, colliders, missingness indicators  

### MIMIC Data
- Preprocessed **MIMIC-IV demo subset**.
- Labs from first 24h (e.g., Sodium, Glucose) aggregated by median.
- Merged with patient demographics and admission info.
- Includes missing values naturally present in clinical data.

---

## MVPC



### Steps
1. **Detect parents of missingness indicators**  
   - Create binary missingness indicators and find influencing variables.
2. **Initial skeleton**  
   - Build undirected graph using a **base CI test (TD)**.
3. **Corrected skeleton**  
   - Refine skeleton using **DRW or PermC**, adjusting for missingness.

### Conditional Independence (CI) Tests
| Test | Purpose |
|------|---------|
| **TD (Test-wise Deletion)** | Ignores missing rows in variables tested |
| **PermC (Permutation Correction)** | Corrects missingness via residual permutation |
| **DRW (Double Robust Weighting)** | Weighted CI test using missingness-parent info |


---

## Notebooks

- **00_ci_tests_validation.ipynb:** Quick verification that all CI tests (TD, PermC, DRW) work correctly on small synthetic Gaussian and binary chains. 

- **01_generate_synthetic_data.ipynb:** Generates synthetic datasets using random DAGs and linear Gaussian SEMs, then injects MAR or MNAR missingness. Saves complete, missing, and MCAR reference datasets along with ground truth DAG structures. 

- **02_mvpc_synthetic_data.ipynb:** Runs MVPC with TD, PermC, and DRW CI tests on synthetic data. Computes evaluation metrics such as skeleton SHD and F1 score across sample sizes, repetitions, and missingness modes. 

- **03_mvpc_mimic_data.ipynb:** Applies MVPC to the MIMIC dataset, introduces MAR/MNAR missingness, runs the three CI tests, and aggregates edge frequencies over bootstrapped samples. Visualizes network structures and densities. 


---

## References 

Original Paper Tu et al., 2020 – Causal discovery in the presence of missing data: MVPC algorithm

Original Implementation: https://github.com/TURuibo/MVPC