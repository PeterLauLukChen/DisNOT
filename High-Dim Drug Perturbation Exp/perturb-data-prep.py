import numpy as np
import scanpy as sc

"""
This module processes cell perturbation data for use in optimal transport (OT) matching tasks. 
The final processed features for OT are saved as `{specified_drug}_control_features.npy` and 
`{specified_drug}_perturbed_features.npy`, which can be directly used in Neural OT solvers.

Prerequisite: 
Users should download the public dataset available at:
https://www.research-collection.ethz.ch/handle/20.500.11850/609681
and place the cell perturbation dataset (4i folder) in the current working directory 
before running this script.
"""

# Load the 4i cell perturbation dataset
file_path = "./8h.h5ad"
adata = sc.read_h5ad(file_path)

"""
Specify the drug used for cell perturbation experiments. 
By modifying `specified_drug`, users can analyze the effect of different perturbations. 
`control_label` represents the baseline condition.
"""
specified_drug = "melphalan" 
control_label = "control"

# Extract indices for control and perturbed cells based on the experimental conditions
control_indices = adata.obs[adata.obs['drug'] == control_label].index
perturbed_indices = adata.obs[adata.obs['drug'] == specified_drug].index

# Extract feature matrices for control and perturbed cells
control_features = adata[control_indices, :].X
perturbed_features = adata[perturbed_indices, :].X

# Convert to dense arrays if the feature matrices are stored in a sparse format
if not isinstance(control_features, np.ndarray):
    control_features = control_features.toarray()
if not isinstance(perturbed_features, np.ndarray):
    perturbed_features = perturbed_features.toarray()

# Save the processed feature matrices as .npy files for downstream OT tasks
np.save(f'./{specified_drug}_control_features.npy', control_features)
np.save(f'./{specified_drug}_perturbed_features.npy', perturbed_features)
print(f"Control features shape: {control_features.shape} (cells x features)")
print(f"Perturbed features shape: {perturbed_features.shape} (cells x features)")
