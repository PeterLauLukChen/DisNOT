import numpy as np 
import os

'''
This module generates synthetic high-dimensional datasets to simulate control and 
treated conditions for biological or computational experiments. 
'''

# Define parameters for the dataset
num_cells = 2000            # Number of cells in the dataset
num_genes = 3000            # Number of genes in the dataset
num_drastic_changes = 100   # Number of genes that will have drastic changes due to treatment (Ground Truth Dimensionality)
num_noise_changes = 2000    # Number of genes that will have small noise-induced changes
np.random.seed(42)

control_data = np.random.rand(num_cells, num_genes)
treated_data = control_data.copy()

# Introduce drastic changes to a subset of genes
drastic_indices = np.random.choice(num_genes, num_drastic_changes, replace=False)
treated_data[:, drastic_indices] += np.random.rand(num_cells, num_drastic_changes) * 100

# Introduce noise changes to another subset of genes
noise_indices = np.random.choice(num_genes, num_noise_changes, replace=False)
treated_data[:, noise_indices] += np.random.rand(num_cells, num_noise_changes) * 1

os.makedirs('./data', exist_ok=True)
np.save('./data/high_c.npy', control_data)
np.save('./data/high_t.npy', treated_data)
