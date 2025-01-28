import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

import os
import numpy as np

"""
This module performs optimal transport (OT) matching between two high-dimensional datasets 
using the Sinkhorn algorithm. The source and target datasets are loaded from `.npy` files, 
and displacement vectors are computed to analyze the transport map.

The OT map is computed using standard L2 norm cost.

Prerequisite:
Ensure the required datasets (`high_t.npy` and `high_c.npy`) are stored in the `data` folder 
within the current directory structure.

Hyper-parameters setup:
`epsilon`: Sinkhorn regularization parameter
"""

# Define folder paths and load data
current_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_folder, 'data')

# Load source and target datasets
source = jnp.array(np.load(os.path.join(data_folder, 'high_t.npy')))
target = jnp.array(np.load(os.path.join(data_folder, 'high_c.npy')))
print(f"Source dataset size: {source.size}")
print(f"Target dataset size: {target.size}")

# Set OT hyperparameters for sinkhorn regularization
epsilon = 1e-3 

# Define the point cloud geometry and solve OT via Sinkhorn solver
geom = pointcloud.PointCloud(source, target, epsilon=epsilon)
ot_problem = linear_problem.LinearProblem(geom)
sinkhorn_solver = sinkhorn.Sinkhorn()
ot_solution = sinkhorn_solver(ot_problem)

# Extract dual potentials and compute transported points
dual_potentials = ot_solution.to_dual_potentials()
transported_points = np.array(dual_potentials.transport(source))

# Compute displacement vectors
displacement_vectors = transported_points - np.array(source)

# Further analysis for the displacement vectors could be conducted below, e.g. dimensionality, etc.
