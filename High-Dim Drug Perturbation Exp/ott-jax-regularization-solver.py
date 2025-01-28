import jax
import jax.numpy as jnp
from ott.geometry import costs
from ott.geometry import pointcloud
from ott.geometry.regularizers import ProximalOperator
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

import numpy as np
import os

"""
This module implements a sparse optimal transport (OT) solution using the Sinkhorn 
algorithm with a custom L1 proximal operator for regularization. The OT problem 
is solved between two high-dimensional datasets (`source` and `target`) loaded 
from `.npy` files, and the displacement vectors are computed for further analysis.

Prerequisites:
1. Ensure `control.npy` (source dataset, cells in controlling state) 
   and `perturb.npy` (target dataset, cells after drug perturbation) 
   are stored  in the `data` directory.

2. The proximal operator is designed for sparsity-inducing regularization 
   (L1-regularization).

3. JAX is used for efficient computation, enabling GPU acceleration.

Outputs:
- Displacement vectors after applying the transport map.
- Sparse regularization is controlled by the `sparsity_intensity` parameter.

Hyper-parameters setup:
`sparsity_intensity`: Sparsity-inducing intensity
`epsilon`: Sinkhorn regularization parameter
"""

# Define folder paths and load data; Load source and target datasets
current_folder = os.path.dirname(os.path.abspath(__file__))
data_folder_sicnn = os.path.join(current_folder, 'data')
source = jnp.array(np.load(os.path.join(data_folder_sicnn, 'control.npy')))
target = jnp.array(np.load(os.path.join(data_folder_sicnn, 'perturb.npy')))


# Define a custom L1 proximal operator for regularization
@jax.tree_util.register_pytree_node_class
class L1ProximalOperator(ProximalOperator):
    """
    Implements an L1-regularization proximal operator for sparsity control.
    - `__call__`: Computes the L1-norm of the input vector.
    - `prox`: Applies the soft-thresholding operation, the proximal operator.
    """
    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute the L1-norm of the input."""
        return jnp.sum(jnp.abs(v), axis=-1)

    def prox(self, v: jnp.ndarray, tau: float) -> jnp.ndarray:
        """
        Apply the soft-thresholding operator.
        
        Parameters:
        - v: Input vector.
        - tau: Threshold parameter.

        Returns:
        - Thresholded vector with reduced magnitude based on tau.
        """
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - tau, 0.0)

    def tree_flatten(self):
        """Flatten the class for JAX transformations."""
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the class for JAX transformations."""
        return cls()

# Set the regularization intensity for L1 sparsity control
sparsity_intensity = 1
epsilon = 1e-3

reg_cost = costs.RegTICost(L1ProximalOperator(), lam=sparsity_intensity)

# Define the geometry of the point cloud and create the OT problem
geom = pointcloud.PointCloud(source, target, cost_fn=reg_cost, epsilon=epsilon)
ot_problem = linear_problem.LinearProblem(geom)
sinkhorn_solver = sinkhorn.Sinkhorn()
ot_solution = sinkhorn_solver(ot_problem)

# Extract the transport map and compute displacement vectors
dual_potentials = ot_solution.to_dual_potentials()
transported_points = np.array(dual_potentials.transport(source))
displacement_vectors = transported_points - np.array(source)


