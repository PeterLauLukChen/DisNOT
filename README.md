# **DisNOT: Displacement-Sparse Neural Optimal Transport**

This repository contains the code implementation for the work:

*Displacement-Sparse Neural Optimal Transport*

---

## **Installation**

Before running any specific task, ensure you install all required dependencies by executing:

```bash
pip install -r requirement.txt
```

## **Repository Structure**

This repository is divided into four main components:

### 1) **Low-Dim Constant Spa**
   - Implements DisNOT using a constant sparsity approach for lower-dimensional tasks.
   - Focuses on maintaining consistent sparsity throughout the optimization process.

### 2) **Low-Dim SA**
   - Uses simulated annealing to heuristically adjust sparsity intensity.
   - Designed for low-dimensional tasks with varying user-defined goals and targets for source and target measures.

### 3) **High-Dim SA**
   - Applies simulated annealing in high-dimensional tasks.
   - Aims to find a map that satisfies sparsity constraints while minimizing the Wasserstein divergence to the target measure.

### 4) **High-Dim Drug Perturbation Experiments**
   - Includes all benchmarks and comparisons for drug perturbation experiments.
   - Contains Sinkhorn solver and neural OT solver implementations via DisNOT.

## Low-Dimensional Constant Sparsity

In this section, we prepare toy data in the data file. This module is designed for constant sparsity intensity in a low-dimensional space.

Please set the hyperparameters in `sparsity-induce.py`. Example of running:

```bash
python sparsity-induce.py --SPARSITY_INDUCING_INTENSITY 0.0001 --PENALITY 'stvs'
```

## **Low-Dim SA**

In this section, we use toy data provided in the data file. This module is designed for heuristic adjustments in a low-dimensional space.

Please set the hyperparameters in `heuristic.py`. Example of running: 

```bash
python heuristic.py --INITIAL_SPARSITY_INDUCING_INTENSITY 0.0005 --PENALITY 'stvs' --ALPHA 0.8 --SA_MIN_TEMP 0.1 --SA_TEMPERATURE_DECAY_RATE 0.98
```

## **High-Dim SA**

We provide a high-dimensional toy data generator in `hd-data-gen.py`. Please follow the instructions inside the script to prepare your toy high-dimensional simulated sc-RNA perturbation dataset.

To run high-dimensional simulated annealing, please set the hyperparameters in `hd.py`. Example of running:

```bash
python hd.py --INITIAL_SPARSITY_INDUCING_INTENSITY 0.005 --DIM_CON 150 --SA_MIN_TEMP 0.15 --SA_TEMPERATURE_DECAY_RATE 0.95 --INPUT_DIM 78
```

## **High-Dim Drug Perturbation**

In this section, follow these steps:

1. **Prepare the Data**:
   - Refer to the instructions in `perturb-data-prep.py` to download the public dataset and prepare it for processing.

2. **Solve OT Map (Standard L2 Cost)**:
   - After the data is prepared, follow the instructions in `ott-jax-l2-solver.py` to solve the OT map between control and treated cells under the standard L2 cost.

3. **Sinkhorn Solver Results**:
   - For results using the Sinkhorn solver (Cuturi et al., 2023), follow the instructions in `ott-jax-regularization-solver.py` and set the hyperparameters as detailed in Appendix C of the referenced paper.

4. **Neural OT Results**:
   - For results using the neural OT solver:
     - Refer to `icnn-sparsity-solver.py` to update dataset naming conventions and other configurations.
     - Set the required hyperparameters and run the solver to obtain the final displacement statistics.


## **Reference**

The code implemenation is based on the original implementation from:
```
Makkuva, A., Taghvaei, A., Oh, S., & Lee, J. (2020). Optimal transport mapping via input convex neural networks. In International Conference on Machine Learning (pp. 6672-6681). PMLR.
```
