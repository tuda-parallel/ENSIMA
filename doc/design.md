# Design Concepts

## Overview

`ensima` applies **Bayesian Optimization (BO)** to reduce the number of costly stamping
simulations needed to find optimal process parameters. Instead of exhaustive search,
a surrogate model is used to predict simulation outcomes cheaply, and an acquisition
function guides where to evaluate next.

---

## Bayesian Optimization loop

Each iteration of the BO loop does the following:

1. **Fit a surrogate model** (Gaussian Process) on all observed `(x, y)` pairs.
2. **Score candidate points** using an acquisition function.
3. **Select the best candidate** `x*` and evaluate it (simulation or objective function).
4. **Append** `(x*, y*)` to the dataset and repeat.

The loop terminates based on a configurable end condition (see below).

---

## Gaussian Process Regression (GPR)

The surrogate model is a **Gaussian Process**, which provides both a predicted mean and
an uncertainty estimate at any point in the input space. This uncertainty is what makes
BO sample-efficient — it actively seeks points where the model is uncertain.

Two backends are supported:

- **scikit-learn** (`GaussianProcessRegressor`): fast, single-task, good for small datasets.
- **GPyTorch** (`GPyTorchMultitaskModel`): multi-task, scalable, GPU-accelerated. Uses a
  **Matern 5/2 kernel**, which is suitable for non-smooth physical simulations.

### LatentCoupledMean

The GPyTorch model optionally uses a **LatentCoupledMean** function: a learned linear
projection from a low-dimensional latent space to the output space. This captures
correlations between output targets (e.g., different defect categories in the same
forming process share an underlying physical cause).

---

## Acquisition functions

### Expected Improvement (EI)

EI quantifies how much improvement over the current best observation a candidate point
is expected to yield. It balances **exploitation** (sampling near the known optimum) and
**exploration** (sampling in uncertain regions).

Supports:
- `minimization` and `maximization` modes
- **Scalar EI**: each output treated independently
- **Multi-output EI**: Monte Carlo sampling over the full output covariance

### Human-Guided Active Learning (HGAL)

HGAL (`--method hgal`) incorporates domain expert input into the sampling strategy.
Instead of purely algorithmic selection, the expert can steer the acquisition toward
regions of interest. This is useful when physical intuition can rule out large parts
of the search space.

### Attention coefficients

When multiple outputs are optimized simultaneously, **attention coefficients** weight
each output's contribution to the overall acquisition score. Negative weights indicate
outputs to minimize; positive weights indicate outputs to maximize.

```sh
ensima --attention_coefficients 1 0.8 -0.4 -0.4 0.8 0.8 1
```

---

## Search space

`SearchSpace` generates candidate points for acquisition function evaluation.
Two strategies are supported:

- **Grid**: constructs a regular grid within observed data bounds (optionally expanded
  by `expansion_factor`). Respects discrete constraints (e.g., material grades as a
  fixed list of values).
- **Random**: uniform random sampling within the bounded region.

Constraints can be specified as:
- **Continuous**: `(min, max)` tuple
- **Discrete**: list of allowed values, e.g., `[270, 420, 890]` for material yield strength

---

## Mixture of Experts (MoE)

For **multi-part optimization**, a single GP model may not generalize well across
geometrically diverse parts. The MoE system trains a separate expert GP per part type
and routes new parts to the most relevant expert based on geometry.

### PointNet geometry encoder

Part geometry (`.t52` files) is encoded using a **PointNet-style neural network**,
which processes raw 3D point clouds in a permutation-invariant way. Two modes:

- **Supervised** (`PointNetClassifier`): learns discriminative embeddings that separate
  known part classes. Uses cosine similarity for routing.
- **Unsupervised** (`PointNetAutoEncoder`): learns geometry embeddings without labels,
  then clusters with KMeans. Useful when part categories are not predefined.

### Gating mechanism

When a new part arrives, its geometry is encoded and compared (via cosine similarity
or Euclidean distance) to the embedding library. The closest expert is selected.
The gating can be **hard** (winner-takes-all) or **soft** (weighted blend).

---

## End conditions

The optimization loop terminates when one of the following conditions is met:

| Condition | `--end_condition` | `--end_value` meaning |
|---|---|---|
| No improvement for N iterations | `no_improvement` | Number of stagnant iterations |
| Best value constant for N iterations | `constant_min` | Number of constant iterations |
| Energy budget exhausted | `energy_budget` | Budget in kWh |
| Max iterations reached | *(always active)* | `--iterations` |

---

## Approximate computing

To avoid spending simulation time on clearly suboptimal configurations, `ensima`
supports **approximate computing checks**: if the early simulation output (partial
results) already violates a threshold, the simulation is terminated early.

Controlled via:
- `--approximate_computing_check`: percentage of simulation to wait before checking
- `--approximate_computing_limit`: per-output threshold values

---

## Parallel sampling

Multiple candidates can be evaluated per iteration using `--parallel_samples`.
Candidates are selected using a **batch acquisition strategy** (e.g., crowding distance,
peak-based, or highest-sum selection) to ensure diversity among parallel evaluations.

Selection strategies (`--selection_strategy`):
- `highest_sum`: selects the candidate with the highest weighted sum of outputs
- `peak_based`: selects candidates near acquisition function peaks
- `crowding_distance`: promotes diversity via Pareto crowding distance

---

## Energy and CO2 monitoring

When `--energy_estimation` is enabled, `ensima` measures CPU and GPU energy consumption
during each simulation using:

- **CPU**: dynamic power model based on frequency, utilization, capacitance, and voltage
- **GPU**: NVML (`pynvml`) total energy counter (if available)

CO2 emissions are estimated from energy consumption using a configurable emission factor
(`co2.py`). Both are recorded per simulation run and reported in the final summary.

---

## HPC cluster support

`adjust_args_cluster.py` detects the current hostname and automatically remaps local
file paths to cluster paths. This allows the same script to run unchanged on a
developer workstation and on an HPC cluster.

The license server (`license_server.py`) manages the RLM license daemon lifecycle —
starting it before the first simulation and stopping it when the optimization finishes.
