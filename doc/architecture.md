# Architecture

## Package layout

```
ensima/
├── optimize.py           Entry point — argument parsing and main dispatch
├── classes/              Core abstractions
│   ├── bayesian_optimization.py   Main BO orchestration
│   ├── model.py                   GP model (sklearn / GPyTorch backends)
│   ├── acquisition_function.py    EI and HGAL acquisition functions
│   ├── mixture_of_experts.py      MoE with PointNet geometry encoder
│   ├── expert.py                  Single expert wrapper
│   ├── geometry_nn.py             PointNet encoder/autoencoder
│   ├── search_space.py            Candidate point generation
│   ├── simulation.py              Simulation execution and result parsing
│   ├── file_modifier.py           Read/write OpenForm .dat input files
│   ├── progress_watcher.py        Monitor simulation log files
│   ├── license_server.py          RLM license server lifecycle
│   ├── logger.py                  Logging setup
│   └── print.py                   Rich-based console output
└── helpers/              Stateless utility functions
    ├── parse_args.py              CLI argument definitions
    ├── read_data.py               CSV loading and filtering
    ├── read_geometry.py           .t52 geometry file parser
    ├── objective_function.py      Dummy objectives for testing
    ├── pareto.py                  Pareto frontier computation
    ├── optimum.py                 Best-point extraction
    ├── clustering.py              KMeans clustering for MoE
    ├── complexity.py              Part complexity classification
    ├── adjust_args_cluster.py     HPC path/argument adjustment
    ├── energy.py                  CPU/GPU energy monitoring
    ├── co2.py                     CO2 emission estimation
    ├── plot.py                    Plotting utilities
    ├── parse_results.py           OpenForm output parsing
    ├── serilaize.py               JSON serialization helpers
    └── units.py                   Unit conversion utilities
```

## Class relationships

```
optimize.py
    └── BayesianOptimization
            ├── Model                      (GP backend: sklearn or GPyTorch)
            │       └── GPyTorchMultitaskModel / GaussianProcessRegressor
            ├── AcquisitionFunction        (EI, HGAL)
            ├── SearchSpace                (grid / random candidate generation)
            ├── MoEPointNetSystem          (optional, multi-part mode)
            │       ├── PointNetClassifier / PointNetAutoEncoder
            │       └── Expert (one per part type)
            ├── Simulation                 (optional, runs OpenForm solver)
            │       └── FileModifier       (edits .dat input files)
            └── ProgressWatcher            (optional, tails simulation logs)
```

## Data flow

### Single-part (MLVGP)

```
Initial data (x, y)
        │
        ▼
   Model.fit()              ← GP trained on all observed (x, y)
        │
        ▼
AcquisitionFunction         ← scores candidates from SearchSpace (grid or random)
        │
        ▼
  Best candidate x*         ← selected via selection strategy (EI, HGAL, ...)
        │
        ▼
  Simulation / ObjectiveFunction   ← evaluate x* (real solver or dummy)
        │                            FileModifier patches .dat, ProgressWatcher tails log
        ▼
  New (x*, y*) appended
        │
        └──────────────────────────► repeat for N iterations / until end condition
```

### Multi-part (MoE)

```
Part geometry files (.t52)
        │
        ▼
PointNetEncoder             ← encodes each part into a geometry embedding
        │
        ▼
Gating (cosine similarity)  ← routes new part to closest expert
        │
        ├── Expert_A (trained on part-type A data) ──┐
        ├── Expert_B (trained on part-type B data) ──┤
        └── Expert_C (trained on part-type C data) ──┘
                                                      │
                                                      ▼
                                           Active Expert GP
                                                      │
                                                      ▼
                                           AcquisitionFunction
                                                      │
                                                      ▼
                                             Best candidate x*
                                                      │
                                                      ▼
                                           Simulation / ObjectiveFunction
                                                      │
                                                      ▼
                                           New (x*, y*) added to active expert
                                                      │
                                           (gating re-evaluated each iteration)
                                                      │
                                                      └──► repeat
```

## MoE PointNet modes

`MoEPointNetSystem` supports two operating modes, selected via the `mode` argument:

| Mode | Encoder | Expert definition | Gating |
|---|---|---|---|
| `supervised` | `PointNetClassifier` | One expert per part label | Cosine similarity (hard or soft) |
| `unsupervised` | `PointNetAutoEncoder` | One expert per KMeans cluster | Cosine similarity or Euclidean distance (hard only) |

### Supervised mode

Requires part labels. The `PointNetClassifier` is trained discriminatively to separate
known part classes. At inference time, the embedding of a new geometry is compared to
stored class embeddings via **cosine similarity**; the part is routed to the closest expert.

Two gating strategies are available:

- **Hard gating**: assigns the new part to exactly one expert (nearest embedding).
- **Soft gating**: blends expert predictions weighted by cosine similarity scores —
  only valid in supervised mode.

### Unsupervised mode

Does not require labels. `PointNetAutoEncoder` is trained to reconstruct point clouds,
producing geometry embeddings that capture shape similarity without supervision. Experts
are defined by **KMeans clustering** over the learned embeddings.

- `n_clusters` can be specified explicitly, or inferred automatically using the Elbow
  method / silhouette score over the training embeddings.
- Gating uses either cosine similarity or Euclidean distance; only hard gating is supported.
- Labels are optional and ignored for routing; they may still be used for logging.

### Shared gating logic

Both modes call `get_best_expert(new_embedding, soft=...)`, which:

1. Encodes the new geometry into an embedding.
2. Scores it against all stored expert embeddings using the chosen metric.
3. Returns the best-matching expert (hard) or a weighted combination (soft, supervised only).

Gating is re-evaluated at every iteration, so a new part shape can be rerouted if embeddings
shift after continued training.

## Backends

`Model` provides a unified interface over two GP backends, selected via `args.backend`:

| Backend | Class | Kernel | Use case |
|---|---|---|---|
| `sklearn` | `GaussianProcessRegressor` | Matern 5/2 + WhiteKernel + ConstantKernel | Fast, single-task, small datasets |
| `torch` | `GPyTorchMultitaskModel` | Matern 5/2 + MultitaskKernel | Multi-task, larger datasets, GPU support |

### sklearn backend

Uses scikit-learn's `GaussianProcessRegressor` with a composite kernel:

```
ConstantKernel * Matern(nu=2.5) + WhiteKernel
```

- `optimizer_restart`: number of random restarts for kernel hyperparameter optimization
- `normalize_y`: standardizes outputs before fitting (default: `True`)
- Suitable when outputs are independent or the dataset is small (<1000 points)

### torch (GPyTorch) backend

Uses GPyTorch's `ExactGP` with a **Matern 5/2 kernel** wrapped in a `MultitaskKernel`
for joint modeling of correlated outputs. Key components:

- **`GPyTorchMultitaskModel`**: multi-task GP with per-task noise and shared covariance
- **`LatentCoupledMean`**: optional neural mean function — a small MLP that projects
  inputs through a low-dimensional latent space to the output space. Captures shared
  physical structure across output dimensions (e.g., correlated failure modes).
- **`LatentInputEncoder`**: optional input encoder that compresses high-dimensional
  inputs to a lower-dimensional latent representation before the GP kernel.
- **`SimpleLCMKernel`**: a Linear Model of Coregionalization kernel variant for
  explicit task correlation modeling.

Training parameters:

| Parameter | Description |
|---|---|
| `epochs` | Number of Adam optimizer steps (torch backend) |
| `optimizer_restart` | Random restarts for hyperparameter optimization (sklearn) |
| `normalize_y` | Standardize outputs before fitting |
| `latent_input_dim` | Latent dimension for `LatentInputEncoder` |
| `save_and_load` | Serialize model to disk after training and reload on next run |

### Model persistence

When `save_and_load=True`, the trained model is pickled to disk after fitting.
On subsequent runs with the same filename, the saved model is loaded instead of
retraining — useful for expensive multi-task models on large datasets.
