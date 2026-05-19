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

```
Initial data (x, y)
        │
        ▼
   Model.fit()          ← GP trained on observed (x, y)
        │
        ▼
AcquisitionFunction     ← scores candidate points from SearchSpace
        │
        ▼
  Best candidate x*
        │
        ▼
  Simulation / ObjectiveFunction   ← evaluate x* (real or dummy)
        │
        ▼
  New (x*, y*) appended to dataset
        │
        └──────────────────► repeat for N iterations
```

## Backends

`Model` supports two backends selected via `args.backend`:

| Backend | Class | Use case |
|---|---|---|
| `sklearn` | `GaussianProcessRegressor` | Fast, single-task, small datasets |
| `torch` | `GPyTorchMultitaskModel` | Multi-task, larger datasets, GPU support |

The GPyTorch model uses a **Matern 5/2 kernel** with an optional **LatentCoupledMean** function
that projects outputs from a shared latent space — useful when multiple output targets are
correlated (e.g., different failure modes of the same forming process).
