Organizing and Using Simulation Results

## Overview
This guide explains:
- How simulation results are stored
- How labeled and unlabeled CSV files are used
- How data filtering works
- How data selection changes when using a Mixture of Experts (MoE) model

---

## 1. Storing Simulation Results

- Each simulation run is stored in its own **JSON file**.  
- All results are additionally aggregated into a **single CSV file**.

---

## 2. Using Labeled vs. Unlabeled CSV Files

### Unlabeled CSV
- Can be used for optimization in two ways (for MLVGP):
  - Use **all data points**, or
  - **Filter by type** (e.g., geometry type or global category)

### Labeled CSV (`*_labeled`)
- Contains additional labels for each data point.
- This is always used for MOE and sometimes (filter by parts for MLVGP)
- Enables more precise filtering:
  - **Filter by part**, e.g.:
    - Selecting only roof points
    - Selecting only Seat Shell points, etc.

---

## 3. Using Data with Mixture of Experts (MoE)

- MoE introduces additional complexity.
- Data selection works as follows:
  - All points that correspond to the **currently active expert** are used.
  - When the gating mechanism decides that a **new expert should be activated**, the newly assigned expert receives its corresponding points.
- This means the dataset dynamically changes depending on which expert is active at each stage.

## 4. Reproducing the Results

Each JSON file in `sim_results/` records the complete run configuration under the
`args` key — including all optimization parameters, field names, constraints, and
acquisition function settings used for that specific run. This makes each result
self-documenting. Use `plot_results.ipynb` (included in this folder) to visualize
and compare results across simulation variants.

The `simulation/` folder contains the three Python scripts that were used to launch
the runs. All scripts require `ensima` to be installed (see the repository root
`README.md`) and access to the OpenForm solver and software.

### Prerequisites

- `ensima` installed (run `make install` from the repository root)
- OpenForm solver (`OFSolv`) and OpenForm software (`OpenForm_64_batch`) available on the cluster
- TCO-Benchmark input files present under `TCO-Benchmark/` (included in this artifact)

### Note on cluster execution

All three scripts were executed on a GPU cluster. The `-ofs` and `-ofm` arguments
point to cluster-local installations of the OpenForm solver and software:

```python
"-ofs", "/d/gitlab/.../OFSolv_1.0.4e_eng_linux64.exe",
"-ofm", "/d/gitlab/.../OpenForm_64_batch",
```

Set these to your local paths before running. The `adjust_args_for_cluster` /
`adjust_args_and_parts_for_cluster` helpers handle any remaining cluster-specific
path overrides automatically.

### Running MLVGP

`simulation/run_mlvgp.py` runs single-part Bayesian optimization (MLVGP) on
SeatShell, filtering the labeled CSV to the target part.
Results are in `sim_results/MLVGP/`.

```sh
python artifacts/JIMS/simulation/run_mlvgp.py
```

### Running MoE

`simulation/run_moe.py` runs multi-part optimization with Mixture of Experts on
Laengstraeger_02 as the new target part. The `parts` list defines the expert library
(one GP per part type). Uncomment the corresponding `-p`/`-j`/`-s`/`-g` lines to
switch to a different target part. Results are in `sim_results/MOE/`.

```sh
python artifacts/JIMS/simulation/run_moe.py
```

### Running the no-optimization baseline

`simulation/run_no_optimization.py` runs the solver with fixed expert- or
user-defined parameter sequences — no BO loop. Switch `mode` between `"expert"` and
`"unskilled user"` in the DACH-VWS block and set the active part via `-j`/`-p`/`-s`.
Results are in `sim_results/no_optimization/`.

```sh
python artifacts/JIMS/simulation/run_no_optimization.py
```

---

## 5. Structure

```
artifacts/JIMS/
├── CSV/                  CSV results (labeled, unlabeled, expert)
├── TCO-Benchmark/        Simulation input files (geometry, session, dat files)
├── simulation/           Scripts used to produce sim_results (exact configurations)
│   ├── run_mlvgp.py           Single-part MLVGP optimization
│   ├── run_moe.py             Multi-part MoE optimization
│   └── run_no_optimization.py Expert/user baseline runs (no BO loop)
├── sim_results/
│   ├── MLVGP/            Simulation results with MLVGP (summary, log, JSON, etc.)
│   ├── MOE/              Simulation results with MOE (summary, log, JSON, etc.)
│   └── no_optimization/  Expert and User simulations repeated on the GPU cluster
├── plot_results.ipynb    Notebook to visualise and compare results
└── instructions.md
```
