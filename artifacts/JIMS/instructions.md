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

## 4. Structure

```
artifacts/JIMS/
├── CSV/                  CSV results (labeled, unlabeled, expert)
├── TCO-Benchmark/        Simulation input files (geometry, session, dat files)
├── sim_results/
│   ├── MLVGP/            Simulation results with MLVGP (summary, log, JSON, etc.)
│   ├── MOE/              Simulation results with MOE (summary, log, JSON, etc.)
│   └── no_optimization/  Repeated runs with the same expert/user input configurations,
│                         used for a fair energy comparison on the same machine.
└── instructions.md
```
