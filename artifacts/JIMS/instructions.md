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
"""

## 4. Strucutre
..
├── CSV  --> CSV results
├── instructions.md
├── MLVGP --> Simulations results with MLVGP (inlcudes summary, log, JSON file, etc.)
├── MOE --> Simulations results with MOE (inlcudes summary, log, JSON file, etc.)
└── no_optimization --> Simulations results for the repeated runs with the same input configurations as the user/expert. This was needed for a fair energy comparision, to have the same machine.
