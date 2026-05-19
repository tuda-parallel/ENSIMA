ENSIMA Repository — File Overview
==================================

Top-level layout:

  Directories:
  artifacts/      Simulation artifacts and benchmark data (JIMS experiments)
  ensima/         Python package (source code)
  examples/       Example scripts demonstrating usage
  test/           Test suite and CSV datasets
  tools/          Jupyter notebook and analysis scripts

  Files:
  README.md       Project overview, installation, usage, and acknowledgments
  pyproject.toml  Package metadata, dependencies, and tool configuration
  Makefile        Build, install, test, style-check, and publish targets
  LICENSE         BSD 3-Clause License
  run.sh          Convenience script for launching optimization runs
  files.txt       This file
  .gitignore      Git ignore rules
  .gitattributes  Git attributes (nbstripout filter for notebooks)


================================================================================
artifacts/JIMS/
================================================================================

├── CSV/                         Aggregated CSV datasets used for training and optimization
│   ├── DataSets-AIandML.csv         Unlabeled dataset (all parts)
│   ├── DataSets-AIandML_labeled.csv Labeled dataset (part-level labels for filtering)
│   └── DataSets-AIandML_expert.csv  Expert-guided subset
│
├── sim_results/                 Simulation run results from JIMS experiments
│   │                            Each sub-folder contains per-run directories with:
│   │                            logs, intermediate results, summary, and JSON output.
│   │                            Aggregated results per experiment are in a single JSON file.
│   │
│   ├── MLVGP/                   Results using Multi-Level Variational Gaussian Processes
│   │   ├── SeatShell_1/             SeatShell with 1 initial sample
│   │   ├── SeatShell_168/           SeatShell with 168 initial samples
│   │   └── SeatShell_168_highestpeak_added/  Variant with highest-peak sample added
│   │
│   ├── MOE/                     Results using Mixture of Experts
│   │   ├── DACH-VWS/                Roof panel (DACH-VWS)
│   │   ├── DACH-VWS_6cores/         DACH-VWS run with 6 cores
│   │   ├── DACH-VWS_withpredicted/  Variant including predicted points
│   │   ├── Laengstraeger/           Side beam (Laengstraeger_02)
│   │   ├── Laengstraeger_softLimits/ Variant with soft approximate-computing limits
│   │   ├── SeatShell/               Seat shell part
│   │   └── SeatShell_withpredicted/ Variant including predicted points
│   │
│   └── no_optimization/         Repeated runs using fixed expert/user inputs (no BO)
│       │                        Used for fair energy comparison on the same machine.
│       │                        Each part has: timestamped run folders + aggregated JSON.
│       ├── DACH-VWS_no_optimization.json
│       ├── DACH-VWS_no_optimization_expert_only.json
│       ├── Laengstraeger_02_no_optimization.json
│       ├── Laengstraeger_02_no_optimization_expert_only.json
│       ├── SeatShell_no_optimization.json
│       └── SeatShell_no_optimization_expert_only.json
│
├── TCO-Benchmark/               Simulation input files for all benchmark parts
│   │                            See TCO-Benchmark section below for full tree.
│   ├── PartType_01_Flat/        A-pillar (ASaeule) — flat part
│   ├── PartType_02_Beam/        B-pillar (BSaeule) — beam part
│   ├── PartType_03_Deep/        Wheel house adapter — deep-drawn part
│   ├── PartType_04/             Einleger — insert part
│   ├── new_parts/               Additional parts (DACH-VWS, Laengstraeger_02, SeatShell)
│   └── T52-PartFiles/           Standalone geometry files for additional car body parts
│
└── instructions.md              Guide: result structure, CSV labeling, MoE data selection


================================================================================
TCO-Benchmark — Full Tree
================================================================================

artifacts/JIMS/TCO-Benchmark/
├── new_parts
│   ├── PartType_01
│   │   ├── DACH-VWS.dat
│   │   ├── DACH-VWS-Session.ofs
│   │   ├── DACH-VWS.t51
│   │   ├── DACH-VWS.t52
│   │   └── DACH-VWS.t53
│   ├── PartType_02
│   │   ├── Laengstraeger_02.dat
│   │   ├── Laengstraeger_02-Session.ofs
│   │   ├── Laengstraeger_02.t51
│   │   └── Laengstraeger_02.t52
│   └── PartType_03
│       ├── SeatShell.dat
│       ├── SeatShell-Session.ofs
│       ├── SeatShell.t51
│       ├── SeatShell.t52
│       └── SeatShell.t53
├── PartType_01_Flat
│   ├── ASaeule.dat
│   ├── ASaeule-Session_01.ofs
│   ├── ASaeule.t51
│   ├── ASaeule.t52
│   └── ASaeule.t53
├── PartType_02_Beam
│   ├── BSaeule_DX56D.dat
│   ├── BSaeule_DX56D.ele
│   ├── BSaeule_DX56D.nod
│   ├── BSaeule_DX56D-Session_01.ofs
│   ├── BSaeule_DX56D.t28
│   ├── BSaeule_DX56D.t29
│   ├── BSaeule_DX56D.t30
│   ├── BSaeule_DX56D.t51
│   └── BSaeule_DX56D.t52
├── PartType_03_Deep
│   ├── RadhausAdapter.dat
│   ├── RadhausAdapter-Session_01.ofs
│   ├── RadhausAdapter.t51
│   └── RadhausAdapter.t52
├── PartType_04
│   ├── backup_Einleger.dat
│   ├── Einleger.dat
│   ├── Einleger-Session.ofs
│   ├── Einleger.t51
│   └── Einleger.t52
└── T52-PartFiles
    ├── A-Pilar-Reinforcement.t52
    ├── A-Pilar.t52
    ├── B-Pilar.t52
    ├── CrossBeam.t52
    ├── FrontFender_A.t52
    ├── FrontFender_B.t52
    ├── TailGate.t52
    ├── TankCapInsert.t52
    ├── Tunnel.t52
    ├── WheelhouseAdapter.t52
    └── Wheelhouse.t52

File extensions:
  .dat   OpenForm simulation input (process parameters, material, blank)
  .ofs   OpenForm session file (post-processing script)
  .t51   Material / tool geometry
  .t52   Blank / part geometry (main mesh)
  .t53   Additional geometry component
  .t28   Mesh data (connectivity)
  .t29   Mesh data (node coordinates)
  .t30   Mesh data (element data)
  .nod   Node definitions
  .ele   Element definitions


================================================================================
ensima/  (Python package)
================================================================================

ensima/
├── __init__.py
├── __main__.py
├── optimize.py               Entry point: argument parsing and main optimization loop
├── classes/
│   ├── acquisition_function.py   EI and HGAL acquisition functions
│   ├── bayesian_optimization.py  Main BO orchestration class
│   ├── execute.py                Shell command execution and logging
│   ├── expert.py                 Expert model wrapper
│   ├── file_modifier.py          Reads/writes OpenForm .dat input files
│   ├── geometry_nn.py            PointNet-based geometry encoder
│   ├── license_server.py         RLM license server management
│   ├── logger.py                 Logging setup
│   ├── mixture_of_experts.py     MoE model with geometry-based gating
│   ├── model.py                  GP model (PyTorch / scikit-learn backends)
│   ├── print.py                  Rich-based console output
│   ├── progress_watcher.py       Monitors simulation log files for progress
│   ├── search_space.py           Search space construction (grid/random)
│   └── simulation.py             Simulation execution and result parsing
└── helpers/
    ├── adjust_args_cluster.py    Adjusts paths/args for HPC cluster environments
    ├── clustering.py             Clustering utilities for MoE
    ├── co2.py                    CO2 emission estimation
    ├── complexity.py             Part complexity classification from geometry
    ├── energy.py                 Energy consumption monitoring (CPU/GPU)
    ├── misc.py                   Miscellaneous utilities
    ├── objective_function.py     Dummy objective functions for testing
    ├── optimum.py                Optimum extraction from results
    ├── pareto.py                 Pareto frontier computation
    ├── parse_args.py             CLI argument definitions
    ├── parse_results.py          Parses OpenForm simulation output
    ├── plot.py                   Plotting utilities
    ├── read_data.py              CSV/data loading
    ├── read_geometry.py          Reads .t52 geometry files
    ├── serilaize.py              JSON serialization helpers
    └── units.py                  Unit conversion utilities


================================================================================
examples/
================================================================================

examples/
├── example_optimization.py                Single-part Bayesian optimization
├── example_moe_optimization.py            Multi-part optimization with Mixture of Experts
├── example_expert.py                      Expert-guided optimization with fixed input sets
├── example_filtered_optimization_cluster.py  Filtered optimization on an HPC cluster
├── example_train.py                       Train a GP model on benchmark CSV data
├── example_train_subset.py                Train on a filtered subset of the data
├── example_detailed_training.py           Step-by-step GP training walkthrough
└── example_plot.py                        Plot optimization results


================================================================================
test/
================================================================================

test/
├── Makefile                     Test runner (targets: all, silent, verbose)
├── csv/                         CSV datasets used by tests and examples
│   ├── DataSets-AIandML.csv
│   ├── DataSets-AIandML_20250401.csv
│   ├── DataSets-AIandML_labeled.csv
│   ├── DataSets-AIandML_expert.csv
│   ├── DataSets-AIandML_Asaeule.csv
│   ├── DataSets.csv
│   ├── ASaeule-Results_01.csv
│   ├── BSaeule_DX56D-Results_01.csv
│   └── Results-CylindricalCup_23-05-02.csv
├── test_csv_parser.py           Tests for CSV parsing utilities
├── test_license.py              Tests for license server management
├── test_moe.py                  Tests for Mixture of Experts model
├── test_optimization.py         Integration tests for the full BO workflow
├── test_pareto.py               Tests for Pareto frontier computation
├── test_progress.py             Tests for simulation progress watcher
└── test_sample_space.py         Tests for search space construction


================================================================================
tools/
================================================================================

tools/
├── plot_results.ipynb           Jupyter notebook for plotting JIMS experiment results
├── plot_results.py              Standalone script version of the plotting notebook
├── Import-CSV.c                 C utility for importing CSV results into OpenForm
├── original_Import-CSV.c        Original version of the CSV importer
├── Makefile                     Builds the csv_parser binary from Import-CSV.c
├── csv_parser                   Compiled binary of Import-CSV.c
└── test.ipynb                   Scratch notebook for quick experiments
