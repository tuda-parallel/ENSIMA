![](https://img.shields.io/github/last-commit/tuda-parallel/ENSIMA)
![contributors](https://img.shields.io/github/contributors/tuda-parallel/ENSIMA)
![issues](https://img.shields.io/github/issues/tuda-parallel/ENSIMA)
![](https://img.shields.io/github/languages/code-size/tuda-parallel/ENSIMA)
![](https://img.shields.io/github/languages/top/tuda-parallel/ENSIMA)
![license][license.badge]

<br />
<div align="center">
  <h1 align="center">ENSIMA</h1>
  <p align="center">
    <h3 align="center">AI-driven parameter optimization for metal stamping simulations</h3>
    <a href="https://github.com/tuda-parallel/ENSIMA/issues">Report Bug</a>
    ·
    <a href="https://github.com/tuda-parallel/ENSIMA/issues">Request Feature</a>
  </p>
</div>

`ensima` is the optimization component of the [ENSIMA project](https://www.vi-hps.org/projects/ensima/overview/overview.html).
It applies Bayesian optimization with Gaussian Process Regression to accelerate design parameter selection in sheet metal forming simulations, reducing the number of costly simulation runs needed to find optimal process parameters.

## Overview

`ensima` provides tools for optimizing and managing parameter selection for stamping simulations. Key features include:

- **Bayesian Optimization** with Gaussian Process Regression (GPR) backend supporting both PyTorch and scikit-learn
- **Mixture of Experts (MoE)** with PointNet geometry encoding for multi-part optimization
- **Acquisition functions**: Expected Improvement (EI), Human-Guided Active Learning (HGAL)
- **End conditions**: no improvement, constant minimum, energy budget
- **HPC support**: cluster-aware argument adjustment, license server management, energy monitoring
- **Multi-objective optimization** with Pareto frontier analysis

For a full breakdown of the repository structure and file descriptions, see [files.md](./files.md).

## Installation

To install `ensima` in a virtual environment, use the `Makefile`:

```sh
make install
```

This will:

1. Create a virtual environment in `.venv`.
2. Install `ensima` in editable mode with optional dependencies.

To install optional libraries (ruff, black, nbstripout, numba, etc.):

```sh
make debug
```

## Activating the Virtual Environment

After installation, activate the virtual environment with:

```sh
source .venv/bin/activate
```

## Usage

Once installed, you can use the command-line tool:

```sh
ensima -h
```

### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|---|---|---|---|---|
| `--ofsolver` | `-ofs` | `str` | `None` | Path to the OpenFOAM solver executable. |
| `--jobname` | `-j` | `str` | `None` | Name of the optimization job. |
| `--path` | `-p` | `str` | `"./"` | Path to job directory. |
| `--output` | `-o` | `str` | `None` | Output directory or filename for results. |
| `--cores` | `-c` | `int` | `4` | Number of CPU cores allocated. |
| `--openform` | `-ofm` | `str` | `None` | Path to the OpenForm software. |
| `--session` | `-s` | `str` | `None` | OpenForm session file. |
| `--plot` | `-pl` | flag | `False` | Enable plotting of results. |
| `--iterations` | `-it` | `int` | `20` | Number of optimization iterations. |
| `--parallel_samples` | `-ps` | `int` | `1` | Number of parallel samples per iteration. |
| `--energy_estimation` | `-e` | flag | `False` | Enable energy estimation. |
| `--method` | `-m` | `str` | `bo` | Optimization method (`bo`, `hgal`). |
| `--end_condition` | `-ec` | `str` | `no_improvement` | Termination condition (`no_improvement`, `constant_min`, `energy_budget`). |
| `--end_value` | `-ev` | `float` | `5` | Threshold for the end condition. |
| `--x_fields` | `-x` | `str+` | `["Rp","D","p","Fr","db"]` | Input parameter field names. |
| `--y_fields` | `-y` | `str+` | `["L1","L2","L3","L4","L5","L6"]` | Output field names. |
| `--log_level` | `-l` | `str` | `INFO` | Logging level. |

### Example Usage

Run with a dummy objective function (no simulation required):

```sh
ensima
```

Run with custom parameters:

```sh
ensima --ofsolver /path/to/solver --openform /path/to/openform \
  --jobname "OptimizationJob" --path /path/to/job \
  --cores 8 --session session.ofs --iterations 30 --parallel_samples 2 -e
```

#### Energy Budget End Condition

```sh
ensima --iterations 40 -e --end_condition energy_budget --end_value 0.8
```

#### No Improvement Condition

```sh
ensima -it 40 --end_condition no_improvement --end_value 3
```

#### Constant Minimum Condition

```sh
ensima -it 40 --end_condition constant_min --end_value 3
```

### Using the Optimization in a Python Script

```python
from ensima.helpers.parse_args import parse_arguments
from ensima.classes.bayesian_optimization import BayesianOptimization
import numpy as np

def dummy_objective_function(x):
    return np.sin(x).sum(axis=1).reshape(-1, 1)

args = parse_arguments(["--jobname", "PythonOptimization", "--iterations", "10"])

x_init = np.random.rand(5, 1) * 10
y_init = dummy_objective_function(x_init)

optimizer = BayesianOptimization(
    x=x_init, y=y_init, args=args, objective_function=dummy_objective_function
)
optimizer.optimize(n_iters=args.iterations)
```

<p align="right"><a href="#ensima">⬆</a></p>

## Testing

A test suite is provided under [test](./test). Run all tests with:

```sh
cd test && make
```

Or silently:

```sh
cd test && make silent
```

To run a specific test directly:

```sh
cd test && python3 test_optimization.py
```

<p align="right"><a href="#ensima">⬆</a></p>

## Examples

Several examples are provided under [examples](./examples):

| Script | Description |
|---|---|
| `example_optimization.py` | Basic single-part Bayesian optimization |
| `example_moe_optimization.py` | Multi-part optimization with Mixture of Experts |
| `example_expert.py` | Expert-based filtered optimization |
| `example_filtered_optimization_cluster.py` | Filtered optimization on an HPC cluster |
| `example_train.py` | Training a GP model on benchmark CSV data |
| `example_train_subset.py` | Training on a filtered subset |
| `example_detailed_training.py` | Detailed GP training walkthrough |
| `example_plot.py` | Plotting optimization results |

Experiment data and results used for the JIMS paper are available under [artifacts/JIMS](./artifacts/JIMS), organized by experiment type (`MLVGP`, `MOE`, `no_optimization`) and part geometry.

<p align="right"><a href="#ensima">⬆</a></p>

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/tuda-parallel/ENSIMA/issues).

<p align="right"><a href="#ensima">⬆</a></p>

## Contact

- Ahmad Tarraf — <ahmad.tarraf@tu-darmstadt.de>

<p align="right"><a href="#ensima">⬆</a></p>

## License

![license][license.badge]

Distributed under the BSD 3-Clause License. See [LICENSE](./LICENSE) for more information.

<p align="right"><a href="#ensima">⬆</a></p>

## Acknowledgments

The optimization loop in `ensima` is developed and maintained by the [Parallel Programming](https://www.parallel.informatik.tu-darmstadt.de/) group at TU Darmstadt.
The main author of this project is Ahmad Tarraf.

This software is part of the [ENSIMA project](https://www.vi-hps.org/projects/ensima/overview/overview.html), funded by the German Federal Ministry of Research, Technology, and Space (BMBF), grant period 2022–2025. The project aims to accelerate design parameter optimization in sheet metal forming through AI-based methods, approximate computing, and heterogeneous hardware.

<p align="right"><a href="#ensima">⬆</a></p>

[license.badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
