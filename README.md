# ensima-hpc

## Overview

`ensima-hpc` is a Python package designed for high-performance computing (HPC) containing the software from the ENSIMA project. It provides tools for optimizing and managing parameter selection for process simulation.

## Installation

To install `ensima-hpc` in a virtual environment, use the `Makefile` commands:

```sh
make install
```

This will:

1. Create a virtual environment in `.venv`.
2. Install `ensima-hpc` in editable mode.

## Activating the Virtual Environment

After installation, activate the virtual environment with:

```sh
source .venv/bin/activate
```

## Usage

Once installed, you can use the command-line tool:

```sh
ensima_optimize -h 
```

### Command-Line Arguments

You can customize the execution of Bayesian optimization using the following command-line arguments:


| Argument              | Short Option | Type   | Default Value | Description |
|-----------------------|--------------|--------|---------------|-------------|
| `--ofsolver`          | `-ofs`       | `str`  | `/rwthfs/rz/cluster/home/rwth1453/OFSolv/Software/OFSolv_V2.16.0-E/bin/OFSolv_V2.16.0-E_amd64.exe` | Path to the OpenFOAM solver executable. |
| `--jobname`           | `-j`         | `str`  | `None`        | Name of the optimization job. |
| `--path`              | `-p`         | `str`  | `"./"`        | Path to job directory. |
| `--csv_file`          | `-i`         | `str`  | `None`        | CSV file name for results. |
| `--output`            | `-o`         | `str`  | `None`        | Output directory or filename for results. |
| `--cores`             | `-c`         | `int`  | `4`           | Number of CPU cores allocated. |
| `--openform`          | `-ofm`       | `str`  | `/home/cg021604/rwth1453/OpenForm_daily_linux64/OpenForm_64_batch` | Path to the OpenForm software. |
| `--session`           | `-s`         | `str`  | `None`        | OpenForm session file. |
| `--plot`              | `-pl`        | `flag` | `False`       | Flag to enable plotting of the results. |
| `--iterations`        | `-it`        | `int`  | `20`          | Number of iterations for the optimization process. |
| `--parallel_samples`  | `-ps`        | `int`  | `1`           | Number of parallel samples to evaluate in each iteration. |
| `--energy_estimation` | `-e`         | `bool` | `False`       | Energy estimation if True. |
| `--method`            | `-m`         | `str`  | `bo`          | Optimization method to use. Supported methods are Bayesian Optimization ('bo') and Human-Guided Active Learning ('hgal'). |
| `--end_condition`     | `-ec`        | `str`  | `no_improvement` | Specifies the condition to terminate the optimization process (no_improvement, constant_min, energy_budget). |
| `--end_value`         | `-ev`        | `float`| `5`           | Value used to terminate the optimization process, depending on the selected --end_condition. |

### Example Usage

To run the optimization with default parameters (launches a dummy version with a preset objective function):

```sh
ensima_optimize
```

To specify custom parameters, use the following format:

```sh
ensima_optimize --ofsolver /path/to/solver --openform /path/to/openform --jobname "OptimizationJob" --path /path/to/OptimizationJob --output results.extension --cores 8 --session "session.obs" --iterations 30 --parallel_samples 2 --energy_estimation True --method bo
```

### Dummy Examples with a Preset Objective Function

#### Running the Dummy Objective Function

To run the optimization with a dummy objective function for testing purposes, you can use the following command:

```sh
ensima_optimize --jobname "DummyOptimization" --iterations 10 --method bo
```

This will run the optimization for 10 iterations using the Bayesian Optimization method with a dummy objective function.

#### Running with Human-Guided Active Learning (HGAL)

To run the optimization using the Human-Guided Active Learning method, you can use the following command:

```sh
ensima_optimize --jobname "HGAL_Optimization" --iterations 10 --method hgal
```

This will run the optimization for 10 iterations using the Human-Guided Active Learning method, where you will be prompted to input the next sample points manually.

### Examples with end condicitions

#### Example: Energy Budget End Condition
To run the optimization with energy estimation, where the optimization ends if the energy exceeds 0.8 joules, you can use the following command:

```sh
ensima_optimize --iterations 40 -e  --end_condition energy_budget --end_value 0.8
```

This will run the optimization for 30 iterations using the Bayesian Optimization method, with energy estimation enabled. The optimization will terminate if the total energy exceeds 0.8 joules.

#### Example: No Improvement Condition
To run the optimization with the termination condition set to "no_improvement" (i.e., terminate if the simulation returns the same value for X consecutive iterations), use the following command:
```bash
ensima_optimize  -it 40 --end_condition no_improvement --end_value 3
```
This will run the optimization for 10 iterations, and the optimization will stop if the same result is returned for 3 consecutive iterations.

#### Example : Constant Minimum Condition
To terminate the optimization when the same minimum is found for 3 consecutive iterations, use the following command:

```bash
ensima_optimize  -it 40 --end_condition constant_min --end_value 3
```
In this case, the optimization will stop if the minimum value of the objective function does not change for 3 consecutive iterations

### Using the Optimization in a Python Script

You can also use the optimization functionality within a Python script. Here is an example:

```python
from ensima_optimize.helpers.parse_args import parse_arguments
from ensima_optimize.classes.bayesian_optimization import BayesianOptimization
import numpy as np

# Define a dummy objective function
def dummy_objective_function(x):
    return np.sin(x).sum(axis=1).reshape(-1, 1)

# Parse arguments (you can also pass a list of arguments)
args = parse_arguments(["--jobname", "PythonOptimization", "--iterations", "10", "--method", "bo"])

# Initial samples (x) and corresponding outputs (y)
x_init = np.random.rand(5, 1) * 10  # 5 samples, 1 feature
y_init = dummy_objective_function(x_init)

# Create a BayesianOptimization instance
optimizer = BayesianOptimization(x=x_init, y=y_init, args=args, objective_function=dummy_objective_function)

# Run the optimization
optimizer.optimize(n_iters=args.iterations)

# Optionally, plot the results
if args.plot:
    optimizer.plot()
```

This script demonstrates how to set up and run the optimization process programmatically using the `BayesianOptimization` class and a dummy objective function.

For more details, visit the official repository.
