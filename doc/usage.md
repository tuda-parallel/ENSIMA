# Usage

## Command-line interface

Once installed, the `ensima` command is available:

```sh
ensima [options]
```

Running without arguments launches a dummy optimization with a built-in objective function — no simulation software required:

```sh
ensima
```

### Key arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--ofsolver` | `-ofs` | `None` | Path to the OpenForm solver executable |
| `--openform` | `-ofm` | `None` | Path to the OpenForm software |
| `--jobname` | `-j` | `None` | Name of the optimization job |
| `--path` | `-p` | `./` | Path to the job directory |
| `--session` | `-s` | `None` | OpenForm session file |
| `--output` | `-o` | `None` | Output CSV file for results |
| `--cores` | `-c` | `4` | Number of CPU cores |
| `--iterations` | `-it` | `20` | Number of optimization iterations |
| `--parallel_samples` | `-ps` | `1` | Parallel samples per iteration |
| `--energy_estimation` | `-e` | `False` | Enable energy monitoring |
| `--method` | `-m` | `bo` | Optimization method: `bo` or `hgal` |
| `--end_condition` | `-ec` | `no_improvement` | Stop condition: `no_improvement`, `constant_min`, `energy_budget` |
| `--end_value` | `-ev` | `5` | Threshold for the end condition |
| `--x_fields` | `-x` | `["Rp","D","p","Fr","db"]` | Input parameter names |
| `--y_fields` | `-y` | `["L1",...,"L6"]` | Output field names |
| `--log_level` | `-l` | `INFO` | Logging level |

### Examples

Run with energy budget termination:

```sh
ensima --iterations 40 -e --end_condition energy_budget --end_value 0.8
```

Stop after 3 iterations without improvement:

```sh
ensima -it 40 --end_condition no_improvement --end_value 3
```

Stop when the minimum stays constant for 3 iterations:

```sh
ensima -it 40 --end_condition constant_min --end_value 3
```

Full run with a simulation solver:

```sh
ensima --ofsolver /path/to/OFSolv \
       --openform /path/to/OpenForm \
       --jobname MyJob --path /path/to/job \
       --session MyJob-Session.ofs \
       --cores 8 --iterations 30 -e
```

---

## Python API

### Basic optimization

```python
from ensima.helpers.parse_args import parse_arguments
from ensima.classes.bayesian_optimization import BayesianOptimization
import numpy as np

def objective(x):
    return np.sin(x).sum(axis=1).reshape(-1, 1)

args = parse_arguments(["--jobname", "MyJob", "--iterations", "10"])

x_init = np.random.rand(5, 1) * 10
y_init = objective(x_init)

optimizer = BayesianOptimization(
    args=args,
    x=x_init,
    y=y_init,
    objective_function=objective,
)
optimizer.optimize(n_iters=args.iterations)
```

### Loading data from CSV

```python
from ensima.helpers.read_data import read_data

x, y = read_data(
    "test/csv/DataSets-AIandML_20250401.csv",
    x_fields=["Fr", "p", "D"],
    y_fields=["L1", "L2", "L3", "L4", "L5", "L6"],
)
```

### Mixture of Experts (multi-part)

See `examples/example_moe_optimization.py` for a full working example. Key difference
from single-part optimization is passing `parts` and `types` to `BayesianOptimization`:

```python
optimizer = BayesianOptimization(
    args=args,
    x=x, y=y,
    parts=[("PartA", "/path/to/PartA"), ("PartB", "/path/to/PartB")],
    types=type_labels,
)
```

---

## Running tests

```sh
cd test && make          # run all tests
cd test && make silent   # suppress output
cd test && make verbose  # verbose output
```

## Style checks

```sh
make check_style   # run black + ruff (auto-fixes where possible)
```
