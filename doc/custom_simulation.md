# Using ENSIMA with a Custom Simulation

ENSIMA is not limited to OpenForm. Any simulation can be plugged in. This guide explains what you need to implement and what aspects matter for the optimization to work correctly.

---

## Overview

The optimization loop needs three things from you:

| What | Required | Purpose |
|---|---|---|
| **Results CSV** | yes | Supplies initial training data for the surrogate model |
| **Simulation callable** | yes | Runs one simulation and returns outputs as `np.ndarray` |
| **File modifier** | no | Writes the proposed parameters into your simulator's input files |

A small **run script** (rather than a long command line) is the recommended way to start a run — see [section 5](#5-the-run-script).

---

## 1. The results CSV

ENSIMA reads initial training data from a CSV at startup. A ready-to-use template is provided at [`doc/example_results.csv`](example_results.csv). The file must follow this structure:

```
$---------------------------------------------------------------
$ Row Format:
$ param1 , param2 , output1 , output2 ,
$---------------------------------------------------------------
START
0.5 , 1.2 , 3.41 , 0.82
1.0 , 0.9 , 2.10 , 1.54
END
```

Rules:
- A `$ Row Format:` line must be present, followed immediately by a `$`-prefixed line listing column names separated by commas.
- Data rows sit between `START` and `END` markers.
- **Column names must match `--x_fields` and `--y_fields` exactly** — the parser maps columns by name.
- Extra columns are allowed and ignored.

**Cold start.** If you have no prior results, the CSV can start with an empty data section — just the headers and the `START`/`END` markers and no data rows. The optimizer detects the shortage of points and falls back to random sampling until enough data has been collected to fit the surrogate model reliably.

**Appending new results.** Each time your simulation runs it must append its result to this file so the surrogate model stays current across restarts. Insert the new row before the `END` marker:

```python
def append_result(csv_path: str, x_row: list, y_row: list):
    with open(csv_path) as f:
        lines = f.readlines()
    end_idx = next(i for i, l in enumerate(lines) if l.strip() == "END")
    lines.insert(end_idx, ",".join(str(v) for v in x_row + y_row) + "\n")
    with open(csv_path, "w") as f:
        f.writelines(lines)
```

> **Parallel runs.** When `--parallel_samples > 1` the optimizer launches simulations in separate processes via `ProcessPoolExecutor`. CSV writes from concurrent processes will corrupt the file unless you serialise them. The simplest safe choice is to keep `--parallel_samples 1` for custom simulations. If you need parallelism, protect the append with a file lock (e.g. `filelock`).

---

## 2. `simulation.py` — the callable

The optimizer expects a callable with this signature:

```python
def run_simulation(x: np.ndarray) -> np.ndarray:
    ...
```

- `x` — shape `(1, n_inputs)`: one proposed parameter point.
- return value — shape `(1, n_outputs)`: the corresponding simulation outputs.

**The shape contract is strict.** A wrong shape causes a silent model update with misaligned data.

> **Important:** `BayesianOptimization` only uses a custom `objective_function` when `args.jobname` is not set. Do not pass `--jobname` in your run script.

A class-based implementation (preferred because it is picklable and carries `args`):

```python
import subprocess
import numpy as np


class Simulation:
    def __init__(self, args):
        self.args = args

    def __call__(self, x: np.ndarray) -> np.ndarray:
        params = dict(zip(self.args.x_fields, x[0].tolist()))

        # 1. Write parameters into input files (optional, see FileModifier below)
        modifier = FileModifier("/path/to/input_template.txt", params)
        modifier.write()

        # 2. Run the simulation
        y = self._run(params)

        # 3. Persist the result
        append_result(self.args.output, x[0].tolist(), y[0].tolist())

        return y

    def _run(self, params: dict) -> np.ndarray:
        """
        Launch the simulation and return outputs as shape (1, n_outputs).

        Typical patterns:
          - subprocess call:  subprocess.run(["my_solver", "--config", "input.txt"], check=True)
          - Python API call:  result = my_solver_lib.run(params)
          - REST / gRPC call: result = client.evaluate(params)

        Parse the solver's output (file, stdout, return value) and return it
        as a (1, n_outputs) array whose columns match --y_fields in order.
        """
        raise NotImplementedError
```

**What `_run` must do:**
1. Invoke the simulator (subprocess, library call, remote API — any mechanism works).
2. Wait for it to finish.
3. Read the result from wherever the simulator writes it (file, stdout, database row, …).
4. Return `np.array([[out1, out2, ...]])` — shape `(1, n_outputs)`, columns in the same order as `--y_fields`.

If the simulation can fail, return `np.full((1, len(self.args.y_fields)), np.nan)` on failure so the surrogate model skips the point gracefully rather than crashing.

---

## 3. `file_modifier.py` — modifying input files (optional)

If your simulator reads parameters from a configuration or input file, provide a class that writes the new values before each run:

```python
class FileModifier:
    def __init__(self, template_path: str, params: dict):
        self.template_path = template_path
        self.params = params

    def write(self):
        with open(self.template_path) as f:
            content = f.read()
        for key, value in self.params.items():
            # adapt the replacement pattern to your file format
            content = content.replace(f"${{{key}}}", str(value))
        with open(self.template_path, "w") as f:
            f.write(content)
```

The replacement strategy depends entirely on your file format — regex, keyword substitution, or a templating library are all valid.

---

## 4. `args.py` — custom command-line arguments (optional)

If your simulation needs flags that are not in ENSIMA's built-in argument list, extend `parse_arguments`:

```python
from ensima.helpers.parse_args import parse_arguments as _parse_arguments
import argparse


def parse_arguments(arg_list=None):
    args = _parse_arguments(arg_list)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solver_binary", type=str, required=True)
    parser.add_argument("--config_template", type=str, required=True)
    extra, _ = parser.parse_known_args(arg_list)
    args.__dict__.update(vars(extra))

    return args
```

---

## 5. The run script

Rather than typing a long command line each time, prepare a small Python script that hard-codes all settings for your specific use case. This is the same pattern used by the scripts in [`examples/`](../examples/).

```python
# my_run.py
import numpy as np
from ensima.classes.bayesian_optimization import BayesianOptimization
from ensima.helpers.parse_args import parse_arguments   # or your extended version
from ensima.helpers.read_data import read_data_type

if __name__ == "__main__":
    args = parse_arguments([
        # do NOT pass --jobname; it must be unset for a custom simulation
        "--x_fields",         "param1", "param2",
        "--y_fields",         "output1", "output2",
        "--output",           "/path/to/results.csv",
        "--iterations",       "20",
        "--parallel_samples", "1",       # keep at 1 unless CSV writes are locked
        "--log_level",        "INFO",
    ])

    # load initial training data from the CSV
    x, y, _ = read_data_type(args.output, args.x_fields, args.y_fields)

    # create the simulation callable
    simulation = Simulation(args)

    # run the optimisation
    bayes_opt = BayesianOptimization(args, x, y, objective_function=simulation)
    bayes_opt.optimize(args.iterations, args.parallel_samples)
```

Run it with:

```sh
python my_run.py
```

The optimizer calls `simulation(x)` at each iteration with the proposed point, waits for the result, updates the surrogate model, and selects the next point. Everything else — GP fitting, acquisition function, convergence check — is handled by ENSIMA.
