# Using ENSIMA with a Custom Simulation

ENSIMA is not limited to OpenForm. Any simulation can be plugged in by subclassing two abstract base classes and passing them to `main()`.

---

## Overview

Three steps are required to integrate a custom simulation:

1. **[Prepare the results CSV](#1-the-results-csv)** — a structured file that stores past simulation results and is read at startup to provide initial training data.
2. **[Implement `simulation.py`](#2-simulationpy--subclass-basesimulation)** — subclass `BaseSimulation`, implement `run()`, and pass your class to `main()`.
3. **[Implement `file_modifier.py`](#3-file_modifierpy--subclass-basefilemodifier-optional)** — subclass `BaseFileModifier` and implement `set_design_parameters()`. Only needed if your simulator reads parameters from an input file.

Two further steps are optional:

4. **[Custom args](#4-argspy--custom-command-line-arguments-optional)** — extend `parse_arguments` with simulation-specific flags.
5. **[Run script](#5-the-run-script)** — a small Python script that hard-codes all settings, avoiding a long command line.

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

**Cold start.** If you have no prior results the CSV can have an empty data section — headers and `START`/`END` with no data rows. The optimizer falls back to random sampling until enough data has been collected to fit the surrogate model.

**Appending results.** Each simulation run must append its result to the CSV so the surrogate model stays current across restarts. This is the responsibility of `run()` — call the helper below before returning `y`:

```python
def append_result(csv_path: str, x_row: list, y_row: list):
    with open(csv_path) as f:
        lines = f.readlines()
    end_idx = next(i for i, line in enumerate(lines) if line.strip() == "END")
    lines.insert(end_idx, ",".join(str(v) for v in x_row + y_row) + "\n")
    with open(csv_path, "w") as f:
        f.writelines(lines)
```

---

## 2. `simulation.py` — subclass `BaseSimulation`

Create a file in your project and subclass `BaseSimulation` from `ensima.classes.base_simulation`:

```python
# my_project/simulation.py
import subprocess
import numpy as np
from ensima.classes.base_simulation import BaseSimulation


class MySimulation(BaseSimulation):

    def run(self, file_modifier, lock, type_filter: bool = False) -> np.ndarray:
        params = dict(zip(self.args.x_fields, self.next_sample[-1].tolist()))

        # 1. Write parameters into input files (if a file modifier is provided)
        if file_modifier is not None:
            file_modifier.set_design_parameters(params)

        # 2. Run the simulation
        y = self._run(params)

        # 3. Append result to the CSV so the model stays current across restarts
        append_result(self.args.output, self.next_sample[-1].tolist(), y[0].tolist())

        return y

    def _run(self, params: dict) -> np.ndarray:
        """
        Invoke the simulator and return shape (1, n_outputs).

        Typical patterns:
          subprocess:   subprocess.run(["my_solver", "--config", "input.cfg"], check=True)
          Python API:   result = my_solver_lib.run(params)
          remote call:  result = client.evaluate(params)

        Read the result from wherever the simulator writes it (output file, stdout,
        database row, …) and return np.array([[out1, out2, ...]]).

        On failure, return np.full((1, n_outputs), np.nan) so the surrogate model
        skips the point gracefully instead of crashing.
        """
        raise NotImplementedError
```

**Important:** `--jobname` must be set in your run script. It is the simulation job name used as a label in the CSV. Without it, ENSIMA runs a built-in dummy loop instead of calling your simulation.

---

## 3. `file_modifier.py` — subclass `BaseFileModifier` (optional)

Create a file in your project and subclass `BaseFileModifier` from `ensima.classes.base_file_modifier`:

```python
# my_project/file_modifier.py
from ensima.classes.base_file_modifier import BaseFileModifier


class MyFileModifier(BaseFileModifier):

    def set_design_parameters(self, parameters: dict) -> None:
        """Write the proposed parameter values into the simulator's input file."""
        with open(self.input_file) as f:
            content = f.read()
        for key, value in parameters.items():
            # adapt to your file format — regex, keyword substitution, etc.
            content = content.replace(f"${{{key}}}", str(value))
        with open(self.input_file, "w") as f:
            f.write(content)
```

`BayesianOptimization` constructs the file modifier as:

```python
MyFileModifier(
    os.path.join(args.path, args.jobname + ".dat"),
    log_level=args.log_level,
)
```

So `args.path` and `args.jobname` determine which file is opened. Adjust `set_design_parameters` to match your input file format.

---

## 4. `args.py` — custom command-line arguments (optional)

If your simulation needs flags beyond ENSIMA's built-in list, extend `parse_arguments`:

```python
# my_project/args.py
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

Rather than a long command line, prepare a small Python script that hard-codes all settings for your use case — the same pattern used by the scripts in [`examples/`](../examples/):

```python
# my_run.py
from ensima.helpers.parse_args import parse_arguments  # or your extended version
from ensima.optimize import main

from my_project.simulation import MySimulation
from my_project.file_modifier import MyFileModifier  # omit if not needed

if __name__ == "__main__":
    args = parse_arguments([
        "-j",           "my_job",       # required: triggers the real simulation path
        "--x_fields",   "param1", "param2",
        "--y_fields",   "output1", "output2",
        "--path",       "/path/to/job/directory",
        "--output",     "/path/to/results.csv",
        "--iterations", "20",
        "--log_level",  "INFO",
    ])

    main(
        args=args,
        simulation_class=MySimulation,
        file_modifier_class=MyFileModifier,  # omit if not needed
    )
```

Run it with:

```sh
python my_run.py
```
