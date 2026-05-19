# Contributing

## Getting started

1. Clone the repository and install in editable mode:

   ```sh
   git clone https://github.com/tuda-parallel/ENSIMA.git
   cd ENSIMA
   make install
   source .venv/bin/activate
   ```

2. Create a feature branch and make your changes.

3. Run style checks before committing:

   ```sh
   make check_style
   ```

4. Run the test suite:

   ```sh
   cd test && make
   ```

5. Open a pull request on [GitHub](https://github.com/tuda-parallel/ENSIMA/pulls).

---

## Code style

`ensima` uses **black** for formatting and **ruff** for linting. Both are enforced by
a pre-commit hook installed automatically with the development environment.

Line length is 90 characters. Import order follows `isort` conventions
(`known-first-party = ["ensima"]`).

To fix style issues automatically:

```sh
make check_style
```

---

## Tests

Tests live in `test/` and use `pytest`. The `Makefile` provides three targets:

```sh
cd test && make          # run all tests (default output)
cd test && make silent   # suppress stdout/stderr
cd test && make verbose  # full verbose output
```

Key test files:

| File | What it tests |
|---|---|
| `test_optimization.py` | Full BO workflow, CSV loading, dummy objective |
| `test_moe.py` | Mixture of Experts model |
| `test_sample_space.py` | Search space construction (grid, random) |
| `test_pareto.py` | Pareto frontier computation |
| `test_progress.py` | Simulation progress watcher |
| `test_csv_parser.py` | CSV parsing utilities |
| `test_license.py` | License server management |

---

## Adding a new acquisition function

1. Add a static method to `ensima/classes/acquisition_function.py`.
2. Wire it into `BayesianOptimization._select_next_point()` in `bayesian_optimization.py`.
3. Expose it as a `--method` option in `ensima/helpers/parse_args.py`.
4. Add a test in `test/test_optimization.py`.

## Adding a new end condition

1. Add the condition name to the `--end_condition` choices in `parse_args.py`.
2. Implement the check in `BayesianOptimization.optimize()`.

---

## Publishing

Build and upload to PyPI:

```sh
make pypi       # build + upload to PyPI
make testpypi   # build + upload to TestPyPI (for testing)
```
