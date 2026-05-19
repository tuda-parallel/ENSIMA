# Installation

## Requirements

- Python 3.8 or higher
- A virtual environment (recommended)

## Install via Makefile

The recommended way to install `ensima` is through the provided `Makefile`, which
creates a virtual environment and installs all dependencies:

```sh
make install
```

This will:
1. Create a virtual environment in `.venv/`
2. Install `ensima` in editable mode with all optional dependencies

Activate the virtual environment afterwards:

```sh
source .venv/bin/activate
```

## Optional dependencies

To install only the optional tools (linters, notebook stripping, GPU monitoring):

```sh
make debug
```

This adds: `ruff`, `black`, `isort`, `nbstripout`, `colorlog`, `numba`, `pynvml`.

## Editable install (manual)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[optional-libs]'
```

## Verify installation

```sh
ensima -h
```

## Uninstall

```sh
make clean_all
```

This removes the virtual environment, egg-info, build artifacts, and the installed package.
