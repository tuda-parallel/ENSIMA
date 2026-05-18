PYTHON = .venv/bin/python3
SHELL := /bin/bash


# Check if python exists in venv, otherwise fallback to default
ifeq ("$(PYTHON)",".venv/bin/python3")
ifeq ("$(wildcard ${PYTHON})","")
$(warning Python not found in .venv, falling back to default)
PYTHON=python3
endif
else
$(info Using python: $(PYTHON))
endif

REQUIRED_PYTHON_VERSION := 3.8
PYTHON_VERSION := $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

version_check = $(shell python3 -c 'import sys; required_version = tuple(map(int, "$(REQUIRED_PYTHON_VERSION)".split("."))); \
    sys.exit((sys.version_info.major, sys.version_info.minor) < required_version)')

# Terminate the make process if the version is below the required version
ifeq ($(version_check),1)
$(error Python $(REQUIRED_PYTHON_VERSION) or higher is required. Installed version: $(PYTHON_VERSION))
else
$(info Python version is satisfied: $(PYTHON_VERSION))
endif


all: install

install: development msg

debug: venv
	.venv/bin/python -m pip install -e '.[optional-libs]'

development: venv
	# Install ROCm PyTorch only on electric
	@if [ "$$(hostname)" = "electric" ]; then \
		echo "Installing ROCm PyTorch for electric"; \
		.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.4; \
	fi
	# Install optional libraries
	.venv/bin/python -m pip install -e '.[optional-libs]'

venv:
	$(PYTHON) -m venv .venv
	@echo -e "Environment created. Using python from .venv/bin/python3"

msg:
	@echo -e "\nensima_optimize was installed in a python environment in .venv"
	@echo -e "To activate python from this venv, call:\nsource $(PWD)/.venv/bin/activate\n"
	@echo -e "Afterwards, you can just call 'ensima_optimize [args]'"


# test
test:
	cd test && ../$(PYTHON) -m pytest

test_parallel:
	@$(PYTHON) -m pip show pytest-xdist > /dev/null 2>&1 || $(PYTHON) -m pip install pytest-xdist
	cd test && ../$(PYTHON) -m pytest -n 4

test_failed:
	cd test && ../$(PYTHON) -m pytest --ff


# style
check_style: check_tools
	black .
	ruff check --fix

check_tools:
	@command -v black >/dev/null 2>&1 || (echo "black not found, installing..." && $(PYTHON) -m pip install black)
	@command -v ruff >/dev/null 2>&1 || (echo "ruff not found, installing..." && $(PYTHON) -m pip install ruff)


# build / publish
build: pack

pack:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

testpypi: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

testpypi-install:
	$(PYTHON) -m pip install --index-url https://test.pypi.org/simple/ --no-deps ensima-hpc

pypi: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload dist/*
	$(PYTHON) -m pip install ensima-hpc


# clean
clean_project:
	$(PYTHON) -m pip uninstall --yes ensima-hpc || echo "no installation of ensima-hpc found"

clean: clean_project
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -f optimization_log.out

clean_all: clean
	rm -rf .venv
	rm -rf ensima_optimize.egg-info
	rm -rf dist
	rm -rf build


.PHONY: all install debug development venv msg test test_parallel test_failed check_style check_tools build pack testpypi testpypi-install pypi clean clean_project clean_all
