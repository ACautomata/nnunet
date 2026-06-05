# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This repository is being initialized as a Python project for nnUNet-related medical imaging work. At the time this file was created, the codebase was empty except for Claude session artifacts, so the guidance below records the agreed project constraints rather than existing application architecture.

## Environment and Dependency Constraints

- Use the conda environment named `nnunet`.
- If the environment does not exist, create it with Python 3.11:
  ```bash
  conda create -n nnunet python=3.11 -y
  ```
- Activate the environment before running project commands:
  ```bash
  conda activate nnunet
  ```
- Manage Python packaging through `pyproject.toml`.
- Use `hatchling` as the build backend.
- Use a `src/` layout for importable Python packages.
- Core ML dependencies are PyTorch, MONAI, Hydra, and `stable-pretraining`.

## Common Commands

Run these from the repository root after activating the conda environment.

### Install for Development

```bash
python -m pip install -e ".[dev]"
```

### Build Package

```bash
python -m build
```

### Run Tests

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_example.py
```

Run a single test by node id:

```bash
pytest tests/test_example.py::test_import_package
```

### Lint and Format

```bash
ruff check .
ruff format .
```

### Type Check

```bash
mypy src
```

## High-Level Structure

Expected project structure:

- `pyproject.toml` — package metadata, hatchling build backend, runtime dependencies, and dev tooling configuration.
- `environment.yml` — conda environment definition for reproducible local setup.
- `src/nnunet/` — importable application/library code.
- `configs/` — Hydra configuration files for experiments and training runs.
- `tests/` — pytest test suite.

Keep implementation code under `src/nnunet/` rather than placing importable modules at the repository root.

## Current Notes for Future Claude Instances

- This repository is not currently a git repository.
- Do not infer architecture from the upstream nnU-Net project unless the user explicitly asks to mirror or integrate it.
- When adding training or preprocessing code, prefer small, explicit modules first; avoid introducing broad framework abstractions before concrete workflows exist.
