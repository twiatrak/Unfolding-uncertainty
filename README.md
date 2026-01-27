# Unfolding (Inverse Problems) with Uncertainty Propagation

This project is a small study of inverse problems: how to estimate an underlying “true” distribution from a noisy, smeared observation.
It focuses on two things that make inverse problems interesting in practice:

- **Ill-posedness**: naive inversion is unstable.
- **Uncertainty**: the estimate is only useful if you can propagate uncertainties and validate coverage.

## Motivation
I built this as a learning project because unfolding sits right at the intersection of linear algebra, statistics, and scientific computing.
It’s easy to write down the forward model, but getting a stable inverse with honest uncertainties is where the real work is.

Concretely, I wanted to:

- implement standard regularized solutions from scratch (so the math is not a black box),
- propagate covariance through the estimator,
- validate the whole pipeline with toy Monte Carlo (bias/variance, pulls).

## What’s included

- `src/unfolding/`: a small Python library
	- Tikhonov unfolding (ridge-style regularization)
	- truncated SVD (TSVD) unfolding
	- covariance propagation utilities
	- toy-MC validation helpers
- `notebooks/demo_unfolding.ipynb`: a runnable demo scanning regularization strength and showing stability / bias–variance / pull behavior
- `docs/analysis_note.tex`: a short write-up template (figures saved by the notebook)

## Quickstart

Create a virtualenv and install:

- `python -m venv .venv`
- Activate it (Windows PowerShell): `.\.venv\Scripts\Activate.ps1`
- `python -m pip install -U pip`
- `pip install -e .[dev]`

Run tests:

- `pytest`

Open the notebook:

- `jupyter lab`

Build the note (requires LaTeX):

- `cd docs`
- `latexmk -pdf analysis_note.tex`

## Ideas for future work

If I continue developing this, I’d like to add:

- **Better regularization selection**: L-curve variants, generalized cross-validation, and closure/coverage-driven tuning.
- **More realistic forward models**: explicit acceptance/efficiency, non-square response matrices, and backgrounds.
- **Systematic uncertainties**: propagate both statistical and systematic covariance, and study their interplay.
- **Alternative unfolding methods**: iterative/EM-style approaches (with early stopping as regularization).
- **More validation tooling**: automated coverage plots, stress tests vs. model mismatch, and documented “failure modes”.
