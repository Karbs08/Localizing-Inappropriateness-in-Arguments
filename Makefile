# Makefile for Localizing-Inappropriateness-in-Arguments
#
# The default initialization workflow creates a local virtual environment,
# installs all Python dependencies, and prepares the canonical dataset.
#
# The virtual environment does not need to be activated when using the
# Makefile because all commands explicitly use the Python executable inside
# `.venv`.
#
# Typical usage:
#
#   make init
#   make jupyter
#
# The Gemini API key is intentionally not part of `make init`. It is only
# required for methods/llm_spans.ipynb and should be stored in a project-level
# `.env` file as GEMINI_API_KEY.

PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_JUPYTER := $(VENV)/bin/jupyter

REQUIREMENTS := requirements.txt
PREPARED_DATA := data/processed/appropriateness_prepared.parquet
PREPARED_METADATA := data/processed/appropriateness_prepared_metadata.json

.PHONY: help init venv install prepare-data force-prepare-data kernel jupyter clean-data clean-venv

help:
	@echo "Available targets:"
	@echo "  make init               Create venv, install dependencies, and prepare data"
	@echo "  make venv               Create the local .venv environment"
	@echo "  make install            Install/update dependencies from requirements.txt"
	@echo "  make prepare-data       Prepare the dataset if processed files are missing"
	@echo "  make force-prepare-data Recreate the prepared dataset explicitly"
	@echo "  make kernel             Register the .venv as a Jupyter kernel"
	@echo "  make jupyter            Start JupyterLab from the project root"
	@echo "  make clean-data         Remove generated raw and processed data"
	@echo "  make clean-venv         Remove the local virtual environment"

# Complete local project initialization.
init: install prepare-data
	@echo ""
	@echo "Initialization complete."
	@echo "Start JupyterLab with: make jupyter"
	@echo "Optional: register a named Jupyter kernel with: make kernel"

# Create the virtual environment only if it does not already exist.
$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

venv: $(VENV_PYTHON)

# Install project dependencies.
# This target intentionally runs on every explicit `make install` or `make init`
# so changes to requirements.txt are picked up automatically.
install: $(VENV_PYTHON) $(REQUIREMENTS)
	$(VENV_PYTHON) -m pip install -r $(REQUIREMENTS)

# Prepare the shared dataset only when the expected processed outputs are absent.
$(PREPARED_DATA) $(PREPARED_METADATA): $(VENV_PYTHON)
	$(VENV_PYTHON) -m src.prepare_data

prepare-data: $(PREPARED_DATA) $(PREPARED_METADATA)

# Explicitly rerun data preparation, for example after changing normalization,
# classifier settings, dataset versions, or preprocessing logic.
force-prepare-data: $(VENV_PYTHON)
	$(VENV_PYTHON) -m src.prepare_data

# Register the project environment as an optional named Jupyter kernel.
kernel: install
	$(VENV_PYTHON) -m ipykernel install --user \
		--name localizing-inappropriateness \
		--display-name "Python (Localizing Inappropriateness)"

# Start JupyterLab from the repository root using the project environment.
jupyter: install
	$(VENV_JUPYTER) lab

# Remove locally generated dataset files.
# The next `make prepare-data` or `make init` recreates them.
clean-data:
	rm -rf data/raw/appropriateness_corpus
	rm -rf data/processed

# Remove the complete local Python environment.
clean-venv:
	rm -rf $(VENV)


# Human study
.PHONY: study

study:
	python human_study/prepare_survey_data.py
	python human_study/generate_limesurvey.py

study-evaluate:
	python human_study/evaluate_survey_results.py \
		--responses human_study/survey_results/survey_results_raw.csv \
		--entity-reference-method llm
