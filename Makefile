# ============================================================
# SeismoAI golden path — clone to running app
#   make setup → data → validate → train → app
# ============================================================

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
SEISMO := $(BIN)/seismo

.PHONY: help setup data validate features train evaluate app notebooks test lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install pinned dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

data: ## Fetch USGS catalog into data/raw/ (resumable)
	$(SEISMO) data fetch

validate: ## Clean raw data, run validation gates, build processed dataset
	$(SEISMO) data validate

status: ## Show fetch manifest summary
	$(SEISMO) data status

features: ## Build ML feature dataset
	$(SEISMO) features build

train: ## Train all models
	$(SEISMO) model train --all

evaluate: ## Evaluate models and write reports
	$(SEISMO) model evaluate

app: ## Run the Streamlit application
	$(BIN)/streamlit run app/streamlit_app.py

notebooks: ## Convert and execute all notebooks (outputs visible on GitHub)
	$(PYTHON) tools/build_notebooks.py

test: ## Run tests
	$(BIN)/pytest

lint: ## Run linter
	$(BIN)/ruff check src tests

clean: ## Remove generated data and caches
	rm -rf data/raw data/processed data/features reports/logs .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
