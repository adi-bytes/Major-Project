# PG-SBD Designer Root Makefile

# Override on Windows, e.g.:  make serve PYTHON=.venv/Scripts/python.exe
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff

.PHONY: help data calibrate train eval ui serve test demo lint

help:
	@echo "Available commands:"
	@echo "  make data       - Fetch raw dataset from repository and audit"
	@echo "  make calibrate  - Calibrate differentiable Lambert-W physics core"
	@echo "  make train      - Train ML baselines and hybrid physics-guided ensemble"
	@echo "  make eval       - Evaluate models and generate validation report"
	@echo "  make serve      - Run FastAPI server"
	@echo "  make test       - Run complete test suite with pytest"
	@echo "  make demo       - Run end-to-end demo script"
	@echo "  make lint       - Run ruff linter checks"

data:
	$(PYTHON) scripts/fetch_data.py
	$(PYTHON) scripts/audit_data.py
	$(PYTHON) -c "from pgsbd.data.load import load_dataset; from pgsbd.data.splits import create_grouped_splits; df, _, _, _ = load_dataset(); create_grouped_splits(df, seed=42)"

calibrate:
	$(PYTHON) scripts/calibrate.py

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/evaluate.py
	$(PYTHON) scripts/eval_reverse.py

ui:
	cd web && npm install && npm run build

serve:
	$(PYTHON) scripts/serve.py

test:
	$(PYTEST) -v

demo:
	$(PYTHON) scripts/demo.py

lint:
	$(RUFF) check .
	$(RUFF) format --check .
