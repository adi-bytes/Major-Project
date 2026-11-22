# PG-SBD Designer Root Makefile

PYTHON := .venv/Scripts/python.exe
PYTEST := .venv/Scripts/pytest.exe
RUFF := .venv/Scripts/ruff.exe

.PHONY: help data calibrate train eval serve test demo lint

help:
	@echo "Available commands:"
	@echo "  make data       - Fetch raw dataset from repository and audit"
	@echo "  make calibrate  - Calibrate differentiable physics core (Phase 3)"
	@echo "  make train      - Train ML baselines and hybrid ensemble (Phases 2 & 4)"
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

serve:
	$(PYTHON) scripts/serve.py

test:
	$(PYTEST) -v

demo:
	$(PYTHON) scripts/demo.py

lint:
	$(RUFF) check .
	$(RUFF) format --check .
