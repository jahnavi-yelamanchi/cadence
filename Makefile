.PHONY: install dev server frontend lint test data-download data-label train eval export

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync
	cd frontend && npm install

# ── Dev (runs server + frontend concurrently) ─────────────────────────────

dev:
	@echo "Starting Cadence dev environment..."
	@trap 'kill %1 %2' INT; \
	  PYTHONPATH=. uv run uvicorn server.main:app --reload --port 8000 & \
	  cd frontend && npm run dev & \
	  wait

server:
	PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn server.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

# ── Code quality ──────────────────────────────────────────────────────────

lint:
	uv run ruff check . --fix
	cd frontend && npm run lint

test:
	uv run pytest tests/ -v

# ── Data pipeline ─────────────────────────────────────────────────────────

data-download:
	PYTHONPATH=. uv run python data/download.py

data-label:
	PYTHONPATH=. uv run python data/label.py

data-split:
	PYTHONPATH=. uv run python data/split.py

# ── Model ─────────────────────────────────────────────────────────────────

train:
	PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python model/train.py

eval:
	PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python model/eval.py

export:
	PYTHONPATH=. uv run python model/export_onnx.py

# ── Notebooks ─────────────────────────────────────────────────────────────

notebook:
	uv run jupyter lab notebooks/
