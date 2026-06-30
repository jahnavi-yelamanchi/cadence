# Contributing to Cadence

## Setup

```bash
# Requires: Python 3.11+, Node 20+, uv
make install
cp .env.example .env
```

## Running locally

```bash
make dev        # server on :8000, frontend on :5173
make test       # pytest
make lint       # ruff + eslint
```

## Training your own model

1. `make data-download` — fetches CANDOR + AMI (~50GB total, takes a while)
2. `make data-label` — auto-labels pause events (~5 min)
3. `make data-split` — speaker-disjoint splits
4. `make train` — fine-tunes on MPS/CUDA/CPU, logs to W&B
5. `make eval` — prints comparison table, saves plots to `notebooks/figures/`
6. `make export` — ONNX export, then `make dev` to test live

## Commit style

`type: short description` — e.g. `feat: add confidence threshold flag`, `fix: resample buffer edge case`

Types: `feat`, `fix`, `data`, `model`, `docs`, `ci`, `refactor`

## Adding a new corpus

1. Add a download function in `data/download.py`
2. Add a `process_<corpus>()` function in `data/label.py` that returns a DataFrame with the standard columns (`audio_path`, `pause_start_ms`, `pause_end_ms`, `gap_ms`, `label`, `speaker_id`, `session_id`, `corpus`)
3. Call it from `main()` in `label.py`
