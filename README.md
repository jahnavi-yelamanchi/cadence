# Cadence

**Smart turn-taking for voice agents** — fine-tuned wav2vec2 that classifies conversational pauses as *turn_end* vs *mid_thought*, eliminating false interruptions and dead air.

**[→ Live demo](https://cadence-demo.vercel.app)** · **[Model card](model/MODEL_CARD.md)**

---

## The problem

Voice agents still feel broken. VAD (voice activity detection) only answers *"is there audio?"* — not *"is the speaker done?"* Human conversation runs on a 200–300ms response window, and exceeding it breaks the flow. The hard part is knowing whether a pause is a thought-collecting *"um…"* or a genuine end of turn.

## What Cadence does

```
You: "I think the answer is…  [pause]  …forty-two."
VAD:     ────────────────────── TURN END ❌  (interrupts you mid-thought)
Cadence: ────────────────────── MID-THOUGHT ✓  (waits correctly)
```

Fine-tune `wav2vec2-base` on 2-second audio windows around pause events, labelled from conversational speech corpora (CANDOR + AMI). Single-token binary output: `turn_end` | `mid_thought`.

## Results

| Metric | silero-VAD | Cadence | Δ |
|---|---|---|---|
| False Interruption Rate | — | — | — |
| ROC AUC | — | — | — |
| Inference latency (CPU) | — | — ms | — |

*(Filled in after training — see [notebooks/03_error_analysis.ipynb](notebooks/03_error_analysis.ipynb))*

## Architecture

```
Browser mic (44.1kHz)
  └─ AudioWorklet (resample → 16kHz)
       └─ WebSocket (20ms PCM chunks)
            └─ FastAPI server
                 ├─ Cadence (ONNX, ~18ms/chunk on CPU)
                 └─ silero-VAD (baseline)
                      └─ JSON event → React UI
```

## Quick start

```bash
# 1. Clone
git clone https://github.com/jahnaviyelamanchi/cadence
cd cadence

# 2. Install (requires uv and Node ≥ 20)
make install

# 3. Run dev server + frontend
make dev
# → http://localhost:5173
```

> **Note:** `make dev` expects a trained ONNX model at `model/onnx/cadence.onnx`.
> To run the full pipeline from scratch: `make data-download data-label data-split train export`

## Full pipeline

```bash
make data-download   # download CANDOR + AMI corpora (~50GB)
make data-label      # auto-label pause events from turn annotations
make data-split      # speaker-disjoint train/val/test split
make train           # fine-tune on MPS/CUDA/CPU (~4h on M3 Pro)
make eval            # compare Cadence vs silero-VAD, save plots
make export          # export to ONNX + INT8 quantization
make dev             # live demo at localhost:5173
```

## Notebooks

| Notebook | What it shows |
|---|---|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | Label distribution, gap duration histograms, audio samples |
| [02_training_curves](notebooks/02_training_curves.ipynb) | Loss, FIR, accuracy over epochs (W&B) |
| [03_error_analysis](notebooks/03_error_analysis.ipynb) | Confusion matrix, ROC curves, calibration, false-positive examples |

## Project structure

```
cadence/
├── data/               # curation + labeling scripts
│   ├── download.py     # fetch CANDOR + AMI
│   ├── label.py        # auto-label pause events
│   └── split.py        # speaker-disjoint splits
├── model/
│   ├── cadence_model.py  # wav2vec2 + classification head
│   ├── dataset.py        # PyTorch Dataset + augmentation
│   ├── train.py          # training loop (MPS/CUDA/CPU)
│   ├── eval.py           # FIR / AUC vs silero-VAD
│   ├── export_onnx.py    # ONNX + INT8 export
│   └── MODEL_CARD.md
├── server/
│   ├── main.py           # FastAPI + WebSocket
│   ├── endpointer.py     # ONNX inference + sliding window
│   ├── baseline.py       # silero-VAD wrapper
│   └── Dockerfile
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── hooks/useAudioStream.ts
│       └── components/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_curves.ipynb
│   └── 03_error_analysis.ipynb
└── Makefile
```

## Stack

- **Model:** PyTorch + HuggingFace Transformers, ONNX Runtime
- **Training:** `facebook/wav2vec2-base`, AdamW, OneCycleLR, W&B logging
- **Server:** FastAPI, WebSockets
- **Frontend:** React, TypeScript, Tailwind CSS, Web Audio API
- **Deploy:** Fly.io (backend) + Vercel (frontend) + HuggingFace Hub (model weights)

---

*Jahnavi Yelamanchi · jy4857@nyu.edu*
