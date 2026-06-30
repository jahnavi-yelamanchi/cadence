# Cadence — Model Card

## Model description

Fine-tuned `facebook/wav2vec2-base` for conversational turn-taking detection.
Classifies 2-second audio windows (ending at a detected silence onset) as:

| Label | Meaning |
|---|---|
| `turn_end` | Speaker has finished their turn |
| `mid_thought` | Speaker paused but is still speaking |

## Training data

| Corpus | Hours | License |
|---|---|---|
| AMI Meeting Corpus (IHM) | ~100h | CC BY 4.0 |

Silence events > 150ms were auto-labelled from speaker-turn annotations.
Split is speaker-disjoint (no speaker appears in both train and test).

## Training config

```
backbone:              facebook/wav2vec2-base
frozen layers:         feature extractor + bottom 8 transformer blocks
fine-tuned params:     ~15M / 95M total
optimizer:             AdamW, lr=2e-5, weight_decay=0.01
schedule:              OneCycleLR with 10% warmup
epochs:                10
batch size:            16
augmentation:          gain jitter ±20%, additive noise σ=0.005
```

## Evaluation results (test set, speaker-disjoint)

| Metric | silero-VAD | Cadence | Δ |
|---|---|---|---|
| False Interruption Rate | — | — | — |
| ROC AUC | — | — | — |
| Accuracy | — | — | — |
| Inference latency (CPU) | — | — ms | — |

*(Fill in after training)*

## Inference

```python
from model.cadence_model import CadenceModel, load_processor
import torch, soundfile as sf

processor = load_processor()
model = CadenceModel()
model.load_state_dict(torch.load("model/checkpoints/best.pt")["model_state"])

audio, sr = sf.read("pause_window.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
result = model.predict(inputs.input_values)
# {"label": "turn_end", "confidence": 0.91, "probs": {...}}
```

## Limitations

- Trained on English conversational speech; may underperform on other languages
- Input must be 16kHz mono audio
- Model is calibrated for ~160ms silence threshold at inference time
- MPS (Apple Silicon) inference not tested for production — use ONNX/CPU for serving
