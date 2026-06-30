"""Cadence turn-detection model.

Architecture: wav2vec2-base encoder (frozen or partially frozen) +
a 2-layer MLP classification head.

Input:  raw 16kHz audio, 2 seconds  (32 000 samples)
Output: logits for [turn_end, mid_thought]
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

BACKBONE = "facebook/wav2vec2-base"
LABELS = ["turn_end", "mid_thought"]
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}


@dataclass
class CadenceConfig:
    backbone: str = BACKBONE
    hidden_dim: int = 256
    dropout: float = 0.1
    freeze_feature_extractor: bool = True
    freeze_transformer_layers: int = 8  # freeze bottom N of 12 transformer blocks


class CadenceModel(nn.Module):
    def __init__(self, config: CadenceConfig = CadenceConfig()):
        super().__init__()
        self.config = config
        self.backbone = Wav2Vec2Model.from_pretrained(config.backbone)

        if config.freeze_feature_extractor:
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False

        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < config.freeze_transformer_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        backbone_dim = self.backbone.config.hidden_size  # 768 for wav2vec2-base
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(LABELS)),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        # Mean-pool over the time dimension
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.head(hidden)

    def predict(self, input_values: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            logits = self(input_values)
            probs = torch.softmax(logits, dim=-1)
            pred_id = probs.argmax(dim=-1).item()
        return {
            "label": ID2LABEL[pred_id],
            "confidence": probs[0, pred_id].item(),
            "probs": {ID2LABEL[i]: probs[0, i].item() for i in range(len(LABELS))},
        }


def load_processor() -> Wav2Vec2Processor:
    return Wav2Vec2Processor.from_pretrained(BACKBONE)
