"""
model.py

Wav2Vec2-based binary classifier for real vs AI-generated (spoofed) voice
detection. Uses a pretrained Wav2Vec2 encoder as a feature extractor and
adds a lightweight classification head on top.

Label convention (used throughout this project):
    0 = genuine / bona fide human voice
    1 = spoof / AI-generated (TTS, voice conversion, replay, etc.)

(Unchanged from the training notebook.)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2SpoofClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 0,
        num_labels: int = 2,
        classifier_hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def predict_fake_probability(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Convenience method: returns P(spoof) for each item in the batch."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_values, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            return probs[:, 1]
