"""
infer.py

Model-loading and single-clip inference helpers, shared by the API layer.
(Unchanged in logic from the training notebook's infer.py.)
"""

import torch
from transformers import Wav2Vec2FeatureExtractor

from model import Wav2Vec2SpoofClassifier
from preprocess import preprocess_pipeline, TARGET_SR


def load_model(checkpoint_path, pretrained_model="facebook/wav2vec2-base", device="cpu"):
    model = Wav2Vec2SpoofClassifier(pretrained_model_name=pretrained_model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_fake_probability(audio_path, model, processor, device="cpu", max_duration=4.0):
    waveform = preprocess_pipeline(audio_path, max_duration=max_duration)
    inputs = processor(waveform.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    fake_prob = model.predict_fake_probability(input_values).item()
    return fake_prob
