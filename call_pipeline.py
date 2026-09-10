"""
call_pipeline.py

Ties the three layers together end to end:
    Detect  -> spoof classifier gives a fake-probability score
    Verify  -> if score is high, issue a liveness challenge
    Protect -> allow the call, or block + report it

(Unchanged in logic from the training notebook; this is what main.py calls.)
"""

import torch

from infer import load_model, get_fake_probability
from liveness_challenge import LivenessVerifier, generate_random_phrase


class CallFraudPreventionPipeline:
    def __init__(self, checkpoint_path, pretrained_model="facebook/wav2vec2-base",
                 fake_prob_threshold=0.6, whisper_model_size="base", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model)
        self.model = load_model(checkpoint_path, pretrained_model, self.device)
        self.fake_prob_threshold = fake_prob_threshold
        self.liveness_verifier = LivenessVerifier(whisper_model_size, device="cpu")

    def detect(self, incoming_audio_path: str) -> float:
        return get_fake_probability(incoming_audio_path, self.model, self.processor, self.device)

    def issue_challenge(self) -> str:
        return generate_random_phrase()

    def verify(self, response_audio_path: str, expected_phrase: str) -> dict:
        phrase_result = self.liveness_verifier.verify(response_audio_path, expected_phrase)
        response_fake_prob = self.detect(response_audio_path)

        phrase_result["response_fake_probability"] = response_fake_prob
        phrase_result["response_flagged_as_synthetic"] = response_fake_prob > self.fake_prob_threshold
        return phrase_result

    def process_call(self, incoming_audio_path: str, response_audio_path: str = None) -> dict:
        fake_prob = self.detect(incoming_audio_path)
        result = {
            "fake_probability": round(fake_prob, 4),
            "risk_level": "high" if fake_prob > self.fake_prob_threshold else "low",
        }

        if fake_prob <= self.fake_prob_threshold:
            result["decision"] = "ALLOW"
            result["reason"] = "Low fake-voice probability; call proceeds normally."
            return result

        if response_audio_path is None:
            challenge_phrase = self.issue_challenge()
            result["decision"] = "CHALLENGE_REQUIRED"
            result["challenge_phrase"] = challenge_phrase
            result["reason"] = "High fake-voice probability; liveness verification needed before allowing the call."
            return result

        challenge_phrase = self.issue_challenge()
        verify_result = self.verify(response_audio_path, challenge_phrase)
        result["liveness_check"] = verify_result

        if verify_result["passed"] and not verify_result["response_flagged_as_synthetic"]:
            result["decision"] = "ALLOW"
            result["reason"] = "Liveness challenge passed; caller appears to be a real, present human."
        else:
            result["decision"] = "BLOCK_AND_REPORT"
            result["reason"] = "Liveness challenge failed or response audio also flagged as synthetic."

        return result
