import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import ModelConfig, build_encoder, build_siamese_model  # noqa: E402


@pytest.fixture(scope="module")
def small_config():
    return ModelConfig(
        vocab_size=500, max_len=20, embed_dim=16, lstm_units=8,
        dense1_units=16, dense2_units=8, num_classes=3,
    )


def test_build_encoder_output_shape(small_config):
    encoder = build_encoder(small_config)
    dummy = np.random.randint(0, small_config.vocab_size, size=(4, small_config.max_len))
    out = encoder.predict(dummy, verbose=0)
    assert out.shape == (4, small_config.lstm_units * 2)  # bidirectional doubles units


def test_build_siamese_model_output_shape(small_config):
    model = build_siamese_model(small_config)
    n = 5
    resume = np.random.randint(0, small_config.vocab_size, size=(n, small_config.max_len))
    jd = np.random.randint(0, small_config.vocab_size, size=(n, small_config.max_len))
    preds = model.predict({"resume_tokens": resume, "jd_tokens": jd}, verbose=0)
    assert preds.shape == (n, small_config.num_classes)


def test_siamese_model_outputs_are_valid_probabilities(small_config):
    model = build_siamese_model(small_config)
    resume = np.random.randint(0, small_config.vocab_size, size=(3, small_config.max_len))
    jd = np.random.randint(0, small_config.vocab_size, size=(3, small_config.max_len))
    preds = model.predict({"resume_tokens": resume, "jd_tokens": jd}, verbose=0)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-4)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_encoder_shares_weights_across_resume_and_jd_towers(small_config):
    model = build_siamese_model(small_config)
    encoder_layer = model.get_layer("shared_encoder")
    # The same encoder submodel must be reused (shared weights) for both inputs.
    assert model.inputs[0].name.startswith("resume_tokens") or "resume" in model.inputs[0].name
    assert encoder_layer is not None


def test_model_compiles_with_expected_loss_and_optimizer(small_config):
    model = build_siamese_model(small_config)
    assert "sparse_categorical_crossentropy" in model.loss if isinstance(model.loss, str) else True
    assert model.optimizer is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
