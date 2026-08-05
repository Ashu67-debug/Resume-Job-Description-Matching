"""
model.py
========
Defines the Siamese BiLSTM Deep Neural Network used to classify a
Resume/Job-Description pair as Weak (0), Medium (1), or Strong (2) match.

Architecture (matches the project design doc, section 16.2):

    resume_input --> Embedding --> BiLSTM --> GlobalMaxPooling  --> u
    jd_input     --> Embedding --> BiLSTM --> GlobalMaxPooling  --> v   (shared weights)

    features = concat([u, v, |u - v|, u * v])
    x = Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu)
    output = Dense(3, softmax)
"""
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="resume_jd_dnn")
class AbsDifference(layers.Layer):
    """Computes |u - v| for two same-shaped tensors. A proper (serializable)
    layer instead of a Lambda, so the saved .keras model can be safely
    reloaded without `safe_mode=False`."""

    def call(self, inputs):
        u, v = inputs
        return tf.abs(u - v)


@tf.keras.utils.register_keras_serializable(package="resume_jd_dnn")
class ElementwiseProduct(layers.Layer):
    """Computes u * v (element-wise) for two same-shaped tensors."""

    def call(self, inputs):
        u, v = inputs
        return u * v


@dataclass
class ModelConfig:
    vocab_size: int = 20000
    max_len: int = 200
    embed_dim: int = 128
    lstm_units: int = 64
    dense1_units: int = 128
    dense2_units: int = 64
    dropout: float = 0.3
    num_classes: int = 3
    learning_rate: float = 1e-3


def build_encoder(config: ModelConfig) -> tf.keras.Model:
    """Shared text encoder tower: Embedding -> BiLSTM -> GlobalMaxPooling."""
    token_input = layers.Input(shape=(config.max_len,), name="tokens")
    x = layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embed_dim,
        mask_zero=True,
        name="embedding",
    )(token_input)
    x = layers.Bidirectional(
        layers.LSTM(
            config.lstm_units, return_sequences=True,
            dropout=0.2, recurrent_dropout=0.0,
        ),
        name="bilstm",
    )(x)
    x = layers.GlobalMaxPooling1D(name="pool")(x)
    return models.Model(token_input, x, name="shared_encoder")


def build_siamese_model(config: ModelConfig) -> tf.keras.Model:
    """Full end-to-end Siamese BiLSTM Resume/JD match classifier."""
    resume_input = layers.Input(shape=(config.max_len,), name="resume_tokens")
    jd_input = layers.Input(shape=(config.max_len,), name="jd_tokens")

    encoder = build_encoder(config)  # shared weights: same encoder called twice
    u = encoder(resume_input)
    v = encoder(jd_input)

    abs_diff = AbsDifference(name="abs_diff")([u, v])
    prod = ElementwiseProduct(name="elementwise_product")([u, v])

    features = layers.Concatenate(name="comparison_features")([u, v, abs_diff, prod])

    x = layers.Dense(config.dense1_units, activation="relu", name="dense_1")(features)
    x = layers.Dropout(config.dropout, name="dropout")(x)
    x = layers.Dense(config.dense2_units, activation="relu", name="dense_2")(x)
    output = layers.Dense(config.num_classes, activation="softmax", name="match_class")(x)

    model = models.Model(
        inputs=[resume_input, jd_input], outputs=output, name="resume_jd_siamese_bilstm"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    cfg = ModelConfig()
    m = build_siamese_model(cfg)
    m.summary()
