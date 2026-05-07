"""
lstm_architecture.py
────────────────────────────────────────────────────────────────
Bidirectional LSTM + Temporal Attention model for ASL classification.

Input  : (batch, SEQ_LEN, 126)  — positions + velocity landmarks
Output : (batch, num_classes)   — softmax probabilities

Architecture:
    Input
    → BiLSTM(128, return_sequences=True) → LayerNorm → Dropout
    → MultiHeadAttention (self-attention, 4 heads) + Residual → LayerNorm
    → BiLSTM(64, return_sequences=False) → Dropout
    → Dense(256, relu) → BN → Dropout
    → Dense(128, relu) → Dropout
    → Dense(num_classes, softmax)

Why attention?
    Signs have a few key frames (the hold / peak position). Attention
    lets the model weight those frames more heavily instead of treating
    every frame equally like a plain LSTM.
────────────────────────────────────────────────────────────────
"""

import tensorflow as tf
from keras import layers, regularizers, Model, Input


def build_lstm_model(num_classes   : int,
                     seq_len       : int   = 30,
                     landmark_dim  : int   = 126,
                     dropout_rate  : float = 0.5) -> Model:
    """
    Build and return the BiLSTM + Attention model (uncompiled).

    Sized for small datasets (~200 training clips):
      - Smaller hidden dims (64/32) to reduce overfitting
      - Stronger dropout (0.5)
      - Lighter attention (2 heads)
    """
    inp = Input(shape=(seq_len, landmark_dim), name="landmark_sequence")

    # ── Layer 1: BiLSTM ───────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(2e-4)),
        name="bilstm_1"
    )(inp)
    x = layers.LayerNormalization(name="ln_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    # ── Temporal Self-Attention (light: 2 heads) ──────────────
    attn_out = layers.MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.1, name="temporal_attn"
    )(x, x)
    x = layers.Add(name="attn_residual")([x, attn_out])
    x = layers.LayerNormalization(name="ln_2")(x)

    # ── Layer 2: BiLSTM ───────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False,
                    kernel_regularizer=regularizers.l2(2e-4)),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    # ── Classification head ───────────────────────────────────
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(2e-4),
                     name="fc_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="drop_3")(x)

    out = layers.Dense(num_classes, activation="softmax",
                       name="predictions")(x)

    return Model(inputs=inp, outputs=out, name="ASL_BiLSTM_Attn")


def compile_lstm_model(model          : Model,
                       num_classes    : int,
                       learning_rate  : float = 1e-3,
                       label_smoothing: float = 0.1) -> None:
    """Compile model in-place with Adam + label-smoothed crossentropy."""
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
