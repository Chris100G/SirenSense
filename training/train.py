#!/usr/bin/env python3
"""Train and export a small binary siren classifier for SirenSense.

Install training/requirements.txt, then provide an extracted UrbanSound8K
directory containing metadata/UrbanSound8K.csv and audio/foldN/*.wav.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16_000
WINDOW = SAMPLE_RATE
HOP = 250
FFT_SIZE = 512
MELS = 64
FRAMES = 64
SEED = 42


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_bins() -> list[int]:
    low, high = hz_to_mel(40.0), hz_to_mel(SAMPLE_RATE / 2)
    return [
        int((FFT_SIZE + 1) * mel_to_hz(low + (high - low) * i / (MELS + 1)) / SAMPLE_RATE)
        for i in range(MELS + 2)
    ]


MEL_FILTER_BINS = mel_bins()


def log_mel(samples: np.ndarray) -> np.ndarray:
    """Keep windowing, FFT bins, log floor, and normalization in sync with app.js."""
    padded = np.zeros(WINDOW, dtype=np.float32)
    clipped = samples[-WINDOW:]
    padded[-len(clipped):] = clipped
    window = np.hanning(FFT_SIZE).astype(np.float32)
    features = np.empty((FRAMES, MELS), dtype=np.float32)
    for frame_index in range(FRAMES):
        start = frame_index * HOP
        frame = np.zeros(FFT_SIZE, dtype=np.float32)
        frame_end = min(start + FFT_SIZE, WINDOW)
        frame[:frame_end - start] = padded[start:frame_end]
        spectrum = np.fft.rfft(frame * window)
        power = (spectrum.real**2 + spectrum.imag**2) / FFT_SIZE
        for band in range(MELS):
            left, center, right = MEL_FILTER_BINS[band:band + 3]
            center = max(center, left + 1)
            right = max(right, center + 1)
            total = 0.0
            for index in range(left, min(center, len(power))):
                total += power[index] * (index - left) / (center - left)
            for index in range(center, min(right, len(power))):
                total += power[index] * (right - index) / (right - center)
            features[frame_index, band] = np.log(max(float(total), 1e-10))
    mean, std = float(features.mean()), float(features.std())
    return ((features - mean) / (std if std else 1.0))[..., np.newaxis]


def read_metadata(dataset: Path) -> list[dict[str, str]]:
    metadata = dataset / "metadata" / "UrbanSound8K.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata}")
    with metadata.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"fold", "classID", "slice_file_name"}.issubset(rows[0]):
        raise ValueError(f"Unexpected UrbanSound8K metadata format: {metadata}")
    return rows


def extract_examples(dataset: Path, rows: list[dict[str, str]], folds: set[int], name: str):
    features, labels = [], []
    selected = [row for row in rows if int(row["fold"]) in folds]
    random.Random(SEED).shuffle(selected)
    for index, row in enumerate(selected, start=1):
        path = dataset / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Audio file listed in metadata is missing: {path}")
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(audio) > WINDOW:
            if name == "train":
                # One deterministic random crop per recording keeps memory modest.
                randomizer = random.Random(f"{SEED}:{row['slice_file_name']}")
                start = randomizer.randrange(len(audio) - WINDOW + 1)
            else:
                start = (len(audio) - WINDOW) // 2
            audio = audio[start:start + WINDOW]
        if len(audio) < WINDOW:
            audio = np.pad(audio, (0, WINDOW - len(audio)))
        features.append(log_mel(audio))
        labels.append(1 if int(row["classID"]) == 8 else 0)
        if index % 500 == 0:
            print(f"{name}: processed {index}/{len(selected)} clips")
    if not features:
        raise ValueError(f"No recordings found for {name} folds {sorted(folds)}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    print(f"{name}: {len(y)} clips; siren={int(y.sum())}, other={int((y == 0).sum())}")
    return x, y


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(FRAMES, MELS, 1), name="log_mel")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="sirensense_small_cnn")


def siren_recall(y_true, y_pred):
    labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    true_siren = tf.equal(labels, 1)
    predicted_siren = tf.equal(predictions, 1)
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(true_siren, predicted_siren), tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(true_siren, tf.float32))
    return tf.math.divide_no_nan(true_positives, actual_positives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted UrbanSound8K root")
    parser.add_argument("--output", type=Path, default=Path("model"), help="TensorFlow.js output folder")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    rows = read_metadata(args.dataset)
    train_x, train_y = extract_examples(args.dataset, rows, set(range(1, 9)), "train")
    val_x, val_y = extract_examples(args.dataset, rows, {9}, "validation")
    test_x, test_y = extract_examples(args.dataset, rows, {10}, "test")

    counts = np.bincount(train_y, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training folds must contain both siren and non-siren examples.")
    class_weight = {label: len(train_y) / (2.0 * int(count)) for label, count in enumerate(counts)}
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.MeanMetricWrapper(siren_recall, name="siren_recall")],
    )
    model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )
    print("Held-out fold 10 metrics:", model.evaluate(test_x, test_y, verbose=2))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    keras_path = output / "sirensense.h5"
    model.save(keras_path, include_optimizer=False)
    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        raise RuntimeError(
            "tensorflowjs_converter was not found. Install training/requirements.txt "
            "in a compatible Python environment and retry."
        )
    command = [
        converter, "--input_format=keras", "--output_format=tfjs_layers_model",
        "--quantize_uint8", "--weight_shard_size_bytes=4194304",
        str(keras_path), str(output),
    ]
    print("Exporting quantized TensorFlow.js model:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Model exported to {output}. Review model.json and weight shards before deployment.")
    print("Check recall and false-positive rates under realistic noise before any safety-related use.")


if __name__ == "__main__":
    main()
