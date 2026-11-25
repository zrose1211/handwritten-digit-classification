"""CLI demonstration using a trained pipeline. Supply a numpy file with 64 pixel features."""
import argparse
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'reports' / 'digit_classifier.joblib'


def predict_digit(pixel_values: np.ndarray, model_path: Path = MODEL_PATH) -> int:
    pixels = np.asarray(pixel_values, dtype=float)
    if pixels.shape != (64,):
        raise ValueError(f'Expected a flat array of 64 pixels; received {pixels.shape}')
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels > 16):
        raise ValueError('Pixel values must be finite and between 0 and 16 (dataset scale)')
    model = joblib.load(model_path)  # Only load models you trust; joblib uses pickle.
    return int(model.predict(pixels.reshape(1, -1))[0])


def main():
    parser = argparse.ArgumentParser(description='Classify an 8x8 grayscale digit represented as 64 features')
    parser.add_argument('npy_file', type=Path, help='Path to a .npy file containing one 64-element array')
    args = parser.parse_args()
    pixels = np.load(args.npy_file, allow_pickle=False)
    print(f'Predicted digit: {predict_digit(pixels)}')


if __name__ == '__main__':
    main()
