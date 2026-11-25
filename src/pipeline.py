"""Reproducible handwritten digit classification with a sealed test set."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
RANDOM_STATE = 42


def get_data_and_split():
    digits = load_digits()
    indices = np.arange(len(digits.target))
    development_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=digits.target
    )
    return digits, np.asarray(development_idx), np.asarray(test_idx)


def model_candidates() -> dict[str, Pipeline]:
    return {
        'logistic_regression': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(C=1.0, max_iter=2500, random_state=RANDOM_STATE)),
        ]),
        'svm_rbf_c1': Pipeline([
            ('scale', StandardScaler()), ('model', SVC(kernel='rbf', C=1.0, gamma='scale')),
        ]),
        'svm_rbf_c10': Pipeline([
            ('scale', StandardScaler()), ('model', SVC(kernel='rbf', C=10.0, gamma='scale')),
        ]),
        'pca95_svm': Pipeline([
            ('scale', StandardScaler()), ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('model', SVC(kernel='rbf', C=10.0, gamma='scale')),
        ]),
    }


def evaluate_candidates(x: np.ndarray, y: np.ndarray):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = model_candidates()
    ranking = []
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, x, y, cv=folds, scoring='f1_macro')
        ranking.append({
            'model': name,
            'mean_cv_macro_f1': round(float(scores.mean()), 4),
            'std_cv_macro_f1': round(float(scores.std()), 4),
        })
    ranking.sort(key=lambda item: (-item['mean_cv_macro_f1'], item['model']))
    return ranking, candidates[ranking[0]['model']]


def plot_results(y: np.ndarray, pred: np.ndarray, x: np.ndarray) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    conf = confusion_matrix(y, pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(conf, cmap='Blues', interpolation='nearest')
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(conf[row, col]), ha='center', va='center',
                    fontsize=8, color='white' if conf[row, col] > conf.max() * .45 else '#1a2939')
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel='Predicted digit', ylabel='Actual digit')
    ax.set_title('Held-out test confusion matrix', fontsize=15, weight='bold', loc='left')
    fig.colorbar(im, ax=ax, label='Images', shrink=.75)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'test_confusion_matrix.png', dpi=190)
    plt.close(fig)

    errors = np.flatnonzero(y != pred)
    fig, axs = plt.subplots(2, 5, figsize=(10.6, 5.2))
    for i, ax in enumerate(axs.ravel()):
        ax.axis('off')
        if i < len(errors):
            idx = errors[i]
            ax.imshow(x[idx].reshape(8, 8), cmap='gray_r', interpolation='nearest')
            ax.set_title(f'Actual {y[idx]} / Pred {pred[idx]}', fontsize=10)
    fig.suptitle('Representative held-out classification errors', x=.04, y=.995,
                 ha='left', fontsize=14, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT / 'figures' / 'error_gallery.png', dpi=190)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'figures').mkdir(exist_ok=True, parents=True)
    digits, dev_idx, test_idx = get_data_and_split()
    x_dev, y_dev = digits.data[dev_idx], digits.target[dev_idx]
    x_test, y_test = digits.data[test_idx], digits.target[test_idx]
    ranking, best_model = evaluate_candidates(x_dev, y_dev)
    best_model.fit(x_dev, y_dev)
    pred = best_model.predict(x_test)
    metrics = {
        'dataset': 'scikit-learn digits (1797 examples of 8x8 handwritten digits)',
        'development_samples': len(dev_idx),
        'held_out_test_samples': len(test_idx),
        'selection_criterion': '5-fold stratified development-set cross-validation, macro F1',
        'cross_validation_ranking': ranking,
        'selected_model': ranking[0]['model'],
        'held_out_test_accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'held_out_test_macro_f1': round(float(f1_score(y_test, pred, average='macro')), 4),
        'test_errors': int(np.count_nonzero(y_test != pred)),
        'test_per_digit': classification_report(y_test, pred, output_dict=True, zero_division=0),
        'limitations': [
            'Small 8x8 benchmark images; this is not a production OCR system.',
            'Images are drawn from the same dataset distribution, so real-world transfer is unmeasured.',
            'The held-out test set was not used to select the model.',
        ],
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    predictions = pd.DataFrame({'test_index': test_idx, 'actual_digit': y_test, 'predicted_digit': pred})
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    joblib.dump(best_model, OUT / 'digit_classifier.joblib')
    plot_results(y_test, pred, x_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'test_per_digit'}, indent=2))
    return metrics


if __name__ == '__main__':
    main()
