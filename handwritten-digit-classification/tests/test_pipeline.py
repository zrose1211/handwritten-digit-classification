import numpy as np
import pytest
from sklearn.datasets import load_digits
from src.pipeline import get_data_and_split, evaluate_candidates
from src.infer import predict_digit


def test_stratified_split_is_deterministic_and_disjoint():
    digits, dev, test = get_data_and_split()
    assert len(dev) == 1437 and len(test) == 360
    assert not set(dev) & set(test)
    _, dev2, test2 = get_data_and_split()
    assert np.array_equal(dev, dev2) and np.array_equal(test, test2)
    assert len(np.unique(digits.target[test])) == 10


def test_cv_selects_from_declared_candidates():
    digits, dev, _ = get_data_and_split()
    ranked, best = evaluate_candidates(digits.data[dev], digits.target[dev])
    assert len(ranked) == 4
    assert all(0 <= row['mean_cv_macro_f1'] <= 1 for row in ranked)
    assert hasattr(best, 'fit')


def test_inference_validates_input_shape():
    with pytest.raises(ValueError, match='64 pixels'):
        predict_digit(np.zeros(8))
    with pytest.raises(ValueError, match='between 0 and 16'):
        predict_digit(np.full(64, 17))
