"""Large-context fitted state survives persistence and sklearn-style copying."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch

from synthefy_nori.api import NoriRegressor
from synthefy_nori.inference.large_context import predictor_call_fn
from synthefy_nori.inference.policies import SharedTrainState


class _PicklePredictor:
    quantile_collapse = "mean"
    bar_point_estimator = "mean"

    def budget_n_features(self, X):
        return X.shape[1]

    def max_context_rows(self, X, *, budget_n_features):
        return 16

    def predict(self, X_context, y_context, X_query):
        context = np.column_stack((X_context, np.ones(len(X_context))))
        query = np.column_stack((X_query, np.ones(len(X_query))))
        coefficients = np.linalg.solve(context.T @ context + 1e-3 * np.eye(context.shape[1]), context.T @ y_context)
        return torch.tensor(query @ coefficients).reshape(-1, 1).requires_grad_()


def _pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize("roundtrip", [_pickle_roundtrip, copy.deepcopy], ids=["pickle", "deepcopy"])
def test_predictor_adapter_roundtrip_preserves_tensor_conversion(roundtrip):
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(6, dtype=np.float64)
    call = predictor_call_fn(_PicklePredictor())
    expected = call(X, y, X[:1])

    actual = roundtrip(call)(X, y, X[:1])

    assert actual.shape == (1,)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
