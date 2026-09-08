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


@pytest.mark.parametrize("protocol", [4, 5])
def test_populated_train_state_roundtrip_preserves_hit_accounting(protocol):
    state = SharedTrainState()
    state["earlier"] = np.arange(4)
    state.begin_call()
    state.get("earlier")
    state["current"] = np.arange(2)

    restored = pickle.loads(pickle.dumps(state, protocol=protocol))

    assert restored.hits == 1
    np.testing.assert_array_equal(restored["current"], [0, 1])
    assert restored.hits == 1
    np.testing.assert_array_equal(restored["earlier"], [0, 1, 2, 3])
    assert restored.hits == 2
    restored.begin_call()
    restored.get("current")
    assert restored.hits == 1
    assert state.hits == 1


@pytest.mark.parametrize("roundtrip", [_pickle_roundtrip, copy.deepcopy], ids=["pickle", "deepcopy"])
@pytest.mark.parametrize(
    "policy,reuses_state",
    [
        (None, False),
        ("random", False),
        ("target_rank[cap=8]", False),
        ("cluster_route", True),
        ("cluster_route_g4", True),
        ("safeboost", True),
        ("boost", True),
        (["random", "target_rank[cap=8]"], True),
    ],
)
def test_fitted_policy_roundtrip_preserves_predictions_and_warm_cache(policy, reuses_state, roundtrip):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(192, 4)).astype(np.float32)
    y = X[:, 0] - 2 * X[:, 1] + 0.3 * X[:, 2] ** 2
    query = rng.normal(size=(12, 4)).astype(np.float32)
    estimator = NoriRegressor(
        model_path="unused", device="cpu", large_context_policy=policy, large_context_threshold=16
    )
    estimator._predictor = _PicklePredictor()
    estimator.fit(X, y)
    estimator.predict(query)
    expected = estimator.predict(query)
    expected_report = copy.deepcopy(estimator.large_context_report_)

    restored = roundtrip(estimator)

    np.testing.assert_array_equal(restored.predict(query), expected)
    assert restored.large_context_report_ == expected_report
    if policy is not None:
        assert restored.large_context_report_["reused_train_state"] is reuses_state
    new_query = query[::-1].copy()
    np.testing.assert_array_equal(restored.predict(new_query), estimator.predict(new_query))
    restored.fit(X, y + 10)
    np.testing.assert_allclose(restored.predict(query), expected + 10, atol=1e-5, rtol=1e-5)
