"""Outcome metrics reported by FAIR-SHEPHERD."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .structural import MetricResult


def auc(y_true: Any, scores: Any) -> float:
    """Return binary ROC-AUC."""

    y, score = _outcome_inputs(y_true, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("AUC requires both outcome classes.")
    return float(roc_auc_score(y, score))


def wg_auc(
    y_true: Any,
    scores: Any,
    groups: Mapping[Hashable, Sequence[bool]] | Sequence[Hashable],
) -> MetricResult:
    """Return the minimum group ROC-AUC."""

    y, score = _outcome_inputs(y_true, scores)
    details = {}
    for name, mask in _group_masks(groups, len(y)).items():
        if int(mask.sum()) >= 2 and len(np.unique(y[mask])) == 2:
            details[name] = float(roc_auc_score(y[mask], score[mask]))
    if not details:
        raise ValueError("No group contains both outcome classes.")
    return _result(min(details.values()), details)


def eod(
    y_true: Any,
    y_pred: Any,
    groups: Mapping[Hashable, Sequence[Hashable]] | Sequence[Hashable],
) -> MetricResult:
    """Return the maximum pairwise true-positive-rate difference."""

    y, predicted = _binary_pair(y_true, y_pred)
    attributes = groups if isinstance(groups, Mapping) else {"group": groups}
    details = {}
    for attribute, values in attributes.items():
        labels = np.asarray(values, dtype=object)
        if labels.ndim != 1 or len(labels) != len(y):
            raise ValueError(f"Group labels for {attribute!r} must match y_true.")
        names = _unique(labels)
        rates = {}
        for name in names:
            mask = labels == name
            positives = mask & (y == 1)
            rates[name] = float(np.mean(predicted[positives] == 1)) if positives.any() else 0.0
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                details[(attribute, left, right)] = abs(rates[left] - rates[right])
    if not details:
        raise ValueError("EOD requires at least two groups.")
    return _result(max(details.values()), details)


def _outcome_inputs(y_true, scores):
    y = _binary(y_true, "y_true")
    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or len(score) != len(y) or not np.isfinite(score).all():
        raise ValueError("scores must be one finite value per outcome.")
    return y, score


def _binary_pair(y_true, y_pred):
    y, predicted = _binary(y_true, "y_true"), _binary(y_pred, "y_pred")
    if len(y) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length.")
    return y, predicted


def _binary(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be a one-dimensional binary array.")
    return array.astype(np.int8, copy=False)


def _group_masks(groups, length):
    if isinstance(groups, Mapping):
        masks = {}
        for name, values in groups.items():
            mask = np.asarray(values)
            if mask.ndim != 1 or len(mask) != length or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Group mask {name!r} must be binary and row-aligned.")
            masks[name] = mask.astype(bool)
        return masks
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1 or len(labels) != length:
        raise ValueError("Group labels must contain one value per outcome.")
    return {name: labels == name for name in _unique(labels)}


def _unique(values):
    try:
        return list(dict.fromkeys(values.tolist()))
    except TypeError as error:
        raise ValueError("Group labels must be hashable.") from error


def _result(value, details):
    return MetricResult(float(value), MappingProxyType(dict(details)))
