"""FAIR-SHEPHERD structural metrics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HorizontalPair:
    """A group pair and the harmful direction from group B toward group A."""

    name: str
    group_key: str
    group_a: Hashable
    group_b: Hashable
    harmful_sign: int

    def __post_init__(self) -> None:
        if not self.name or not self.group_key or self.group_a == self.group_b:
            raise ValueError("A pair needs a name, group key, and two different groups.")
        if isinstance(self.harmful_sign, bool) or self.harmful_sign not in (-1, 1):
            raise ValueError("harmful_sign must be -1 or +1.")


@dataclass(frozen=True)
class ProxyPair:
    """A proxy-group comparison used by GradPen."""

    name: str
    group_key: str
    group_a: Hashable
    group_b: Hashable

    def __post_init__(self) -> None:
        if not self.name or not self.group_key or self.group_a == self.group_b:
            raise ValueError("A proxy pair needs a name, group key, and two different groups.")


@dataclass(frozen=True)
class StructuralPolicy:
    """Feature partition, ordered levels, and horizontal group pairs."""

    feature_names: tuple[str, ...]
    normative_features: tuple[str, ...]
    vulnerable_features: tuple[str, ...]
    level_order: tuple[Hashable, ...]
    horizontal_pairs: tuple[HorizontalPair, ...]

    def __post_init__(self) -> None:
        for field in (
            "feature_names",
            "normative_features",
            "vulnerable_features",
            "level_order",
            "horizontal_pairs",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.feature_names or len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be non-empty and unique.")
        if not self.normative_features or not self.vulnerable_features:
            raise ValueError("Both feature subspaces must be non-empty.")
        normative, vulnerable, known = (
            set(self.normative_features),
            set(self.vulnerable_features),
            set(self.feature_names),
        )
        if len(normative) != len(self.normative_features) or len(vulnerable) != len(
            self.vulnerable_features
        ):
            raise ValueError("Feature names cannot repeat within a subspace.")
        if normative & vulnerable or normative | vulnerable != known:
            raise ValueError("Normative and vulnerable features must partition feature_names.")
        if len(self.level_order) < 2 or len(self.level_order) != len(set(self.level_order)):
            raise ValueError("level_order must contain at least two unique levels.")
        if not self.horizontal_pairs or len({p.name for p in self.horizontal_pairs}) != len(
            self.horizontal_pairs
        ):
            raise ValueError("At least one uniquely named horizontal pair is required.")

    @property
    def normative_indices(self) -> np.ndarray:
        return np.array([self.feature_names.index(name) for name in self.normative_features])

    @property
    def vulnerable_indices(self) -> np.ndarray:
        return np.array([self.feature_names.index(name) for name in self.vulnerable_features])


@dataclass(frozen=True)
class FittedPolicy:
    """Unit vulnerable directions and group counts fitted on reference data."""

    policy: StructuralPolicy
    directions: Mapping[tuple[Hashable, str], np.ndarray]
    reference_counts: Mapping[tuple[Hashable, str], tuple[int, int]]

    def __post_init__(self) -> None:
        expected = {
            (level, pair.name)
            for level in self.policy.level_order
            for pair in self.policy.horizontal_pairs
        }
        if set(self.directions) != expected or set(self.reference_counts) != expected:
            raise ValueError("Every declared level/pair needs one direction and count pair.")
        directions, counts = {}, {}
        dimension = len(self.policy.vulnerable_features)
        for key in expected:
            direction = np.asarray(self.directions[key], dtype=np.float64).copy()
            if direction.shape != (dimension,) or not np.isfinite(direction).all():
                raise ValueError(f"Invalid direction for {key!r}.")
            if not np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Direction for {key!r} must have unit norm.")
            n_a, n_b = map(int, self.reference_counts[key])
            if n_a < 1 or n_b < 1:
                raise ValueError(f"Invalid reference counts for {key!r}.")
            direction.setflags(write=False)
            directions[key], counts[key] = direction, (n_a, n_b)
        object.__setattr__(self, "directions", MappingProxyType(directions))
        object.__setattr__(self, "reference_counts", MappingProxyType(counts))

    def direction(self, level: Hashable, pair_name: str) -> np.ndarray:
        return self.directions[(level, pair_name)]


@dataclass(frozen=True)
class MetricResult:
    """A reported value and the component values used to aggregate it."""

    value: float
    details: Mapping[Hashable, float]


@dataclass(frozen=True)
class StructuralReport:
    """The five FAIR-SHEPHERD structural metric results."""

    vcs: MetricResult
    hls: MetricResult
    shls: MetricResult
    shls_projection: MetricResult
    vcs_dispersion: MetricResult

    @property
    def summary(self) -> dict[str, float]:
        return {name: getattr(self, name).value for name in self.__dataclass_fields__}


def fit_policy(
    policy: StructuralPolicy,
    reference_X: Any,
    reference_levels: Sequence[Hashable],
    reference_groups: Mapping[str, Sequence[Hashable]],
    *,
    feature_names: Sequence[str] | None = None,
    min_group_size: int = 2,
    zero_tolerance: float = 1e-8,
) -> FittedPolicy:
    """Fit each direction as unit(mean(group A) - mean(group B))."""

    X = _feature_matrix(reference_X, policy.feature_names, feature_names, "reference_X")
    levels = np.asarray(reference_levels, dtype=object)
    if levels.ndim != 1 or len(levels) != len(X) or not np.isfinite(X).all():
        raise ValueError("reference_X and reference_levels must be finite and row-aligned.")
    if isinstance(min_group_size, bool) or not isinstance(min_group_size, int) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer.")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative.")
    group_arrays = {}
    for pair in policy.horizontal_pairs:
        if pair.group_key not in reference_groups:
            raise ValueError(f"Missing reference group array: {pair.group_key}")
        values = np.asarray(reference_groups[pair.group_key], dtype=object)
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError(f"Group array {pair.group_key!r} must match reference_X.")
        group_arrays[pair.group_key] = values

    directions, counts = {}, {}
    vulnerable_X = X[:, policy.vulnerable_indices]
    for level, level_mask in _level_masks(levels, policy.level_order, "reference_levels").items():
        for pair in policy.horizontal_pairs:
            groups = group_arrays[pair.group_key]
            mask_a = level_mask & (groups == pair.group_a)
            mask_b = level_mask & (groups == pair.group_b)
            n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
            if n_a < min_group_size or n_b < min_group_size:
                raise ValueError(
                    f"Cannot fit {pair.name!r} in {level!r}: counts ({n_a}, {n_b}) "
                    f"are below min_group_size={min_group_size}."
                )
            direction = vulnerable_X[mask_a].mean(0) - vulnerable_X[mask_b].mean(0)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= zero_tolerance:
                raise ValueError(f"Direction for {(level, pair.name)!r} has zero norm.")
            directions[(level, pair.name)] = direction / norm
            counts[(level, pair.name)] = (n_a, n_b)
    return FittedPolicy(policy, directions, counts)


def vcs(
    gradients: Any,
    levels: Sequence[Hashable],
    policy: StructuralPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> MetricResult:
    """Return the minimum adjacent-level cosine of mean normative gradients."""

    return _vcs(*_prepare(gradients, levels, policy, gradient_feature_names, zero_tolerance), policy)


def hls(
    gradients: Any,
    levels: Sequence[Hashable],
    fitted_policy: FittedPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> MetricResult:
    """Return the maximum level/pair mean absolute vulnerable-gradient cosine."""

    data = _prepare(
        gradients, levels, fitted_policy.policy, gradient_feature_names, zero_tolerance
    )
    return _horizontal(*data, fitted_policy, "hls")


def shls(
    gradients: Any,
    levels: Sequence[Hashable],
    fitted_policy: FittedPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> MetricResult:
    """Return the maximum mean rectified harmful cosine alignment."""

    data = _prepare(
        gradients, levels, fitted_policy.policy, gradient_feature_names, zero_tolerance
    )
    return _horizontal(*data, fitted_policy, "shls")


def shls_projection(
    gradients: Any,
    levels: Sequence[Hashable],
    fitted_policy: FittedPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> MetricResult:
    """Return the maximum mean rectified harmful projection magnitude."""

    data = _prepare(
        gradients, levels, fitted_policy.policy, gradient_feature_names, zero_tolerance
    )
    return _horizontal(*data, fitted_policy, "projection")


def vcs_dispersion(
    gradients: Any,
    levels: Sequence[Hashable],
    policy: StructuralPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> MetricResult:
    """Return the maximum level mean of one minus individual-to-mean cosine."""

    return _dispersion(
        *_prepare(gradients, levels, policy, gradient_feature_names, zero_tolerance), policy
    )


def evaluate(
    gradients: Any,
    levels: Sequence[Hashable],
    fitted_policy: FittedPolicy,
    *,
    gradient_feature_names: Sequence[str] | None = None,
    zero_tolerance: float = 1e-8,
) -> StructuralReport:
    """Calculate all five structural metrics."""

    policy = fitted_policy.policy
    data = _prepare(gradients, levels, policy, gradient_feature_names, zero_tolerance)
    return StructuralReport(
        _vcs(*data, policy),
        _horizontal(*data, fitted_policy, "hls"),
        _horizontal(*data, fitted_policy, "shls"),
        _horizontal(*data, fitted_policy, "projection"),
        _dispersion(*data, policy),
    )


def _prepare(gradients, levels, policy, feature_names, tolerance):
    matrix = _feature_matrix(gradients, policy.feature_names, feature_names, "gradients")
    level_values = np.asarray(levels, dtype=object)
    if level_values.ndim != 1 or len(level_values) != len(matrix):
        raise ValueError("levels must contain one value per gradient row.")
    if not np.isfinite(matrix).all() or tolerance < 0:
        raise ValueError("Gradients must be finite and zero_tolerance non-negative.")
    return matrix, _level_masks(level_values, policy.level_order, "levels"), tolerance


def _vcs(gradients, masks, tolerance, policy):
    normative = gradients[:, policy.normative_indices]
    means = {level: normative[mask].mean(0) for level, mask in masks.items()}
    details = {
        (left, right): _cosine(means[left], means[right], tolerance)
        for left, right in zip(policy.level_order[:-1], policy.level_order[1:])
    }
    return _result(min(details.values()), details)


def _horizontal(gradients, masks, tolerance, fitted, kind):
    policy, details = fitted.policy, {}
    vulnerable = gradients[:, policy.vulnerable_indices]
    for level, mask in masks.items():
        rows = vulnerable[mask]
        for pair in policy.horizontal_pairs:
            direction = fitted.direction(level, pair.name)
            if kind == "projection":
                values = np.maximum(0.0, pair.harmful_sign * (rows @ direction))
            else:
                alignment = _row_cosines(rows, direction, tolerance)
                values = (
                    np.abs(alignment)
                    if kind == "hls"
                    else np.maximum(0.0, pair.harmful_sign * alignment)
                )
            details[(level, pair.name)] = float(values.mean())
    return _result(max(details.values()), details)


def _dispersion(gradients, masks, tolerance, policy):
    normative, details = gradients[:, policy.normative_indices], {}
    for level, mask in masks.items():
        rows = normative[mask]
        mean, row_norms = rows.mean(0), np.linalg.norm(rows, axis=1)
        mean_norm, valid = float(np.linalg.norm(mean)), row_norms > tolerance
        if mean_norm <= tolerance or not valid.any():
            raise ValueError(f"VCS dispersion is undefined for level {level!r}.")
        cosines = (rows[valid] @ mean) / (row_norms[valid] * mean_norm)
        details[level] = float(np.mean(1.0 - np.clip(cosines, -1.0, 1.0)))
    return _result(max(details.values()), details)


def _feature_matrix(values, expected_names, supplied_names, field_name):
    expected = tuple(expected_names)
    columns = getattr(values, "columns", None)
    inferred = tuple(map(str, columns)) if columns is not None else None
    supplied = tuple(map(str, supplied_names)) if supplied_names is not None else None
    if inferred is not None and supplied is not None and inferred != supplied:
        raise ValueError(f"{field_name} columns and supplied feature names disagree.")
    actual = inferred if inferred is not None else supplied
    if actual is None or actual != expected:
        raise ValueError(f"{field_name} feature names and order must match the policy.")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(expected):
        raise ValueError(f"{field_name} must match the policy feature schema.")
    return matrix


def _level_masks(values, level_order, field_name):
    masks = {level: np.asarray(values == level, dtype=bool) for level in level_order}
    if np.any(np.sum(np.stack(tuple(masks.values())), axis=0) != 1):
        raise ValueError(f"{field_name} must assign every row to exactly one declared level.")
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"{field_name} must contain every declared level.")
    return masks


def _cosine(left, right, tolerance):
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= tolerance or right_norm <= tolerance:
        return 0.0
    return float(np.clip(left @ right / (left_norm * right_norm), -1, 1))


def _row_cosines(rows, direction, tolerance):
    norms, values = np.linalg.norm(rows, axis=1), np.zeros(len(rows), dtype=np.float64)
    valid = norms > tolerance
    values[valid] = (rows[valid] @ direction) / norms[valid]
    return np.clip(values, -1, 1)


def _result(value, details):
    return MetricResult(float(value), MappingProxyType(dict(details)))
