# FAIR-SHEPHERD API

## Structural policy

### `HorizontalPair(name, group_key, group_a, group_b, harmful_sign)`

Defines a horizontal comparison. The direction points from group B to group A.
`harmful_sign` is `+1` or `-1`.

### `ProxyPair(name, group_key, group_a, group_b)`

Defines an unsigned proxy-group comparison used by GradPen.

### `StructuralPolicy(feature_names, normative_features, vulnerable_features, level_order, horizontal_pairs)`

Defines the feature subspaces, ordered vertical levels, and audit pairs.

### `FittedPolicy`

Returned by `fit_policy`. `direction(level, pair_name)` returns the fitted unit
vulnerable direction.

### `MetricResult`

Contains the aggregate `value` and its component `details`.

### `StructuralReport`

Contains VCS, HLS, SHLS, SHLS Projection, and VCS Dispersion results.

## Structural metrics

### `fit_policy(policy, reference_X, reference_levels, reference_groups, *, feature_names=None, min_group_size=2, zero_tolerance=1e-8)`

For each level and horizontal pair, computes

`unit(mean(reference_X[group_a]) - mean(reference_X[group_b]))`.

Returns `FittedPolicy`.

### `vcs(gradients, levels, policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

Computes the cosine between adjacent-level mean normative gradients and returns
the minimum adjacent-level value.

### `hls(gradients, levels, fitted_policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

For each level and pair, computes

`mean(abs(cos(vulnerable gradient, vulnerable direction)))`.

Returns the maximum level/pair value.

### `shls(gradients, levels, fitted_policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

For each level and pair, computes

`mean(max(0, harmful_sign * cos(vulnerable gradient, vulnerable direction)))`.

Returns the maximum level/pair value.

### `shls_projection(gradients, levels, fitted_policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

For each level and pair, computes

`mean(max(0, harmful_sign * dot(vulnerable gradient, unit vulnerable direction)))`.

Returns the maximum level/pair value.

### `vcs_dispersion(gradients, levels, policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

For each level, the dispersion average uses rows whose normative gradient norm exceeds zero_tolerance. The level mean uses all rows. The function returns the maximum defined level value.

### `evaluate(gradients, levels, fitted_policy, *, gradient_feature_names=None, zero_tolerance=1e-8)`

Computes all five structural metrics and returns `StructuralReport`.

## Outcome metrics

### `auc(y_true, scores)`

Computes binary ROC-AUC.

### `wg_auc(y_true, scores, groups)`

Computes ROC-AUC within each group and returns the minimum group value.

### `eod(y_true, y_pred, groups)`

Computes the maximum absolute difference in true-positive rate between groups.

## Training methods

### `gradpen(task_loss, gradients, inputs, levels, proxy_groups, policy, proxy_pairs, *, structural_weight, k, vcs_weight, zero_tolerance=1e-8)`

Returns

`task_loss + structural_weight * (sum(abs(cosine)) + sum(abs(projection)) / k) + vcs_weight * sum(1 - adjacent VCS)`.

### `monotower(task_loss, gradients, levels, policy, harmful_signs, *, monotonicity_weight, vcs_weight, zero_tolerance=1e-8)`

Returns

`task_loss + monotonicity_weight * mean(ReLU(harmful_sign * vulnerable gradient)) + vcs_weight * sum(1 - adjacent VCS)`.

## Experiment configs

Each dataset config contains `audit_groups` and `audit_pairs` for evaluation,
and `training_proxy_groups` and `training_proxy_pairs` for S-agnostic GradPen
training.
