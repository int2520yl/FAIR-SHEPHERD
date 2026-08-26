"""Public FAIR-SHEPHERD API."""

from .methods import gradpen, monotower
from .outcomes import auc, eod, wg_auc
from .structural import (
    FittedPolicy,
    HorizontalPair,
    MetricResult,
    ProxyPair,
    StructuralPolicy,
    StructuralReport,
    evaluate,
    fit_policy,
    hls,
    shls,
    shls_projection,
    vcs,
    vcs_dispersion,
)

__all__ = [
    "FittedPolicy",
    "HorizontalPair",
    "MetricResult",
    "ProxyPair",
    "StructuralPolicy",
    "StructuralReport",
    "auc",
    "eod",
    "evaluate",
    "fit_policy",
    "gradpen",
    "hls",
    "monotower",
    "shls",
    "shls_projection",
    "vcs",
    "vcs_dispersion",
    "wg_auc",
]
