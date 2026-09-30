"""Chemical Checker Protocols: fit, evaluate and compare bioactivity signature spaces."""

from .config import DatasetConfig, RunConfig, load_run_config
from .evaluation import (
    cosine_nn_recapitulation_auroc,
    shared_key_recapitulation,
    signature_recovery,
)
from .signature_fitting import run_signature_pipeline

__all__ = [
    "DatasetConfig",
    "RunConfig",
    "cosine_nn_recapitulation_auroc",
    "load_run_config",
    "run_signature_pipeline",
    "shared_key_recapitulation",
    "signature_recovery",
]
