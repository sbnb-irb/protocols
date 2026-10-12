"""Binary profiles that define similar molecules for a custom sign3 triplet sampler.

``BinJaccardTripletSampler`` compares molecules by the Jaccard similarity of binary
profiles. For continuous data the profiles are built here, before chemicalchecker sees
them: from the raw input (so missing values stay missing, instead of the per-feature
median chemicalchecker's sign0 sanitizer fills in) or from a fitted signature. The
result is wrapped in :class:`ProfileSignature`, which ``sign3.fit`` accepts in place of
a chemicalchecker signature.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from .config import BinarizationConfig, DatasetConfig
from .data_loaders import load_wide_matrix

logger = logging.getLogger(__name__)


class ProfileSignature:
    """
    In-memory binary profiles standing in for a chemicalchecker signature.

    ``sign3.fit(triplet_sign=...)`` hands its triplet signature to the sampler, which
    reads ``as_dataframe()`` and ``data_path``, and uses ``shape[0]`` itself.

    Parameters
    ----------
    profiles : pandas.DataFrame
        0/1 values, one row per molecule (indexed by InChIKey).
    description : str
        Where the profiles come from; logged by the sampler as its data path.
    """

    def __init__(self, profiles: pd.DataFrame, description: str) -> None:
        self.profiles = profiles
        self.data_path = description

    @property
    def keys(self) -> np.ndarray:
        """InChIKeys of the profiled molecules."""
        return self.profiles.index.to_numpy()

    @property
    def shape(self) -> tuple[int, int]:
        """(molecules, features)."""
        return self.profiles.shape

    def as_dataframe(self) -> pd.DataFrame:
        """The profiles, as chemicalchecker signatures return their matrix."""
        return self.profiles


def binarize_profiles(
    values: pd.DataFrame, binarize: BinarizationConfig
) -> pd.DataFrame:
    """
    Turn continuous values into up/down binary profiles.

    Parameters
    ----------
    values : pandas.DataFrame
        One row per molecule, one column per feature; NaN for missing values.
    binarize : BinarizationConfig
        The rule (see its docstring).

    Returns
    -------
    pandas.DataFrame
        0/1 values with a ``<feature>_up`` and a ``<feature>_down`` column per feature.

    Raises
    ------
    ValueError
        If a ``percentile`` rule gives a cut-off that is not positive.
    """
    matrix = values.to_numpy(dtype=float)
    if binarize.percentile is not None:
        cutoff = np.nanpercentile(matrix, binarize.percentile)
        if cutoff <= 0:
            raise ValueError(
                f"The {binarize.percentile} percentile of the values is {cutoff:.3g}; "
                "use a higher percentile so that up and down calls cannot overlap"
            )
        up, down = matrix > cutoff, matrix < -cutoff
        rule = f"|value| beyond the P{binarize.percentile:g} cut-off {cutoff:.4g}"
    else:
        up = np.zeros(matrix.shape, dtype=bool)
        down = np.zeros(matrix.shape, dtype=bool)
        rules = []
        if binarize.log2fc is not None:
            up |= matrix >= binarize.log2fc
            down |= matrix <= -binarize.log2fc
            rules.append(f"|value| >= {binarize.log2fc:g}")
        if binarize.zscore is not None:
            deviation = matrix - np.nanmean(matrix, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                zscores = np.abs(deviation) / np.nanstd(matrix, axis=0, ddof=1)
            called = (
                zscores >= binarize.zscore
            )  # NaN (missing or constant) is never called
            up |= called & (deviation > 0)
            down |= called & (deviation < 0)
            rules.append(f"per-feature |z| >= {binarize.zscore:g}")
        rule = " or ".join(rules)

    profiles = pd.DataFrame(
        np.hstack([up, down]).astype(np.int8),
        index=values.index,
        columns=[f"{c}_up" for c in values.columns]
        + [f"{c}_down" for c in values.columns],
    )
    per_molecule = profiles.sum(axis=1)
    logger.info(
        "Binarized %d molecules x %d features (%s): %.2f%% of values called, "
        "median %d features per molecule, %d molecules without any",
        *values.shape,
        rule,
        100 * (up | down).mean(),
        int(per_molecule.median()),
        int((per_molecule == 0).sum()),
    )
    return profiles


def build_triplet_signature(
    cc_instance: Any,
    dataset_config: DatasetConfig,
    signatures: dict[str, Any],
    holdout_keys: Collection[str] | None = None,
) -> Any:
    """
    Return what a custom sampler compares molecules by, for ``sign3.fit(triplet_sign=...)``.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_config : DatasetConfig
        The dataset; its ``triplet_sampler`` must be set.
    signatures : dict
        Signatures already fitted or loaded in this run, by type.
    holdout_keys : collection of str, optional
        Molecules held out of the space; removed from ``raw`` profiles.

    Returns
    -------
    chemicalchecker signature or ProfileSignature
        The fitted signature itself when no binarization is requested, otherwise
        the binary profiles built from it (or from the raw input).
    """
    # Imported here to avoid a circular import: signature_fitting uses this module.
    from .signature_fitting import load_fitted_signature

    sampler_config = dataset_config.triplet_sampler
    source = sampler_config.triplet_signature
    dataset_code = dataset_config.dataset_code
    if source == "raw":
        values = load_wide_matrix(dataset_config.source)
        if holdout_keys is not None:
            values = values.loc[~values.index.isin(list(holdout_keys))]
        # chemicalchecker's sign0 averages duplicated InChIKeys; do the same.
        values = values.groupby(level=0).mean()
        description = f"raw {dataset_config.source.path}"
    else:
        signature = signatures.get(source)
        if signature is None:
            signature = load_fitted_signature(cc_instance, dataset_code, source)
        if sampler_config.binarize is None:
            return signature
        values = signature.as_dataframe()
        description = f"{source} {signature.data_path}"

    if sampler_config.binarize is None:
        return ProfileSignature(values, description)
    profiles = binarize_profiles(values, sampler_config.binarize)
    return ProfileSignature(profiles, f"binarized {description}")
