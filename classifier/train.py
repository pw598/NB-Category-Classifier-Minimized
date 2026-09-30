"""Filter the labelled data and fit on all of it.

There is no holdout. That is right for a model being deployed and wrong
for measuring one: it has seen every row, so any accuracy computed on this
data would be meaningless. Measurement belongs to the separate
experimentation project.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from .cleaning import DescriptionCleaner
from .model import HierarchicalClassifier
from .settings import ID_LEVELS, TEXT_COL, ModelSettings


def drop_unusable(
    df: pd.DataFrame,
    level_columns: List[str] = ID_LEVELS,
    text_col: str = TEXT_COL,
    min_path_count: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows that cannot be trained on, and categories too rare to learn.

    Two filters. Rows missing a description or any label are unusable.
    Categories seen fewer than min_path_count times give the model almost
    no evidence to work from -- raising the threshold lifts accuracy, but
    the products removed still have to be classified by someone, so the
    count is reported rather than left implicit.
    """
    out = df.copy()
    n_start = len(out)

    out = out[out[text_col].notna() & (out[text_col].astype(str).str.strip() != "")]
    out = out.dropna(subset=level_columns)
    n_after_null = len(out)

    paths_before = out[level_columns].drop_duplicates().shape[0]
    if min_path_count > 1:
        counts = out.groupby(level_columns, dropna=False)[level_columns[0]].transform("size")
        out = out[counts >= min_path_count]
    n_after_rare = len(out)
    paths_after = out[level_columns].drop_duplicates().shape[0]

    if verbose:
        print(f"[filter] {n_start:>9,}  rows in")
        print(f"[filter] {n_after_null:>9,}  after dropping null text or labels "
              f"({n_start - n_after_null:,} dropped)")
        print(f"[filter] {n_after_rare:>9,}  after dropping paths seen "
              f"< {min_path_count} times ({n_after_null - n_after_rare:,} dropped, "
              f"{paths_before:,} -> {paths_after:,} categories)")

    if not len(out):
        raise ValueError("No rows survive filtering; lower min_path_count.")
    return out.reset_index(drop=True)


def train(
    df: pd.DataFrame,
    settings: ModelSettings,
    level_columns: List[str] = ID_LEVELS,
    text_col: str = TEXT_COL,
    verbose: bool = True,
) -> Tuple[HierarchicalClassifier, pd.DataFrame]:
    """Filter, then fit on everything that survives.

    Returns the classifier and the frame it was trained on, so the caller
    can record the row count and build the ID-to-name lookup from exactly
    the rows the model saw.
    """
    usable = drop_unusable(
        df, level_columns=level_columns, text_col=text_col,
        min_path_count=settings.min_path_count, verbose=verbose,
    )
    if verbose:
        print(f"[train] fitting on all {len(usable):,} rows (no holdout)")

    clf = HierarchicalClassifier(level_columns=level_columns, settings=settings)
    clf.fit(usable[text_col], usable[level_columns])
    return clf, usable
