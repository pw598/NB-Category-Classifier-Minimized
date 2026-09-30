"""Validate, clean, gate and predict a batch of unlabelled products.

The order matters and each step can refuse:

1. **Validate.** Two columns, an identifier and a description, with no
   duplicate identifiers. A duplicate identifier makes the output
   impossible to join back, so it stops the run rather than being
   deduplicated silently.

2. **Clean**, with the recipe saved alongside the model. The vocabulary
   was fitted on cleaned text, so anything else looks mostly unknown --
   and fails silently, as a raised out-of-vocabulary rate rather than an
   error.

3. **Gate on readability.** A description with no word the model
   recognises still produces a confident-looking answer, drawn from how
   common each category is rather than from the product. Those are always
   refused, at every level, whatever the confidence says.

4. **Optionally threshold**, walking from Level 1 down and stopping at the
   first level whose confidence falls short. Everything below a refused
   level is refused too: a Level 3 category means nothing under an unknown
   Level 2.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model import PREFIX_PROB_FMT, HierarchicalClassifier
from .settings import ID_LEVELS, NAME_LEVELS, THRESHOLDS_PATH

# Refusal reasons. Blank means the level was predicted.
OUT_OF_VOCABULARY = "out of vocabulary"
BELOW_THRESHOLD = "below threshold"
NO_PARENT = "no parent prediction"
NOT_REQUESTED = "not requested"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def read_input(path: str, delimiter: str = "|") -> pd.DataFrame:
    """Read a two-column pipe-delimited file of id and description."""
    return pd.read_csv(path, delimiter=delimiter, dtype=str)


def from_records(records) -> pd.DataFrame:
    """Build the input frame from literal pairs, for a quick single run.

    Accepts a single (id, description) tuple or a sequence of them.
    """
    if isinstance(records, tuple) and len(records) == 2 and isinstance(records[0], str):
        records = [records]
    return pd.DataFrame(list(records), columns=["id", "description"], dtype=str)


def validate(df: pd.DataFrame) -> Tuple[str, str]:
    """Check the shape and the identifiers. Returns (id_col, desc_col).

    Raises on anything that would make the output unusable, so a bad file
    stops here rather than producing a plausible-looking result.
    """
    if df is None or not len(df):
        raise ValueError("Input is empty: nothing to predict.")
    if df.shape[1] < 2:
        raise ValueError(
            f"Input needs at least 2 columns (identifier, description); "
            f"found {df.shape[1]}: {list(df.columns)}"
        )

    id_col, desc_col = df.columns[0], df.columns[1]

    blank_ids = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    if blank_ids.any():
        raise ValueError(
            f"{int(blank_ids.sum()):,} row(s) have a blank identifier in column "
            f"{id_col!r}; every row needs one to be joined back."
        )

    dupes = df[id_col].astype(str).duplicated(keep=False)
    if dupes.any():
        examples = df.loc[dupes, id_col].astype(str).unique()[:5]
        raise ValueError(
            f"{int(dupes.sum()):,} row(s) share a duplicate identifier in column "
            f"{id_col!r}, e.g. {list(examples)}. Deduplicate the input: a "
            "duplicated id makes the output impossible to join back."
        )

    n_blank_desc = int(
        (df[desc_col].isna() | (df[desc_col].astype(str).str.strip() == "")).sum()
    )
    print(f"[input] {len(df):,} rows | id column {id_col!r} | "
          f"description column {desc_col!r}")
    if n_blank_desc:
        print(f"[input] {n_blank_desc:,} blank description(s); these will be "
              "refused as out of vocabulary")
    return id_col, desc_col


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def load_thresholds(path: str = THRESHOLDS_PATH) -> Dict[int, float]:
    """Per-level confidence thresholds, keyed by depth.

    Kept in a file rather than in code so they can be retuned without a
    change here. A level omitted from the file is never predicted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No threshold file at {path}. Either add one or run with "
            "thresholding switched off."
        )
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)

    raw = record.get("thresholds", record)
    thresholds = {int(k): float(v) for k, v in raw.items()}
    if not thresholds:
        raise ValueError(f"{path} contains no thresholds.")
    return thresholds


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    clf: HierarchicalClassifier,
    df: pd.DataFrame,
    cleaner,
    lookup: Optional[pd.DataFrame] = None,
    all_levels: bool = True,
    thresholds: Optional[Dict[int, float]] = None,
    level_columns: List[str] = ID_LEVELS,
    name_columns: List[str] = NAME_LEVELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predict a validated batch, one output row per input row.

    all_levels=False reports Level 4 only; the shallower levels are
    refused as 'not requested' rather than dropped, so the output shape
    never changes.

    thresholds=None predicts wherever the description is readable.
    Supplying them adds the level-by-level walk described in the module
    docstring.
    """
    id_col, desc_col = df.columns[0], df.columns[1]
    depth = len(level_columns)

    cleaned = cleaner.clean(df[desc_col])
    scored = clf.predict(cleaned)
    quality = clf.known_word_counts(cleaned)

    readable = (quality["n_known"] > 0).to_numpy()
    n = len(df)

    # Which levels are wanted at all.
    wanted = {d: (all_levels or d == depth) for d in range(1, depth + 1)}

    # Walk from the shallowest level down, refusing everything below the
    # first level that falls short.
    reasons = {d: np.full(n, "", dtype=object) for d in range(1, depth + 1)}
    accepted = {d: np.zeros(n, dtype=bool) for d in range(1, depth + 1)}

    parent_ok = np.ones(n, dtype=bool)
    for d in range(1, depth + 1):
        conf = scored[PREFIX_PROB_FMT.format(d=d)].to_numpy(dtype=float)
        passes = (
            np.ones(n, dtype=bool) if thresholds is None
            else conf >= thresholds.get(d, np.inf)
        )
        ok = readable & parent_ok & passes

        reasons[d] = np.where(
            ~readable, OUT_OF_VOCABULARY,
            np.where(~parent_ok, NO_PARENT,
                     np.where(~passes, BELOW_THRESHOLD, "")),
        )

        if wanted[d]:
            accepted[d] = ok
            # Refusing a level refuses everything under it: a Level 3
            # category means nothing beneath an unknown Level 2.
            parent_ok = parent_ok & ok
        else:
            # Not being asked for a level is not a refusal, so it does not
            # cascade -- and the reason only replaces a blank, since a
            # genuine refusal is more informative than "not requested".
            accepted[d] = np.zeros(n, dtype=bool)
            reasons[d] = np.where(ok, NOT_REQUESTED, reasons[d])

    out = pd.DataFrame({id_col: df[id_col].astype(str).to_numpy()})
    for d in range(1, depth + 1):
        level = level_columns[d - 1]
        ids = scored[level].astype(str).to_numpy()
        out[f"Predicted {level}"] = np.where(accepted[d], ids, "")

    if lookup is not None:
        for d in range(1, depth + 1):
            level, name = level_columns[d - 1], name_columns[d - 1]
            mapping = (
                lookup[[level, name]].astype(str)
                .drop_duplicates(subset=[level]).set_index(level)[name]
            )
            predicted = out[f"Predicted {level}"]
            out[f"Predicted {name}"] = np.where(
                predicted != "", predicted.map(mapping).fillna(""), ""
            )

    for d in range(1, depth + 1):
        out[f"L{d} Confidence"] = np.where(
            accepted[d], scored[PREFIX_PROB_FMT.format(d=d)].to_numpy(dtype=float),
            np.nan,
        )

    for d in range(1, depth + 1):
        out[f"L{d} Refusal Reason"] = reasons[d]

    if verbose:
        print(f"\n[predict] {n:,} rows")
        print(f"[predict] {int((~readable).sum()):,} refused as out of vocabulary")
        for d in range(1, depth + 1):
            print(f"[predict] L{d}: {int(accepted[d].sum()):,} predicted, "
                  f"{int((~accepted[d]).sum()):,} refused")
    return out


def summarise(predictions: pd.DataFrame, depth: int = len(ID_LEVELS)) -> pd.DataFrame:
    """Counts per level of predictions and each refusal reason."""
    rows = []
    n = len(predictions)
    for d in range(1, depth + 1):
        reasons = predictions[f"L{d} Refusal Reason"]
        row = {"level": f"L{d}", "predicted": int((reasons == "").sum())}
        for reason in (OUT_OF_VOCABULARY, BELOW_THRESHOLD, NO_PARENT, NOT_REQUESTED):
            row[reason] = int((reasons == reason).sum())
        row["share_predicted"] = row["predicted"] / n if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
