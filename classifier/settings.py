"""Fixed choices, and the model configuration.

The hierarchy columns, the prediction depth and the file layout do not
vary between runs, so they live here rather than being restated in every
notebook. Model parameters are collected in ModelSettings, which the
training notebook edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# What is predicted
# ---------------------------------------------------------------------------

# IDs are the labels the model learns. A category *name* can repeat under
# different parents, and paths are matched by string equality, so training
# on names would silently merge distinct categories.
ID_LEVELS: List[str] = [
    "Level 1 ID",
    "Level 2 ID",
    "Level 3 ID",
    "Level 4 ID",
]

# Carried alongside for readability; predictions are mapped back to these
# through the lookup saved with the model.
NAME_LEVELS: List[str] = [
    "Level 1 Name",
    "Level 2 Name",
    "Level 3 Name",
    "Level 4 Name",
]

N_LEVELS: int = len(ID_LEVELS)

TEXT_COL: str = "FullDesc"
ORIG_TEXT_COL: str = "FullDesc_orig"
ID_COL: str = "Vallen ID"

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
THRESHOLDS_PATH = os.path.join(CONFIG_DIR, "thresholds.json")


def raw_data_path(timestamp: str) -> str:
    return os.path.join(DATA_DIR, f"raw_data_{timestamp}.txt")


def cleaned_data_path(timestamp: str) -> str:
    return os.path.join(DATA_DIR, f"cleaned_data_{timestamp}.txt")


def model_dir(timestamp: str) -> str:
    return os.path.join(OUTPUT_DIR, f"model_{timestamp}")


def predictions_path(timestamp: str) -> str:
    return os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.txt")


def timestamp() -> str:
    """One stamp per run, shared by the data files and the model directory."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


DELIMITER = "|"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class VectorizerSettings:
    """Every vectorizer parameter, including the ones sklearn would default.

    Nothing is left implicit: a value not stated here is a value nobody
    reviews. The defaults below are what this model was built with, which
    are not always sklearn's own -- sublinear_tf in particular.
    """

    kind: str = "tfidf"                       # "tfidf" or "count"

    # ---- applies to both kinds -----------------------------------------
    ngram_range: Tuple[int, int] = (1, 1)
    # int is a document count, float is a proportion of documents. They are
    # not interchangeable: min_df=1 keeps every term, min_df=1.0 would
    # demand a term appear in every description and prune the vocabulary
    # to nothing.
    min_df: int = 1
    max_df: float = 1.0
    stop_words: Optional[str] = "english"
    max_features: Optional[int] = 75_000
    lowercase: bool = True
    # sklearn's default. Note it requires two or more characters, so a
    # single-character token surviving the cleaning is discarded here.
    token_pattern: str = r"(?u)\b\w\w+\b"
    # True counts presence rather than frequency.
    binary: bool = False

    # ---- tfidf only ----------------------------------------------------
    # Dampens repeated words with 1 + log(tf). Not sklearn's default, and
    # it usually helps on short descriptions where a repeated word says
    # little.
    sublinear_tf: bool = True
    # Scales each row to unit length, so a long description does not
    # outweigh a short one. None leaves the weights as they are.
    norm: Optional[str] = "l2"
    # Weight by inverse document frequency, so a word common to every
    # description counts for less.
    use_idf: bool = True
    # Adds one to every document frequency, avoiding a divide by zero on a
    # term unseen at fit time.
    smooth_idf: bool = True

    def build(self):
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            TfidfVectorizer,
        )

        if isinstance(self.min_df, float) and self.min_df >= 1.0:
            raise ValueError(
                f"min_df={self.min_df!r} is a float, read as a proportion, so "
                "it requires a term in every description. Use an int for a "
                "document count: min_df=1 keeps every term."
            )
        if isinstance(self.max_df, int) and not isinstance(self.max_df, bool) \
                and self.max_df <= 1:
            raise ValueError(
                f"max_df={self.max_df!r} is an int, read as a document count, "
                "so it drops any term appearing more than once. Use a float "
                "for a proportion: max_df=1.0 keeps every term."
            )

        common = dict(
            ngram_range=tuple(self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            max_features=self.max_features,
            lowercase=self.lowercase,
            token_pattern=self.token_pattern,
            binary=self.binary,
        )
        if self.kind == "count":
            return CountVectorizer(**common)
        if self.kind == "tfidf":
            return TfidfVectorizer(
                sublinear_tf=self.sublinear_tf,
                norm=self.norm,
                use_idf=self.use_idf,
                smooth_idf=self.smooth_idf,
                **common,
            )
        raise ValueError(f"kind must be 'tfidf' or 'count', got {self.kind!r}")


@dataclass
class NBSettings:
    alpha: float = 0.001                      # additive smoothing
    fit_prior: bool = True                    # learn class frequencies
    # Explicit class priors. None means they come from the data when
    # fit_prior is True, or are uniform when it is False.
    class_prior: Optional[List[float]] = None

    def build(self):
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior,
        )


@dataclass
class ModelSettings:
    """Everything the training notebook sets.

    min_path_count drops categories with fewer than this many products
    before training. Raising it lifts accuracy by removing categories the
    model has little evidence for; the products removed still have to be
    classified by someone.

    batch_size and fit_chunk_size bound memory only. Chunked fitting
    accumulates the same counts as a single pass, so neither changes a
    prediction.

    prediction_mode chooses how the per-level evidence is combined.

        "joint_path"        one model per level, each trained on every
            row, and the per-level posteriors multiplied together. The
            product is not a probability distribution over paths: it does
            not sum to one, which is why the figures are reported as
            scores.

        "conditional_path"  the chain rule instead --
            P(c1|x) * P(c2|c1,x) * P(c3|c1,c2,x) * P(c4|c1,c2,c3,x). Each
            factor comes from a model trained only on the rows beneath its
            parent, over that parent's children alone. Paths are scored
            and compared exactly as joint_path does, so the output columns
            and the thresholding are unchanged.

            This one *is* a probability distribution over the leaves: the
            scores sum to one across valid paths by construction. Its weak
            point is the opposite of joint_path's -- each factor is
            estimated from a subset of the rows, so deep nodes with little
            data are where it suffers. min_node_rows is the guard.

    min_node_rows: conditional_path only. A parent whose subtree holds
        fewer than this many training rows gets no model, and its children
        are treated as equally likely. 0 models every parent that has at
        least two children. A confident wrong factor high in the tree
        propagates to every path below it, so admitting ignorance beats
        guessing from a handful of rows.

    fit_flat_levels: conditional_path only. The per-level models are still
        fitted by default, so a single fitted object can be scored either
        way and the two compared. Set False to skip them -- the deepest
        flat fit is the largest allocation in training, so that is where
        the memory saving is, at the cost of making joint_path unavailable
        until you refit.
    """

    min_path_count: int = 10
    batch_size: int = 2_000
    fit_chunk_size: int = 20_000
    prediction_mode: str = "joint_path"
    min_node_rows: int = 0
    fit_flat_levels: bool = True
    # Precision of the path-score matrix during prediction. float32 halves
    # its memory against float64 at negligible cost to the result.
    score_dtype: str = "float32"
    vectorizer: VectorizerSettings = field(default_factory=VectorizerSettings)
    nb: NBSettings = field(default_factory=NBSettings)

    def __post_init__(self):
        if self.prediction_mode not in ("joint_path", "conditional_path"):
            raise ValueError(
                f"prediction_mode must be 'joint_path' or 'conditional_path', "
                f"got {self.prediction_mode!r}. Per-level prediction is not "
                "offered: it cannot produce the per-level confidence figures "
                "this pipeline reports, because those come from summing whole "
                "hierarchy paths."
            )
