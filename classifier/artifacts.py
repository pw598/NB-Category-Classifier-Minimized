"""Save and reload everything needed to score new products.

What is kept, and what is not:

    keep   the fitted vectorizer -- its vocabulary and IDF weights are what
           map a new description into the feature space the models were
           trained in. Without it the models are meaningless.
    keep   one model per level.
    keep   the path index: which hierarchy paths were seen in training, and
           how each level's labels map to model columns. Path scoring
           cannot run without it.
    keep   the cleaning recipe, word list included, so scoring cleans
           exactly as training did.
    keep   the ID-to-name lookup, so predictions are readable later
           without re-querying the source.
    drop   the document-term matrix. Derived data for the training rows;
           regenerating it from text costs seconds, and a stored copy can
           only fall out of step with the vectorizer that made it.

The size choices are fixed rather than exposed. Models are compressed, and
each model's feature_count_ is discarded: only feature_log_prob_ is needed
to predict, and the counts exist so partial_fit can continue, which a
deployed model never does. Together those keep the bundle roughly a
quarter of its uncompressed size, which is what makes it fit a workspace
file limit.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import HierarchicalClassifier

MANIFEST = "manifest.json"
_COMPRESS = 3


def _level_tag(level: str) -> str:
    """A filesystem-safe stem from a column name like 'Level 3 ID'."""
    return "".join(ch if ch.isalnum() else "_" for ch in level).strip("_")


def _fingerprint(clf: HierarchicalClassifier) -> str:
    """A hash of the shapes that must agree across the files.

    Not security. It is here so that a vectorizer from one training run
    and models from another are rejected loudly instead of producing
    confident nonsense.
    """
    parts = [
        f"features={len(clf.vectorizer.vocabulary_)}",
        f"levels={','.join(clf.level_columns)}",
        f"paths={len(clf.valid_paths_)}",
    ]
    parts += [
        f"{lv}:{len(clf.models[lv].classes_)}"
        for lv in clf.level_columns
        if lv in clf.models
    ]
    stack = getattr(clf, "node_stack_", None)
    if stack:
        parts += [
            f"node{d}:{0 if s['W'] is None else s['W'].shape[1]}"
            for d, s in sorted(stack.items())
        ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _settings_dict(settings) -> dict:
    """The settings as plain JSON types, for the manifest."""
    import dataclasses

    return dataclasses.asdict(settings)


def _slim(model):
    """A copy carrying only what prediction needs."""
    import copy

    out = copy.copy(model)
    if hasattr(out, "feature_count_"):
        # class_count_ is kept: it is small, and the class priors derive
        # from it.
        out.feature_count_ = np.zeros((0, 0), dtype=np.float32)
    return out


def save(
    clf: HierarchicalClassifier,
    directory: str,
    cleaner,
    hierarchy_lookup: pd.DataFrame,
    n_training_rows: int,
) -> Dict[str, str]:
    """Write the bundle. Returns {name: path} for everything written."""
    import joblib

    if not getattr(clf, "fitted_", False):
        raise RuntimeError("Refusing to save an unfitted classifier.")
    os.makedirs(directory, exist_ok=True)
    written = {}

    def put(name, filename, obj):
        path = os.path.join(directory, filename)
        joblib.dump(obj, path, compress=_COMPRESS)
        written[name] = path

    put("vectorizer", "vectorizer.pkl", clf.vectorizer)
    put("settings", "settings.pkl", clf.settings)
    put("cleaner", "cleaner.pkl", cleaner)
    put("path_index", "path_index.pkl", {
        "level_columns": clf.level_columns,
        "valid_paths_": clf.valid_paths_,
        "label_to_col": clf.label_to_col,
        "_path_idx": clf._path_idx,
    })
    for lv in clf.level_columns:
        if lv in clf.models:
            put(f"model::{lv}", f"model_{_level_tag(lv)}.pkl", _slim(clf.models[lv]))

    # conditional_path only. Kept in its own file so a reloaded bundle can
    # still score, rather than looking complete and then failing to find
    # its per-parent factors.
    if getattr(clf, "node_stack_", None):
        put("node_stack", "node_stack.pkl", clf.node_stack_)

    lookup_path = os.path.join(directory, "hierarchy_lookup.csv")
    hierarchy_lookup.drop_duplicates().to_csv(lookup_path, index=False)
    written["hierarchy_lookup"] = lookup_path

    import sklearn

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": _fingerprint(clf),
        "level_columns": clf.level_columns,
        "n_features": len(clf.vectorizer.vocabulary_),
        "n_paths": int(len(clf.valid_paths_)),
        "classes_per_level": {lv: int(len(clf.models[lv].classes_))
                              for lv in clf.level_columns if lv in clf.models},
        "n_training_rows": int(n_training_rows),
        # Every setting, readable without unpickling anything. settings.pkl
        # is what reloads; this is what answers "what was it built with?"
        # six months from now.
        "settings": _settings_dict(clf.settings),
        "versions": {"scikit-learn": sklearn.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__},
        "files": {k: os.path.basename(v) for k, v in written.items()},
    }
    path = os.path.join(directory, MANIFEST)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    written["manifest"] = path
    return written


def load(directory: str, verify: bool = True):
    """Reload a bundle as (classifier, cleaner, lookup, manifest).

    A fingerprint mismatch means the files came from different training
    runs, which is worth an exception rather than a warning: mixed
    artifacts predict confidently and wrongly.
    """
    import joblib

    index = joblib.load(os.path.join(directory, "path_index.pkl"))
    settings = joblib.load(os.path.join(directory, "settings.pkl"))

    clf = HierarchicalClassifier(
        level_columns=index["level_columns"], settings=settings
    )
    clf.vectorizer = joblib.load(os.path.join(directory, "vectorizer.pkl"))
    clf.models = {}
    for lv in index["level_columns"]:
        # Absent when the run was fitted with fit_flat_levels=False.
        piece = os.path.join(directory, f"model_{_level_tag(lv)}.pkl")
        if os.path.exists(piece):
            clf.models[lv] = joblib.load(piece)
    clf.label_to_col = index["label_to_col"]
    clf.valid_paths_ = index["valid_paths_"]
    clf._path_idx = index["_path_idx"]
    node_path = os.path.join(directory, "node_stack.pkl")
    if os.path.exists(node_path):
        clf.node_stack_ = joblib.load(node_path)
    clf.fitted_ = True

    cleaner = joblib.load(os.path.join(directory, "cleaner.pkl"))

    lookup_path = os.path.join(directory, "hierarchy_lookup.csv")
    lookup = pd.read_csv(lookup_path, dtype=str) if os.path.exists(lookup_path) else None

    manifest = {}
    manifest_path = os.path.join(directory, MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    if verify and manifest.get("fingerprint"):
        got = _fingerprint(clf)
        if got != manifest["fingerprint"]:
            raise ValueError(
                f"Fingerprint mismatch: manifest says {manifest['fingerprint']}, "
                f"the loaded files give {got}. These artifacts are from "
                "different training runs."
            )
    return clf, cleaner, lookup, manifest


def latest_model_dir(output_dir: str) -> str:
    """The most recent model_* directory, so notebook 02 needs no path."""
    candidates = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith("model_") and os.path.isdir(os.path.join(output_dir, d))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No model_* directory in {output_dir}. Run notebook 01 first."
        )
    return os.path.join(output_dir, candidates[-1])
