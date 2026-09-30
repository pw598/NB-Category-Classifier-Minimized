"""Naive Bayes models combined into hierarchy path scores.

A prediction is a whole hierarchy path, chosen by summing per-level
log-probabilities over the paths actually observed in training. Two
consequences: a predicted path is always a real path, never an L4 paired
with an L1 it never appears under; and confidence can be reported at any
depth by summing the paths that share a prefix.

Two ways to produce those per-level factors, set by
ModelSettings.prediction_mode:

    joint_path        one model per level, trained on every row, and the
                      per-level marginals multiplied together.
    conditional_path  one model per parent node, each trained only on the
                      rows beneath it, combined by the chain rule.

They differ only in how the (rows x n_paths) log-score matrix is built --
see _joint_log_from_batch. Everything after that is shared, which is why
the output columns, the thresholds and the refusal logic are identical
either way.

Confidence at depth d is the summed score over the paths sharing that
prefix, reported exactly as the models produce it, with nothing divided
out. Under joint_path the figures are small in absolute terms -- four
posteriors multiplied -- and do not sum to one across prefixes; treat
them as scores, not probabilities of being correct. Under
conditional_path they do sum to one, being a genuine distribution over
the leaves.

The figures fall as depth grows, since getting four things right is harder
than getting one, and they are not comparable between levels: each depth's
scores come from a different distribution. Compare a level against its own
threshold, never against another level's.

Memory: the document-term matrix stays sparse; only observed paths are
scored rather than the full cross product; training and inference both
run in chunks, so peak memory does not scale with row count.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .settings import ModelSettings

# Floor applied before taking logs, to avoid log(0).
EPS = 1e-12

PATH_SCORE_COL = "Path Score"
PREFIX_PROB_FMT = "L{d} Confidence"
PRED_FMT = "L{d} Predicted {level}"


class HierarchicalClassifier:
    """Fit one model per level; predict whole paths with per-depth confidence."""

    def __init__(self, level_columns: List[str], settings: Optional[ModelSettings] = None):
        self.settings = settings or ModelSettings()
        self.level_columns = list(level_columns)
        self.vectorizer = None
        self.models: Dict[str, object] = {}
        self.label_to_col: Dict[str, Dict[str, int]] = {}
        self.valid_paths_: Optional[pd.DataFrame] = None
        self._path_idx: Optional[Dict[str, np.ndarray]] = None
        self._prefix_cache: Optional[Dict[int, dict]] = None
        # conditional_path only: one stacked matrix per depth, built by
        # _fit_node_models.
        self.node_stack_: Optional[Dict[int, dict]] = None
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, texts: pd.Series, labels: pd.DataFrame) -> "HierarchicalClassifier":
        t0 = time.time()
        self.vectorizer = self.settings.vectorizer.build()
        X = self.vectorizer.fit_transform(texts)
        print(f"[fit] vectorized {X.shape[0]:,} rows into {X.shape[1]:,} features "
              f"(sparse, nnz={X.nnz:,}) in {time.time() - t0:.1f}s")

        conditional = self.settings.prediction_mode == "conditional_path"
        # getattr: settings unpickled from a bundle saved before these
        # fields existed will not carry them.
        flat = getattr(self.settings, "fit_flat_levels", True) or not conditional

        if flat:
            for level in self.level_columns:
                t1 = time.time()
                y = labels[level].astype(str).values
                model = self._fit_level(X, y)
                self.models[level] = model
                self.label_to_col[level] = {
                    lbl: i for i, lbl in enumerate(model.classes_)
                }
                print(f"[fit] trained '{level}': {len(model.classes_):,} classes "
                      f"in {time.time() - t1:.1f}s")
        else:
            print("[fit] fit_flat_levels=False: skipping the per-level models, "
                  "so joint_path is unavailable on this object")

        self._build_path_index(labels, flat=flat)

        if conditional:
            self._fit_node_models(X, labels)

        del X
        gc.collect()
        self.fitted_ = True
        return self

    def _fit_level(self, X, y: np.ndarray):
        """Fit one level, in row chunks when configured.

        sklearn's MultinomialNB.fit densifies the labels into an
        (n_rows x n_classes) float64 array, which is fine at 20 classes and
        enormous at a few thousand -- what makes the deep levels fall over
        on a large catalogue. partial_fit accumulates the same counts chunk
        by chunk and recomputes the log-probabilities from those totals, so
        the fitted model is identical to a single fit() call. Passing the
        full class list up front is what makes it exact: the model
        allocates for every class immediately, so a chunk containing no
        examples of a rare class still leaves room for it.
        """
        model = self.settings.nb.build()
        chunk = self.settings.fit_chunk_size
        n = X.shape[0]

        if not chunk or chunk >= n:
            model.fit(X, y)
            return model

        classes = np.unique(y)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            model.partial_fit(X[start:end], y[start:end], classes=classes)
        return model

    def _build_path_index(self, labels: pd.DataFrame, flat: bool = True) -> None:
        self.valid_paths_ = (
            labels[self.level_columns].astype(str).drop_duplicates()
            .reset_index(drop=True)
        )
        if not flat:
            # The path index maps a path to a column of each flat model's
            # predict_proba output. Without those models there is nothing
            # to index into.
            self._path_idx = None
            print(f"[fit] {len(self.valid_paths_):,} paths observed in training")
            return

        self._path_idx = {
            level: self.valid_paths_[level].map(self.label_to_col[level]).to_numpy()
            for level in self.level_columns
        }
        n_full = 1
        for level in self.level_columns:
            n_full *= len(self.models[level].classes_)
        print(f"[fit] {len(self.valid_paths_):,} paths observed in training vs. "
              f"{n_full:,} possible combinations")

    # ------------------------------------------------------------------
    # conditional_path: one model per parent node
    # ------------------------------------------------------------------

    @staticmethod
    def _group_bounds(keys: np.ndarray):
        """Row positions grouped by key, from one stable sort.

        Rows of group i are order[bounds[i]:bounds[i + 1]].
        """
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        bounds = np.r_[starts, len(sorted_keys)]
        return order, sorted_keys, starts, bounds

    def _parent_keys(self, frame: pd.DataFrame, d: int) -> np.ndarray:
        """Each row's parent-prefix key at depth d. Depth 1 has one root."""
        parents = self.level_columns[: d - 1]
        if not parents:
            return np.full(len(frame), "", dtype="<U1")
        return frame[parents].astype(str).agg("\x00".join, axis=1).to_numpy()

    def _fit_node_models(self, X, labels: pd.DataFrame) -> None:
        """One MultinomialNB per parent, stacked into a matrix per depth.

        Each parent's model sees only its own rows and predicts among its
        own children, which is what makes the factor conditional rather
        than marginal. The estimators are discarded afterwards: scoring
        needs only their feature_log_prob_ and class_log_prior_, and one
        stacked matrix per depth beats holding a thousand small models.

        Layout per depth, in self.node_stack_[d]:
            W             (n_features x C_d) log-probabilities, sibling
                          groups in contiguous column blocks
            b             (C_d,) class log-priors, same column order
            group_starts  first column of each block, for reduceat
            group_sizes   width of each block
            path_col      each valid path's column in W, or -1
            fallback      each valid path's log-probability when its
                          parent has no model

        A parent goes unmodelled when it has one child (the factor is then
        log 1) or fewer rows than min_node_rows. Unmodelled parents spread
        their mass evenly over their children rather than contributing
        nothing, which would silently favour those subtrees.
        """
        levels = self.level_columns
        lab = labels[levels].astype(str).reset_index(drop=True)
        paths = self.valid_paths_
        n_paths = len(paths)
        min_rows = max(int(getattr(self.settings, "min_node_rows", 0) or 0), 0)
        score_dtype = np.dtype(self.settings.score_dtype)

        self.node_stack_ = {}
        for d in range(1, len(levels) + 1):
            t0 = time.time()
            child_col = levels[d - 1]

            row_keys = self._parent_keys(lab, d)
            order, sorted_keys, starts, bounds = self._group_bounds(row_keys)
            child_of_row = lab[child_col].to_numpy()

            path_keys = self._parent_keys(paths, d)
            path_child = paths[child_col].astype(str).to_numpy()

            blocks: List[np.ndarray] = []
            priors: List[np.ndarray] = []
            sizes: List[int] = []
            col_of: Dict[tuple, int] = {}
            next_col = 0
            n_modelled = n_skipped = 0

            for gi in range(len(starts)):
                rows = order[bounds[gi]: bounds[gi + 1]]
                key = sorted_keys[starts[gi]]
                y = child_of_row[rows]
                if len(np.unique(y)) < 2 or len(rows) < min_rows:
                    n_skipped += 1
                    continue
                model = self.settings.nb.build()
                model.fit(X[rows], y)
                blocks.append(model.feature_log_prob_.T.astype(score_dtype))
                priors.append(model.class_log_prior_.astype(score_dtype))
                for j, lbl in enumerate(model.classes_):
                    col_of[(key, str(lbl))] = next_col + j
                next_col += len(model.classes_)
                sizes.append(len(model.classes_))
                n_modelled += 1
                del model

            # Children per parent, counted over the path table that the
            # score matrix is indexed by.
            n_children: Dict[str, set] = {}
            for k, c in zip(path_keys, path_child):
                n_children.setdefault(k, set()).add(c)

            path_col = np.full(n_paths, -1, dtype=np.int64)
            fallback = np.zeros(n_paths, dtype=np.float64)
            for i in range(n_paths):
                key = path_keys[i]
                col = col_of.get((key, path_child[i]))
                if col is None:
                    fallback[i] = -np.log(max(len(n_children.get(key, ())), 1))
                else:
                    path_col[i] = col

            if blocks:
                W = np.hstack(blocks)
                b = np.concatenate(priors)
                size_arr = np.asarray(sizes, dtype=np.int64)
                gstarts = np.concatenate(([0], np.cumsum(size_arr)[:-1])).astype(np.int64)
            else:
                W = b = size_arr = gstarts = None
            del blocks, priors

            self.node_stack_[d] = {
                "W": W,
                "b": b,
                "group_starts": gstarts,
                "group_sizes": size_arr,
                "path_col": path_col,
                "fallback": fallback,
            }
            width = 0 if W is None else W.shape[1]
            print(f"[fit] depth {d}: {n_modelled:,} parent model(s), "
                  f"{n_skipped:,} left uniform, {width:,} stacked columns "
                  f"in {time.time() - t0:.1f}s")
            gc.collect()

    @staticmethod
    def _segmented_log_softmax(s, starts, sizes):
        """log-softmax within each contiguous block of columns.

        The max-shift is per block, not per row: each sibling group is its
        own distribution, and a shared shift would not cancel.
        """
        m_full = np.repeat(np.maximum.reduceat(s, starts, axis=1), sizes, axis=1)
        e = np.exp(s - m_full)
        z = np.add.reduceat(e, starts, axis=1)
        del e
        out = s - m_full - np.repeat(np.log(z), sizes, axis=1)
        del m_full
        return out

    def _conditional_joint_log(self, X_batch, score_dtype) -> np.ndarray:
        """log P(c1|x) + log P(c2|c1,x) + ... for every valid path."""
        from sklearn.utils.extmath import safe_sparse_dot

        joint = np.zeros(
            (X_batch.shape[0], len(self.valid_paths_)), dtype=score_dtype
        )
        for d in sorted(self.node_stack_):
            stack = self.node_stack_[d]
            joint += stack["fallback"].astype(score_dtype)
            if stack["W"] is None:
                continue
            # One matmul for the whole depth, then normalise inside each
            # sibling block -- the same reduceat trick the prefix sums use.
            s = np.asarray(
                safe_sparse_dot(X_batch, stack["W"]), dtype=score_dtype
            ) + stack["b"]
            logp = self._segmented_log_softmax(
                s, stack["group_starts"], stack["group_sizes"]
            )
            del s
            cols = stack["path_col"]
            mask = cols >= 0
            joint[:, mask] += logp[:, cols[mask]]
            del logp
        return joint

    def _flat_joint_log(self, X_batch, score_dtype) -> np.ndarray:
        """log P1(c1|x) + log P2(c2|x) + ... -- marginals, multiplied."""
        joint = np.zeros(
            (X_batch.shape[0], len(self.valid_paths_)), dtype=score_dtype
        )
        for level in self.level_columns:
            proba = np.clip(
                self.models[level].predict_proba(X_batch).astype(score_dtype),
                EPS, 1.0,
            )
            joint += np.log(proba)[:, self._path_idx[level]]
            del proba
        return joint

    def _joint_log_from_batch(self, X_batch, score_dtype) -> np.ndarray:
        """The (rows x n_paths) log-score matrix predict() works from.

        The only difference between the two modes lives here, which is why
        nothing downstream -- prefix sums, confidence, thresholds,
        refusals -- needs to know which one is in use.
        """
        if self.settings.prediction_mode == "conditional_path":
            if not self.node_stack_:
                raise RuntimeError(
                    "conditional_path needs the per-parent models, and this "
                    "classifier was fitted without them. Set "
                    "settings.prediction_mode = 'conditional_path' before "
                    "training."
                )
            return self._conditional_joint_log(X_batch, score_dtype)
        if self._path_idx is None:
            raise RuntimeError(
                "The per-level models were skipped at fit time "
                "(fit_flat_levels=False), so conditional_path is the only "
                "mode available on this object."
            )
        return self._flat_joint_log(X_batch, score_dtype)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predicting.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def transform(self, texts: pd.Series):
        self._check_fitted()
        return self.vectorizer.transform(texts.fillna("").astype(str))

    def _iter_batches(self, X):
        n = X.shape[0]
        for start in range(0, n, self.settings.batch_size):
            yield start, min(start + self.settings.batch_size, n), \
                X[start:min(start + self.settings.batch_size, n)]

    def _prefix_groups(self) -> Dict[int, dict]:
        """Path-to-prefix grouping per depth. Built once, cached."""
        if self._prefix_cache is not None:
            return self._prefix_cache

        cache = {}
        paths = self.valid_paths_
        for d in range(1, len(self.level_columns) + 1):
            cols = self.level_columns[:d]
            key = paths[cols].astype(str).agg("\x00".join, axis=1).to_numpy()
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
            first_rows = order[starts]
            cache[d] = {
                "order": order,
                "starts": starts,
                "labels": [paths[c].astype(str).to_numpy()[first_rows] for c in cols],
            }
        self._prefix_cache = cache
        return cache

    def predict(self, texts: pd.Series) -> pd.DataFrame:
        """The best path per row, and the confidence of every prefix of it.

        Returns the predicted label for each level, one confidence column
        per depth, and the whole-path score.

        A shallower depth's own best prefix is not always the first d
        levels of the full-depth answer -- summing over a prefix's children
        can let a different prefix win -- so each depth's confidence
        belongs to that depth's best prefix. Where the two part company,
        truncating would change the answer rather than shorten it.
        """
        self._check_fitted()
        X = self.transform(texts)
        n = X.shape[0]
        depths = range(1, len(self.level_columns) + 1)
        groups = self._prefix_groups()

        conf = {d: np.empty(n, dtype=np.float64) for d in depths}
        winner = {d: np.empty(n, dtype=np.int64) for d in depths}

        score_dtype = np.dtype(self.settings.score_dtype)
        for start, end, X_batch in self._iter_batches(X):
            joint_log = self._joint_log_from_batch(X_batch, score_dtype)

            # Shift by the row max before exponentiating: the joint log
            # probabilities underflow float64 well before depth 4.
            row_max = joint_log.max(axis=1, keepdims=True)
            weights = np.exp((joint_log - row_max).astype(np.float64))
            # Undoing the shift is the whole of the arithmetic here: the
            # figures are reported as the summed products the level models
            # produce, with nothing divided out. The EPS clip above bounds
            # row_max at 4*log(EPS), so exp(-row_max) cannot overflow.
            total = np.exp(-row_max.astype(np.float64))
            del joint_log

            for d in depths:
                order, starts = groups[d]["order"], groups[d]["starts"]
                # Paths pre-sorted by prefix, so contiguous blocks sum with
                # one reduceat rather than a scatter-add per group.
                summed = np.add.reduceat(weights[:, order], starts, axis=1)
                summed /= total
                best = np.argmax(summed, axis=1)
                conf[d][start:end] = summed[np.arange(end - start), best]
                winner[d][start:end] = best
                del summed
            del weights, total
            gc.collect()

        deepest = len(self.level_columns)
        out = {}
        for d in depths:
            labels = groups[d]["labels"]
            for i, level in enumerate(self.level_columns[:d]):
                out[PRED_FMT.format(d=d, level=level)] = labels[i][winner[d]]
            out[PREFIX_PROB_FMT.format(d=d)] = conf[d]
        frame = pd.DataFrame(out)

        for level in self.level_columns:
            frame[level] = frame[PRED_FMT.format(d=deepest, level=level)]
        frame[PATH_SCORE_COL] = conf[deepest]
        return frame

    def known_word_counts(self, texts: pd.Series) -> pd.DataFrame:
        """How much of each description the model can actually read.

        Confidence answers "how sure is the model?"; it does not answer
        "did the model have anything to work with?". A description whose
        every word is unknown still produces a confident-looking answer,
        drawn from how common each category is rather than from the
        product, and nothing in the score reveals that.
        """
        self._check_fitted()
        analyzer = self.vectorizer.build_analyzer()
        vocab = self.vectorizer.vocabulary_

        texts = texts.fillna("").astype(str)
        n_tokens = np.empty(len(texts), dtype=np.int32)
        n_known = np.empty(len(texts), dtype=np.int32)
        for i, t in enumerate(texts):
            toks = analyzer(t)
            n_tokens[i] = len(toks)
            n_known[i] = sum(1 for w in toks if w in vocab)

        return pd.DataFrame({
            "n_tokens": n_tokens,
            "n_known": n_known,
            "oov_rate": np.where(n_tokens > 0, (n_tokens - n_known) / np.maximum(n_tokens, 1), 0.0),
        })
