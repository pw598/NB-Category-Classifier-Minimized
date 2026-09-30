"""The description-cleaning procedure.

Aggressive by design: as well as colours, sizes and units it removes pack
counts, dimension strings, fractions, standalone numbers and all
punctuation. What survives is close to bare nouns and adjectives, which is
the point -- a category is determined by what a thing *is*, not by how big
it is or how many come in the box.

Two steps deserve explanation.

first-word stripping
    Descriptions often open with a vendor code or part number
    ('QJX8A TORQUE HEAD BOXEND'). If the leading token is not a
    recognisable English word it is dropped: it identifies the individual
    product rather than its category, and a token unique to one SKU is
    noise the model would otherwise try to learn from.

the two-phase regex pass
    The order is not interchangeable. Phase 1 removes compound patterns
    while the punctuation delimiting them is still present -- '1-1/4X3-1/2'
    is only recognisable as a dimension while it has its slashes. Phase 2
    then removes the reference-list terms and finally the punctuation and
    standalone numbers. Reversed, the dimension strings fragment into
    stray digits the earlier patterns no longer match.

The text is not uppercased. The patterns and reference lists are
uppercase, so this assumes the ERP descriptions arrive that way.

DescriptionCleaner is saved with the model, word list included, so scoring
applies exactly the recipe training used. The vocabulary is fitted on this
output; a scoring path that cleaned differently would see mostly unknown
words and fail silently, as a raised out-of-vocabulary rate rather than an
error.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LIST_DIR = Path(__file__).parent / "cleaning_lists"

# Units that may legitimately trail a number inside a dimension string, as
# opposed to the tokens making up the dimension itself.
SPEC_UNITS = ["FT", "IN", "CM", "MM", "MIL", "TPI", "TP", "T"]


# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------

def load_lists() -> Dict[str, List[str]]:
    """Every .txt in cleaning_lists/ as {stem: [uppercase terms]}.

    The first line of each file is a header and is skipped. Raises rather
    than returning an empty dict, since silently empty lists produce
    silently uncleaned text.
    """
    if not LIST_DIR.is_dir():
        raise FileNotFoundError(f"Cleaning-list directory not found: {LIST_DIR}")

    data: Dict[str, List[str]] = {}
    for path in sorted(LIST_DIR.iterdir()):
        if path.suffix.lower() not in (".txt", ".csv"):
            continue
        with path.open("r", encoding="utf-8-sig") as fh:
            if path.suffix.lower() == ".csv":
                reader = csv.reader(fh)
                next(reader, None)
                values = [r[0].strip() for r in reader if r and r[0].strip()]
            else:
                values = [ln.strip() for ln in fh.readlines()[1:] if ln.strip()]
        data[path.stem] = [v.upper() for v in values]

    if not data:
        raise ValueError(f"No .txt or .csv files found in {LIST_DIR}")
    return data


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

def build_patterns(lists: Dict[str, List[str]]) -> Dict[str, object]:
    """The four pattern groups, built from the reference lists.

    Note the lists are interpolated raw rather than regex-escaped, except
    for `sizes` in the SZ pattern; a term containing a regex metacharacter
    would therefore behave as a pattern rather than a literal.
    """
    colors = lists["colors"]
    color_abbrevs = lists["color_abbrevs"]
    sizes = lists["sizes"]
    units = lists["units"]
    unit_abbrevs = lists["unit_abbrevs"]
    unit_plurals = lists["unit_plurals"]

    spec_units_pattern = "|".join(SPEC_UNITS)
    non_spec_token = r"\d+|\.|/|-|X"
    all_tokens = rf"{non_spec_token}|{spec_units_pattern}"
    sizes_pattern = "|".join(re.escape(s) for s in sizes)

    deletion_1 = [
        # DV or DC indicators
        r"\(DV\)|\[DV\]|\(DC\)|\[DC\]",
        # number followed by a slash and then PK, etc.
        r"(?<=\s)\d+/(CT|CNT|CA|PK|BX|BG|CASE|PACK|BOX|BAG|RL|ROLL)",
        # number (or number followed by a dash) followed by CNT/PK, etc.
        r"(?<=\s)\d+(?:-)?/(CNT|CT)/(CA|PK|BX|BG|RL|ROLL)",
        # combos of digits, "X", units like FT/IN, dashes and slashes
        rf"(?<=\s)(?:{non_spec_token})(?:{all_tokens}){{1,}}(?=\s|$)",
        # periods
        r"[.]",
        r"(?:\d+-)?\d+/\d+[Xx]\d+",
        r"\d+/\d+[Xx]\d+-\d+/\d+",
        r"\d+[Xx]\d+-\d+/\d+",
        # #/#X#
        r"\d+/\d+[Xx]\d+",
        r"\d+-\d+/\d+",
        # #/#
        r"\d+/\d+",
    ]

    replacement_1 = {
        # replace apostrophe with a space
        r"[']": " ",
    }

    deletion_2 = [
        # #X#<unit> or #X#X#<unit>
        r"(?<=\s)\d+(?:(?:X\d+)|(" + "|".join(unit_abbrevs) + r"))*("
        + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by a unit abbreviation
        r"(?<=\s)\d+(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # PSI, HP, or a number followed by PSI or HP
        r"(?<=\s)(?:\d+\s*(?:PSI|HP)|PSI|HP)",
        # colour abbreviations
        r"(?<=\s)(" + "|".join(color_abbrevs) + r")(?=\s|$)",
        # colours
        r"(?<=\s)(" + "|".join(colors) + r")(?=\s|$)",
        # sizes
        r"(?:^|\s)(" + "|".join(sizes) + r")/(" + "|".join(sizes) + r")(?=\s|$)",
        r"(?<=\s)(" + "|".join(sizes) + r")(?=\s|$)",
        # unit plurals
        r"(?<=\s)(" + "|".join(unit_plurals) + r")(?=\s|$)",
        # units
        r"(?<=\s)(" + "|".join(units) + r")(?=\s|$)",
        # unit abbreviations
        r"(?<=\s)(" + "|".join(unit_abbrevs) + r")(?=\s|$)",
        # numbers followed by CT, CNT, CA, etc.
        r"(?<=\s)\d+(CT|CNT|CA|PK|BX|BG)(?=\s|$)",
        # "W/"
        r"W/",
        # #X# mixed with slashes and/or dashes
        r"(?i)(?<=\s)(?:(?:\d+(?:\.\d+)?|X|/|-)){2,}(?=\s|$)",
        # things like 1-1/4X3-1/2
        r"\d+-\d+/\d+\s*X\s*\d+-\d+/\d+",
        # things like 1-1/2
        r"\b\d+-\d+/\d+",
        # special characters
        r"[-\./()<>#&+\"!@$%^\*]",
        # #X#
        r"\d+X\d+",
        # standalone numbers
        r"(?<=\s)\d+(?=\s|$)",
        # numbers followed by "T"
        r"(?<=\s)\d+T(?=\s|$)",
        # stray "X"
        r"(?<=\s)X(?=\s|$)",
        # stray "SZ", or SZ followed by numbers
        rf"SZ(?=\s|$)|SZ\d\S*|\d+SZ\S*|SZ[./\-:;]\S*|SZ(?:{sizes_pattern})\b",
    ]

    replacement_2 = {
        # collapse runs of whitespace
        r"\s{2,}": " ",
    }

    return {
        "deletion_1": deletion_1,
        "replacement_1": replacement_1,
        "deletion_2": deletion_2,
        "replacement_2": replacement_2,
    }


def apply_patterns(series: pd.Series, patterns: Dict[str, object]) -> pd.Series:
    """The two-phase pass. Training and scoring both come through here."""
    out = series.astype(str)
    for pattern in patterns["deletion_1"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_1"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    for pattern in patterns["deletion_2"]:
        out = out.str.replace(pattern, "", regex=True)
    for pattern, replacement in patterns["replacement_2"].items():
        out = out.str.replace(pattern, replacement, regex=True)
    return out


# ---------------------------------------------------------------------------
# First-word stripping
# ---------------------------------------------------------------------------

def first_word_is_real(text, vocabulary: set) -> int:
    """1 when the leading token looks like an English word, else 0.

    Three ways to score 0, each deliberate: two characters or fewer (too
    short to be a meaningful noun); contains a digit (part numbers and
    sizes); not in the vocabulary (vendor codes, abbreviations, models).
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    first = text.split(maxsplit=1)[0].strip('.,;:!?"\'()[]{}').lower()
    if len(first) <= 2:
        return 0
    if any(ch.isdigit() for ch in first):
        return 0
    return 1 if first in vocabulary else 0


# ---------------------------------------------------------------------------
# The recipe as one portable object
# ---------------------------------------------------------------------------

@dataclass
class DescriptionCleaner:
    """The whole recipe, picklable, carrying its own word list.

    Saved with the model so scoring cannot use a different recipe -- or a
    different English word list, which would shift which leading tokens
    get stripped.
    """

    lists: Dict[str, List[str]]
    vocabulary: set = field(default_factory=set)
    patterns: Optional[Dict[str, object]] = None

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = build_patterns(self.lists)

    def clean(self, series: pd.Series) -> pd.Series:
        """Clean without dropping anything.

        Row-preserving on purpose: scoring needs one output per input, and
        a description reduced to nothing is a decision for the
        out-of-vocabulary gate rather than something to discard here.
        """
        out = series.fillna("").astype(str)
        out = out.map(self._strip_leading)
        out = apply_patterns(out, self.patterns)
        return out.str.strip()

    def _strip_leading(self, text: str) -> str:
        if first_word_is_real(text, self.vocabulary):
            return text
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def describe(self) -> str:
        return (
            f"DescriptionCleaner(lists={sorted(self.lists)}, "
            f"vocabulary={len(self.vocabulary):,} words)"
        )


def build_cleaner(verbose: bool = True) -> DescriptionCleaner:
    """A cleaner with the packaged lists and the NLTK word list."""
    from .bootstrap import ensure_word_corpus

    return DescriptionCleaner(
        lists=load_lists(), vocabulary=ensure_word_corpus(verbose=verbose)
    )


# ---------------------------------------------------------------------------
# Training-time cleaning
# ---------------------------------------------------------------------------

def clean_training_data(
    df: pd.DataFrame,
    cleaner: DescriptionCleaner,
    text_col: str,
    orig_col: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Preserve the original, clean, and drop rows left with no text.

    Unlike scoring, rows are dropped here: a description reduced to
    nothing cannot contribute to training. Counts are printed because the
    steps drop rows, and a total that moves without explanation is what
    goes unnoticed.
    """
    out = df.copy()
    out[orig_col] = out[text_col]

    n_start = len(out)
    out = out.dropna(subset=[text_col])
    out = out[out[text_col].astype(str).str.strip() != ""]
    n_after_blank = len(out)

    out[text_col] = cleaner.clean(out[text_col])
    out = out[out[text_col] != ""]
    n_end = len(out)

    if verbose:
        print(f"[clean] {n_start:>9,}  loaded")
        print(f"[clean] {n_after_blank:>9,}  after dropping blank descriptions")
        print(f"[clean] {n_end:>9,}  after cleaning emptied "
              f"{n_after_blank - n_end:,} more")
    return out.reset_index(drop=True)
