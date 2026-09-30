"""Make the NLTK word list available, installing NLTK if it is absent.

Imported for its side effect, so the notebooks do not need a pip cell.

Two things have to be present: the `nltk` package, and its `words` corpus.
Both are handled here, and both are cached -- the install is skipped when
the import already works, and the download is skipped once the corpus is
on disk. Repeat runs cost nothing.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

_NLTK_SPEC = "nltk>=3.8"
_CORPUS = "words"


def ensure_nltk(verbose: bool = True):
    """Import nltk, pip-installing it first if that fails.

    A runtime pip install needs network access from wherever this runs. It
    also only takes effect for a package not already imported in this
    session, which nltk will not have been if the import above failed --
    so no kernel restart is needed here.
    """
    try:
        import nltk

        return nltk
    except ImportError:
        pass

    if verbose:
        print(f"[bootstrap] installing {_NLTK_SPEC}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", _NLTK_SPEC]
    )
    importlib.invalidate_caches()
    import nltk

    return nltk


def ensure_word_corpus(verbose: bool = True) -> set:
    """The English word list as a set of lowercase words.

    Checked before downloading, so this is a no-op after the first run on
    a given machine.
    """
    nltk = ensure_nltk(verbose=verbose)

    try:
        from nltk.corpus import words

        words.words()
    except LookupError:
        if verbose:
            print(f"[bootstrap] downloading the '{_CORPUS}' corpus")
        nltk.download(_CORPUS, quiet=True)
        from nltk.corpus import words

    from nltk.corpus import words

    vocabulary = {w.lower() for w in words.words()}
    if verbose:
        print(f"[bootstrap] {len(vocabulary):,} English words available")
    return vocabulary
