"""Product category classifier: train on labelled data, predict on new."""

from .settings import (
    ID_LEVELS,
    NAME_LEVELS,
    N_LEVELS,
    ID_COL,
    TEXT_COL,
    ORIG_TEXT_COL,
    ModelSettings,
    NBSettings,
    VectorizerSettings,
    DELIMITER,
    cleaned_data_path,
    model_dir,
    predictions_path,
    raw_data_path,
    timestamp,
    DATA_DIR,
    OUTPUT_DIR,
    THRESHOLDS_PATH,
)

__all__ = [
    "ID_LEVELS", "NAME_LEVELS", "N_LEVELS", "ID_COL", "TEXT_COL",
    "ORIG_TEXT_COL", "ModelSettings", "NBSettings", "VectorizerSettings",
    "DELIMITER", "raw_data_path", "cleaned_data_path", "model_dir",
    "predictions_path", "timestamp", "DATA_DIR", "OUTPUT_DIR",
    "THRESHOLDS_PATH",
]
