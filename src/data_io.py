import hashlib
from pathlib import Path

import pandas as pd

from .config import BASE_COLUMNS, EXPECTED, RAW_FILE


def sha256(path):
    """Return the SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df):
    """Validate the raw dataset schema and range checks."""
    if list(df.columns) != BASE_COLUMNS:
        raise ValueError(f"Expected columns {BASE_COLUMNS}, got {list(df.columns)}")
    if df.shape != (EXPECTED["rows"], EXPECTED["cols"]):
        raise ValueError(f"Expected shape {(EXPECTED['rows'], EXPECTED['cols'])}, got {df.shape}")
    if df.isna().sum().sum() != 0:
        raise ValueError("Dataset contains null values")
    if not all(dtype == "int64" for dtype in df.dtypes.astype(str).tolist()):
        raise ValueError("All columns must have int64 dtype")
    if not df["patient_id"].is_unique:
        raise ValueError("patient_id must be unique")
    if df["patient_id"].min() != 1 or df["patient_id"].max() != EXPECTED["rows"]:
        raise ValueError("patient_id range is unexpected")
    if not set(df["GENDER"].unique()).issubset({0, 1}):
        raise ValueError("GENDER values must be subset of {0,1}")
    if not set(df["LUNG_CANCER"].unique()).issubset({0, 1}):
        raise ValueError("LUNG_CANCER values must be subset of {0,1}")
    if df["AGE"].lt(18).any() or df["AGE"].gt(110).any():
        raise ValueError("AGE values must be between 18 and 110")
    symptom_cols = [c for c in BASE_COLUMNS if c not in {"patient_id", "GENDER", "AGE", "LUNG_CANCER"}]
    for col in symptom_cols:
        if not set(df[col].unique()).issubset({1, 2}):
            raise ValueError(f"{col} values must be subset of {{1,2}}")
    return df


def load_raw(path=RAW_FILE, verify=True):
    """Load the raw file, validate schema and checksum by default.

    Set verify=False only when intentionally using a known checksum-mismatched
    snapshot under an explicitly documented project override.
    """
    df = pd.read_csv(path)
    validate_schema(df)
    if verify:
        prefix = sha256(path)[:16]
        if prefix != EXPECTED["sha256_prefix"]:
            raise ValueError(
                f"Checksum prefix mismatch: expected {EXPECTED['sha256_prefix']}, got {prefix}"
            )
    return df
