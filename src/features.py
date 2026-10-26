"""Leakage-conscious feature construction for descriptive analysis."""

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ID,
    PROCESSED,
    SYMPTOMS_OTHER,
    SYMPTOMS_RESP,
    TARGET,
    TAB,
)
from .stats_utils import benjamini_hochberg

RAW_FEATURES = [
    "GENDER_CODE",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]

ENGINEERED_FOR_F1 = [
    "AGE_Z",
    "RESP_SYMPTOM_COUNT",
    "SYMPTOM_COUNT_ALL",
    "SMOKER_IN_SCREEN_AGE",
]

FEATURE_SETS = {
    "F0_raw15": RAW_FEATURES,
    "F1_raw_plus_engineered": RAW_FEATURES + ENGINEERED_FOR_F1,
    "F2_prior_driven": [
        "AGE_Z",
        "SMOKING",
        "IN_SCREEN_AGE",
        "RESP_SYMPTOM_COUNT",
        "SMOKER_IN_SCREEN_AGE",
    ],
}


def recode_binary(df):
    """Map symptom codes 1/2 to 0/1 and rename GENDER to GENDER_CODE."""
    result = df.copy()
    flag_columns = [
        column
        for column in result.columns
        if column not in {ID, "GENDER", "AGE", TARGET}
    ]
    for column in flag_columns:
        if not result[column].isin([1, 2]).all():
            raise ValueError(f"{column} must contain only codes 1 and 2")
        result[column] = result[column].eq(2).astype("int8")
    return result.rename(columns={"GENDER": "GENDER_CODE"})


def fit_age_scaler(train_age):
    """Fit a population-standard-deviation scaler on the supplied age values."""
    values = np.asarray(train_age, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("train_age must contain finite values")
    scale = float(values.std(ddof=0))
    if scale == 0:
        raise ValueError("train_age must have nonzero variance")
    return {"mean": float(values.mean()), "scale": scale}


def add_features(df, scaler):
    """Create engineered descriptors using a separately supplied AGE scaler."""
    result = recode_binary(df)
    required = set(RAW_FEATURES) | {TARGET}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "mean" not in scaler or "scale" not in scaler or scaler["scale"] <= 0:
        raise ValueError("scaler must contain a valid mean and positive scale")

    result["AGE_Z"] = (result["AGE"] - scaler["mean"]) / scaler["scale"]
    result["AGE_BAND"] = pd.cut(
        result["AGE"],
        bins=[0, 50, 60, 70, 80, np.inf],
        labels=["<50", "50-59", "60-69", "70-79", "80+"],
        right=False,
    )
    result["IN_SCREEN_AGE"] = result["AGE"].between(50, 80).astype("int8")
    result["RESP_SYMPTOM_COUNT"] = result[SYMPTOMS_RESP].sum(axis=1).astype("int8")
    all_symptoms = SYMPTOMS_RESP + SYMPTOMS_OTHER
    result["SYMPTOM_COUNT_ALL"] = result[all_symptoms].sum(axis=1).astype("int8")
    result["SMOKER_IN_SCREEN_AGE"] = (
        result["SMOKING"] & result["IN_SCREEN_AGE"]
    ).astype("int8")
    return result.drop(columns=[ID], errors="ignore")


def verify(fe):
    """Validate engineered feature values, ranges, and leakage exclusions."""
    if ID in fe.columns:
        raise AssertionError("patient_id must not appear in engineered features")
    if TARGET not in fe.columns:
        raise AssertionError("LUNG_CANCER recorded label must be retained separately")
    if not fe[TARGET].isin([0, 1]).all():
        raise AssertionError("LUNG_CANCER must contain only 0/1")
    binary_columns = [
        column
        for column in fe.columns
        if column in RAW_FEATURES and column not in {"AGE", "GENDER_CODE"}
    ] + ["IN_SCREEN_AGE", "SMOKER_IN_SCREEN_AGE"]
    for column in binary_columns:
        if not fe[column].isin([0, 1]).all():
            raise AssertionError(f"{column} must be binary")
    if not fe["RESP_SYMPTOM_COUNT"].between(0, 4).all():
        raise AssertionError("RESP_SYMPTOM_COUNT must be in [0,4]")
    if not fe["SYMPTOM_COUNT_ALL"].between(0, 6).all():
        raise AssertionError("SYMPTOM_COUNT_ALL must be in [0,6]")
    if fe["AGE_BAND"].isna().any():
        raise AssertionError("AGE_BAND must be defined for every row")
    for feature_set, columns in FEATURE_SETS.items():
        missing = set(columns) - set(fe.columns)
        if missing:
            raise AssertionError(f"{feature_set} is missing columns: {sorted(missing)}")
    return True


def univariate_screening(fe, feature_columns):
    """Create descriptive, BH-adjusted univariate tests without selecting features."""
    rows = []
    label = fe[TARGET].to_numpy()
    for column in feature_columns:
        values = fe[column]
        if values.nunique() <= 2:
            table = pd.crosstab(values, label)
            statistic, p_value, _, _ = stats.chi2_contingency(
                table, correction=False
            )
            effect = float(np.sqrt(statistic / len(fe)))
            test = "chi-square"
        else:
            statistic, p_value = stats.spearmanr(values, label)
            effect = float(statistic)
            test = "Spearman"
        rows.append(
            {
                "feature": column,
                "test": test,
                "statistic": float(statistic),
                "effect_size": effect,
                "p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["q_BH"] = benjamini_hochberg(result["p"].to_numpy())
    return result


def build_feature_outputs(view_b):
    """Save the B-view feature sets, spec outputs, and descriptive screening table."""
    scaler = fit_age_scaler(view_b["AGE"])
    engineered = add_features(view_b, scaler)
    verify(engineered)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for name, features in FEATURE_SETS.items():
        table = engineered[features + [TARGET]].copy()
        table.to_csv(PROCESSED / f"features_B_{name}.csv", index=False)
        outputs[name] = table

    screening = univariate_screening(engineered, RAW_FEATURES + ENGINEERED_FOR_F1)
    screening.to_csv(TAB / "univariate_screening.csv", index=False)
    counts = engineered["RESP_SYMPTOM_COUNT"].value_counts().sort_index().rename_axis(
        "RESP_SYMPTOM_COUNT"
    ).reset_index(name="count")
    counts.to_csv(TAB / "resp_symptom_count_distribution_B.csv", index=False)
    return engineered, scaler, outputs, screening, counts
