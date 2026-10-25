"""Pre-specified statistical comparisons for the recorded label."""

import numpy as np
import pandas as pd
from scipy import stats

from .audit import source_shift
from .config import SYMPTOMS_RESP, TAB
from .stats_utils import (
    benjamini_hochberg,
    cramers_v,
    feature_family,
    odds_ratio_ci,
    permutation_p,
    risk_difference_ci,
)


def h1_smoking(views, n_perms=5000, seed=42):
    """Compare recorded-label proportions for smoking codes 2 and 1."""
    rows = []
    for view_name, frame in views.items():
        smoker = frame["SMOKING"].eq(2).astype("int8").to_numpy()
        label = frame["LUNG_CANCER"].astype("int8").to_numpy()
        table = pd.crosstab(smoker, label).reindex(
            index=[1, 0], columns=[1, 0], fill_value=0
        )
        chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
        odds_ratio, or_low, or_high = odds_ratio_ci(smoker, label)
        risk_difference, rd_low, rd_high = risk_difference_ci(smoker, label)
        expected_min = float(stats.contingency.expected_freq(table.to_numpy()).min())
        rows.append(
            {
                "view": view_name,
                "hypothesis": "H1",
                "test": "Pearson chi-square; label permutation",
                "statistic": float(chi2),
                "p": float(p_value),
                "permutation_p": float(
                    permutation_p(smoker, label, n_perms=n_perms, seed=seed)
                ),
                "effect_size": float(odds_ratio),
                "ci_low": float(or_low),
                "ci_high": float(or_high),
                "risk_difference": float(risk_difference),
                "risk_difference_ci_low": float(rd_low),
                "risk_difference_ci_high": float(rd_high),
                "cramers_v": float(cramers_v(table)),
                "n": int(len(frame)),
                "min_expected_count": expected_min,
                "interpretation_flag": (
                    "expected_counts_ok" if expected_min >= 5 else "check_expected_counts"
                ),
            }
        )
    return pd.DataFrame(rows)


def _cles(pos, neg):
    ordered_neg = np.sort(neg)
    below = np.searchsorted(ordered_neg, pos, side="left")
    through = np.searchsorted(ordered_neg, pos, side="right")
    ties = through - below
    return float(np.mean((below + 0.5 * ties) / len(ordered_neg)))


def h2_age(views, bootstrap_reps=500, seed=42):
    """Compare AGE distributions by label and bootstrap the CLES interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for view_name, frame in views.items():
        pos = frame.loc[frame["LUNG_CANCER"].eq(1), "AGE"].to_numpy()
        neg = frame.loc[frame["LUNG_CANCER"].eq(0), "AGE"].to_numpy()
        statistic, p_value = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        cles = float(statistic / (len(pos) * len(neg)))
        boot = np.empty(bootstrap_reps)
        for i in range(bootstrap_reps):
            boot_pos = rng.choice(pos, size=len(pos), replace=True)
            boot_neg = rng.choice(neg, size=len(neg), replace=True)
            boot[i] = _cles(boot_pos, boot_neg)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "view": view_name,
                "hypothesis": "H2",
                "test": "Mann-Whitney U",
                "statistic": float(statistic),
                "p": float(p_value),
                "effect_size": cles,
                "ci_low": float(low),
                "ci_high": float(high),
                "positive_mean_age": float(pos.mean()),
                "negative_mean_age": float(neg.mean()),
                "positive_median_age": float(np.median(pos)),
                "negative_median_age": float(np.median(neg)),
                "n": int(len(frame)),
                "interpretation_flag": "rank_based",
            }
        )
    return pd.DataFrame(rows)


def h3_respiratory_count(views, bootstrap_reps=300, seed=42):
    """Test respiratory symptom count association with label by Spearman rho."""
    rng = np.random.default_rng(seed)
    rows = []
    for view_name, frame in views.items():
        count = frame[SYMPTOMS_RESP].eq(2).sum(axis=1).to_numpy()
        label = frame["LUNG_CANCER"].to_numpy()
        rho, p_value = stats.spearmanr(count, label)
        boot = np.empty(bootstrap_reps)
        n = len(frame)
        for i in range(bootstrap_reps):
            sampled = rng.integers(0, n, size=n)
            boot[i] = stats.spearmanr(count[sampled], label[sampled]).statistic
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "view": view_name,
                "hypothesis": "H3",
                "test": "Spearman correlation",
                "statistic": float(rho),
                "p": float(p_value),
                "effect_size": float(rho),
                "ci_low": float(low),
                "ci_high": float(high),
                "n": int(n),
                "interpretation_flag": "rank_correlation",
            }
        )
    return pd.DataFrame(rows)


def _cramers_v_ci(table, bootstrap_reps, rng):
    counts = np.asarray(table, dtype=float)
    if counts.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 contingency table, got {counts.shape}")
    probabilities = counts.ravel() / counts.sum()
    samples = rng.multinomial(int(counts.sum()), probabilities, size=bootstrap_reps)
    samples = samples.reshape(-1, 2, 2)
    row_totals = samples.sum(axis=2)
    column_totals = samples.sum(axis=1)
    denominator = (
        row_totals[:, 0]
        * row_totals[:, 1]
        * column_totals[:, 0]
        * column_totals[:, 1]
    )
    numerator = samples[:, 0, 0] * samples[:, 1, 1] - samples[:, 0, 1] * samples[:, 1, 0]
    values = np.zeros(bootstrap_reps, dtype=float)
    valid = denominator > 0
    values[valid] = np.abs(numerator[valid]) / np.sqrt(denominator[valid])
    return np.quantile(values, [0.025, 0.975])


def run_hypothesis_suite(df, views, n_perms=5000, bootstrap_reps=300, seed=42):
    """Run H1-H5 and return one normalized table across views and features."""
    binary_features = ["GENDER"] + [
        column
        for column in df.columns
        if column not in {"patient_id", "GENDER", "AGE", "LUNG_CANCER"}
    ]
    h1 = h1_smoking(views, n_perms=n_perms, seed=seed)
    h2 = h2_age(views, bootstrap_reps=bootstrap_reps, seed=seed)
    h3 = h3_respiratory_count(
        views, bootstrap_reps=max(100, bootstrap_reps // 2), seed=seed
    )
    for family in (h1, h2, h3):
        family["q_BH"] = benjamini_hochberg(family["p"].to_numpy())
    h1["feature"] = "SMOKING"
    h2["feature"] = "AGE"
    h3["feature"] = "RESP_SYMPTOM_COUNT"

    h4_rows = []
    rng = np.random.default_rng(seed)
    for view_name, frame in views.items():
        family = feature_family(frame, binary_features)
        intervals = [
            _cramers_v_ci(
                pd.crosstab(frame[column], frame["LUNG_CANCER"]),
                bootstrap_reps,
                rng,
            )
            for column in binary_features
        ]
        family["view"] = view_name
        family["hypothesis"] = "H4"
        family["test"] = "Chi-square family; Benjamini-Hochberg"
        family["statistic"] = family["chi2"]
        family["effect_size"] = family["v"]
        family["ci_low"] = [interval[0] for interval in intervals]
        family["ci_high"] = [interval[1] for interval in intervals]
        family["n"] = len(frame)
        family["interpretation_flag"] = "descriptive_family_test"
        h4_rows.append(family)
    h4 = pd.concat(h4_rows, ignore_index=True)

    fp = int(df.index[df["LUNG_CANCER"].eq(1)].min())
    h5 = source_shift(df, fp)
    h5["view"] = "integrity_check"
    h5["hypothesis"] = "H5"
    h5["test"] = "Chi-square; Cramer's V; bootstrap interval"
    h5["p"] = h5["p_value"]
    h5["q_BH"] = benjamini_hochberg(h5["p"].to_numpy())
    h5["effect_size"] = h5["cramers_v"]
    h5_intervals = [
        _cramers_v_ci(
            np.array(
                [
                    [
                        row.leading_block_yes_n,
                        row.leading_block_n - row.leading_block_yes_n,
                    ],
                    [
                        row.later_negatives_yes_n,
                        row.later_negatives_n - row.later_negatives_yes_n,
                    ],
                ]
            ),
            bootstrap_reps,
            rng,
        )
        for row in h5.itertuples(index=False)
    ]
    h5["ci_low"] = [interval[0] for interval in h5_intervals]
    h5["ci_high"] = [interval[1] for interval in h5_intervals]
    h5["n"] = h5["leading_block_n"] + h5["later_negatives_n"]
    h5["interpretation_flag"] = "integrity_shift"
    combined = pd.concat([h1, h2, h3, h4, h5], ignore_index=True, sort=False)
    TAB.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TAB / "hypotheses.csv", index=False)
    return combined
