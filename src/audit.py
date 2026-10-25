import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from .config import DOCS, FIG, RAW_FILE, TAB
from .data_io import load_raw, sha256
from .stats_utils import cramers_v


def leading_block_end(df):
    """Return the first index where the recorded label becomes positive."""
    positives = df.index[df["LUNG_CANCER"] == 1]
    return int(positives.min()) if len(positives) else len(df)


def duplicate_report(df, fp=None):
    """Summarise duplicate patterns and label conflicts, excluding patient_id."""
    dedup_cols = [c for c in df.columns if c != "patient_id"]
    all_dup = df[dedup_cols].duplicated(keep="first").sum()
    feature_cols = [c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}]
    feature_dup = df[feature_cols].duplicated(keep="first").sum()
    grouped = df.groupby(feature_cols, dropna=False)["LUNG_CANCER"].agg(["nunique", "size"])
    conflicts = grouped[grouped["nunique"] > 1]
    lead_patterns = None
    if fp is not None:
        lead_patterns = int(df.iloc[:fp][feature_cols].drop_duplicates().shape[0])
    return {
        "exact_duplicates": int(all_dup),
        "feature_only_duplicates": int(feature_dup),
        "distinct_feature_patterns": int(df[feature_cols].drop_duplicates().shape[0]),
        "leading_block_distinct_patterns": lead_patterns,
        "conflicting_patterns": int(len(conflicts)),
        "conflicting_rows": int(conflicts["size"].sum()),
        "max_pattern_multiplicity": int(grouped["size"].max()),
    }


def source_shift(df, fp):
    """Cramer's V for leading-block negatives versus later negatives by binary feature."""
    lead = df.iloc[:fp]
    later_negatives = df.iloc[fp:]
    lead = lead[lead["LUNG_CANCER"] == 0]
    later_negatives = later_negatives[later_negatives["LUNG_CANCER"] == 0]
    result = []
    for col in [c for c in df.columns if c not in {"patient_id", "AGE", "LUNG_CANCER"}]:
        source = pd.Series(
            ["leading_block"] * len(lead) + ["later_negatives"] * len(later_negatives),
            name="source",
        )
        values = pd.concat([lead[col], later_negatives[col]], ignore_index=True)
        tab = pd.crosstab(source, values)
        chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
        v = cramers_v(tab)
        yes_value = 2 if col != "GENDER" else 1
        leading_yes_n = int(lead[col].eq(yes_value).sum())
        later_yes_n = int(later_negatives[col].eq(yes_value).sum())
        result.append(
            {
                "feature": col,
                "leading_block_n": int(len(lead)),
                "later_negatives_n": int(len(later_negatives)),
                "leading_block_yes_n": leading_yes_n,
                "later_negatives_yes_n": later_yes_n,
                "leading_block_yes_share": float(leading_yes_n / len(lead)),
                "later_negatives_yes_share": float(later_yes_n / len(later_negatives)),
                "statistic": float(chi2),
                "cramers_v": float(v),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(result)


def plot_order_audit(df, window=300):
    """Save rolling rates by row position and return their backing table."""
    fp = leading_block_end(df)
    rolling = pd.DataFrame(
        {
            "row_index": np.arange(len(df)),
            "recorded_label_1_rate": df["LUNG_CANCER"].rolling(window, min_periods=window).mean(),
            "smoking_yes_rate": df["SMOKING"].eq(2).rolling(window, min_periods=window).mean(),
            "gender_code_1_rate": df["GENDER"].eq(1).rolling(window, min_periods=window).mean(),
        }
    )

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(TAB / "f1_order_rolling.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(0, fp, color="#8A94A6", alpha=0.18, label="Leading negative block")
    ax.plot(rolling["row_index"], rolling["recorded_label_1_rate"], color="#1F3A5F", label="Recorded label 1")
    ax.plot(rolling["row_index"], rolling["smoking_yes_rate"], color="#2A9D8F", label="SMOKING = 2")
    ax.plot(rolling["row_index"], rolling["gender_code_1_rate"], color="#E9A23B", label="GENDER code 1")
    ax.axvline(fp, color="#C8553D", linestyle="--", label=f"First positive (index {fp})")
    ax.annotate(
        f"First positive: {fp}",
        xy=(fp, 0.98),
        xytext=(fp + len(df) * 0.04, 0.98),
        arrowprops={"arrowstyle": "->", "color": "#C8553D"},
        color="#C8553D",
    )
    ax.set(
        title="Row order reveals a leading all-negative block and changing feature rates",
        xlabel="Row index (0-based)",
        ylabel=f"Rolling proportion (window={window} rows)",
        ylim=(0, 1.05),
    )
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "f1_order.png", dpi=200)
    plt.close(fig)
    return rolling


def auc_of_id(df):
    labels = df["LUNG_CANCER"].to_numpy()
    ids = df["patient_id"].to_numpy()
    rank = stats.rankdata(ids)
    n1 = (labels == 1).sum()
    n0 = (labels == 0).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((rank[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def independence_report(df_bulk):
    """Summarise feature redundancy in a cleaned view."""
    x = df_bulk.drop(columns=["LUNG_CANCER", "patient_id"], errors="ignore")
    corr = x.corr().abs()
    n = x.shape[1]
    max_r = float(corr.where(~np.eye(n, dtype=bool)).max().max())
    mean_r = float(corr.where(~np.eye(n, dtype=bool)).stack().mean())

    pairwise_pvals = []
    binary_cols = [col for col in x.columns if col != "AGE"]
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i + 1 :]:
            table = pd.crosstab(x[c1], x[c2])
            if table.shape == (1, 1):
                continue
            chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
            pairwise_pvals.append(p)
    share_sig = float(np.mean(np.array(pairwise_pvals) < 0.05)) if pairwise_pvals else 0.0

    z = (x - x.mean()) / x.std(ddof=0)
    vifs = []
    for col in x.columns:
        y = x[col]
        others = x.drop(columns=[col])
        model = np.linalg.lstsq(np.column_stack([np.ones(len(others)), others.to_numpy()]), y.to_numpy(), rcond=None)[0]
        y_hat = np.column_stack([np.ones(len(others)), others.to_numpy()]) @ model
        r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
        vifs.append(1 / (1 - r2))

    pca = np.linalg.svd(z.to_numpy(), compute_uv=False)
    ratios = pca ** 2 / np.sum(pca ** 2)
    return {
        "max_abs_pairwise_corr": max_r,
        "mean_abs_pairwise_corr": mean_r,
        "pairwise_chisq_tests": int(len(pairwise_pvals)),
        "share_sig_pairwise_chisq": share_sig,
        "vif_min": float(np.min(vifs)),
        "vif_max": float(np.max(vifs)),
        "pca_first_component": float(ratios[0]),
        "pca_flat_reference": float(1 / n),
        "age_skew": float(df_bulk["AGE"].skew()),
        "age_kurtosis": float(df_bulk["AGE"].kurtosis()),
        "n": int(len(df_bulk)),
    }


def chi_square_pair(table):
    table = np.asarray(table)
    if table.shape[0] == 1 or table.shape[1] == 1:
        return 0.0, 1.0, 0, 0
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p, table.shape[0] - 1, (table.shape[0] - 1) * (table.shape[1] - 1)


def expected_duplicates_under_independence(df_bulk, reps=20, seed=42):
    """Approximate expected duplicates under independent feature marginals."""
    rng = np.random.default_rng(seed)
    features = [c for c in df_bulk.columns if c not in {"patient_id", "LUNG_CANCER"}]
    n = len(df_bulk)
    counts = []
    for _ in range(reps):
        sim = pd.DataFrame(index=range(n))
        for col in features:
            vals = df_bulk[col].value_counts(normalize=True)
            sim[col] = rng.choice(vals.index.to_numpy(), size=n, p=vals.to_numpy())
        counts.append(sim.duplicated().sum())
    return float(np.mean(counts))


def _safe_pearson_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    """Run the standalone integrity audit and save its machine-readable summary."""
    actual_prefix = sha256(RAW_FILE)[:16]
    if actual_prefix != "4ad2333067fcb0a9":
        warnings.warn(
            "Using the explicitly approved raw-file checksum override: "
            f"expected 4ad2333067fcb0a9, observed {actual_prefix}.",
            RuntimeWarning,
        )
    df = load_raw(path=RAW_FILE, verify=False)
    first_positive = leading_block_end(df)
    duplicates = duplicate_report(df, first_positive)
    shifts = source_shift(df, first_positive)
    result = {
        "sha256": sha256(RAW_FILE),
        "sha256_prefix": actual_prefix,
        "shape": list(df.shape),
        "null_count": int(df.isna().sum().sum()),
        "first_positive_index": first_positive,
        "leading_block_all_negative": bool(
            df.iloc[:first_positive]["LUNG_CANCER"].eq(0).all()
        ),
        "patient_id_auc": auc_of_id(df),
        "duplicate_report": duplicates,
        "source_shift_v_min": float(shifts["cramers_v"].min()),
        "source_shift_v_max": float(shifts["cramers_v"].max()),
        "source_shift_p_max": float(shifts["p_value"].max()),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    (DOCS / "audit_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    shifts.to_csv(TAB / "source_shift.csv", index=False)
    print(
        f"shape={tuple(result['shape'])} nulls={result['null_count']} "
        f"first_positive_index={first_positive} "
        f"leading_patterns={duplicates['leading_block_distinct_patterns']} "
        f"exact_duplicates={duplicates['exact_duplicates']} "
        f"conflicting_patterns={duplicates['conflicting_patterns']} "
        f"id_auc={result['patient_id_auc']:.4f}"
    )
    return result


if __name__ == "__main__":
    main()
