import sys
import warnings
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .audit import auc_of_id, duplicate_report
from .config import INTERIM, RAW_FILE, TAB
from .data_io import load_raw, sha256


def make_views(df):
    """Create the four project views: A raw, B deduplicated, C label-conflict removed, D leading block removed."""
    views = {}
    views["A_raw"] = df.copy()

    dedup_cols = [c for c in df.columns if c != "patient_id"]
    B = df.drop_duplicates(subset=dedup_cols, keep="first").copy()
    views["B_dedup"] = B

    feature_cols = [c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}]
    grouped = B.groupby(feature_cols, dropna=False)
    conflict_rows = []
    for _, group in grouped:
        if group["LUNG_CANCER"].nunique() > 1:
            conflict_rows.extend(group.index.tolist())
    C = B.drop(index=conflict_rows).copy()
    views["C_dedup_noconflict"] = C

    fp = int(df.index[df["LUNG_CANCER"] == 1].min()) if (df["LUNG_CANCER"] == 1).any() else len(df)
    D = B.loc[B.index >= fp].copy()
    views["D_dedup_nolead"] = D
    return views


def summarize(views):
    rows = []
    comparison = {
        "A_raw": None,
        "B_dedup": "A_raw",
        "C_dedup_noconflict": "B_dedup",
        "D_dedup_nolead": "B_dedup",
    }
    for name, view in views.items():
        n = len(view)
        pos = int(view["LUNG_CANCER"].sum())
        pos_rate = float(pos / n) if n else 0.0
        min_age = float(view["AGE"].min()) if n else float("nan")
        mean_age = float(view["AGE"].mean()) if n else float("nan")
        reference = comparison[name]
        delta = n - len(views[reference]) if reference is not None else 0
        rows.append({
            "view": name,
            "n": n,
            "positives": pos,
            "pos_rate": pos_rate,
            "min_age": min_age,
            "mean_age": mean_age,
            "delta_n_vs_prev": delta,
        })
    out = Path(TAB) / "cleaning_log.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def write_cleaning_outputs(df, views):
    """Write the prescribed data views and preprocessing treatment table."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    for name, view in views.items():
        view.to_csv(INTERIM / f"{name}.csv", index=False)

    flagged = age_iqr_flag(df)
    flagged.to_csv(TAB / "age_iqr_flags.csv", index=False)
    q1 = float(df["AGE"].quantile(0.25))
    q3 = float(df["AGE"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    feature_cols = [c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}]
    conflicts = df.groupby(feature_cols, dropna=False)["LUNG_CANCER"].nunique().gt(1)
    conflict_patterns = int(conflicts.sum())
    conflict_rows = int(df.groupby(feature_cols, dropna=False).size()[conflicts].sum())
    fp = int(df.index[df["LUNG_CANCER"].eq(1)].min())
    duplicates = duplicate_report(df, fp)

    rows = [
        {
            "issue": "Missing values",
            "diagnostic": "Total null count",
            "treatment": "No treatment required",
            "rows_before": len(df),
            "rows_after": len(df),
            "rationale": "The raw dataset contains no missing values.",
        },
        {
            "issue": "Types and categories",
            "diagnostic": "All columns int64; category value-set checks",
            "treatment": "Preserve raw codes in interim views; recode for feature tables",
            "rows_before": len(df),
            "rows_after": len(df),
            "rationale": "Keep raw data unchanged and make coding transformations explicit.",
        },
        {
            "issue": "Exact duplicates",
            "diagnostic": f"{duplicates['exact_duplicates']} repeated feature-plus-label rows",
            "treatment": "Remove duplicates, keep first (view B)",
            "rows_before": len(df),
            "rows_after": len(views["B_dedup"]),
            "rationale": "Avoid repeated records driving the primary descriptive estimates.",
        },
        {
            "issue": "Label conflicts",
            "diagnostic": f"{conflict_patterns} feature patterns across {conflict_rows} raw rows have both labels",
            "treatment": "Retain in B; remove conflicting feature patterns in view C",
            "rows_before": len(views["B_dedup"]),
            "rows_after": len(views["C_dedup_noconflict"]),
            "rationale": "Keep the primary view minimally altered and test a conflict-free sensitivity view.",
        },
        {
            "issue": "Order and leading block",
            "diagnostic": f"First positive at index {fp}; {fp} earlier rows are negative",
            "treatment": "Retain in B; remove earlier rows in view D",
            "rows_before": len(views["B_dedup"]),
            "rows_after": len(views["D_dedup_nolead"]),
            "rationale": "Quantify sensitivity to the ordered block without replacing the primary view.",
        },
        {
            "issue": "AGE outliers",
            "diagnostic": f"IQR fences {lower:g} to {upper:g}; {int(flagged['AGE_OUTLIER'].sum())} flagged",
            "treatment": "Keep all rows; add AGE_OUTLIER flag table",
            "rows_before": len(df),
            "rows_after": len(df),
            "rationale": "Flag unusual ages without discarding plausible observations.",
        },
        {
            "issue": "Identifier leakage",
            "diagnostic": f"patient_id-only AUC {auc_of_id(df):.4f}",
            "treatment": "Exclude patient_id and row position from all feature sets",
            "rows_before": len(df),
            "rows_after": len(df),
            "rationale": "The sequential identifier is associated with label order and is not a patient feature.",
        },
        {
            "issue": "Integration and lineage",
            "diagnostic": "One raw CSV supplied; lineage not confirmed",
            "treatment": "No integration performed; lineage remains OPEN",
            "rows_before": len(df),
            "rows_after": len(df),
            "rationale": "Do not infer or invent source lineage.",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "treatments.csv", index=False)
    return table


def age_iqr_flag(df):
    q1 = df["AGE"].quantile(0.25)
    q3 = df["AGE"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = df.assign(AGE_OUTLIER=(df["AGE"] < lower) | (df["AGE"] > upper))
    return out[["patient_id", "AGE", "AGE_OUTLIER"]]


def main():
    """Run standalone view construction and write its row-count summary."""
    actual_prefix = sha256(RAW_FILE)[:16]
    if actual_prefix != "4ad2333067fcb0a9":
        warnings.warn(
            "Using the explicitly approved raw-file checksum override: "
            f"expected 4ad2333067fcb0a9, observed {actual_prefix}.",
            RuntimeWarning,
        )
    df = load_raw(path=RAW_FILE, verify=False)
    views = make_views(df)
    summary = summarize(views)
    write_cleaning_outputs(df, views)
    for row in summary.itertuples(index=False):
        print(f"{row.view}: n={row.n:,}, recorded_label_1_rate={row.pos_rate:.4f}")
    return summary


if __name__ == "__main__":
    main()
