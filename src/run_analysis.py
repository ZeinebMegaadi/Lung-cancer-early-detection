"""Run the reproducible data-integrity and descriptive analysis pipeline."""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .audit import (
    auc_of_id,
    duplicate_report,
    expected_duplicates_under_independence,
    independence_report,
    leading_block_end,
    plot_order_audit,
    source_shift,
)
from .clean import make_views, summarize, write_cleaning_outputs
from .config import DOCS, RAW_FILE, TAB
from .data_io import load_raw, sha256
from .features import build_feature_outputs
from .figures import build_guide_figures
from .hypotheses import run_hypothesis_suite


def run():
    """Generate checked audit, view, hypothesis, figure, and feature outputs."""
    actual_prefix = sha256(RAW_FILE)[:16]
    warnings.warn(
        f"Raw checksum override in effect: expected project prefix differs "
        f"(expected 4ad2333067fcb0a9; observed {actual_prefix}).",
        RuntimeWarning,
    )
    df = load_raw(path=RAW_FILE, verify=False)
    fp = leading_block_end(df)
    views = make_views(df)

    duplicates = duplicate_report(df, fp)
    source = source_shift(df, fp)
    source.to_csv(TAB / "source_shift.csv", index=False)
    later_features = [
        column for column in df.columns if column not in {"patient_id", "LUNG_CANCER"}
    ]
    bulk = df.iloc[fp:].drop_duplicates(
        subset=later_features + ["LUNG_CANCER"]
    )
    independence = independence_report(bulk)
    independence_frame = pd.DataFrame([independence])
    independence_frame.to_csv(TAB / "independence_report.csv", index=False)
    simulation_mean = expected_duplicates_under_independence(
        df.iloc[fp:], reps=10, seed=42
    )
    audit_summary = {
        "sha256": sha256(RAW_FILE),
        "sha256_prefix": actual_prefix,
        "checksum_matches_contract": actual_prefix == "4ad2333067fcb0a9",
        "shape": list(df.shape),
        "null_count": int(df.isna().sum().sum()),
        "first_positive_index": fp,
        "leading_block_all_negative": bool(
            df.iloc[:fp]["LUNG_CANCER"].eq(0).all()
        ),
        "patient_id_auc": auc_of_id(df),
        "duplicate_report": duplicates,
        "expected_later_feature_duplicates_under_independence_10_reps": simulation_mean,
        "observed_later_feature_duplicates": int(
            df.iloc[fp:].duplicated(subset=later_features).sum()
        ),
        "source_shift_v_min": float(source["cramers_v"].min()),
        "source_shift_v_max": float(source["cramers_v"].max()),
        "source_shift_p_max": float(source["p_value"].max()),
        "independence_report": independence,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    with (DOCS / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    view_summary = summarize(views)
    write_cleaning_outputs(df, views)
    plot_order_audit(df, window=300)
    figures = build_guide_figures(df, views)
    hypothesis_table = run_hypothesis_suite(
        df, views, n_perms=5000, bootstrap_reps=300, seed=42
    )
    engineered, scaler, feature_tables, screening, count_table = build_feature_outputs(
        views["B_dedup"]
    )

    print(f"Raw checksum prefix: {actual_prefix} (contract match: {audit_summary['checksum_matches_contract']})")
    print(f"Dataset shape: {df.shape}; nulls: {audit_summary['null_count']}")
    print(f"First positive index: {fp}; preceding rows all negative: {audit_summary['leading_block_all_negative']}")
    print(f"Exact duplicates: {duplicates['exact_duplicates']}; feature-only duplicates: {duplicates['feature_only_duplicates']}")
    print(f"Views: {view_summary[['view', 'n', 'pos_rate']].to_dict(orient='records')}")
    print(f"Hypothesis rows: {len(hypothesis_table)}")
    print(f"Figures generated: {len(figures)}")
    print(f"Feature sets: { {name: table.shape[1] - 1 for name, table in feature_tables.items()} }")
    print(f"AGE scaler (descriptive B only): {scaler}")
    print(f"Screening rows: {len(screening)}; engineered rows: {len(engineered)}")
    print(f"Respiratory count distribution: {dict(zip(count_table['RESP_SYMPTOM_COUNT'], count_table['count']))}")
    return {
        "df": df,
        "views": views,
        "audit": audit_summary,
        "hypotheses": hypothesis_table,
        "feature_tables": feature_tables,
        "figures": figures,
    }


if __name__ == "__main__":
    run()
