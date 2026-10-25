"""Consolidate analysis outputs into the report/slides numeric source."""

import json
import re

import numpy as np
import pandas as pd

from .audit import (
    auc_of_id,
    duplicate_report,
    expected_duplicates_under_independence,
    independence_report,
    leading_block_end,
    source_shift,
)
from .clean import age_iqr_flag, make_views, summarize
from .config import DOCS, FIG, RAW_FILE, TAB
from .data_io import load_raw, sha256
from .hypotheses import run_hypothesis_suite
from .stats_utils import min_detectable_difference


def collect_results():
    """Regenerate the numeric result document from the raw file and analyses."""
    df = load_raw(path=RAW_FILE, verify=False)
    views = make_views(df)
    fp = leading_block_end(df)
    exact = duplicate_report(df, fp)
    shift = source_shift(df, fp)
    bulk_features = [
        c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}
    ]
    later = df.iloc[fp:]
    bulk = later.drop_duplicates(subset=bulk_features + ["LUNG_CANCER"])
    independence = independence_report(bulk)
    view_table = summarize(views)
    binary_features = ["GENDER"] + [
        c for c in df.columns if c not in {"patient_id", "GENDER", "AGE", "LUNG_CANCER"}
    ]

    hypothesis_results = run_hypothesis_suite(
        df, views, n_perms=5000, bootstrap_reps=300, seed=42
    )
    h1 = hypothesis_results[hypothesis_results["hypothesis"].eq("H1")].set_index("view")
    h2 = hypothesis_results[hypothesis_results["hypothesis"].eq("H2")].set_index("view")
    h3 = hypothesis_results[hypothesis_results["hypothesis"].eq("H3")].set_index("view")
    h4 = {
        name: hypothesis_results.loc[
            hypothesis_results["hypothesis"].eq("H4")
            & hypothesis_results["view"].eq(name)
        ]
        for name in views
    }
    h5 = hypothesis_results[hypothesis_results["hypothesis"].eq("H5")]

    q1, q3 = df["AGE"].quantile([0.25, 0.75])
    iqr = q3 - q1
    age_flags = age_iqr_flag(df)
    D = views["D_dedup_nolead"]
    d_smoker_count = int(D["SMOKING"].eq(2).sum())
    d_nonsmoker_count = int(D["SMOKING"].eq(1).sum())
    source_smoking = shift.set_index("feature").loc["SMOKING"]
    feature_smoke = ["COUGHING", "WHEEZING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]
    resp_count = (
        views["B_dedup"][feature_smoke].eq(2).sum(axis=1).value_counts().sort_index()
    )
    actual_hash = sha256(RAW_FILE)
    captions_path = FIG / "captions.json"
    if not captions_path.is_file():
        raise FileNotFoundError(
            f"Figure captions are missing; run `python -m src.run_analysis` first: "
            f"{captions_path}"
        )
    figure_metadata = json.loads(captions_path.read_text(encoding="utf-8"))
    for filename in figure_metadata:
        if not (FIG / filename).is_file():
            raise FileNotFoundError(
                f"Registered figure is missing; run `python -m src.run_analysis` first: "
                f"{FIG / filename}"
            )

    results = {
        "dataset": {
            "n_raw": int(len(df)),
            "n_columns": int(df.shape[1]),
            "null_count": int(df.isna().sum().sum()),
            "sha256": actual_hash,
            "sha256_prefix": actual_hash[:16],
            "expected_sha256_prefix": "4ad2333067fcb0a9",
            "checksum_matches_contract": actual_hash.startswith("4ad2333067fcb0a9"),
            "patient_id_unique": bool(df["patient_id"].is_unique),
            "confidence_interval_percent": 95,
            "gender_code_1_share": float(df["GENDER"].eq(1).mean()),
            "binary_flag_yes_shares": {
                column: float(df[column].eq(2).mean())
                for column in df.columns
                if column not in {"patient_id", "GENDER", "AGE", "LUNG_CANCER"}
            },
        },
        "target": {
            "negative_n": int(df["LUNG_CANCER"].eq(0).sum()),
            "positive_n": int(df["LUNG_CANCER"].eq(1).sum()),
            "positive_rate": float(df["LUNG_CANCER"].mean()),
            "always_positive_accuracy_reference": float(df["LUNG_CANCER"].mean()),
        },
        "age": {
            "min": int(df["AGE"].min()),
            "max": int(df["AGE"].max()),
            "mean": float(df["AGE"].mean()),
            "std": float(df["AGE"].std()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr_fence_low": float(q1 - 1.5 * iqr),
            "iqr_fence_high": float(q3 + 1.5 * iqr),
            "outlier_n": int(age_flags["AGE_OUTLIER"].sum()),
            "under_40_n": int(df["AGE"].lt(40).sum()),
        },
        "integrity": {
            "first_positive_index": fp,
            "leading_block_n": fp,
            "leading_block_all_negative": bool(df.iloc[:fp]["LUNG_CANCER"].eq(0).all()),
            "leading_block_distinct_patterns": exact["leading_block_distinct_patterns"],
            "exact_duplicates": exact["exact_duplicates"],
            "feature_only_duplicates": exact["feature_only_duplicates"],
            "distinct_feature_patterns": exact["distinct_feature_patterns"],
            "conflicting_patterns": exact["conflicting_patterns"],
            "conflicting_rows": exact["conflicting_rows"],
            "max_pattern_multiplicity": exact["max_pattern_multiplicity"],
            "later_duplicates_simulated_mean": float(
                expected_duplicates_under_independence(later, reps=10, seed=42)
            ),
            "later_duplicates_observed": int(
                later.duplicated(subset=bulk_features).sum()
            ),
            "patient_id_auc": auc_of_id(df),
            "source_shift_v_min": float(shift["cramers_v"].min()),
            "source_shift_v_max": float(shift["cramers_v"].max()),
            "source_shift_p_max": float(shift["p_value"].max()),
            "source_shift_smoking_lead_yes": float(
                source_smoking["leading_block_yes_share"]
            ),
            "source_shift_smoking_later_yes": float(
                source_smoking["later_negatives_yes_share"]
            ),
            "source_shift_age_lead_min": int(df.iloc[:fp]["AGE"].min()),
            "source_shift_age_later_negative_min": int(
                df.iloc[fp:].loc[df.iloc[fp:]["LUNG_CANCER"].eq(0), "AGE"].min()
            ),
            "independence": independence,
        },
        "views": {
            row["view"]: {
                "n": int(row["n"]),
                "positives": int(row["positives"]),
                "positive_rate": float(row["pos_rate"]),
                "mean_age": float(row["mean_age"]),
                "min_age": float(row["min_age"]),
            }
            for row in view_table.to_dict(orient="records")
        },
        "h1": {
            view: {
                "odds_ratio": float(row["effect_size"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p": float(row["p"]),
                "q_BH": float(row["q_BH"]),
                "permutation_p": float(row["permutation_p"]),
                "risk_difference": float(row["risk_difference"]),
                "risk_difference_ci_low": float(row["risk_difference_ci_low"]),
                "risk_difference_ci_high": float(row["risk_difference_ci_high"]),
                "cramers_v": float(row["cramers_v"]),
            }
            for view, row in h1.iterrows()
        },
        "h2": {
            view: {
                "cles": float(row["effect_size"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p": float(row["p"]),
                "q_BH": float(row["q_BH"]),
            }
            for view, row in h2.iterrows()
        },
        "h3": {
            view: {
                "rho": float(row["effect_size"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p": float(row["p"]),
                "q_BH": float(row["q_BH"]),
            }
            for view, row in h3.iterrows()
        },
        "h4": {
            view: {
                "min_p": float(table["p"].min()),
                "min_q": float(table["q_BH"].min()),
                "max_v": float(table["v"].max()),
                "all_q_below_005": bool((table["q_BH"] < 0.05).all()),
            }
            for view, table in h4.items()
        },
        "feature_family_size": len(binary_features),
        "h5": {
            row["feature"]: {
                "statistic": float(row["statistic"]),
                "p": float(row["p"]),
                "q_BH": float(row["q_BH"]),
                "cramers_v": float(row["effect_size"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
            for _, row in h5.iterrows()
        },
        "hypothesis_results": [
            {
                key: (
                    None
                    if pd.isna(value)
                    else int(value)
                    if key == "n"
                    else value.item()
                    if isinstance(value, (np.integer, np.floating))
                    else value
                )
                for key, value in row.items()
            }
            for row in hypothesis_results[
                [
                    "view",
                    "hypothesis",
                    "feature",
                    "test",
                    "statistic",
                    "p",
                    "q_BH",
                    "effect_size",
                    "ci_low",
                    "ci_high",
                    "n",
                    "permutation_p",
                ]
            ].to_dict(orient="records")
        ],
        "power": {
            "view_D_smokers_n": d_smoker_count,
            "view_D_nonsmokers_n": d_nonsmoker_count,
            "alpha": 0.05,
            "power": 0.8,
            "minimum_detectable_risk_difference": min_detectable_difference(
                float(D["LUNG_CANCER"].mean()), d_smoker_count, d_nonsmoker_count
            ),
        },
        "features": {
            "F0_raw15": 15,
            "F1_raw_plus_engineered": 19,
            "F2_prior_driven": 5,
            "age_band_edges": [50, 59, 60, 69, 70, 79, 80],
            "resp_symptom_count_distribution_B": {
                str(key): int(value) for key, value in resp_count.items()
            },
        },
        "figure_files": list(figure_metadata),
        "figures": figure_metadata,
        "external_context": {
            "uspstf": {
                "year": 2021,
                "age_min": 50,
                "age_max": 80,
                "pack_years_min": 20,
                "quit_years_max": 15,
                "frequency_per_year": 1,
            },
            "references": [
                {
                    "number": 1,
                    "year": 2021,
                    "title": "Screening for Lung Cancer: US Preventive Services Task Force Recommendation Statement",
                    "doi": "10.1001/jama.2021.1117",
                    "verified_via": "USPSTF recommendation page and PubMed metadata",
                    "volume": 325,
                    "issue": 10,
                    "page_start": 962,
                    "page_end": 970,
                },
                {
                    "number": 2,
                    "year": 2013,
                    "title": "Selection Criteria for Lung-Cancer Screening",
                    "doi": "10.1056/NEJMoa1211776",
                    "verified_via": "PubMed and OpenAlex metadata",
                    "volume": 368,
                    "issue": 8,
                    "page_start": 728,
                    "page_end": 736,
                },
                {
                    "number": 3,
                    "year": 2023,
                    "title": "Leakage and the reproducibility crisis in machine-learning-based science",
                    "doi": "10.1016/j.patter.2023.100804",
                    "verified_via": "OpenAlex and Elsevier DOI landing metadata",
                    "volume": 4,
                    "issue": 9,
                    "article_number": 100804,
                },
                {
                    "number": 4,
                    "year": 1995,
                    "title": "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
                    "doi": "10.1111/j.2517-6161.1995.tb02031.x",
                    "verified_via": "OpenAlex metadata",
                    "volume": 57,
                    "issue": 1,
                    "page_start": 289,
                    "page_end": 300,
                },
                {
                    "number": 5,
                    "year": 2000,
                    "title": "CRISP-DM 1.0: Step-by-step data mining guide",
                    "doi": None,
                    "verified_via": "Bibliographic record was not independently retrieved; verify before formal submission",
                },
            ],
        },
        "deliverable_spec": {
            "report_sections": 6,
            "report_pages_min": 10,
            "report_pages_max": 14,
            "numbered_figures": 7,
            "main_slides": 14,
            "backup_slides": 4,
            "presentation_minutes": 20,
            "rehearsal_max_minutes": 19.5,
            "qa_answers": 35,
            "slide_numbers": list(range(1, 19)),
        },
    }
    return results


def write_results():
    results = collect_results()
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "results.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {path}")
    print(f"Dataset: {results['dataset']['n_raw']} rows x {results['dataset']['n_columns']} columns")
    print(f"View D smoking OR: {results['h1']['D_dedup_nolead']['odds_ratio']:.2f}")
    print(f"View D minimum q: {results['h4']['D_dedup_nolead']['min_q']:.4f}")
    return results


def check_consistency():
    """Flag report/slide numeric literals not present among JSON number values."""
    results_path = DOCS / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    numeric_values = []

    def collect(value):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(results)
    misses = []
    for path in [DOCS.parent / "report" / "report.md", DOCS.parent / "slides" / "slides.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"10\.\d+/\S+", "", text)
        text = text.replace("SHA-256", "SHA")
        for checksum in (
            results["dataset"]["sha256"],
            results["dataset"]["sha256_prefix"],
            results["dataset"]["expected_sha256_prefix"],
        ):
            text = text.replace(checksum, "")
        pattern = r"(?<![A-Za-z0-9])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?"
        for match in re.finditer(pattern, text, re.I):
            token = match.group(0)
            is_percent = token.endswith("%")
            bare = token.rstrip("%").replace(",", "")
            try:
                value = float(bare)
            except ValueError:
                continue
            mantissa = re.split(r"e", bare, flags=re.I)[0]
            decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
            scientific = "e" in bare.lower()
            candidates_match = False
            for candidate in numeric_values:
                candidate_value = candidate * 100 if is_percent else candidate
                if scientific:
                    candidate_text = f"{candidate_value:.{decimals}e}"
                    target_text = f"{value:.{decimals}e}"
                else:
                    candidate_text = f"{candidate_value:.{decimals}f}"
                    target_text = f"{value:.{decimals}f}"
                if candidate_text == target_text:
                    candidates_match = True
                    break
            if candidates_match:
                continue
            if value in {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}:
                continue
            misses.append((path.name, token))
    if misses:
        print("Unmatched numeric literals:")
        for filename, token in misses:
            print(f"{filename}: {token}")
        return False
    print("Report/slides numeric consistency: PASS")
    return True


if __name__ == "__main__":
    write_results()
