import os
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.audit import (
    auc_of_id,
    duplicate_report,
    expected_duplicates_under_independence,
    independence_report,
    leading_block_end,
    plot_order_audit,
    source_shift,
)
from src.config import FIG, TAB
from src.clean import make_views, summarize, write_cleaning_outputs
from src.config import INTERIM, TAB
from src.build_report import make_report
from src.build_slides import build_deck
from src.data_io import load_raw, validate_schema
from src.figures import build_guide_figures
from src.results import write_results
from src.features import (
    FEATURE_SETS,
    add_features,
    build_feature_outputs,
    fit_age_scaler,
    verify,
)
from src.hypotheses import (
    h1_smoking,
    h2_age,
    h3_respiratory_count,
    run_hypothesis_suite,
)
from src.stats_utils import benjamini_hochberg, feature_family, min_detectable_difference


@pytest.mark.parametrize(
    ("script", "expected_output"),
    [
        ("audit.py", "first_positive_index=1455"),
        ("clean.py", "A_raw: n=22,457"),
        ("run_analysis.py", "Dataset shape: (22457, 17)"),
    ],
)
def test_handbook_direct_script_entrypoints(script, expected_output):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, str(root / "src" / script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert expected_output in completed.stdout


def test_raw_load_and_schema():
    df = load_raw(verify=False)
    validate_schema(df)
    assert df.shape == (22457, 17)
    assert df.isna().sum().sum() == 0
    assert df["patient_id"].is_unique
    assert df["LUNG_CANCER"].isin([0, 1]).all()
    assert df["AGE"].between(18, 110).all()


def test_recorded_label_prevalence_and_trivial_accuracy():
    df = load_raw(verify=False)
    counts = df["LUNG_CANCER"].value_counts()
    positive_prevalence = df["LUNG_CANCER"].eq(1).mean()

    assert counts[1] == 18434
    assert counts[0] == 4023
    assert positive_prevalence == pytest.approx(0.8209, abs=0.0001)
    assert positive_prevalence == pytest.approx((df["LUNG_CANCER"] == 1).mean())


def test_schema_rejects_age_outside_allowed_bounds():
    df = load_raw(verify=False)
    df.loc[df.index[0], "AGE"] = 17
    with pytest.raises(ValueError, match="between 18 and 110"):
        validate_schema(df)


def test_load_raw_verifies_checksum_by_default():
    with pytest.raises(ValueError, match="Checksum prefix mismatch"):
        load_raw()


def test_integrity_numbers():
    df = load_raw(verify=False)
    fp = leading_block_end(df)
    dup = duplicate_report(df, fp)
    assert dup["exact_duplicates"] == 2573
    assert dup["feature_only_duplicates"] == 2706
    assert dup["distinct_feature_patterns"] == 19751
    assert dup["leading_block_distinct_patterns"] == 103
    assert dup["conflicting_patterns"] == 133
    assert dup["conflicting_rows"] == 315
    assert dup["max_pattern_multiplicity"] == 86
    assert fp == 1455
    assert auc_of_id(df) == pytest.approx(0.67, abs=0.05)


def test_expected_duplicates_under_independent_marginals():
    df = load_raw(verify=False)
    fp = leading_block_end(df)
    later_rows = df.iloc[fp:]
    expected = expected_duplicates_under_independence(later_rows, reps=10, seed=42)
    assert expected == pytest.approx(594, abs=60)
    feature_columns = [c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}]
    assert later_rows.duplicated(subset=feature_columns).sum() == 1352


def test_independence_report_on_later_deduplicated_rows():
    df = load_raw(verify=False)
    fp = leading_block_end(df)
    feature_columns = [c for c in df.columns if c not in {"patient_id", "LUNG_CANCER"}]
    bulk = df.iloc[fp:].drop_duplicates(subset=feature_columns + ["LUNG_CANCER"])
    report = independence_report(bulk)

    assert report["n"] == 19781
    assert report["pairwise_chisq_tests"] == 91
    assert report["max_abs_pairwise_corr"] == pytest.approx(0.02154, abs=0.001)
    assert report["share_sig_pairwise_chisq"] == pytest.approx(4 / 91, abs=1e-12)
    assert 1.0 <= report["vif_min"] < 1.01
    assert 1.0 <= report["vif_max"] < 1.05
    assert report["pca_first_component"] == pytest.approx(0.07, abs=0.002)
    assert report["pca_flat_reference"] == pytest.approx(1 / 15)
    assert report["age_skew"] == pytest.approx(-0.006, abs=0.002)
    assert report["age_kurtosis"] == pytest.approx(-0.06, abs=0.01)


def test_leading_block_and_source_shift():
    df = load_raw(verify=False)
    fp = leading_block_end(df)
    assert fp == 1455
    assert df.iloc[:fp]["LUNG_CANCER"].eq(0).all()

    shifts = source_shift(df, fp).set_index("feature")
    smoking = shifts.loc["SMOKING"]
    assert smoking["leading_block_n"] == 1455
    assert smoking["later_negatives_n"] == 2568
    assert smoking["leading_block_yes_share"] == pytest.approx(0.2928, abs=0.001)
    assert smoking["later_negatives_yes_share"] == pytest.approx(0.5604, abs=0.001)
    assert shifts["cramers_v"].min() == pytest.approx(0.176, abs=0.002)
    assert shifts["cramers_v"].max() == pytest.approx(0.539, abs=0.002)
    assert shifts["p_value"].max() < 1e-25
    assert auc_of_id(df) == pytest.approx(0.67, abs=0.01)


def test_order_audit_figure_and_table():
    df = load_raw(verify=False)
    rolling = plot_order_audit(df, window=300)
    assert len(rolling) == len(df)
    assert (FIG / "f1_order.png").is_file()
    assert (TAB / "f1_order_rolling.csv").is_file()


def test_clean_views_and_logging():
    df = load_raw(verify=False)
    views = make_views(df)
    assert len(views["B_dedup"]) == 19884
    assert len(views["C_dedup_noconflict"]) == 19618
    assert len(views["D_dedup_nolead"]) == 19781
    summary = summarize(views)
    assert list(summary["view"]) == ["A_raw", "B_dedup", "C_dedup_noconflict", "D_dedup_nolead"]
    assert summary["n"].tolist() == [22457, 19884, 19618, 19781]
    assert summary["pos_rate"].tolist() == pytest.approx([0.8209, 0.8661, 0.8711, 0.8706], abs=0.0002)
    assert summary["delta_n_vs_prev"].tolist() == [0, -2573, -266, -103]
    treatments = write_cleaning_outputs(df, views)
    assert len(treatments) == 8
    assert (TAB / "treatments.csv").is_file()
    assert (TAB / "age_iqr_flags.csv").is_file()
    for name in views:
        assert (INTERIM / f"{name}.csv").is_file()
    age_flags = pd.read_csv(TAB / "age_iqr_flags.csv")
    assert int(age_flags["AGE_OUTLIER"].sum()) == 221


def test_h1_smoking_across_views():
    df = load_raw(verify=False)
    result = h1_smoking(make_views(df), n_perms=5000, seed=42).set_index("view")

    expected = {
        "A_raw": (1.49, 1.39, 1.59, 4.5e-30, 0.0),
        "B_dedup": (1.03, 0.95, 1.12, 0.52, 0.51),
        "C_dedup_noconflict": (1.04, 0.96, 1.14, 0.30, 0.30),
        "D_dedup_nolead": (1.00, 0.92, 1.08, 0.91, 0.91),
    }
    for view, (odds, low, high, p, perm_p) in expected.items():
        row = result.loc[view]
        assert row["effect_size"] == pytest.approx(odds, abs=0.01)
        assert row["ci_low"] == pytest.approx(low, abs=0.01)
        assert row["ci_high"] == pytest.approx(high, abs=0.01)
        assert row["p"] == pytest.approx(p, rel=0.1, abs=1e-31)
        assert row["permutation_p"] == pytest.approx(perm_p, abs=0.03)
        assert row["min_expected_count"] >= 5


def test_benjamini_hochberg_and_view_d_feature_family():
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.02]
    )
    df = load_raw(verify=False)
    views = make_views(df)
    features = ["GENDER"] + [c for c in df.columns[3:-1]]
    d_family = feature_family(views["D_dedup_nolead"], features)
    a_family = feature_family(views["A_raw"], features)

    assert len(d_family) == 14
    assert d_family["p"].min() == pytest.approx(0.0327, abs=0.001)
    assert d_family["q_BH"].min() == pytest.approx(0.4583, abs=0.002)
    assert d_family["v"].max() == pytest.approx(0.0152, abs=0.001)
    assert (a_family["q_BH"] < 0.05).all()


def test_h1_minimum_detectable_difference():
    assert min_detectable_difference(0.870633, 11080, 8701) == pytest.approx(
        0.0135, abs=0.0001
    )


def test_h2_age_and_h3_respiratory_count_across_views():
    df = load_raw(verify=False)
    views = make_views(df)
    h2 = h2_age(views, bootstrap_reps=50, seed=42).set_index("view")
    h3 = h3_respiratory_count(views, bootstrap_reps=50, seed=42).set_index("view")

    expected_h2 = {
        "A_raw": (6.72e-9, 0.529),
        "B_dedup": (0.0052, 0.517),
        "C_dedup_noconflict": (0.0024, 0.519),
        "D_dedup_nolead": (0.0511, 0.512),
    }
    expected_h3 = {
        "A_raw": (0.191, 0.0),
        "B_dedup": (0.0148, 0.036),
        "C_dedup_noconflict": (0.0207, 0.0038),
        "D_dedup_nolead": (-0.0021, 0.773),
    }
    for view, (p, cles) in expected_h2.items():
        assert h2.loc[view, "p"] == pytest.approx(p, rel=0.03, abs=1e-10)
        assert h2.loc[view, "effect_size"] == pytest.approx(cles, abs=0.001)
        assert h2.loc[view, "ci_low"] <= cles <= h2.loc[view, "ci_high"]
    for view, (rho, p) in expected_h3.items():
        assert h3.loc[view, "effect_size"] == pytest.approx(rho, abs=0.001)
        if p == 0.0:
            assert h3.loc[view, "p"] < 1e-100
        else:
            assert h3.loc[view, "p"] == pytest.approx(p, rel=0.05, abs=1e-5)
        assert h3.loc[view, "ci_low"] <= rho <= h3.loc[view, "ci_high"]


def test_hypothesis_suite_has_complete_effects_intervals_and_adjusted_p_values():
    df = load_raw(verify=False)
    result = run_hypothesis_suite(
        df, make_views(df), n_perms=100, bootstrap_reps=50, seed=42
    )

    assert result["hypothesis"].value_counts().to_dict() == {
        "H4": 56,
        "H5": 14,
        "H1": 4,
        "H2": 4,
        "H3": 4,
    }
    assert result[["statistic", "p", "q_BH", "effect_size", "ci_low", "ci_high"]].notna().all().all()
    assert result["q_BH"].between(0, 1).all()
    assert (result["ci_low"] <= result["ci_high"]).all()
    assert result.loc[result["hypothesis"].isin(["H4", "H5"]), "ci_low"].ge(0).all()
    assert result.loc[result["hypothesis"].isin(["H4", "H5"]), "ci_high"].le(1).all()
    assert result.loc[result["hypothesis"].eq("H1"), "permutation_p"].notna().all()
    assert set(result.loc[result["hypothesis"].eq("H5"), "view"]) == {"integrity_check"}
    assert (TAB / "hypotheses.csv").is_file()
    saved = pd.read_csv(TAB / "hypotheses.csv")
    assert len(saved) == len(result)


def test_requested_visualizations_have_registered_200_dpi_artifacts():
    df = load_raw(verify=False)
    views = make_views(df)
    plot_order_audit(df, window=300)
    hypothesis_table = run_hypothesis_suite(
        df, views, n_perms=100, bootstrap_reps=50, seed=42
    )
    build_feature_outputs(views["B_dedup"])
    figures = build_guide_figures(df, views, hypothesis_table)

    assert len(figures) == 19
    captions = json.loads((FIG / "captions.json").read_text(encoding="utf-8"))
    assert len(captions) == 19
    schema = pd.read_csv(TAB / "v2_schema_summary.csv").set_index("column")
    assert schema["missing"].sum() == 0
    assert all(
        schema.loc[column, "unique_values"] == "1, 2"
        for column in df.columns[3:-1]
    )
    yes_shares = pd.read_csv(TAB / "v6_yes_share_by_block.csv")
    assert set(yes_shares["feature"]) == set(df.columns[3:-1])
    assert set(captions) == {f"{name}.png" for name in [
        "f1_order", "fig5_views", "fig2_source_shift", "fig4_age", "fig3_correlation",
        "fig6_cramers_v", "fig7_smoking_forest", "v1_class_balance",
        "v2_schema_summary", "v4_duplicate_multiplicity", "v6_yes_share_by_block",
        "v7_age_profiles", "v8_pca_scree", "s3_h1_permutation_null",
        "s4_h4_p_q", "s5_view_d_feature_or", "s6_power_curve",
        "s7_respiratory_count", "s8_age_scaling_feature_sets",
    ]}
    for filename, details in captions.items():
        path = FIG / filename
        assert path.is_file() and path.stat().st_size > 0
        assert details["question"].strip()
        assert details["takeaway"].strip()
        with Image.open(path) as image:
            assert image.info["dpi"][0] == pytest.approx(200, abs=1)
            assert image.info["dpi"][1] == pytest.approx(200, abs=1)
    notebook = json.loads(
        (Path("notebooks") / "01_explore.ipynb").read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "figure_files" in notebook_source and "display(Image(filename" in notebook_source
    results = write_results()
    assert set(results["figure_files"]) == set(captions)
    assert results["figures"] == captions


def test_handbook_figure_mapping_matches_report_and_slides():
    results = json.loads((Path("docs") / "results.json").read_text(encoding="utf-8"))
    report = make_report(results)
    expected_report_mapping = [
        ("### Figure 1.", "f1_order.png"),
        ("### Figure 2.", "v6_yes_share_by_block.png"),
        ("### Figure 3.", "fig3_correlation.png"),
        ("### Figure 4.", "v7_age_profiles.png"),
        ("### Figure 5.", "fig5_views.png"),
        ("### Figure 6.", "fig6_cramers_v.png"),
        ("### Figure 7.", "fig7_smoking_forest.png"),
    ]
    for index, (heading, filename) in enumerate(expected_report_mapping):
        section_start = report.index(heading)
        section_end = (
            report.index(expected_report_mapping[index + 1][0], section_start + 1)
            if index + 1 < len(expected_report_mapping)
            else report.index("### Hypothesis results across views", section_start)
        )
        assert f"../reports/figures/{filename}" in report[section_start:section_end]

    slides, _ = build_deck(results)
    assert [path.name for path in slides[5]["figures"]] == [
        "f1_order.png",
        "v6_yes_share_by_block.png",
    ]
    assert slides[7]["figure"].name == "fig5_views.png"
    assert [path.name for path in slides[8]["figures"]] == [
        "fig3_correlation.png",
        "v7_age_profiles.png",
    ]
    assert [path.name for path in slides[10]["figures"]] == [
        "fig7_smoking_forest.png",
        "fig6_cramers_v.png",
    ]
    slide_source = (Path("slides") / "slides.md").read_text(encoding="utf-8")
    assert "Figure 1: row order, Figure 2: YES-share profiles" in slide_source
    assert "Figure 3: feature correlations, Figure 4: AGE profiles" in slide_source
    assert "Figure 7: smoking odds ratio by view, Figure 6: Cramer's V by feature" in slide_source


def test_feature_engineering_scaling_sets_and_outputs():
    df = load_raw(verify=False)
    views = make_views(df)
    fit_rows = views["B_dedup"].iloc[:100]
    scaler = fit_age_scaler(fit_rows["AGE"])
    transformed_fit = add_features(fit_rows, scaler)
    assert transformed_fit["AGE_Z"].mean() == pytest.approx(0.0, abs=1e-12)
    assert transformed_fit["AGE_Z"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)

    engineered, _, tables, screening, count_table = build_feature_outputs(
        views["B_dedup"]
    )
    assert verify(engineered)
    assert "patient_id" not in engineered.columns
    assert "LUNG_CANCER" in engineered.columns
    assert {key: len(value) for key, value in FEATURE_SETS.items()} == {
        "F0_raw15": 15,
        "F1_raw_plus_engineered": 19,
        "F2_prior_driven": 5,
    }
    assert {name: table.shape[1] - 1 for name, table in tables.items()} == {
        "F0_raw15": 15,
        "F1_raw_plus_engineered": 19,
        "F2_prior_driven": 5,
    }
    assert engineered["RESP_SYMPTOM_COUNT"].value_counts().sort_index().to_dict() == {
        0: 663, 1: 3416, 2: 7110, 3: 6426, 4: 2269
    }
    assert len(screening) == 19
    assert count_table["count"].sum() == len(views["B_dedup"])
    for feature_set in FEATURE_SETS:
        assert (Path("data/processed") / f"features_B_{feature_set}.csv").is_file()
