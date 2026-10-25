"""Build the project data and statistical visualization set."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .audit import source_shift
from .config import FIG, TAB
from .features import FEATURE_SETS, add_features, fit_age_scaler
from .stats_utils import (
    feature_family,
    min_detectable_difference,
    odds_ratio_ci,
)


def _save(frame, stem):
    TAB.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TAB / f"{stem}.csv", index=False)


def _save_figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _feature_columns(df):
    return [
        column
        for column in df.columns
        if column not in {"patient_id", "AGE", "LUNG_CANCER"}
    ]


def _record_figure(captions, filename, question, takeaway):
    captions[filename] = {"question": question, "takeaway": takeaway}


def _save_captions(captions):
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "captions.json").write_text(
        json.dumps(captions, indent=2), encoding="utf-8"
    )


def _yes_value(column):
    return 1 if column == "GENDER" else 2


def plot_views(views):
    rows = [
        {
            "view": name,
            "n": len(frame),
            "positive_rate": frame["LUNG_CANCER"].mean(),
        }
        for name, frame in views.items()
    ]
    data = pd.DataFrame(rows)
    _save(data, "fig5_views")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = data["view"].str.replace("_", " ")
    axes[0].barh(labels, data["n"], color="#1F3A5F")
    axes[0].set(title="Rows retained by cleaning view", xlabel="Rows", ylabel="View")
    axes[1].barh(labels, data["positive_rate"], color="#2A9D8F")
    axes[1].set(
        title="Recorded-label-1 rate by view",
        xlabel="Proportion of rows",
        ylabel="View",
        xlim=(0, 1),
    )
    _save_figure(fig, "fig5_views.png")
    return data


def plot_source_shift(df):
    fp = int(df.index[df["LUNG_CANCER"].eq(1)].min())
    data = source_shift(df, fp).sort_values("cramers_v")
    _save(data, "fig2_source_shift")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["feature"], data["cramers_v"], color="#E9A23B")
    ax.set(
        title="Leading-block negatives differ from later negatives",
        xlabel="Cramer's V (feature association with block membership)",
        ylabel="Binary feature",
    )
    _save_figure(fig, "fig2_source_shift.png")
    return data


def plot_age_distribution(view, view_name="D"):
    age = view["AGE"].to_numpy()
    mean = float(np.mean(age))
    sd = float(np.std(age, ddof=1))
    x = np.linspace(float(age.min()), float(age.max()), 300)
    density = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    data = pd.DataFrame({"age": x, "normal_overlay_density": density})
    _save(data, "fig4_age_normal_overlay")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(age, bins=30, density=True, alpha=0.65, color="#2A9D8F", label=f"View {view_name} AGE")
    ax.plot(x, density, color="#C8553D", linewidth=2, label="Normal overlay using observed mean and SD")
    ax.set(
        title="AGE in the later deduplicated view is approximately bell-shaped",
        xlabel="AGE (recorded units; year interpretation unverified)",
        ylabel="Density",
    )
    ax.legend()
    _save_figure(fig, "fig4_age.png")
    return data


def plot_feature_correlation(view):
    features = view.drop(columns=["patient_id", "LUNG_CANCER"], errors="ignore")
    corr = features.corr()
    data = corr.rename_axis("feature").reset_index()
    _save(data, "fig3_correlation")
    max_abs = float(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max())
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title(f"Feature correlation heatmap (colour clipped at ±0.10; max |r|={max_abs:.4f})")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    _save_figure(fig, "fig3_correlation.png")
    return data


def plot_cramers_v(raw_view, cleaned_view, features):
    raw = feature_family(raw_view, features).assign(view="A raw")
    clean = feature_family(cleaned_view, features).assign(view="D cleaned")
    data = pd.concat([raw, clean], ignore_index=True)
    _save(data, "fig6_cramers_v")
    pivot = data.pivot(index="feature", columns="view", values="v").sort_values("A raw")
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(pivot))
    ax.hlines(y, pivot["A raw"], pivot["D cleaned"], color="#8A94A6", linewidth=1.5)
    ax.scatter(pivot["A raw"], y, color="#C8553D", label="A raw", zorder=3)
    ax.scatter(pivot["D cleaned"], y, color="#2A9D8F", label="D cleaned", zorder=3)
    ax.set_yticks(y, pivot.index)
    ax.set(
        title="Feature–label associations shrink after removing the leading block",
        xlabel="Cramer's V",
        ylabel="Feature",
    )
    ax.legend()
    _save_figure(fig, "fig6_cramers_v.png")
    return data


def plot_smoking_forest(views):
    rows = []
    for name, frame in views.items():
        smoker = frame["SMOKING"].eq(2).to_numpy()
        odds_ratio, low, high = odds_ratio_ci(smoker, frame["LUNG_CANCER"].to_numpy())
        rows.append(
            {"view": name, "effect_size": odds_ratio, "ci_low": low, "ci_high": high}
        )
    data = pd.DataFrame(rows)
    _save(data, "fig7_smoking_forest")
    labels = data["view"].str.replace("_", " ")
    y = np.arange(len(data))
    errors = np.vstack(
        [data["effect_size"] - data["ci_low"], data["ci_high"] - data["effect_size"]]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(data["effect_size"], y, xerr=errors, fmt="o", color="#1F3A5F", capsize=4)
    ax.axvline(1, color="#C8553D", linestyle="--", label="No association (OR = 1)")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set(
        title="Smoking odds ratio changes across cleaning views",
        xlabel="Odds ratio for recorded label (log scale; 95% Wald CI)",
        ylabel="View",
    )
    ax.legend()
    _save_figure(fig, "fig7_smoking_forest.png")
    return data


def plot_class_balance(df):
    counts = df["LUNG_CANCER"].value_counts().reindex([0, 1], fill_value=0)
    data = pd.DataFrame(
        {
            "recorded_label": counts.index,
            "n": counts.to_numpy(),
            "share": counts.to_numpy() / len(df),
        }
    )
    _save(data, "v1_class_balance")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["0", "1"], data["share"], color=["#8A94A6", "#2A9D8F"])
    ax.axhline(data["share"].max(), color="#C8553D", linestyle="--",
               label="Always-positive reference")
    ax.set(
        title="Recorded-label class balance and always-positive reference",
        xlabel="Recorded label code (meaning unverified)",
        ylabel="Share of rows",
        ylim=(0, 1),
    )
    for bar, row in zip(bars, data.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, row.share + 0.02,
                f"{row.share:.1%}", ha="center")
    ax.legend()
    _save_figure(fig, "v1_class_balance.png")
    return data


def plot_schema_summary(df):
    rows = []
    for column in df.columns:
        values = sorted(df[column].dropna().unique().tolist())
        if len(values) > 20:
            value_summary = f"{values[0]}-{values[-1]} ({len(values)} unique)"
        else:
            value_summary = ", ".join(map(str, values))
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique_values": value_summary,
                "value_count": len(values),
            }
        )
    data = pd.DataFrame(rows)
    _save(data, "v2_schema_summary")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    table = ax.table(
        cellText=data[["column", "dtype", "missing", "unique_values"]].values,
        colLabels=["Column", "dtype", "Missing rows", "Observed values/range"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.28, 0.12, 0.15, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title("Schema and observed value sets (binary flags should be 1/2)")
    _save_figure(fig, "v2_schema_summary.png")
    return data


def plot_duplicate_multiplicity(df):
    features = _feature_columns(df)
    groups = df.groupby(features, dropna=False).agg(
        multiplicity=("LUNG_CANCER", "size"),
        label_count=("LUNG_CANCER", "nunique"),
    )
    data = groups.reset_index(drop=True)
    _save(data, "v4_duplicate_multiplicity")
    conflicts = int(groups["label_count"].gt(1).sum())
    conflict_rows = int(groups.loc[groups["label_count"].gt(1), "multiplicity"].sum())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    max_multiplicity = int(data["multiplicity"].max())
    bins = np.arange(0.5, max_multiplicity + 1.5, 1)
    axes[0].hist(data["multiplicity"], bins=bins, color="#2A9D8F")
    axes[0].set(
        title="Multiplicity of feature patterns",
        xlabel="Rows per distinct feature pattern",
        ylabel="Number of patterns",
    )
    axes[1].bar(["Conflicting\npatterns", "Rows in\nconflicts"],
                [conflicts, conflict_rows], color=["#C8553D", "#E9A23B"])
    axes[1].set(
        title="Feature patterns observed with both recorded labels",
        ylabel="Count",
    )
    axes[1].text(0, conflicts, f"{conflicts}", ha="center", va="bottom")
    axes[1].text(1, conflict_rows, f"{conflict_rows}", ha="center", va="bottom")
    _save_figure(fig, "v4_duplicate_multiplicity.png")
    return data


def plot_yes_share_by_block(df):
    fp = int(df.index[df["LUNG_CANCER"].eq(1)].min())
    later = df.iloc[fp:]
    groups = {
        "Leading negatives": df.iloc[:fp],
        "Later negatives": later.loc[later["LUNG_CANCER"].eq(0)],
        "Later positives": later.loc[later["LUNG_CANCER"].eq(1)],
    }
    features = list(df.columns[3:-1])
    rows = []
    for group_name, group in groups.items():
        for column in features:
            rows.append(
                {
                    "group": group_name,
                    "feature": column,
                    "yes_share": float(group[column].eq(2).mean()),
                    "n": len(group),
                }
            )
    data = pd.DataFrame(rows)
    _save(data, "v6_yes_share_by_block")
    pivot = data.pivot(index="feature", columns="group", values="yes_share")
    pivot = pivot.sort_values("Leading negatives")
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(pivot))
    offsets = [-0.22, 0, 0.22]
    colors = ["#C8553D", "#1F3A5F", "#2A9D8F"]
    for offset, group_name, color in zip(offsets, groups, colors):
        ax.scatter(pivot[group_name], y + offset, label=group_name, color=color, s=35)
    ax.set_yticks(y, pivot.index)
    ax.set(
        title="YES-share profiles for 13 binary flags across row groups",
        xlabel="Share coded YES (value 2)",
        ylabel="Symptom/lifestyle flag",
        xlim=(0, 1),
    )
    ax.legend()
    _save_figure(fig, "v6_yes_share_by_block.png")
    return data


def plot_age_profiles(df):
    fp = int(df.index[df["LUNG_CANCER"].eq(1)].min())
    later = df.iloc[fp:]
    groups = {
        "Leading negatives": df.iloc[:fp],
        "Later negatives": later.loc[later["LUNG_CANCER"].eq(0)],
        "Later positives": later.loc[later["LUNG_CANCER"].eq(1)],
    }
    rows = []
    for group_name, group in groups.items():
        for value in group["AGE"]:
            rows.append({"group": group_name, "AGE": int(value)})
    data = pd.DataFrame(rows)
    _save(data, "v7_age_profiles")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#C8553D", "#1F3A5F", "#2A9D8F"]
    bins = np.arange(df["AGE"].min() - 0.5, df["AGE"].max() + 1.5, 2)
    x = np.linspace(df["AGE"].min(), df["AGE"].max(), 250)
    for (label, group), color in zip(groups.items(), colors):
        values = group["AGE"].to_numpy()
        axes[0].hist(values, bins=bins, density=True, histtype="step",
                     linewidth=1.7, label=label, color=color)
        mean, sd = values.mean(), values.std(ddof=1)
        density = stats.norm.pdf(x, mean, sd)
        axes[0].plot(x, density, color=color, linestyle=":", linewidth=1.2)
    axes[0].set(title="AGE by row-order and recorded-label group",
                xlabel="AGE (recorded units)", ylabel="Density")
    axes[0].legend(fontsize=7)

    labels = [0, 1]
    axes[1].boxplot(
        [df.loc[df["LUNG_CANCER"].eq(label), "AGE"] for label in labels],
        tick_labels=["0", "1"],
        showfliers=False,
    )
    axes[1].set(title="AGE by recorded label",
                xlabel="Recorded label code", ylabel="AGE (recorded units)")

    age_band = pd.cut(
        df["AGE"],
        bins=[-np.inf, 49, 59, 69, 79, np.inf],
        labels=["<50", "50-59", "60-69", "70-79", "80+"],
    )
    band = pd.DataFrame({"AGE_BAND": age_band, "LUNG_CANCER": df["LUNG_CANCER"]})
    summary = band.groupby("AGE_BAND", observed=False)["LUNG_CANCER"].agg(
        n="size", positive_rate="mean"
    ).reset_index()
    _save(summary, "v7_age_band_positive_rate")
    axes[2].bar(summary["AGE_BAND"].astype(str), summary["positive_rate"],
                color="#E9A23B")
    axes[2].set(title="Recorded-label-1 rate by AGE_BAND",
                xlabel="AGE_BAND (recorded AGE units)", ylabel="Share with label 1",
                ylim=(0, 1))
    _save_figure(fig, "v7_age_profiles.png")
    return data, summary


def plot_pca_scree(df):
    features = df[_feature_columns(df)].to_numpy(dtype=float)
    standard_deviation = features.std(axis=0, ddof=0)
    if np.any(standard_deviation == 0):
        raise ValueError("PCA scree requires non-constant feature columns")
    standardized = (features - features.mean(axis=0)) / standard_deviation
    singular_values = np.linalg.svd(standardized, full_matrices=False,
                                   compute_uv=False)
    eigenvalues = singular_values**2 / (len(standardized) - 1)
    explained = eigenvalues / eigenvalues.sum()
    data = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": np.cumsum(explained),
        }
    )
    _save(data, "v8_pca_scree")
    flat_reference = 1 / len(explained)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["component"], data["explained_variance_ratio"], marker="o",
            color="#1F3A5F")
    ax.axhline(flat_reference, color="#C8553D", linestyle="--",
               label="Flat-spectrum reference")
    ax.set(
        title="Standardized-feature PCA scree is close to a flat reference",
        xlabel="Principal component",
        ylabel="Share of total variance",
        xticks=data["component"],
    )
    ax.legend()
    _save_figure(fig, "v8_pca_scree.png")
    return data


def plot_h1_permutation_null(view):
    smoker = view["SMOKING"].eq(2).to_numpy(dtype=np.int8)
    label = view["LUNG_CANCER"].to_numpy(dtype=np.int8)
    observed = stats.chi2_contingency(
        pd.crosstab(smoker, label), correction=False
    )[0]
    rng = np.random.default_rng(42)
    permutations = 1000
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = stats.chi2_contingency(
            pd.crosstab(smoker, rng.permutation(label)), correction=False
        )[0]
    data = pd.DataFrame({"permutation": np.arange(permutations), "chi_square": null})
    _save(data, "s3_h1_permutation_null")
    empirical_p = (int(np.count_nonzero(null >= observed)) + 1) / (permutations + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=35, color="#8A94A6", alpha=0.85)
    ax.axvline(observed, color="#C8553D", linewidth=2,
               label="Observed chi-square")
    ax.set(
        title="H1 label-permutation null distribution (view A)",
        xlabel="Pearson chi-square statistic",
        ylabel="Permutation count",
    )
    ax.legend(title=f"Empirical p ≈ {empirical_p:.4g}")
    _save_figure(fig, "s3_h1_permutation_null.png")
    return data


def plot_h4_p_q(hypothesis_table):
    data = hypothesis_table.loc[hypothesis_table["hypothesis"].eq("H4")].copy()
    data = data.loc[data["view"].isin(["A_raw", "D_dedup_nolead"])]
    _save(data, "s4_h4_p_q")
    fig, ax = plt.subplots(figsize=(8, 6))
    for view, marker, color in [
        ("A_raw", "o", "#C8553D"),
        ("D_dedup_nolead", "s", "#2A9D8F"),
    ]:
        selected = data.loc[data["view"].eq(view)]
        ax.scatter(selected["p"], selected["q_BH"], marker=marker,
                   color=color, label=view)
    ax.plot([1e-110, 1], [1e-110, 1], color="#8A94A6", linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        title="H4 nominal p-values and BH q-values in views A and D",
        xlabel="Nominal p-value (log scale)",
        ylabel="BH-adjusted q-value (log scale)",
    )
    ax.legend()
    _save_figure(fig, "s4_h4_p_q.png")
    return data


def plot_h4_odds_ratio_forest(view):
    features = _feature_columns(view)
    rows = []
    for column in features:
        yes = view[column].eq(_yes_value(column)).to_numpy(dtype=np.int8)
        ratio, low, high = odds_ratio_ci(yes, view["LUNG_CANCER"].to_numpy())
        rows.append({"feature": column, "odds_ratio": ratio,
                     "ci_low": low, "ci_high": high})
    data = pd.DataFrame(rows).sort_values("odds_ratio")
    _save(data, "s5_view_d_feature_or")
    y = np.arange(len(data))
    errors = np.vstack(
        [data["odds_ratio"] - data["ci_low"], data["ci_high"] - data["odds_ratio"]]
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(data["odds_ratio"], y, xerr=errors, fmt="o",
                color="#1F3A5F", capsize=3)
    ax.axvline(1, color="#C8553D", linestyle="--", label="OR = 1")
    ax.set_xscale("log")
    ax.set_yticks(y, data["feature"])
    ax.set(title="Feature odds ratios in view D",
            xlabel="Odds ratio for recorded label (log scale; 95% Wald CI)",
            ylabel="Feature")
    ax.legend()
    _save_figure(fig, "s5_view_d_feature_or.png")
    return data


def plot_power_curve(view, h1_row):
    p = float(view["LUNG_CANCER"].mean())
    n_yes = int(view["SMOKING"].eq(2).sum())
    n_no = int(view["SMOKING"].eq(1).sum())
    scale = np.linspace(0.25, 2.5, 60)
    mde = np.array(
        [
            min_detectable_difference(p, max(1, int(n_yes * factor)),
                                      max(1, int(n_no * factor)))
            for factor in scale
        ]
    )
    data = pd.DataFrame({"sample_size_multiplier": scale,
                         "minimum_detectable_risk_difference": mde})
    _save(data, "s6_power_curve")
    low, high = h1_row["risk_difference_ci_low"], h1_row["risk_difference_ci_high"]
    observed = h1_row["risk_difference"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scale, mde, color="#1F3A5F", label="Approximate minimum detectable difference")
    ax.axhspan(low, high, color="#E9A23B", alpha=0.25,
               label="Observed H1 risk-difference 95% CI")
    ax.axhline(observed, color="#C8553D", linestyle="--",
               label="Observed H1 risk difference")
    ax.axvline(1, color="#8A94A6", linestyle=":")
    ax.set(
        title="Approximate H1 power sensitivity and observed uncertainty",
        xlabel="Per-group sample-size multiplier relative to view D",
        ylabel="Absolute risk difference (proportion)",
    )
    ax.legend(fontsize=8)
    _save_figure(fig, "s6_power_curve.png")
    return data


def _wilson_interval(successes, total):
    z = stats.norm.ppf(0.975)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def plot_respiratory_count(view):
    respiratory = ["COUGHING", "WHEEZING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]
    count = view[respiratory].eq(2).sum(axis=1)
    data = pd.DataFrame({"RESP_SYMPTOM_COUNT": count,
                         "LUNG_CANCER": view["LUNG_CANCER"]})
    summary = data.groupby("RESP_SYMPTOM_COUNT").agg(
        n=("LUNG_CANCER", "size"), positives=("LUNG_CANCER", "sum")
    ).reset_index()
    summary["positive_rate"] = summary["positives"] / summary["n"]
    intervals = [
        _wilson_interval(int(row.positives), int(row.n))
        for row in summary.itertuples(index=False)
    ]
    summary["ci_low"] = [interval[0] for interval in intervals]
    summary["ci_high"] = [interval[1] for interval in intervals]
    _save(summary, "s7_respiratory_count")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(summary["RESP_SYMPTOM_COUNT"], summary["n"], color="#1F3A5F")
    axes[0].set(title="Respiratory-count distribution",
                xlabel="YES symptoms among four (count)", ylabel="Rows")
    errors = np.vstack(
        [summary["positive_rate"] - summary["ci_low"],
         summary["ci_high"] - summary["positive_rate"]]
    )
    axes[1].errorbar(summary["RESP_SYMPTOM_COUNT"], summary["positive_rate"],
                     yerr=errors, fmt="o-", capsize=4, color="#2A9D8F")
    axes[1].set(title="Recorded-label-1 rate by count (Wilson 95% CI)",
                xlabel="YES symptoms among four (count)",
                ylabel="Share with recorded label 1", ylim=(0, 1))
    _save_figure(fig, "s7_respiratory_count.png")
    return summary


def plot_age_scaling_and_feature_sets(view):
    scaler = fit_age_scaler(view["AGE"])
    transformed = add_features(view, scaler)
    rows = []
    for set_name, columns in FEATURE_SETS.items():
        rows.append({"feature_set": set_name, "predictor_count": len(columns)})
    counts = pd.DataFrame(rows)
    _save(counts, "s8_feature_sets")
    age_data = pd.DataFrame(
        {
            "AGE": view["AGE"].to_numpy(),
            "AGE_Z": transformed["AGE_Z"].to_numpy(),
        }
    )
    _save(age_data.describe().transpose().reset_index(names="variable"),
          "s8_age_scaling_summary")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(age_data["AGE"], bins=30, density=True, color="#1F3A5F")
    axes[0].set(title="AGE before scaling",
                xlabel="AGE (recorded units)", ylabel="Density")
    axes[1].hist(age_data["AGE_Z"], bins=30, density=True, color="#2A9D8F")
    axes[1].set(title="AGE_Z after descriptive scaling",
                xlabel="AGE_Z (standard deviations)", ylabel="Density")
    axes[2].barh(counts["feature_set"], counts["predictor_count"],
                 color=["#1F3A5F", "#2A9D8F", "#E9A23B"])
    axes[2].set(title="Candidate feature-set sizes (not model results)",
                xlabel="Predictor columns (count)", ylabel="Feature set")
    for bar, value in zip(axes[2].patches, counts["predictor_count"]):
        axes[2].text(value + 0.2, bar.get_y() + bar.get_height() / 2,
                     str(value), va="center")
    _save_figure(fig, "s8_age_scaling_feature_sets.png")
    return age_data, counts


def build_guide_figures(df, views, hypothesis_table):
    """Generate all requested non-model visualization artifacts."""
    feature_columns = ["GENDER"] + [
        column
        for column in df.columns
        if column not in {"patient_id", "GENDER", "AGE", "LUNG_CANCER"}
    ]
    captions = {}
    figures = {
        "fig1_order": pd.read_csv(TAB / "f1_order_rolling.csv"),
        "fig5_views": plot_views(views),
        "fig2_source_shift": plot_source_shift(df),
        "fig4_age": plot_age_distribution(views["D_dedup_nolead"]),
        "fig3_correlation": plot_feature_correlation(views["D_dedup_nolead"]),
        "fig6_cramers_v": plot_cramers_v(
            views["A_raw"], views["D_dedup_nolead"], feature_columns
        ),
        "fig7_smoking_forest": plot_smoking_forest(views),
        "v1_class_balance": plot_class_balance(df),
        "v2_schema_summary": plot_schema_summary(df),
        "v4_duplicate_multiplicity": plot_duplicate_multiplicity(df),
        "v6_yes_share_by_block": plot_yes_share_by_block(df),
        "v7_age_profiles": plot_age_profiles(df),
        "v8_pca_scree": plot_pca_scree(df),
        "s3_h1_permutation_null": plot_h1_permutation_null(views["A_raw"]),
        "s4_h4_p_q": plot_h4_p_q(hypothesis_table),
        "s5_view_d_feature_or": plot_h4_odds_ratio_forest(views["D_dedup_nolead"]),
        "s6_power_curve": plot_power_curve(
            views["D_dedup_nolead"],
            hypothesis_table.loc[
                hypothesis_table["hypothesis"].eq("H1")
                & hypothesis_table["view"].eq("D_dedup_nolead")
            ].iloc[0],
        ),
        "s7_respiratory_count": plot_respiratory_count(views["B_dedup"]),
        "s8_age_scaling_feature_sets": plot_age_scaling_and_feature_sets(
            views["B_dedup"]
        ),
    }
    metadata = {
        "f1_order.png": (
            "Does row order show a leading block with a different recorded-label and feature composition?",
            "Rolling label and feature shares vary across row index; this file's leading rows are not exchangeable with later rows.",
        ),
        "fig5_views.png": (
            "How much data remains under each cleaning policy?",
            "Row count and recorded-label-1 share vary across the four declared views.",
        ),
        "fig2_source_shift.png": (
            "Do leading-block negatives differ from later negatives?",
            "Feature distributions vary between the row groups; provenance remains unknown.",
        ),
        "fig4_age.png": (
            "What is the AGE distribution in the later deduplicated view?",
            "The observed AGE distribution is compared with a normal overlay parameterized by its sample mean and standard deviation.",
        ),
        "fig3_correlation.png": (
            "Are later-view features correlated with each other?",
            "The clipped heatmap displays weak pairwise linear relationships without establishing independence.",
        ),
        "fig6_cramers_v.png": (
            "Do feature-label effect sizes persist after removing the leading block?",
            "Cramer's V values generally attenuate between raw view A and cleaned view D.",
        ),
        "fig7_smoking_forest.png": (
            "How sensitive is the smoking odds ratio to cleaning policy?",
            "The estimated smoking odds ratio shifts toward one in the cleaned views.",
        ),
        "v1_class_balance.png": (
            "How imbalanced is the recorded label, and what is the always-positive reference?",
            "The dashed reference marks the recorded-label-1 share achieved by an always-positive rule.",
        ),
        "v2_schema_summary.png": (
            "What types, missingness, and observed values appear in each input column?",
            "The schema table shows zero missingness and the observed value sets, including the binary 1/2 codes.",
        ),
        "v4_duplicate_multiplicity.png": (
            "How often are feature patterns repeated, and how many have conflicting labels?",
            "The multiplicity distribution is long-tailed, and some repeated feature patterns occur with both recorded labels.",
        ),
        "v6_yes_share_by_block.png": (
            "Do the 13 symptom/lifestyle YES shares differ across leading negatives, later negatives, and later positives?",
            "Shares coded YES (2) vary across row groups; the comparison is descriptive and source provenance remains unknown.",
        ),
        "v7_age_profiles.png": (
            "How does recorded AGE vary across row groups and labels?",
            "Group histograms, label-wise boxplots, and AGE_BAND label shares describe distinct views of the recorded ages.",
        ),
        "v8_pca_scree.png": (
            "Does the standardized binary-feature PCA spectrum depart from a flat reference?",
            "The scree curve is compared with equal variance per component as a descriptive structure check.",
        ),
        "s3_h1_permutation_null.png": (
            "Is the observed H1 statistic unusual under label permutation in raw view A?",
            "The observed chi-square is marked against a fixed-margin permutation reference distribution.",
        ),
        "s4_h4_p_q.png": (
            "How do nominal and BH-adjusted H4 p-values compare in views A and D?",
            "The plot contrasts nominal evidence with the feature-family adjustment in both views.",
        ),
        "s5_view_d_feature_or.png": (
            "What are the feature-level odds ratios and 95% intervals in view D?",
            "All 14 binary-feature estimates are shown with Wald intervals and the OR=1 reference.",
        ),
        "s6_power_curve.png": (
            "How does approximate H1 detectability change with sample size?",
            "The minimum detectable difference curve is compared with the observed risk-difference estimate and interval.",
        ),
        "s7_respiratory_count.png": (
            "What are the respiratory-count distribution and recorded-label-1 shares by count?",
            "Counts and Wilson intervals describe the label share at each four-symptom count.",
        ),
        "s8_age_scaling_feature_sets.png": (
            "How does descriptive AGE scaling change the scale, and how large are F0/F1/F2?",
            "AGE_Z is standardized on view B for descriptive exports; feature-set sizes are shown without fitting a model.",
        ),
    }
    for filename, (question, takeaway) in metadata.items():
        _record_figure(captions, filename, question, takeaway)
    _save_captions(captions)
    return figures
