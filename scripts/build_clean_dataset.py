"""
Build a clean lung-cancer prediction dataset from the US National Health Interview
Survey (NHIS) adult files, 2019-2023.

Input : data/raw/nhis/adult19csv.zip ... adult23csv.zip  (official CDC files, unchanged)
Output: data/clean/nhis_lung_cancer_clean.csv               readable, one row per adult
        data/clean/nhis_lung_cancer_model_ready.csv         final features, numbers only, no missing values, train/test split
        data/clean/nhis_lung_cancer_model_ready_scaled.csv  same, numeric columns standardised (fitted on training rows)
        data/clean/final_features.csv                       every candidate feature: keep or drop, and why
        data/clean/outlier_check.csv                        IQR outlier check and the decision for each column
        data/clean/before_after_summary.csv                 one table: what each step changed
        data/clean/data_dictionary.csv                      what every column means
        data/clean/cleaning_report.txt                      every cleaning step with before/after counts

Run from the repository root (after download_data.py and build_regional_data.py):
    python scripts/build_clean_dataset.py
"""
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import CLEAN, RANDOM_STATE, RAW, REFERENCE, REGIONAL, TAB, YEARS, save_fig, text_log

COMORB = ["copd", "asthma", "hypertension", "heart_disease", "diabetes"]

# Raw NHIS columns we need (same names in all 2019-2023 adult files)
RAW_COLS = ["SRVY_YR", "HHX", "WTFA_A", "AGEP_A", "SEX_A", "HISPALLP_A", "REGION", "EDUC_A",
            "RATCAT_A", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A", "ECIGEV_A", "COPDEV_A", "ASEV_A",
            "HYPEV_A", "CHDEV_A", "DIBEV_A", "CANEV_A", "LUNGCAN_A"]

report, log, _ = text_log()

summary = []        # before/after table: one row per step
def track(step, df, change):
    summary.append({"step": step, "rows": len(df), "columns": df.shape[1],
                    "missing_cells": int(df.isna().sum().sum()), "what_changed": change})


def vif_table(X):
    """Variance inflation factor of every column of X, with an intercept (so the VIFs are centred).
    An exact linear dependency has an infinite VIF; floating point returns a huge number instead,
    so values above 1e4 are stored as inf and flagged (here the exact ones are 4e4 or more, the largest
    non-exact one is about 640)."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    # the 'before' matrix is rank-deficient on purpose (that is what is being measured), so statsmodels'
    # rank-deficiency warnings are expected here and silenced
    with np.errstate(divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.array([variance_inflation_factor(A, i + 1) for i in range(X.shape[1])])
    exact = v > 1e4
    t = pd.DataFrame({"feature": X.columns, "vif": np.where(exact, np.inf, v), "exact_dependency": exact})
    return t.sort_values("vif", ascending=False, kind="stable").reset_index(drop=True)


def near_zero_eigenvalues(X, limit=0.01):
    """Number of eigenvalues of the correlation matrix below `limit` (= near-exact linear dependencies)."""
    return int((np.linalg.eigvalsh(np.corrcoef(X.to_numpy(float), rowvar=False)) < limit).sum())


# ---------------------------------------------------------------- 1. load and stack
def load_year(year):
    path = RAW / "nhis" / f"adult{str(year)[-2:]}csv.zip"
    with zipfile.ZipFile(path) as z:
        member = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(member) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.upper()
    df = df.rename(columns={"EDUCP_A": "EDUC_A"})   # renamed EDUCP_A from 2021 (same codes 1-10)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        log(f"  WARNING {year}: columns not found, filled as missing: {missing}")
        for c in missing:
            df[c] = np.nan
    df = df[RAW_COLS].copy()
    if df["SRVY_YR"].isna().all():
        df["SRVY_YR"] = year
    return df

def load_and_stack():
    log("STEP 1 - Load each year and stack the rows")
    frames = []
    for y in YEARS:
        d = load_year(y)
        log(f"  {y}: {len(d):,} adults")
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    log(f"  Stacked: {len(raw):,} rows x {raw.shape[1]} columns")
    track("1. Load + stack 5 years", raw, "5 yearly files stacked; 20 needed columns kept (LUNGCAN_A is blank by design for people who never had cancer)")
    return raw


# ---------------------------------------------------------------- 2. duplicates
def remove_duplicates(raw):
    log("\nSTEP 2 - Duplicates")
    raw["person_id"] = raw["SRVY_YR"].astype(int).astype(str) + "_" + raw["HHX"].astype(str)
    n_dup_id = raw["person_id"].duplicated().sum()
    log(f"  Duplicate person IDs (year + household): {n_dup_id}")
    raw = raw.drop_duplicates(subset="person_id").reset_index(drop=True)
    log(f"  Rows after removing duplicate IDs: {len(raw):,}")
    track("2. Remove duplicates", raw, f"{n_dup_id} duplicate person IDs removed")
    return raw


# ---------------------------------------------------------------- 3. target
def build_target(raw):
    log("\nSTEP 3 - Build the target: lung_cancer (1 = ever told they had lung cancer)")
    # LUNGCAN_A is only asked to people who ever had cancer (CANEV_A = 1).
    # Codes: 1 = yes, 2 = no, 7/8/9 = refused / not ascertained / don't know.
    can, lung = raw["CANEV_A"], raw["LUNGCAN_A"]
    target = pd.Series(np.nan, index=raw.index)
    target[can == 2] = 0                       # never had any cancer
    target[(can == 1) & (lung == 2)] = 0       # had cancer, but not lung cancer
    target[(can == 1) & (lung == 1)] = 1       # had lung cancer
    raw["lung_cancer"] = target
    n_unknown = target.isna().sum()
    log(f"  Unknown target (refused / don't know): {n_unknown:,} rows -> removed (cannot guess the answer)")
    raw = raw[target.notna()].copy()
    raw["lung_cancer"] = raw["lung_cancer"].astype(int)
    log(f"  Rows with a known target: {len(raw):,}")
    log(f"  Lung cancer cases: {raw.lung_cancer.sum():,} ({100 * raw.lung_cancer.mean():.2f}%)")
    by_year = raw.groupby("SRVY_YR")["lung_cancer"].agg(["sum", "size"])
    log("  By year (cases / adults): " + ", ".join(f"{int(y)}: {int(r['sum'])}/{int(r['size']):,}" for y, r in by_year.iterrows()))
    track("3. Build target", raw, f"{n_unknown:,} rows with unknown answer removed")
    return raw


# ---------------------------------------------------------------- 4. recode to readable values
def recode(raw):
    log("\nSTEP 4 - Recode NHIS codes into clear values (7/8/9, 97/98/99 = unknown -> missing)")
    def yes_no(s):
        return s.map({1: 1, 2: 0})          # anything else (7, 8, 9) becomes missing

    clean = pd.DataFrame({
        "person_id": raw["person_id"],
        "year": raw["SRVY_YR"].astype(int),
        "survey_weight": raw["WTFA_A"],
        "age": raw["AGEP_A"].where(raw["AGEP_A"].between(18, 85)),     # 85 means 85 or older
        "sex": raw["SEX_A"].map({1: "Male", 2: "Female"}),
        "race_ethnicity": raw["HISPALLP_A"].map({1: "Hispanic", 2: "White", 3: "Black", 4: "Asian",
                                                 5: "AIAN", 6: "AIAN and other", 7: "Other or multiple"}),
        "region": raw["REGION"].map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}),
        "education": raw["EDUC_A"].map({0: "Less than high school", 1: "Less than high school",
                                        2: "Less than high school", 3: "High school / GED", 4: "High school / GED",
                                        5: "Some college", 6: "Some college", 7: "Some college",
                                        8: "Bachelor", 9: "Graduate", 10: "Graduate",
                                        11: "Graduate"}),   # 11 = doctoral degree (2019-2020 coding only)
        "income_group": pd.cut(raw["RATCAT_A"].where(raw["RATCAT_A"].between(1, 14)), [0, 3, 7, 11, 14],
                               labels=["Below poverty", "Low", "Middle", "High"]).astype(object),
        "general_health": raw["PHSTAT_A"].map({1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}),
        "bmi_category": raw["BMICAT_A"].map({1: "Underweight", 2: "Healthy", 3: "Overweight", 4: "Obese"}),
        "smoking_status": raw["SMKCIGST_A"].map({1: "Current", 2: "Current", 5: "Current", 3: "Former", 4: "Never"}),
        "ever_ecigarette": yes_no(raw["ECIGEV_A"]),
        "copd": yes_no(raw["COPDEV_A"]),
        "asthma": yes_no(raw["ASEV_A"]),
        "hypertension": yes_no(raw["HYPEV_A"]),
        "heart_disease": yes_no(raw["CHDEV_A"]),
        "diabetes": yes_no(raw["DIBEV_A"]),
        "lung_cancer": raw["lung_cancer"],
    })

    n_codes = int(raw[["AGEP_A", "SEX_A", "SMKCIGST_A", "COPDEV_A"]].isin([7, 8, 9, 97, 98, 99]).sum().sum())
    track("4. Recode codes -> values", clean, f"number codes turned into words; e.g. {n_codes:,} 'unknown' codes in 4 key columns -> missing")
    return clean


# ---------------------------------------------------------------- 5. new columns
def add_new_columns(clean):
    log("\nSTEP 5 - New columns (feature engineering)")
    clean["age_group"] = pd.cut(clean["age"], [17, 44, 54, 64, 74, 85],
                                labels=["18-44", "45-54", "55-64", "65-74", "75+"]).astype(object)
    clean["ever_smoker"] = clean["smoking_status"].map({"Current": 1, "Former": 1, "Never": 0})
    # Rough proxy of the US lung-cancer screening group (age 50-80 and ever smoked).
    # Pack-years are not in NHIS, so this is only an approximation.
    clean["screening_age_smoker"] = np.where(clean["age"].isna() | clean["ever_smoker"].isna(), np.nan,
                                             ((clean["age"].between(50, 80)) & (clean["ever_smoker"] == 1)).astype(float))
    clean["comorbidity_count"] = clean[COMORB].sum(axis=1, min_count=1)
    log("  Added: age_group, ever_smoker, screening_age_smoker, comorbidity_count")
    track("5. New columns", clean, "added age_group, ever_smoker, screening_age_smoker, comorbidity_count")
    return clean


# ---------------------------------------------------------------- 5b. join regional data
def join_regional(clean):
    log("\nSTEP 5b - Join the regional table (EPA air pollution, BRFSS smoking rate, US lung cancer rate)")
    ctx_path = CLEAN / "regional_context.csv"
    if not ctx_path.exists():
        raise SystemExit("Run scripts/build_regional_data.py first (it creates data/clean/regional_context.csv)")
    ctx = pd.read_csv(ctx_path)
    n_before = len(clean)
    clean = clean.merge(ctx, on=["region", "year"], how="left", validate="many_to_one")
    assert len(clean) == n_before, "The join must add columns, never rows"
    log(f"  Rows before / after join: {n_before:,} / {len(clean):,}  (join key: region + year)")
    for c in REGIONAL:
        log(f"  {c}: {clean[c].isna().sum()} rows without a match (only if region is unknown)")
    track("5b. Join regional table", clean, "3 columns added (pm25_mean, smoking_rate, lung_cancer_incidence); 0 rows added")
    return clean


# ---------------------------------------------------------------- 5c. outliers
def outlier_check(clean):
    log("\nSTEP 5c - Outlier check (IQR rule: below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR)")
    OUTLIER_DECISION = {
        "age": "KEEP - every age 18-85 is a real adult; 85 means '85 or older' (NHIS top-code), not an error",
        "comorbidity_count": "KEEP - a count from 0 to 5; high values are real people with several diseases",
        "pm25_mean": "KEEP - official EPA measurements; only 20 region-year values, none impossible",
        "smoking_rate": "KEEP - official CDC percentages between 0 and 100",
        "lung_cancer_incidence": "KEEP - official registry rates; differences between regions are real",
        "survey_weight": "KEEP - NHIS design weight, used for population estimates only, not as a feature",
    }
    rows = []
    for c, decision in OUTLIER_DECISION.items():
        x = clean[c].dropna()
        q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        rows.append({"column": c, "min": x.min(), "q1": q1, "median": x.median(), "q3": q3, "max": x.max(),
                     "lower_bound": lo, "upper_bound": hi, "n_outliers": n_out,
                     "pct_outliers": round(100 * n_out / len(x), 2), "decision": decision})
        log(f"  {c:<22} bounds [{lo:.1f}, {hi:.1f}]  outliers: {n_out:,} ({100 * n_out / len(x):.2f}%)  -> {decision}")
    pd.DataFrame(rows).round(3).to_csv(CLEAN / "outlier_check.csv", index=False)

    UNITS = {"age": "years", "comorbidity_count": "conditions (0-5)", "pm25_mean": "PM2.5 (µg/m³)",
             "smoking_rate": "% of adults", "lung_cancer_incidence": "cases per 100,000", "survey_weight": "weight"}
    # matplotlib defaults inside a context, so the figure is the same whether or not another module
    # (e.g. seaborn in the notebook) has changed the global style
    with plt.style.context("default"), plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        for ax, c in zip(axes.ravel(), OUTLIER_DECISION):
            ax.boxplot(clean[c].dropna(), widths=0.6, flierprops={"markersize": 2, "alpha": 0.3})
            n_out = int(pd.DataFrame(rows).set_index("column").loc[c, "n_outliers"])
            ax.set_title(f"{c}\n{n_out:,} beyond 1.5 x IQR", fontsize=13); ax.set_xticks([]); ax.set_ylabel(UNITS[c], fontsize=12)
        fig.suptitle("Fig 3. Outlier check (IQR rule): every flagged value is plausible, so all values are kept", y=1.01, fontsize=14)
        save_fig("fig03_outlier_boxplots")
    log("  Box plots saved to results/figures/fig03_outlier_boxplots.png")
    track("5c. Outlier check", clean, "IQR check on 6 numeric columns; all values plausible -> kept")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. checks
def checks_and_save(clean):
    log("\nSTEP 6 - Checks")
    log(f"  Final rows: {len(clean):,} | columns: {clean.shape[1]}")
    log(f"  Duplicate person IDs: {clean['person_id'].duplicated().sum()}")
    log(f"  Age range: {clean['age'].min()} - {clean['age'].max()}")
    miss = (clean.isna().mean() * 100).round(2)
    log("  Missing values (%) per column:")
    for c, v in miss[miss > 0].sort_values(ascending=False).items():
        log(f"    {c:<22} {v}%")
    log("\n  Lung cancer rate by smoking status (%):")
    log(str((clean.groupby("smoking_status")["lung_cancer"].mean() * 100).round(2).to_dict()))
    log("  Lung cancer rate by COPD (%):")
    log(str((clean.groupby("copd")["lung_cancer"].mean() * 100).round(2).to_dict()))

    clean.to_csv(CLEAN / "nhis_lung_cancer_clean.csv", index=False)
    track("6. Checks + save clean file", clean, "0 duplicate IDs, ages 18-85, missing values listed")


# ---------------------------------------------------------------- 7. model-ready version
def encode_and_fill(clean):
    log("\nSTEP 7 - Model-ready version (numbers only, no missing values, fixed train/test split)")
    train_idx, test_idx = train_test_split(clean.index, test_size=0.20, stratify=clean["lung_cancer"],
                                           random_state=RANDOM_STATE)
    split = pd.Series("train", index=clean.index); split[test_idx] = "test"

    m = pd.DataFrame(index=clean.index)
    m["person_id"], m["split"], m["survey_weight"] = clean["person_id"], split, clean["survey_weight"]

    # Age: missing filled with the TRAINING median only (no leakage) + a flag
    age_median = clean.loc[train_idx, "age"].median()
    m["age"] = clean["age"].fillna(age_median)
    m["age_missing"] = clean["age"].isna().astype(int)
    log(f"  Age median used for filling (from training rows only): {age_median}")

    # Yes/no columns: missing -> 0 plus a '_unknown' flag
    for c in ["ever_smoker", "screening_age_smoker", "ever_ecigarette"] + COMORB:
        m[c] = clean[c].fillna(0).astype(int)
        m[c + "_unknown"] = clean[c].isna().astype(int)
    m["comorbidity_count"] = clean["comorbidity_count"].fillna(0).astype(int)

    # Regional columns: filled with the training median if a person's region is unknown
    for c in REGIONAL:
        m[c] = clean[c].fillna(clean.loc[train_idx, c].median())

    # Categorical columns: one-hot encoding, missing becomes its own 'Unknown' category
    cats = ["sex", "race_ethnicity", "region", "education", "income_group", "general_health",
            "bmi_category", "smoking_status", "age_group"]
    dummies = pd.get_dummies(clean[cats].fillna("Unknown"), prefix=cats, dtype=int)
    m = pd.concat([m, dummies], axis=1)
    m["lung_cancer"] = clean["lung_cancer"]

    assert m.drop(columns=["survey_weight"]).isna().sum().sum() == 0
    log(f"  All candidate features: {m.shape[1] - 4} | train {(split == 'train').sum():,} | test {(split == 'test').sum():,}")
    log(f"  Lung cancer rate - train: {100 * m.loc[split == 'train', 'lung_cancer'].mean():.2f}% | "
        f"test: {100 * m.loc[split == 'test', 'lung_cancer'].mean():.2f}%")
    track("7. Encode + fill gaps (train only)", m, "one-hot encoding, training-median filling, 'unknown' flags -> 0 missing")
    return m, split


# ---------------------------------------------------------------- 8. feature selection
def select_features(m, split):
    log("\nSTEP 8 - Final feature selection (decided on TRAINING rows only)")
    NOT_FEATURES = {"person_id": "ID only, carries no information",
                    "split": "train/test label, not a feature",
                    "survey_weight": "kept as a sampling weight for analysis, not used as a feature",
                    "lung_cancer": "TARGET"}
    candidates = [c for c in m.columns if c not in NOT_FEATURES]
    tr = m.loc[split == "train", candidates]
    decisions = {}

    # Rule 1 - redundant by design (the same information is already in another column)
    REDUNDANT = {"ever_smoker": "same information as the smoking_status columns (Current/Former/Never)",
                 "ever_smoker_unknown": "same as smoking_status_Unknown"}
    for c in [c for c in candidates if c.startswith("age_group_")]:
        REDUNDANT[c] = "age bands duplicate the exact age column (kept: age)"
    for c, why in REDUNDANT.items():
        if c in candidates:
            decisions[c] = ("DROP", "redundant: " + why)

    # Rule 2 - constant in the training rows (no information)
    for c in candidates:
        if c not in decisions and tr[c].nunique() <= 1:
            decisions[c] = ("DROP", "constant in training rows: carries no information")

    # Rule 2b - near-constant (course filter pass): more than 99.5% of training rows share one value.
    # Such a column describes fewer than 1 adult in 200 (e.g. 'COPD answer unknown' = 0.06%), so it
    # adds almost no information and only increases the number of dimensions.
    NEAR_CONSTANT = 0.995
    for c in candidates:
        if c not in decisions:
            top_share = tr[c].value_counts(normalize=True).iloc[0]
            if top_share > NEAR_CONSTANT:
                decisions[c] = ("DROP", f"near-constant: {100 * top_share:.2f}% of training rows have the same value "
                                        f"(limit {100 * NEAR_CONSTANT:.1f}%)")

    # Rule 3 - near-duplicates: |correlation| > 0.95 with a column already kept
    remaining = [c for c in candidates if c not in decisions]
    corr = tr[remaining].corr().abs()
    for i, a in enumerate(remaining):
        if a in decisions:
            continue
        for b in remaining[i + 1:]:
            if b not in decisions and corr.loc[a, b] > 0.95:
                decisions[b] = ("DROP", f"near-duplicate of {a} (|r| = {corr.loc[a, b]:.2f} on training rows)")

    # Rule 4 - linear-model structure (training rows). Rules 1-3 only see pairs of columns; a column can also be
    # an exact or near-exact combination of SEVERAL others, which the variance inflation factor (VIF) detects.
    log("  Rule 4 - multi-column dependencies (VIF, training rows)")
    before = [c for c in candidates if c not in decisions]
    vif_before = vif_table(tr[before])
    vif_before.round(2).to_csv(TAB / "vif_before.csv", index=False)
    n_exact = int(vif_before["exact_dependency"].sum())
    log(f"    VIF after rules 1-3 ({len(before)} features): {n_exact} infinite (exact dependencies), "
        f"{int((vif_before.vif > 10).sum())} above 10; near-zero eigenvalues: {near_zero_eigenvalues(tr[before])}")
    # 4a. comorbidity_count = copd + asthma + hypertension + heart_disease + diabetes (exactly)
    rho = tr["comorbidity_count"].corr(tr["hypertension"], method="spearman")
    decisions["comorbidity_count"] = ("DROP", "exact linear combination of the 5 condition flags; "
                                              f"Spearman {rho:.2f} with hypertension; infinite VIF "
                                              "(kept in the clean dataset and the EDA as a descriptive variable)")
    # 4b. one reference category per one-hot group
    NOTE = {"smoking_status": "; adults with unknown smoking status (smoking_status_Unknown, dropped by rule 3) also "
                              "fall into this reference and are identified only by screening_age_smoker_unknown",
            "general_health": "; the 0.04% with unknown general health (general_health_Unknown, dropped by rule 2b) "
                              "also fall into this reference"}
    for group, ref in REFERENCE.items():
        c = f"{group}_{ref}"
        why = f"reference category of {group} ({ref}): the other {group} dummies are read relative to it"
        if c in decisions:                       # sex_Male and region_West: already dropped by rule 3
            decisions[c] = ("DROP", decisions[c][1] + f"; it is also the {why}")
        else:
            decisions[c] = ("DROP", why + "; dropping one dummy per group removes the exact sum-to-1 dependency "
                                          "(infinite VIF)" + NOTE.get(group, ""))
    # 4c. region dummies vs the 3 regional columns (decision D3 of CHANGES_PLAN.md)
    mid = [c for c in candidates if c not in decisions]
    vif_mid = vif_table(tr[mid]).set_index("feature")["vif"]
    REGION = [c for c in mid if c.startswith("region_")]
    rv = vif_mid[REGION + ["lung_cancer_incidence"]]
    n_zero_mid = near_zero_eigenvalues(tr[mid])
    log(f"    After 4a + 4b ({len(mid)} features): VIF above 10 = " +
        ", ".join(f"{c} {v:.1f}" for c, v in vif_mid[vif_mid > 10].items()) + f"; near-zero eigenvalues: {n_zero_mid}")
    for c in REGION:
        decisions[c] = ("DROP", "the 3 regional columns already encode region x year; region dummies + "
                                f"lung_cancer_incidence were near-collinear (VIF {rv.min():.0f}-{rv.max():.0f}, "
                                f"{n_zero_mid} near-zero PCA eigenvalue)")

    KEEP_REASON = {
        "age": "risk rises strongly with age",
        "screening_age_smoker": "official high-risk group (age 50-80 and ever smoked)",
        "screening_age_smoker_unknown": "'unknown' flag: smoking status unknown (with ever_ecigarette_unknown it marks "
                                        "adults who skipped the smoking questions); the ONLY marker of unknown smoking "
                                        "status, because those adults fall into the Never reference",
        "ever_ecigarette_unknown": "'unknown' flag: e-cigarette use unknown (mostly the same adults as "
                                   "screening_age_smoker_unknown)",
        "copd": "lung disease, strongly linked to lung cancer", "pm25_mean": "regional air pollution (joined)",
        "smoking_rate": "regional smoking level (joined)", "lung_cancer_incidence": "regional baseline risk (joined)",
    }
    for c in candidates:
        if c not in decisions:
            reason = KEEP_REASON.get(c)
            if reason is None:
                group = c.rsplit("_", 1)[0]
                reason = ("'unknown' flag: tells the model the value was missing" if c.endswith(("_unknown", "_missing"))
                          else f"category of {group} (reference: {REFERENCE[group]})" if group in REFERENCE
                          else "health or lifestyle factor linked to lung cancer")
            decisions[c] = ("KEEP", reason)

    after = [c for c in candidates if decisions[c][0] == "KEEP"]
    vif_after = vif_table(tr[after])
    vif_after.round(2).to_csv(TAB / "vif_after.csv", index=False)
    for c, v in vif_after.set_index("feature")["vif"].items():
        if v > 5:
            decisions[c] = ("KEEP", decisions[c][1] + f"; VIF {v:.1f} (below 10)")
    log(f"    VIF of the {len(after)} final features: max {vif_after.vif.max():.2f} ({vif_after.feature[0]}), "
        f"{int((vif_after.vif > 10).sum())} above 10; near-zero eigenvalues: {near_zero_eigenvalues(tr[after])}")
    log("    Saved results/tables/vif_before.csv and vif_after.csv")

    ff = pd.DataFrame([(c, *decisions[c]) for c in candidates] + [(c, "NOT A FEATURE", r) for c, r in NOT_FEATURES.items()],
                      columns=["column", "decision", "reason"])
    ff.to_csv(CLEAN / "final_features.csv", index=False)
    final = [c for c in candidates if decisions[c][0] == "KEEP"]
    log(f"  Candidates: {len(candidates)} | kept: {len(final)} | dropped: {len(candidates) - len(final)}")
    for c, (d, r) in decisions.items():
        if d == "DROP":
            log(f"    DROP {c:<32} {r}")
    log("  Leakage check: 'ever had any cancer' (CANEV_A) was never a candidate - it reveals the answer.")
    m = m[["person_id", "split", "survey_weight"] + final + ["lung_cancer"]]
    m.to_csv(CLEAN / "nhis_lung_cancer_model_ready.csv", index=False)
    track("8. Feature selection", m, f"{len(candidates) - len(final)} redundant / near-constant / near-duplicate / "
                                     f"linearly dependent features dropped; {len(final)} kept (max VIF {vif_after.vif.max():.1f})")
    return m, final


# ---------------------------------------------------------------- 9. scaling
def scale(m, split, final):
    log("\nSTEP 9 - Scaling (StandardScaler fitted on TRAINING rows only)")
    SCALE = [c for c in ["age", "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence"] if c in final]
    scaler = StandardScaler().fit(m.loc[split == "train", SCALE])
    ms = m.copy()
    ms[SCALE] = scaler.transform(m[SCALE])
    pd.DataFrame({"column": SCALE, "train_mean": scaler.mean_, "train_std": scaler.scale_}).round(4) \
      .to_csv(CLEAN / "scaler_parameters.csv", index=False)
    chk = pd.DataFrame({"train_mean": ms.loc[split == "train", SCALE].mean(), "train_std": ms.loc[split == "train", SCALE].std(),
                        "test_mean": ms.loc[split == "test", SCALE].mean()}).round(3)
    log("  Why: these columns use different units (years, µg/m³, %, cases per 100k); scaling puts them on the same scale.")
    log("  Binary 0/1 columns are not scaled (they are already on a 0-1 scale).")
    log(chk.to_string())
    log("  Train means are 0 and SDs 1; test means are not exactly 0 -> the scaler never saw the test rows (no leakage).")
    ms.to_csv(CLEAN / "nhis_lung_cancer_model_ready_scaled.csv", index=False)
    track("9. Scaling (train only)", ms, f"{len(SCALE)} numeric columns standardised")
    return ms


# ---------------------------------------------------------------- 10. before/after table
def save_summary():
    log("\nSUMMARY - What each step changed")
    sm = pd.DataFrame(summary)
    sm.to_csv(CLEAN / "before_after_summary.csv", index=False)
    log(sm.to_string(index=False))
    return sm


# ---------------------------------------------------------------- 8. data dictionary
def write_dictionary():
    dictionary = pd.DataFrame([
        ("person_id", "Unique ID: survey year + NHIS household number", "NHIS HHX + SRVY_YR"),
        ("year", "Survey year (2019-2023)", "SRVY_YR"),
        ("survey_weight", "NHIS sampling weight (use for population estimates)", "WTFA_A"),
        ("age", "Age in years (85 = 85 or older)", "AGEP_A"),
        ("sex", "Male / Female", "SEX_A"),
        ("race_ethnicity", "Hispanic origin and race", "HISPALLP_A"),
        ("region", "US region", "REGION"),
        ("education", "Highest education level (grouped)", "EDUC_A (EDUCP_A from 2021)"),
        ("income_group", "Family income vs poverty line: below (<1), low (1-2), middle (2-4), high (4+)", "RATCAT_A"),
        ("general_health", "Self-rated general health", "PHSTAT_A"),
        ("bmi_category", "Body mass index category", "BMICAT_A"),
        ("smoking_status", "Cigarette smoking: Current / Former / Never", "SMKCIGST_A"),
        ("ever_ecigarette", "Ever used an e-cigarette (1 = yes)", "ECIGEV_A"),
        ("copd", "Ever told COPD, emphysema or chronic bronchitis (1 = yes)", "COPDEV_A"),
        ("asthma", "Ever had asthma (1 = yes)", "ASEV_A"),
        ("hypertension", "Ever told high blood pressure (1 = yes)", "HYPEV_A"),
        ("heart_disease", "Ever told coronary heart disease (1 = yes)", "CHDEV_A"),
        ("diabetes", "Ever told diabetes (1 = yes)", "DIBEV_A"),
        ("age_group", "NEW: age band", "from age"),
        ("ever_smoker", "NEW: current or former smoker (1 = yes)", "from smoking_status"),
        ("screening_age_smoker", "NEW: age 50-80 and ever smoked (proxy of screening eligibility)", "from age + smoking"),
        ("comorbidity_count", "NEW: number of the 5 conditions above", "from copd...diabetes"),
        ("pm25_mean", "JOINED: average fine-particle air pollution (PM2.5, µg/m³) in the person's region that year", "EPA AirData, by region + year"),
        ("smoking_rate", "JOINED: % of adults who currently smoke in the person's region that year", "CDC BRFSS, by region + year"),
        ("lung_cancer_incidence", "JOINED: new lung cancer cases per 100,000 people in the person's region that year", "CDC US Cancer Statistics, by region + year"),
        ("lung_cancer", "TARGET: ever told they had lung cancer (1 = yes)", "CANEV_A + LUNGCAN_A"),
    ], columns=["column", "meaning", "built_from"])
    dictionary.to_csv(CLEAN / "data_dictionary.csv", index=False)
    return dictionary


def main():
    report.clear()
    summary.clear()
    raw = build_target(remove_duplicates(load_and_stack()))
    clean = join_regional(add_new_columns(recode(raw)))
    outlier_check(clean)
    checks_and_save(clean)
    m, split = encode_and_fill(clean)
    m, final = select_features(m, split)
    scale(m, split, final)
    save_summary()
    write_dictionary()
    (CLEAN / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nDone. Files written to", CLEAN)


if __name__ == "__main__":
    main()
