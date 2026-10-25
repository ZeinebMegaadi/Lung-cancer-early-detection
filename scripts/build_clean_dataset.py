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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhis"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)
YEARS = [2019, 2020, 2021, 2022, 2023]
RANDOM_STATE = 42

# Raw NHIS columns we need (same names in all 2019-2023 adult files)
RAW_COLS = ["SRVY_YR", "HHX", "WTFA_A", "AGEP_A", "SEX_A", "HISPALLP_A", "REGION", "EDUC_A",
            "RATCAT_A", "PHSTAT_A", "BMICAT_A", "SMKCIGST_A", "ECIGEV_A", "COPDEV_A", "ASEV_A",
            "HYPEV_A", "CHDEV_A", "DIBEV_A", "CANEV_A", "LUNGCAN_A"]

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

report = []
def log(msg=""):
    print(msg)
    report.append(str(msg))

summary = []        # before/after table: one row per step
def track(step, df, change):
    summary.append({"step": step, "rows": len(df), "columns": df.shape[1],
                    "missing_cells": int(df.isna().sum().sum()), "what_changed": change})


# ---------------------------------------------------------------- 1. load and stack
def load_year(year):
    path = RAW / f"adult{str(year)[-2:]}csv.zip"
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

log("STEP 1 - Load each year and stack the rows")
frames = []
for y in YEARS:
    d = load_year(y)
    log(f"  {y}: {len(d):,} adults")
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
log(f"  Stacked: {len(raw):,} rows x {raw.shape[1]} columns")
track("1. Load + stack 5 years", raw, "5 yearly files stacked; 20 needed columns kept (LUNGCAN_A is blank by design for people who never had cancer)")

# ---------------------------------------------------------------- 2. duplicates
log("\nSTEP 2 - Duplicates")
raw["person_id"] = raw["SRVY_YR"].astype(int).astype(str) + "_" + raw["HHX"].astype(str)
n_dup_id = raw["person_id"].duplicated().sum()
log(f"  Duplicate person IDs (year + household): {n_dup_id}")
raw = raw.drop_duplicates(subset="person_id").reset_index(drop=True)
log(f"  Rows after removing duplicate IDs: {len(raw):,}")
track("2. Remove duplicates", raw, f"{n_dup_id} duplicate person IDs removed")

# ---------------------------------------------------------------- 3. target
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
log("  By year: " + str(raw.groupby("SRVY_YR")["lung_cancer"].agg(["sum", "size"]).to_dict("index")))
track("3. Build target", raw, f"{n_unknown:,} rows with unknown answer removed")

# ---------------------------------------------------------------- 4. recode to readable values
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
                                    8: "Bachelor", 9: "Graduate", 10: "Graduate"}),
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

# ---------------------------------------------------------------- 5. new columns
log("\nSTEP 5 - New columns (feature engineering)")
clean["age_group"] = pd.cut(clean["age"], [17, 44, 54, 64, 74, 85],
                            labels=["18-44", "45-54", "55-64", "65-74", "75+"]).astype(object)
clean["ever_smoker"] = clean["smoking_status"].map({"Current": 1, "Former": 1, "Never": 0})
# Rough proxy of the US lung-cancer screening group (age 50-80 and ever smoked).
# Pack-years are not in NHIS, so this is only an approximation.
clean["screening_age_smoker"] = np.where(clean["age"].isna() | clean["ever_smoker"].isna(), np.nan,
                                         ((clean["age"].between(50, 80)) & (clean["ever_smoker"] == 1)).astype(float))
comorb = ["copd", "asthma", "hypertension", "heart_disease", "diabetes"]
clean["comorbidity_count"] = clean[comorb].sum(axis=1, min_count=1)
log("  Added: age_group, ever_smoker, screening_age_smoker, comorbidity_count")
track("5. New columns", clean, "added age_group, ever_smoker, screening_age_smoker, comorbidity_count")

# ---------------------------------------------------------------- 5b. join regional data
log("\nSTEP 5b - Join the regional table (EPA air pollution, BRFSS smoking rate, US lung cancer rate)")
ctx_path = CLEAN / "regional_context.csv"
if not ctx_path.exists():
    raise SystemExit("Run scripts/build_regional_data.py first (it creates data/clean/regional_context.csv)")
ctx = pd.read_csv(ctx_path)
n_before = len(clean)
clean = clean.merge(ctx, on=["region", "year"], how="left", validate="many_to_one")
assert len(clean) == n_before, "The join must add columns, never rows"
log(f"  Rows before / after join: {n_before:,} / {len(clean):,}  (join key: region + year)")
for c in ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]:
    log(f"  {c}: {clean[c].isna().sum()} rows without a match (only if region is unknown)")
track("5b. Join regional table", clean, "3 columns added (pm25_mean, smoking_rate, lung_cancer_incidence); 0 rows added")

# ---------------------------------------------------------------- 5c. outliers
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

fig, axes = plt.subplots(1, len(OUTLIER_DECISION), figsize=(4 * len(OUTLIER_DECISION), 4.5))
for ax, c in zip(axes, OUTLIER_DECISION):
    ax.boxplot(clean[c].dropna(), widths=0.6, flierprops={"markersize": 2, "alpha": 0.3})
    ax.set_title(c, fontsize=11); ax.set_xticks([])
fig.suptitle("Outlier check: box plots of the numeric columns (all values kept - see outlier_check.csv)", y=1.02)
plt.tight_layout(); plt.savefig(FIG / "prep_outliers_boxplots.png", dpi=150, bbox_inches="tight"); plt.close()
log("  Box plots saved to reports/figures/prep_outliers_boxplots.png")
track("5c. Outlier check", clean, "IQR check on 6 numeric columns; all values plausible -> kept")

# ---------------------------------------------------------------- 6. checks
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
for c in ["ever_smoker", "screening_age_smoker", "ever_ecigarette"] + comorb:
    m[c] = clean[c].fillna(0).astype(int)
    m[c + "_unknown"] = clean[c].isna().astype(int)
m["comorbidity_count"] = clean["comorbidity_count"].fillna(0).astype(int)

# Regional columns: filled with the training median if a person's region is unknown
for c in ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]:
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

# ---------------------------------------------------------------- 8. feature selection
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

# Rule 3 - near-duplicates: |correlation| > 0.95 with a column already kept
remaining = [c for c in candidates if c not in decisions]
corr = tr[remaining].corr().abs()
for i, a in enumerate(remaining):
    if a in decisions:
        continue
    for b in remaining[i + 1:]:
        if b not in decisions and corr.loc[a, b] > 0.95:
            decisions[b] = ("DROP", f"near-duplicate of {a} (|r| = {corr.loc[a, b]:.2f} on training rows)")

KEEP_REASON = {
    "age": "risk rises strongly with age", "comorbidity_count": "summary of 5 conditions (overall health)",
    "screening_age_smoker": "official high-risk group (age 50-80 and ever smoked)",
    "copd": "lung disease, strongly linked to lung cancer", "pm25_mean": "regional air pollution (joined)",
    "smoking_rate": "regional smoking level (joined)", "lung_cancer_incidence": "regional baseline risk (joined)",
}
for c in candidates:
    if c not in decisions:
        reason = KEEP_REASON.get(c)
        if reason is None:
            reason = ("'unknown' flag: tells the model the value was missing" if c.endswith(("_unknown", "_missing"))
                      else "category of " + c.rsplit("_", 1)[0] if "_" in c and c.split("_")[0] in
                      {"sex", "race", "region", "education", "income", "general", "bmi", "smoking"}
                      else "health or lifestyle factor linked to lung cancer")
        decisions[c] = ("KEEP", reason)

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
track("8. Feature selection", m, f"{len(candidates) - len(final)} redundant/constant features dropped; {len(final)} kept")

# ---------------------------------------------------------------- 9. scaling
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

# ---------------------------------------------------------------- 10. before/after table
log("\nSUMMARY - What each step changed")
sm = pd.DataFrame(summary)
sm.to_csv(CLEAN / "before_after_summary.csv", index=False)
log(sm.to_string(index=False))

# ---------------------------------------------------------------- 8. data dictionary
dictionary = pd.DataFrame([
    ("person_id", "Unique ID: survey year + NHIS household number", "NHIS HHX + SRVY_YR"),
    ("year", "Survey year (2019-2023)", "SRVY_YR"),
    ("survey_weight", "NHIS sampling weight (use for population estimates)", "WTFA_A"),
    ("age", "Age in years (85 = 85 or older)", "AGEP_A"),
    ("sex", "Male / Female", "SEX_A"),
    ("race_ethnicity", "Hispanic origin and race", "HISPALLP_A"),
    ("region", "US region", "REGION"),
    ("education", "Highest education level (grouped)", "EDUC_A"),
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

(CLEAN / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nDone. Files written to", CLEAN)
