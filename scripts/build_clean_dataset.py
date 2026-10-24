"""
Build a clean lung-cancer prediction dataset from the US National Health Interview
Survey (NHIS) adult files, 2019-2023.

Input : data/raw/nhis/adult19csv.zip ... adult23csv.zip  (official CDC files, unchanged)
Output: data/clean/nhis_lung_cancer_clean.csv        readable, one row per adult
        data/clean/nhis_lung_cancer_model_ready.csv  numeric, no missing values, with a train/test split column
        data/clean/data_dictionary.csv               what every column means
        data/clean/cleaning_report.txt               every cleaning step with before/after counts

Run from the repository root:  python scripts/build_clean_dataset.py
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

report = []
def log(msg=""):
    print(msg)
    report.append(str(msg))


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

# ---------------------------------------------------------------- 2. duplicates
log("\nSTEP 2 - Duplicates")
raw["person_id"] = raw["SRVY_YR"].astype(int).astype(str) + "_" + raw["HHX"].astype(str)
n_dup_id = raw["person_id"].duplicated().sum()
log(f"  Duplicate person IDs (year + household): {n_dup_id}")
raw = raw.drop_duplicates(subset="person_id").reset_index(drop=True)
log(f"  Rows after removing duplicate IDs: {len(raw):,}")

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

# Categorical columns: one-hot encoding, missing becomes its own 'Unknown' category
cats = ["sex", "race_ethnicity", "region", "education", "income_group", "general_health",
        "bmi_category", "smoking_status", "age_group"]
dummies = pd.get_dummies(clean[cats].fillna("Unknown"), prefix=cats, dtype=int)
m = pd.concat([m, dummies], axis=1)
m["lung_cancer"] = clean["lung_cancer"]

assert m.drop(columns=["survey_weight"]).isna().sum().sum() == 0
log(f"  Model-ready shape: {m.shape} | train {(split == 'train').sum():,} | test {(split == 'test').sum():,}")
log(f"  Lung cancer rate - train: {100 * m.loc[split == 'train', 'lung_cancer'].mean():.2f}% | "
    f"test: {100 * m.loc[split == 'test', 'lung_cancer'].mean():.2f}%")
log("  Note: scale numeric columns inside your model pipeline, fitted on training rows only.")
m.to_csv(CLEAN / "nhis_lung_cancer_model_ready.csv", index=False)

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
    ("lung_cancer", "TARGET: ever told they had lung cancer (1 = yes)", "CANEV_A + LUNGCAN_A"),
], columns=["column", "meaning", "built_from"])
dictionary.to_csv(CLEAN / "data_dictionary.csv", index=False)

(CLEAN / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nDone. Files written to", CLEAN)
