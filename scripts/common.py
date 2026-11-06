"""
Shared paths and constants used by every pipeline script, build_report.py and the notebook.

    from common import ROOT, RAW, CLEAN, FIG, TAB, YEARS, RANDOM_STATE
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                 # original files: nhis/, epa/, brfss/, uscs/, other/
CLEAN = ROOT / "data" / "clean"             # clean and model-ready datasets, cleaning logs
RESULTS = ROOT / "results"                  # pipeline outputs: text results, figures/, tables/
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
for _folder in (CLEAN, FIG, TAB):
    _folder.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023]      # NHIS survey years
RANDOM_STATE = 42                           # every random step (split, sampling, MI, permutations)

# Columns of the model-ready file that are not features
NON_FEATURES = ("person_id", "split", "survey_weight", "lung_cancer")
# The three regional columns joined on region + year
REGIONAL = ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]

# Reference category of every one-hot group: dropped from the model-ready file (Rule 4), so the other
# dummies of the group are read relative to it and the group no longer sums to 1 (no exact collinearity)
REFERENCE = {"race_ethnicity": "White", "education": "High school / GED", "income_group": "High",
             "bmi_category": "Healthy", "general_health": "Excellent", "smoking_status": "Never",
             "sex": "Male", "region": "West"}

# Column type of every column of the 26-column clean file
COLUMN_TYPES = {
    "identifier": ["person_id"], "weight": ["survey_weight"], "target": ["lung_cancer"],
    "numeric": ["age", "year"],
    "ordinal": ["education", "income_group", "general_health", "bmi_category"],
    "nominal": ["race_ethnicity", "region", "smoking_status"],
    "binary": ["sex", "ever_ecigarette", "copd", "asthma", "hypertension", "heart_disease", "diabetes"],
    "engineered": ["age_group", "ever_smoker", "screening_age_smoker", "comorbidity_count"],
    "regional": REGIONAL,
}
# Group of every final feature, by column-name prefix
FEATURE_GROUPS = {
    "demographic": ("age", "sex_", "race_ethnicity_"),
    "socio-economic": ("education_", "income_group_"),
    "smoking": ("smoking_status_", "screening_age_smoker", "ever_ecigarette"),
    "health": ("copd", "asthma", "hypertension", "heart_disease", "diabetes", "general_health_", "bmi_category_"),
    "regional": tuple(REGIONAL),
}
