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
