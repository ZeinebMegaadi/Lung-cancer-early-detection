from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
DOCS = ROOT / "docs"

TARGET = "LUNG_CANCER"
ID = "patient_id"
SEED = 42

EXPECTED = {
    "rows": 22457,
    "cols": 17,
    "sha256_prefix": "4ad2333067fcb0a9",
}

SYMPTOMS_RESP = ["COUGHING", "WHEEZING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]
SYMPTOMS_OTHER = ["FATIGUE", "SWALLOWING_DIFFICULTY"]
BASE_COLUMNS = [
    "patient_id",
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    "LUNG_CANCER",
]

FEATURE_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]

RAW_FILE = RAW / "lung_cancer_combined_22457.csv"
