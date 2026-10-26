# Lineage report (handbook task B-4)

- **Status:** `sources_unavailable`
- **Checked (UTC):** 2026-10-26T08:05:52+00:00
- **Combined file:** `data\raw\lung_cancer_combined_22457.csv`

## Matching rules

Rows match when all 15 handbook features (`GENDER`, `AGE`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC_DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL_CONSUMING`, `COUGHING`, `SHORTNESS_OF_BREATH`, `SWALLOWING_DIFFICULTY`, `CHEST_PAIN`) are equal after:

- Symptom/lifestyle fields recoded to the project `1 = NO`, `2 = YES` scheme.
- `GENDER` mapped from `M`/`F` to `0`/`1` with both orientations attempted.

## Results

| Source | Available | Rows | Matches in combined | Match rate | Gender map | File |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Kaggle Lung Cancer survey (handbook ~309 rows) | no |  |  |  | — | — |
| Kaggle Lung Cancer Dataset (handbook ~3,000 rows) | no |  |  |  | — | — |

## Interpretation

The handbook B-4 Kaggle source CSVs were not present in data/external/. No match counts are invented. Place downloaded files using data/external/README.md, then re-run python -m src.lineage.
