# Data dictionary

The observed values and shares below were computed from
`data/raw/lung_cancer_combined_22457.csv`. A symptom share is the fraction of
rows coded `2`, which the supplied project description defines as YES. The
semantics of `GENDER` and `LUNG_CANCER` are undocumented; their code shares are
reported without inferring a real-world meaning. `unknown` means the value was
not documented in the supplied source materials.

| Column | Type | Coding | Observed range | Share of YES / code share | Notes |
|---|---|---|---:|---:|---|
| `patient_id` | int64 | Sequential row identifier | 1–22,457 | N/A | Unique in this file; not an analysis feature. |
| `GENDER_CODE` (`GENDER`) | int64 | 0/1; mapping unknown | 0–1 | Code 1: 50.69% | Meaning of each code: unknown. Report as `GENDER_CODE`. |
| `AGE` | int64 | Age; unit presumed years, not documented | 21–87 | N/A | Observed range; unit documentation: unknown. |
| `SMOKING` | int64 | 1 = NO, 2 = YES | 1–2 | 54.46% | Code meaning per supplied project description. |
| `YELLOW_FINGERS` | int64 | 1 = NO, 2 = YES | 1–2 | 54.21% | Code meaning per supplied project description. |
| `ANXIETY` | int64 | 1 = NO, 2 = YES | 1–2 | 48.01% | Code meaning per supplied project description. |
| `PEER_PRESSURE` | int64 | 1 = NO, 2 = YES | 1–2 | 47.68% | Code meaning per supplied project description. |
| `CHRONIC_DISEASE` | int64 | 1 = NO, 2 = YES | 1–2 | 48.73% | Code meaning per supplied project description. |
| `FATIGUE` | int64 | 1 = NO, 2 = YES | 1–2 | 66.05% | Code meaning per supplied project description. |
| `ALLERGY` | int64 | 1 = NO, 2 = YES | 1–2 | 52.85% | Code meaning per supplied project description. |
| `WHEEZING` | int64 | 1 = NO, 2 = YES | 1–2 | 52.33% | Code meaning per supplied project description. |
| `ALCOHOL_CONSUMING` | int64 | 1 = NO, 2 = YES | 1–2 | 52.32% | Code meaning per supplied project description. |
| `COUGHING` | int64 | 1 = NO, 2 = YES | 1–2 | 54.90% | Code meaning per supplied project description. |
| `SHORTNESS_OF_BREATH` | int64 | 1 = NO, 2 = YES | 1–2 | 62.97% | Code meaning per supplied project description. |
| `SWALLOWING_DIFFICULTY` | int64 | 1 = NO, 2 = YES | 1–2 | 44.36% | Code meaning per supplied project description. |
| `CHEST_PAIN` | int64 | 1 = NO, 2 = YES | 1–2 | 52.62% | Code meaning per supplied project description. |
| `LUNG_CANCER` | int64 | 0/1; meaning unknown | 0–1 | Recorded label code 1: 82.09% | The meaning and label-generation process are unknown. |

Shares are rounded to two decimal places for display. The underlying proportions
were computed from the loaded file; rows are not assumed to be independent.
