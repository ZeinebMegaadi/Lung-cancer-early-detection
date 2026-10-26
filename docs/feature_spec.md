# Feature specification

## Coding and feature definitions

Symptom and lifestyle flags are recoded from `1 = NO`, `2 = YES` to `0/1`.
`GENDER` is renamed `GENDER_CODE`; the code mapping remains unknown. The target
`LUNG_CANCER` remains a separate recorded-label column. `patient_id` is removed
from all feature tables.

| Feature | Definition | Type / range | Rationale | Leakage note |
|---|---|---|---|---|
| `GENDER_CODE` | Original `GENDER` code, unchanged | Binary 0/1 | Preserve supplied coded variable without guessing its meaning | Not a verified demographic interpretation |
| `AGE` | Original age field | Numeric; observed 21–87 | Preserve recorded age | Units are undocumented |
| Each original symptom/lifestyle flag | Code 2 → 1; code 1 → 0 | Binary 0/1 | Consistent numerical representation of supplied coding | Derived only from the corresponding input flag |
| `AGE_Z` | `(AGE - fitted_mean) / fitted_population_sd` | Numeric | Standardized age for algorithms sensitive to scale | Fit scaler on training folds only in any future modeling |
| `AGE_BAND` | `<50`, `50-59`, `60-69`, `70-79`, or `80+` | Ordered categorical | Descriptive age strata | Cut points are fixed; no label information used |
| `IN_SCREEN_AGE` | 1 when AGE is from 50 through 80 inclusive | Binary 0/1 | Descriptive age-range indicator based on cited screening criteria | Not an eligibility determination; no dose or quit-time fields |
| `RESP_SYMPTOM_COUNT` | Sum of YES recodes for cough, wheezing, shortness of breath, and chest pain | Integer 0–4 | Compact descriptive count of four respiratory/chest symptoms | Input-only summary; not label-derived |
| `SYMPTOM_COUNT_ALL` | Sum of respiratory flags plus fatigue and swallowing difficulty | Integer 0–6 | Broader descriptive symptom count | Input-only summary; not label-derived |
| `SMOKER_IN_SCREEN_AGE` | Recode of smoking YES multiplied by `IN_SCREEN_AGE` | Binary 0/1 | Domain-prior interaction proxy | Not screening eligibility; pack-years and quit time are absent |

## Feature sets

| Set | Number of features | Features |
|---|---:|---|
| `F0_raw15` | 15 | The 15 recoded/raw variables, including `GENDER_CODE` and `AGE` |
| `F1_raw_plus_engineered` | 19 | F0 plus `AGE_Z`, `RESP_SYMPTOM_COUNT`, `SYMPTOM_COUNT_ALL`, and `SMOKER_IN_SCREEN_AGE` |
| `F2_prior_driven` | 5 | `AGE_Z`, `SMOKING`, `IN_SCREEN_AGE`, `RESP_SYMPTOM_COUNT`, `SMOKER_IN_SCREEN_AGE` |

Each exported feature table includes the recorded label as a separate
`LUNG_CANCER` column and excludes `patient_id`.

## Selection and scaling protocol

The univariate screening table is descriptive only; it is not used to select
features. Evidence-based selection is inconclusive because effect sizes are
small in the leading-block-removed view. F2 is therefore a domain-prior-driven
candidate set, while F0 and F1 are retained for comparison.

The current `AGE_Z` values use a scaler fitted on view B for descriptive
feature construction. In any future model evaluation, the scaler must be fitted
inside each training fold only; the descriptive scaler is not a valid
cross-validation preprocessing strategy. No predictive model is trained in
this project.
