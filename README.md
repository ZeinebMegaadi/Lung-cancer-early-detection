# Lung-cancer-early-detection

A clean, real-world dataset for **predicting whether a person has (had) lung cancer** from health and lifestyle information, built from 5 years of the US **National Health Interview Survey (NHIS)**, 2019–2023.

## Folder structure

```
data/
  raw/
    nhis/
      adult19csv.zip ... adult23csv.zip   Official NHIS adult files from CDC (unchanged)
    other/
      lung_cancer_combined_22457.csv      Earlier survey file (kept for reference, NOT used - see below)
  clean/
    nhis_lung_cancer_clean.csv            Clean dataset, readable values, one row per adult
    nhis_lung_cancer_model_ready.csv      Same data, numbers only, no missing values, train/test split
    data_dictionary.csv                   What every column means
    cleaning_report.txt                   Every cleaning step with before/after counts
scripts/
  build_clean_dataset.py                  Builds data/clean/ from data/raw/nhis/
requirements.txt
```

## How to rebuild the clean dataset

```bash
pip install -r requirements.txt
python scripts/build_clean_dataset.py
```

## Data sources

| File | Source | Rows (approx.) |
|---|---|---|
| `adult19csv.zip` … `adult23csv.zip` | US National Center for Health Statistics (CDC), National Health Interview Survey, Sample Adult files: <https://www.cdc.gov/nchs/nhis/> | ~30,000 adults per year |
| `lung_cancer_combined_22457.csv` | Supplied by the team; origin undocumented | 22,457 |

NHIS data are public-domain US government data and may be shared freely. Codebooks (variable meanings) for each year: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/<YEAR>/Adult-summary.pdf`

## How the clean dataset was built

1. **Load and stack** the 5 yearly files (same variable names every year) → one table.
2. **Duplicates:** each person gets an ID = year + household number; duplicate IDs removed.
3. **Target `lung_cancer`:** 1 if the person was ever told they had lung cancer, 0 if they never had cancer or had another cancer. People who refused or didn't know are removed (their answer can't be guessed).
4. **Recode:** NHIS number codes turned into clear values (e.g. `SEX_A` 1/2 → Male/Female); "refused / don't know / not ascertained" codes (7, 8, 9, 97, 98, 99) become missing values instead of fake numbers.
5. **New columns:** `age_group`, `ever_smoker`, `screening_age_smoker` (age 50–80 and ever smoked, a proxy of the US screening group), `comorbidity_count`.
6. **Checks:** row counts, duplicate IDs, value ranges, missing values per column, lung-cancer rate by smoking status and COPD (see `cleaning_report.txt`).
7. **Model-ready file:** fixed 80/20 train/test split (stratified on `lung_cancer`, random seed 42) so everyone in the team uses the same split; missing age filled with the **training** median (+ flag); yes/no columns get an `_unknown` flag; categories one-hot encoded with an `Unknown` category. No scaling is applied — scale inside your model pipeline using training rows only.

## Columns

See `data/clean/data_dictionary.csv`. Summary:

| Column | Meaning |
|---|---|
| person_id, year, survey_weight | ID, survey year, NHIS sampling weight |
| age, age_group, sex, race_ethnicity, region | Demographics |
| education, income_group | Social factors |
| general_health, bmi_category | Self-rated health, body mass index |
| smoking_status, ever_smoker, ever_ecigarette, screening_age_smoker | Smoking |
| copd, asthma, hypertension, heart_disease, diabetes, comorbidity_count | Other conditions |
| **lung_cancer** | **Target** |

## Important notes for modelling

- **Very imbalanced:** only a small fraction of adults (well under 1%) have had lung cancer. Use recall, precision, F1 and PR-AUC — not accuracy — and consider class weights.
- **Leakage:** `CANEV_A` ("ever had any cancer") is NOT included as a feature, because every lung-cancer case has it = yes.
- **Timing:** NHIS asks about *current* habits. Some people stop smoking after their diagnosis, so smoking status may look weaker than it really is (reverse causation).
- **Self-reported** answers, no pack-years, no imaging or clinical tests.
- Each year is a new sample of people, but individuals cannot be linked across years, so a few people might appear in more than one year.

## Why the earlier survey file is not used

`lung_cancer_combined_22457.csv` has the same questions as the 309-person Kaggle "Lung Cancer" survey; 2,573 of its rows are exact repeats and 1,438 rows copy original answers. After removing the repeats, none of its questions is associated with lung cancer (all Cramér's V < 0.02). It is kept in `data/raw/other/` for transparency only.
