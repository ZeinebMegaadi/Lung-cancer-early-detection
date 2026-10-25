# Lung-cancer-early-detection

A clean, real-world dataset to **predict lung cancer (yes / no)** from personal health information **plus** the air pollution, smoking level and lung-cancer rate of the region where the person lives.

## The datasets

| # | Dataset | Source | What it gives | How it is combined |
|---|---|---|---|---|
| 1 | **NHIS 2019–2023** (5 files) | CDC – National Health Interview Survey | ~30,000 real adults per year: age, sex, smoking, COPD… and **lung cancer yes/no** | **Stacked** (rows of the 5 years put together) |
| 2 | **EPA air quality** 2019–2023 | US Environmental Protection Agency – AirData | Fine-particle pollution (PM2.5) per monitor | **Joined** on region + year |
| 3 | **BRFSS smoking rates** | CDC – Behavioral Risk Factor Surveillance System | % of adults who smoke, per state and year | **Joined** on region + year |
| 4 | **US Cancer Statistics** | CDC + National Cancer Institute | New lung cancer cases per state and year | **Joined** on region + year |
| – | `lung_cancer_combined_22457.csv` | Supplied by the team, origin unknown | – | **Audited, not used** (see the end) |

All are free US government data and may be shared.

## Folder structure

```
data/
  raw/                       original data, unchanged (or reduced to the rows we need)
    nhis/   adult19csv.zip ... adult23csv.zip
    epa/    annual_conc_by_monitor_2019.zip ... _2023.zip
    brfss/  brfss_current_smoker_by_state_2019_2023.csv
    uscs/   uscs_lung_incidence_by_area.csv
    other/  lung_cancer_combined_22457.csv
  clean/
    regional_context.csv              4 regions x 5 years: pollution, smoking rate, lung cancer rate
    nhis_lung_cancer_clean.csv        FINAL dataset, readable values, one row per adult
    nhis_lung_cancer_model_ready.csv  same, numbers only, no missing values, train/test split
    nhis_lung_cancer_model_ready_scaled.csv  same, numeric columns standardised (fitted on training rows)
    final_features.csv                every candidate feature: KEEP or DROP, and why
    outlier_check.csv                 IQR outlier check and the decision for each numeric column
    before_after_summary.csv          one table: rows, columns and missing cells after each step
    scaler_parameters.csv             training means / SDs used for scaling
    data_dictionary.csv               meaning of every column
    cleaning_report.txt               every cleaning step with before/after counts
    regional_report.txt               cleaning steps of the 3 regional datasets
scripts/
  download_data.py            step 0: download all raw data
  build_regional_data.py      step 1: clean the 3 regional datasets -> regional_context.csv
  build_clean_dataset.py      step 2: clean NHIS, join the regional table, save the final files
  eda_analysis.py             step 3: charts, statistics and hypothesis tests
reports/
  figures/                    9 charts (fig01 ... fig09)
  tables/                     the numbers behind every chart
  eda_results.txt             all results and tests explained in plain words
run_all.py                    runs steps 0-3 in order
```

## How to rebuild everything

```bash
pip install -r requirements.txt
python run_all.py
```

## What the scripts do (simple version)

**Regional datasets (`build_regional_data.py`)**
1. **EPA:** keep only PM2.5 monitors, one official yearly value per monitor, remove impossible values → average per state → average per region.
2. **BRFSS:** keep the "current smoker = yes" rate per state and year → average per region.
3. **US Cancer Statistics:** keep lung cancer, both sexes, all races, single years → region rate = all cases ÷ all people × 100,000.
4. Map every state to its **US Census region** (Northeast, Midwest, South, West – the same regions NHIS uses). Territories are dropped.
5. Combine into **one table: 4 regions × 5 years = 20 rows**. A missing year is filled from the same region's nearest year.

**Main dataset (`build_clean_dataset.py`)**
1. **Stack** the 5 NHIS years.
2. **Duplicates:** ID = year + household number; duplicates removed.
3. **Target `lung_cancer`:** 1 = ever told lung cancer; 0 = no cancer or another cancer; "don't know/refused" rows removed.
4. **Recode** NHIS number codes into words; codes 7/8/9 (don't know) → missing.
5. **New columns:** `age_group`, `ever_smoker`, `screening_age_smoker`, `comorbidity_count`.
6. **Join** the regional table on **region + year** → adds `pm25_mean`, `smoking_rate`, `lung_cancer_incidence`. Check: same number of rows before and after.
7. **Outliers:** IQR rule + box plots on every numeric column; each value is checked for plausibility and the decision is written in `outlier_check.csv` (all real values kept).
8. **Checks:** row counts, duplicates, ranges, missing values.
9. **Model-ready file:** fixed 80/20 train/test split (seed 42); missing values filled using **training rows only** + "unknown" flags; categories one-hot encoded.
10. **Feature selection (training rows only):** drop redundant columns (e.g. age bands when exact age is kept), constant columns, and near-duplicates (|r| > 0.95). Every decision is in `final_features.csv`.
11. **Scaling:** StandardScaler on the numeric columns, fitted on training rows only → `nhis_lung_cancer_model_ready_scaled.csv`.
12. **Summary:** `before_after_summary.csv` shows what every step changed.

## EDA and hypothesis tests (`eda_analysis.py`)

| Figure | Question | Test |
|---|---|---|
| fig01 | How rare is lung cancer? | – |
| fig02 | What is still missing? | – |
| fig03 | Is smoking linked to lung cancer? | Chi-square + Cramér's V + odds ratio (95% CI) |
| fig04 | Is age linked to lung cancer? | Mann–Whitney U (age is not normal) |
| fig05 | Which conditions go with lung cancer? | Chi-square + odds ratios (95% CI) |
| fig06 | Combined effect of age and smoking | Heatmap of rates |
| fig07 | Which variables move together? | Spearman correlation |
| fig08 | Do regional factors matter? Does NHIS agree with official cancer statistics? | Spearman (20 region-years) |
| fig09 | Which features carry the most information? | Mutual information on training rows only |

All rates are **weighted with NHIS survey weights** (population estimates), and all p-values are **Holm-corrected** for multiple testing.

## Important notes for modelling

- **Very imbalanced:** well under 1% of adults have had lung cancer → use recall, precision, F1, PR-AUC, class weights.
- **Leakage avoided:** "ever had any cancer" (`CANEV_A`) is not a feature.
- **Regional columns have only 20 different values** (4 regions × 5 years), so they add context, not personal detail. The personal columns (smoking, age, COPD) matter most.
- `lung_cancer_incidence` describes the region's population, not the person, so it is not leakage – but say this clearly in the report.
- **Timing:** NHIS asks about *current* habits; some people quit smoking after diagnosis.
- **Self-reported** answers; no pack-years or medical tests.

## Why the earlier survey file is not used

`lung_cancer_combined_22457.csv` copies the 309-person Kaggle "Lung Cancer" survey: 2,573 rows are exact repeats and 1,438 rows copy original answers. Without the repeats, no question is linked to lung cancer (all Cramér's V < 0.02). It is kept in `data/raw/other/` only for transparency.
