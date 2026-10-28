# Lung-cancer-early-detection

A clean, real-world dataset and exploratory analysis of **which personal and regional factors are associated with lung cancer** in US adults. Each adult's personal health information is combined with the air pollution, smoking level and lung-cancer rate of the region where they live.

Scope (midterm): CRISP-DM phases 1-6, from business understanding to feature engineering. **No model is trained**; modelling is described as future work.

## 1. Business understanding

**Context.** Lung cancer is the leading cause of cancer death in the United States. Screening with low-dose CT is recommended only for a high-risk group (adults aged 50-80 with a heavy smoking history), so knowing which factors go with lung cancer helps decide who should be screened first.

**Problem statement.** Using national survey data from 2019-2023, describe how smoking, age, chronic conditions, socio-demographic factors and regional environment relate to having had lung cancer, and prepare a clean, leakage-free dataset for a future risk model.

**Research questions**

| # | Question | Where it is answered |
|---|---|---|
| RQ1 | Is smoking status associated with having had lung cancer? | `eda_results.txt` section 2, Fig 6, Fig 8 |
| RQ2 | Are age, chronic conditions (COPD, heart disease...) and socio-demographic / health factors (sex, race, education, income, BMI, self-rated health) associated with lung cancer? Does the COPD link survive once smoking is taken into account? | sections 3, 4, 4b, 4c, Fig 7, 9, 10, 11 |
| RQ3 | Do regional factors (PM2.5 air pollution, regional smoking rate, registry incidence) add information beyond the personal factors? | section 7 and 9, Fig 13, Fig 14 |

**Measurable success criteria** (criteria 1-4 and 6-8 are checked automatically in `eda_results.txt` section 10 and `reports/tables/success_criteria_check.csv`; criterion 5 is checked by an assertion in `supplementary_analysis.py`, section 11: the saved scaler must equal a refit on training rows only)

1. 0 duplicate person IDs in the clean dataset.
2. 0 missing values in the model-ready dataset.
3. The regional join adds 0 rows (row count identical before and after).
4. Train and test sets keep the same lung cancer rate (stratified split, difference < 0.05 percentage points).
5. No information from the test rows is used for filling missing values, feature selection, scaling or PCA.
6. Every research question is answered with at least one justified hypothesis test, reported with an effect size and a Holm-corrected p-value (alpha = 0.05).
7. Every candidate feature has a written KEEP / DROP reason.
8. No leakage feature: "ever had any cancer" (`CANEV_A`) is never a feature.

## 2. The datasets (data understanding)

| # | Dataset | Source | Granularity (one row =) | What it gives | How it is combined |
|---|---|---|---|---|---|
| 1 | **NHIS 2019–2023** (5 adult files) | CDC – National Health Interview Survey, https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/ | one adult in one survey year | ~30,000 real adults per year: age, sex, smoking, COPD… and **lung cancer yes/no** | **Stacked** (rows of the 5 years put together) |
| 2 | **EPA air quality** 2019–2023 | US EPA – AirData annual concentration by monitor, https://aqs.epa.gov/aqsweb/airdata/ | one monitor, pollutant and year | Fine-particle pollution (PM2.5, µg/m³) | Averaged per state, then per region; **joined** on region + year |
| 3 | **BRFSS smoking rates** | CDC – Behavioral Risk Factor Surveillance System, https://data.cdc.gov/resource/dttw-5yxu | one state and year | % of adults who currently smoke | Averaged per region; **joined** on region + year |
| 4 | **US Cancer Statistics** | CDC + National Cancer Institute, https://www.cdc.gov/cancer/uscs/ | one state and year | New lung cancer cases and population | Region rate = cases ÷ population × 100,000; **joined** on region + year |
| – | `lung_cancer_combined_22457.csv` | Supplied to the team; copies the Kaggle "Lung Cancer" survey (309 people) | – | – | **Audited, not used** (see the end) |

- All are free US government data and may be shared. Raw files were downloaded on 24-25 October 2026 with `scripts/download_data.py`.
- **Granularity:** the final table has one row per adult. Regional values are the same for every adult of the same region and year (4 regions × 5 years = 20 distinct values), because NHIS only tells us the person's region, not the state.
- **Final clean dataset:** 149,928 adults × 26 columns, 619 lung cancer cases (0.41% of the sample, 0.31% of the population after survey weighting).
- Variable meanings, units and the raw NHIS column each one comes from: `data/clean/data_dictionary.csv`.

## 3. Folder structure

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
    nhis_lung_cancer_model_ready.csv  final selected features, numbers only, no missing values, train/test split
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
  build_clean_dataset.py      step 2: clean NHIS, join the regional table, select and scale features
  eda_analysis.py             step 3: charts, statistics, hypothesis tests, success-criteria check
  feature_extraction_pca.py   step 4: PCA feature extraction (fitted on training rows only)
  supplementary_analysis.py   step 5: detailed population profile and supplementary checks
reports/
  figures/                    16 charts (fig01 ... fig16), numbered in the order of the report
  tables/                     the numbers behind every chart and test
  eda_results.txt             all EDA results and tests explained in plain words
  pca_results.txt             PCA results and the keep / reject decision
  supplementary_results.txt   population profile, per-year rates, age-adjusted odds ratios, survey-design check
run_all.py                    runs steps 0-5 in order
```

## 4. How to rebuild everything

Requirements: Python 3.11, 3.12 or 3.13. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Mac / Linux:         source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

The raw data is already in `data/raw/`, so step 0 only downloads files that are missing. A full run takes about two minutes. All paths are relative to the repository, and every random step uses seed 42, so the results are identical on every computer.

## 5. What the scripts do (simple version)

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
7. **Outliers:** IQR rule + box plots (Fig 3) on every numeric column; each value is checked for plausibility and the decision is written in `outlier_check.csv` (all real values kept).
8. **Checks:** row counts, duplicates, ranges, missing values.
9. **Model-ready file:** fixed 80/20 train/test split (seed 42, stratified on the target); missing values filled using **training rows only** + "unknown" flags; categories one-hot encoded.
10. **Feature selection (training rows only):** drop redundant columns (e.g. age bands when exact age is kept), constant columns, **near-constant columns (more than 99.5% of rows with the same value)** and near-duplicates (|r| > 0.95). Result: 67 candidates → 48 kept. Every decision is in `final_features.csv`.
11. **Scaling:** StandardScaler on the numeric columns, fitted on training rows only → `nhis_lung_cancer_model_ready_scaled.csv`.
12. **Summary:** `before_after_summary.csv` shows what every step changed.

**Feature extraction (`feature_extraction_pca.py`)**
1. Standardise all 48 selected features and fit a PCA on **training rows only** (equivalent to a PCA on the correlation matrix).
2. Scree plot (Fig 15), projection (Fig 16), number of components for 80 / 90 / 95% of the variance, Kaiser rule, loadings of PC1-PC3.
3. Check on the test rows that the kept components generalise.
4. Decision: 34 of 48 components are needed for 95% of the variance (no dominant component), so the **original, interpretable features are kept**; PCA stays an option for distance-based models later.

**Population profile and supplementary checks (`supplementary_analysis.py`)**
1. **Who is in the data:** for every category of every variable (sex, age group, race/ethnicity, region, year, education, income, smoking, e-cigarettes, BMI, health, each condition): adults, share of the sample and of the population (weighted), cases, non-cases and lung-cancer rate (`population_profile.csv`); cases vs other adults side by side (`cases_vs_noncases.csv`); age, sex and smoking inside each sex and race group (`group_profile.csv`); cases and rates per survey year (`rate_by_year.csv`).
2. **Confounding:** smoking odds ratio inside each age group and adjusted for age (CMH); COPD adjusted for smoking and for smoking × age group.
3. **RQ3:** the regional correlations split into between-region and within-region parts (the 20 region-years are 4 regions × 5 years).
4. **Benchmark:** recall and precision of the USPSTF-style rule (`screening_age_smoker`), the baseline a future model must beat.
5. **Mutual information** as a share of the target's entropy, compared with a permutation null (training rows).
6. **Survey design:** survey-weighted odds ratios with cluster-robust intervals (NHIS strata and PSUs read from the raw files).
7. **Criterion 5 check:** asserts that the saved scaler equals a refit on training rows only.

## 6. EDA and hypothesis tests (`eda_analysis.py`)

| Figure | Question | Method / test |
|---|---|---|
| fig01 | How rare is lung cancer? | Class balance |
| fig02 | What is still missing? Is missingness linked to the target? | Missing-value chart + chi-square on missing yes/no |
| fig03 | Are there outliers? | IQR rule + box plots (made by `build_clean_dataset.py`) |
| fig04 | How are the numeric variables distributed? Is age normal? | Histogram, Q-Q plot, skewness / kurtosis, Shapiro-Wilk, D'Agostino |
| fig05 | Who is in the data? | Bar charts of every categorical variable |
| fig06 | RQ1 – Is smoking linked to lung cancer? | Chi-square + Cramér's V + odds ratio (95% CI) |
| fig07 | RQ2 – Is age linked to lung cancer? | Mann–Whitney U (age is not normal) + Welch t-test as a check |
| fig08 | Combined effect of age and smoking | Heatmap of weighted rates |
| fig09 | RQ2 – Which conditions go with lung cancer? | Chi-square + odds ratios (95% CI) |
| fig10 | RQ2 – Is the COPD link only due to smoking? | Odds ratios within each smoking group + Cochran-Mantel-Haenszel test + Breslow-Day test |
| fig11 | RQ2 – Socio-demographic and health factors | Weighted rates + chi-square + Cramér's V |
| fig12 | Which variables move together? | Spearman correlation |
| fig13 | RQ3 – Do regional factors matter? Does NHIS agree with official cancer statistics? | Spearman (20 region-years) |
| fig14 | Which features carry the most information? | Mutual information on training rows only |
| fig15, fig16 | Can the features be compressed? | PCA scree plot and projection (`feature_extraction_pca.py`) |

All rates are **weighted with NHIS survey weights** (population estimates). All 18 hypothesis tests are **Holm-corrected** together for multiple testing; 14 stay significant.

**Key findings**
- **RQ1:** ever-smokers have **7.7 times** the odds of having had lung cancer (95% CI 6.2–9.4). Former smokers have a higher rate than current smokers, most likely because people stop smoking after diagnosis.
- **RQ2:** median age 72 vs 54 (Mann–Whitney p < 1e-140). COPD has the strongest link (odds ratio 15.1; on adults with known smoking status the crude 14.8 falls to **9.3 after adjusting for smoking**, and to 5.7 after adjusting for smoking and age group). Smoking itself stays strong after adjusting for age (odds ratio 5.8). Lower education and income go with higher rates; underweight and poor self-rated health have the highest rates (most likely reverse causation). Sex and e-cigarette use are not significant.
- **RQ3:** the NHIS regional rate follows the official registry rate (ρ = 0.79 over 20 region-years). The agreement comes from differences between the 4 regions (the West is low on both; within a region, ρ ≈ 0.03), so it is consistent with the official data rather than a formal validation. PM2.5 and the regional smoking rate show no detectable link. Personal factors carry far more information than regional ones.

## 7. Limitations and notes for modelling

- **Very imbalanced:** well under 1% of adults have had lung cancer → use recall, precision, F1, PR-AUC, class weights; accuracy is useless (99.59% for "always no").
- **Leakage avoided:** "ever had any cancer" (`CANEV_A`) is not a feature; every fitted step (filling, selection, scaling, PCA) uses training rows only.
- **Regional columns have only 20 different values** (4 regions × 5 years), so they add context, not personal detail. Registry rates are crude (not age-adjusted), and state values are averaged without population weights.
- `lung_cancer_incidence` describes the region's population, not the person, so it is not leakage.
- **Timing:** NHIS asks about *current* habits and health; some people quit smoking or lose weight after diagnosis (reverse causation). 2019–2023 pollution cannot explain cancers diagnosed many years earlier.
- **Survey design:** rates use the survey weights, but the hypothesis tests treat adults as independent draws (no strata / clusters).
- **Self-reported** answers; no pack-years or medical tests.
- **Associations, not causes:** only COPD was adjusted for a confounder (smoking).

## 8. Next steps (future modelling, not done in the midterm)

1. Logistic regression with class weights as an interpretable baseline (adjusted odds ratios for all factors at once).
2. Tree-based models (random forest, gradient boosting) compared with stratified cross-validation on the training rows only.
3. Evaluate on the untouched test set with PR-AUC, recall at a fixed precision and calibration.
4. Compare original features vs PCA components for distance-based models.

## 9. Why the earlier survey file is not used

`lung_cancer_combined_22457.csv` copies the 309-person Kaggle "Lung Cancer" survey: 2,573 rows are exact repeats, the first 1,455 rows are all negative, and 1,438 rows copy original answers. Without the repeats, no question is linked to lung cancer (all Cramér's V < 0.02). It is kept in `data/raw/other/` only for transparency.
