# Lung Cancer Risk Factors in US Adults

This project studies which personal and regional factors are associated with a lung-cancer history in US adults. It combines the National Health Interview Survey (NHIS) 2019–2023 (149,928 adults, 619 cases) with regional air-quality, smoking-prevalence and cancer-registry data. The scope is CRISP-DM phases 1–6, from business understanding to feature engineering; no predictive model is trained.

## Quick access

- Report (PDF): [report/Lung_Cancer_Midterm_Report.pdf](report/Lung_Cancer_Midterm_Report.pdf)
- Notebook: [notebooks/lung_cancer_project.ipynb](notebooks/lung_cancer_project.ipynb), saved with all outputs, so it renders directly on GitHub. If GitHub cannot display it, open it on nbviewer: <https://nbviewer.org/github/ZeinebMegaadi/Lung-cancer-early-detection/blob/main/notebooks/lung_cancer_project.ipynb>
- LaTeX source of the report: [report/](report/)
- Results (figures, tables, text results): [results/](results/)

## Key results

- RQ1, smoking: ever-smokers have 7.7 times the odds of a lung-cancer history (95% CI 6.2–9.4); within the same age group the odds ratio is still 5.8 (Mantel–Haenszel, 95% CI 4.7–7.1), and it differs between age groups (Breslow–Day p = 0.005).
- RQ2a, age: median age 72 for cases vs 54 for other adults (Mann–Whitney p = 3.1e-141, rank-biserial r = 0.59); the weighted rate rises from 0.02% at 18–44 to 1.25% at 75+.
- RQ2b, chronic conditions: COPD has the strongest association (odds ratio 15.1); the crude 14.8 falls to 9.3 after adjusting for smoking and to 5.7 after adjusting for smoking × age group. Heart disease, hypertension and diabetes are moderate (2.3–4.6); e-cigarette use is not significant.
- RQ2c, socio-demographic factors: sex is not associated; race/ethnicity, education, income, BMI and self-rated health are (Holm p < 1e-12), all with small Cramér's V.
- RQ3, regional context: the NHIS regional rate follows the official registry incidence (Spearman ρ = 0.79 over 20 region-years, p = 3.5e-05), driven by differences between the 4 regions; PM2.5 and regional smoking prevalence show no detectable link.
- Testing: 14 of the 18 hypothesis tests remain significant after Holm correction.
- Feature set: 67 candidates reduced to 38 features with no leakage and a maximum VIF of 7.49; PCA needs 32 of 38 components for 95% of the variance, so the original features are kept.

## Research questions

Outcome: having ever been told by a doctor that one had lung cancer, self-reported by US adults in NHIS 2019–2023. The full wording is in §1.3 of the [report](report/Lung_Cancer_Midterm_Report.pdf).

| RQ | Question | Where answered |
| --- | --- | --- |
| RQ1 | How much higher are the odds for ever-smokers, and how much remains within age groups? | Report §4; `eda_results.txt` §2; Fig. 6 |
| RQ2a | Are adults with a lung-cancer history older, and how does the rate change with age and smoking? | Report §4; `eda_results.txt` §3, §5; Fig. 7–8 |
| RQ2b | Which chronic conditions (and e-cigarette use) are associated, and does COPD remain after adjusting for smoking? | Report §4; `eda_results.txt` §4, §4b; Fig. 9–10 |
| RQ2c | Are sex, race/ethnicity, education, income, BMI and self-rated health associated? | Report §4; `eda_results.txt` §4c; Fig. 11 |
| RQ3 | Does the regional NHIS rate follow registry incidence, PM2.5 and regional smoking, and how informative are regional features? | Report §4–5; `eda_results.txt` §7, §9; Fig. 13–14 |

## Repository structure

```text
data/raw/                  original NHIS, EPA, BRFSS and US Cancer Statistics files
data/clean/                clean dataset, model-ready (and scaled) dataset, data dictionary, cleaning reports
scripts/                   pipeline steps: download, regional data, cleaning, EDA, PCA, supplementary checks, report build
notebooks/                 project notebook, calling the functions in scripts/
results/                   figures, tables and text results written by the pipeline (index in results/README.md)
results/REPORT_UPDATES.md  record of how the midterm feedback was addressed
report/                    LaTeX source and compiled PDF of the report (see report/README.md)
run_all.py                 runs the whole pipeline in order
requirements.txt           Python dependencies
```

## Reproduce

Requirements: Python 3.11–3.13. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

The raw data is already in `data/raw/`, so only missing files are downloaded. A full run takes about three minutes. Every random step uses seed 42, so the outputs are identical on every machine.

Notebook (re-executes in place; the options keep the saved file byte-identical between runs, about two minutes):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False --NotebookExporter.preprocessors=nbconvert.preprocessors.CoalesceStreamsPreprocessor notebooks/lung_cancer_project.ipynb
```

`REBUILD = False` in the first cell reuses `data/clean/` and `results/`; `REBUILD = True` reruns the pipeline from `data/raw/` first.

Report: install MiKTeX (Windows) or TeX Live (macOS / Linux) so that `pdflatex` is on the PATH, then run

```bash
python scripts/build_report.py
```

This writes `report/generated/` from the pipeline outputs and runs pdflatex twice into `report/build/main.pdf` (about ten seconds). In VS Code, the LaTeX Workshop recipe "pdflatex x2" does the same compile (setup in [report/README.md](report/README.md)). Copy `report/build/main.pdf` to `report/Lung_Cancer_Midterm_Report.pdf` to update the committed PDF.

## Data sources

| Source | Content | Link |
| --- | --- | --- |
| NHIS 2019–2023 (CDC) | one row per adult: demographics, smoking, chronic conditions, lung-cancer history | <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/> |
| EPA AirData | annual PM2.5 concentration by monitor, averaged per region and year | <https://aqs.epa.gov/aqsweb/airdata/> |
| BRFSS (CDC) | adult current-smoking prevalence by state and year | <https://data.cdc.gov/resource/dttw-5yxu> |
| US Cancer Statistics (CDC, NCI) | lung-cancer incidence by state and year | <https://www.cdc.gov/cancer/uscs/> |

The regional sources are joined to NHIS on US Census region and survey year. A dataset supplied earlier (`data/raw/other/lung_cancer_combined_22457.csv`) was rejected: it repeats a 309-person Kaggle survey, and once the repeats are removed no question is associated with lung cancer.

## Authors

Maether Guennichi, Israa Daasi, Mayssa Gloulou, Zeineb Megaadi, Hiba Allah Msallem, Nouha Aouinti, Aicha Lassoued.

CS434 Data Science, instructor Dr. Dorra Louati, MedTech – Mediterranean Institute of Technology (SMU).
