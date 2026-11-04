# Report updates — response to the professor's feedback

**Status: applied.** Every change below is already in the LaTeX report in `report/` (compiled to `report/Lung_Cancer_Midterm_Report.pdf`). This file is kept as the record of how each point of the professor's feedback was answered; it is no longer a to-do list.

Changelog against the first submitted version of the report (CS434 midterm report, 28 October 2026), ordered by report section. For every item: **where**, the **original text** quoted from that version, and the **new text**. Every new number comes from the pipeline outputs (`python run_all.py`); the source file is named where it helps. Table and figure numbers quoted here are those of the original version; in `report/` all tables and figures are numbered by LaTeX (`\label`/`\ref`), with Figures 12b and 12c kept as such.

---

## Abstract

### Paragraph 1, last sentence [A3]

**Current:**
> A detailed profile of the study population (Section 2.5) shows who the cases are: 76% are aged 65 or older, 82% have ever smoked and 47% have COPD, against 31%, 38% and 6% of the other adults.

**Replace with:**
> A detailed profile of the study population (Section 2.5) shows who the cases are: 76% are aged 65 or older, 82% have ever smoked and 47% have COPD, against 31%, 38% and 5% of the other adults.

(5.46% in `cases_vs_noncases.csv`; "6%" came from rounding the 5.5 of Table 10 a second time.)

### Paragraph 2 (whole paragraph) [A3 + RQ3 wording]

**Current:**
> Age, smoking history and COPD stand out clearly. Ever-smokers have 7.7 times the odds of never-smokers (95% CI 6.2–9.4), and still 5.8 times after adjusting for age group; people with lung cancer are much older (median 72 vs 54 years; a random case is older than a random non-case 79.5% of the time). COPD carries an odds ratio of 15.1; among adults with known smoking status the crude 14.8 falls to 9.3 after adjusting for smoking, and to 5.7 after adjusting for smoking and age together. Lower education and income, underweight and poor self-rated health also go with higher rates, the last two most likely through reverse causation. Of 18 hypothesis tests, 14 remain significant after Holm correction, and a survey-weighted, cluster-robust check leaves every key odds ratio far from 1. At the regional level, the survey rate agrees with the official registry rate (Spearman ρ = 0.79 over 20 region-years), but this agreement comes from differences between the four regions rather than from year-to-year changes, so it is consistent with – not proof of – a valid target; PM2.5 and regional smoking prevalence show no detectable association.

**Replace with:**
> Age, smoking history and COPD stand out clearly. Ever-smokers have 7.7 times the odds of never-smokers (95% CI 6.2–9.4), and still 5.8 times after adjusting for age group; people with lung cancer are much older (median 72 vs 54 years; a random case is older than a random non-case 79.5% of the time). COPD carries an odds ratio of 15.1; among adults with known smoking status the crude 14.8 falls to 9.3 after adjusting for smoking, and among those whose age is also known it falls to 5.7 after adjusting for smoking and age group together. Lower education and income, underweight and poor self-rated health also go with higher rates, the last two most likely through reverse causation. Of 18 hypothesis tests, 14 remain significant after Holm correction, and a survey-weighted, cluster-robust check leaves every key odds ratio far from 1. At the regional level, the survey rate agrees with the official registry rate (Spearman ρ = 0.79 over 20 region-years), but this agreement comes from differences between the four regions rather than from year-to-year changes, so it is consistent with – not proof of – a valid target; PM2.5 and regional smoking prevalence show no detectable association, and as features the three regional columns carry a single shared signal (which region-year a person lives in) rather than three separate ones.

### Paragraph 3 (whole paragraph)

**Current:**
> The result is a reproducible pipeline that one command rebuilds in under two minutes, with every cleaning step logged, all learned steps fitted on training rows only (checked by an automated assertion), and a leakage-free set of 48 features selected from 67 candidates. A principal component analysis confirmed that the information is spread over many dimensions (34 of 48 components for 95% of the variance), so we keep the original, interpretable features for the modelling phase.

**Replace with:**
> The result is a reproducible pipeline that one command rebuilds in under two minutes, with every cleaning step logged, all learned steps fitted on training rows only (checked by an automated assertion), and a leakage-free set of 38 features selected from 67 candidates. Feature selection also removes dependencies that span several columns – the comorbidity count, which is the exact sum of five condition flags, one reference category per one-hot group, and the region dummies – so that no feature has a variance inflation factor above 7.5 and a linear model can be fitted without perfect multicollinearity. A principal component analysis confirmed that the information is spread over many dimensions (32 of 38 components for 95% of the variance, and no near-zero component), so we keep the original, interpretable features for the modelling phase.

---

## §1.3 — Research questions (professor's feedback: more specific questions)

The analysis is unchanged: the same 18 tests and the same results. Only the questions, their labels and the success-criteria check change. In the code the questions are `eda_analysis.RESEARCH_QUESTIONS`, and every test in `all_tests_holm.csv` carries its new label.

### Opening paragraph (whole paragraph)

**Current:**
> We fixed these questions at the start of the analysis, so it could not drift toward whatever happened to look significant. They are the same three questions that label every test in the code (reports/tables/all_ tests_holm.csv).

**Replace with:**
> **Outcome.** Having ever been told by a doctor that one had lung cancer, self-reported by US adults in NHIS 2019–2023 (149,928 adults, 619 cases).
>
> We fixed these questions at the start of the analysis, so it could not drift toward whatever happened to look significant, and made each one name the comparison, the measure and the adjustment it relies on. The second question is split into three (RQ2a–c), one per family of factors. The same five labels mark every test in the code (reports/tables/all_tests_holm.csv): RQ1 has 1 test, RQ2a 1, RQ2b 7, RQ2c 6 and RQ3 3, for 18 in total.

### Table 1 (whole table; caption unchanged)

| # | Question | How we answer it | Where |
|---|---|---|---|
| RQ1 | **Smoking.** How much higher are the odds of a lung-cancer history among ever-smokers (current or former) than among never-smokers? How much of that odds ratio remains when ever- and never-smokers are compared within the same age group (Mantel–Haenszel over five age groups), and does its size differ between age groups (Breslow–Day)? | χ² test of independence, Cramér's V; odds ratio with 95% CI; Mantel–Haenszel odds ratio over five age groups with CMH test; Breslow–Day test | §4.2–4.3, Table 19 |
| RQ2a | **Age.** Are adults with a lung-cancer history older than those without, and by how much (Mann–Whitney U, rank-biserial r)? How does the survey-weighted rate change across the five age groups, alone and combined with smoking status? | Mann–Whitney U with rank-biserial r (Welch t-test as a check); survey-weighted rates by age group and by age group × smoking status | §2.5.1, §4.3, Fig. 7–8 |
| RQ2b | **Chronic conditions and e-cigarettes.** Which of five doctor-diagnosed conditions (COPD, heart disease, hypertension, diabetes, asthma), and ever using an e-cigarette, are associated with a lung-cancer history (odds ratios with 95% CIs)? Does the COPD association remain after stratifying by smoking status, and then by smoking status × age group? | χ² tests with odds ratios (95% CI) and Cramér's V; Cochran–Mantel–Haenszel and Breslow–Day tests across smoking groups; Mantel–Haenszel odds ratio over smoking status × age group (15 strata) | §4.4–4.5, Fig. 9–10, Tables 20–22 |
| RQ2c | **Socio-demographic and health factors.** Is a lung-cancer history associated with sex, race/ethnicity, education, family income relative to the poverty line, BMI category and self-rated general health (chi-square tests of independence, Cramér's V)? For the four ordered factors, how does the survey-weighted rate change from the lowest to the highest level (described from the rates, not tested as a trend)? | χ² tests of independence with Cramér's V (smallest race groups merged); survey-weighted rates per level, compared from lowest to highest for the four ordered factors (education, income, BMI, self-rated health) – described, not tested as a trend | §4.6, Fig. 11, Table 23 |
| RQ3 | **Regional context.** Across the 20 region-years (4 US Census regions × 2019–2023), does the survey-weighted NHIS lung-cancer rate follow (a) the official US Cancer Statistics incidence, (b) mean EPA PM2.5, and (c) BRFSS adult smoking prevalence? Is any agreement driven by differences between regions or by year-to-year changes within them? As features, how much information do the three regional columns carry about an individual's lung-cancer history compared with personal factors (mutual information against a permutation null)? | Spearman correlation over 20 region-years, split into between-region (4 regional means) and within-region parts; mutual information on training rows with a permutation null | §4.8, §5.5, Fig. 13–14, Tables 24 and 30 |

---

## Section 2 — new subsection "Dataset composition at a glance"

**Where:** directly after the opening paragraph of Section 2 ("Our final dataset has 149,928 adults and 26 columns. … joined on region and year."), before §2.1. Use an unnumbered subsection (`\subsection*{Dataset composition at a glance}`) so that §2.1–2.7 keep their numbers. They are cited throughout the report.

**Insert:**

> **Dataset composition at a glance**
>
> Every rate, odds ratio and correlation in this report is computed on a stated subset of the adults below, so we first show what the data are made of (Tables 2a–2c; source: `reports/tables/dataset_composition.csv`, produced by `supplementary_analysis.py`). Each survey year contributes between 27,594 and 31,925 adults and between 112 and 145 cases, so no single year dominates (Table 2a). The 26 columns of the clean file are of nine kinds (Table 2b): one identifier, one survey weight and the target, which are never features; two numeric columns, four ordinal, three nominal and seven binary survey variables; four engineered columns; and three regional columns joined on region and year. The model-ready file keeps 38 of the 67 candidate features (Section 5.4), grouped in Table 2c. Finally, the stratified 80/20 split keeps the same 0.41% case rate in the 119,942 training rows and the 29,986 test rows (Table 2a, lower part). Unless stated otherwise, descriptive numbers use all 149,928 adults with the variable known, and anything that ranks or fits features uses the training rows only.
>
> **Table 2a.** Adults and lung-cancer cases per survey year, and in the train / test split (unweighted).
>
> | | Adults | Cases | Rate (sample) |
> |---|---|---|---|
> | 2019 | 31,925 | 112 | 0.35% |
> | 2020 | 31,513 | 131 | 0.42% |
> | 2021 | 29,430 | 115 | 0.39% |
> | 2022 | 27,594 | 116 | 0.42% |
> | 2023 | 29,466 | 145 | 0.49% |
> | **All years** | **149,928** | **619** | **0.41%** |
> | Training rows | 119,942 | 495 | 0.41% |
> | Test rows | 29,986 | 124 | 0.41% |
>
> **Table 2b.** The 26 columns of the clean dataset by type.
>
> | Type | Columns | Which |
> |---|---|---|
> | Identifier | 1 | person_id |
> | Survey weight | 1 | survey_weight |
> | Target | 1 | lung_cancer |
> | Numeric | 2 | age, year |
> | Ordinal | 4 | education, income_group, general_health, bmi_category |
> | Nominal | 3 | race_ethnicity, region, smoking_status |
> | Binary | 7 | sex, ever_ecigarette, copd, asthma, hypertension, heart_disease, diabetes |
> | Engineered | 4 | age_group, ever_smoker, screening_age_smoker, comorbidity_count |
> | Regional (joined) | 3 | pm25_mean, smoking_rate, lung_cancer_incidence |
> | **Total** | **26** | |
>
> **Table 2c.** The 38 final model features by group (full list in Appendix B).
>
> | Group | Features | Which |
> |---|---|---|
> | Demographic | 8 | age; sex_Female; race/ethnicity: AIAN, AIAN and other, Asian, Black, Hispanic, Other or multiple (reference: White) |
> | Socio-economic | 8 | education: Bachelor, Graduate, Less than high school, Some college, Unknown (reference: High school / GED); income: Below poverty, Low, Middle (reference: High) |
> | Smoking | 6 | smoking status: Current, Former (reference: Never); screening_age_smoker and its unknown flag; ever_ecigarette and its unknown flag |
> | Health | 13 | copd, asthma, hypertension, heart_disease, diabetes; general health: Fair, Good, Poor, Very good (reference: Excellent); BMI: Obese, Overweight, Underweight, Unknown (reference: Healthy) |
> | Regional | 3 | pm25_mean, smoking_rate, lung_cancer_incidence |
> | **Total** | **38** | |

---

## §2.5.3 — Smoking bullet [A3]

**Current:**
> • Smoking. 90,937 adults have never smoked (60.7% of all adults, 62.3% of those who answered), 37,452 are former and 17,586 current smokers; 3,953 did not answer. Former smokers have the highest rate (0.98% sample, 0.81% weighted) and are much older (mean age 59.8, against 51.1 for current and 50.3 for never-smokers), a first hint of the reverse causation discussed in §4.2.

**Replace with:**
> • Smoking. 90,937 adults have never smoked (60.7% of all adults, 62.3% of those who answered), 37,452 are former and 17,586 current smokers; 3,953 did not answer. Former smokers have the highest rate (0.98% sample, 0.81% weighted) and are much older (mean age 59.7, against 51.1 for current and 50.3 for never-smokers), a first hint of the reverse causation discussed in §4.2.

(59.746 in `population_profile.csv`; Table 8 already shows 59.7.)

---

## §3.2 — Table 12: new row

**Where:** Table 12 ("Data-quality problems, the treatment applied, the reason, and its measured effect"). Insert after the row "Remaining missing values (at most 2.84%, Fig. 2)" and before "Leakage risk".

**New row:**

| Problem found | What we did | Why | Effect |
|---|---|---|---|
| Features that are exact or near-exact combinations of several other features (comorbidity_count = sum of 5 flags; complete one-hot groups summing to 1; region dummies vs the regional columns) – invisible to a pairwise correlation filter | Rule 4 of feature selection: variance inflation factor (VIF) on training rows; dropped comorbidity_count (kept for the EDA), one reference category per one-hot group and the region dummies | a linear model cannot be fitted with perfect multicollinearity, and its coefficients are unstable with near-perfect multicollinearity | VIF: 28 of 48 features infinite → maximum 7.5 (none above 10); 48 → 38 features |

---

## §3.6 — Effect of each step

### Paragraph, last sentence

**Current:**
> The 52 columns of the model-ready file are the 48 features plus person_id, split, survey_ weight and the target.

**Replace with:**
> The 42 columns of the model-ready file are the 38 features plus person_id, split, survey_weight and the target.

### Table 16: last two rows

**Current rows:**

| Step | Rows | Columns | Missing cells | What changed |
|---|---|---|---|---|
| 8. Feature selection (train only) | 149,928 | 52 | 0 | 19 redundant / near-constant / near-duplicate features dropped; 48 kept |
| 9. Scaling (train only) | 149,928 | 52 | 0 | 5 numeric columns standardised |

**Replace with:**

| Step | Rows | Columns | Missing cells | What changed |
|---|---|---|---|---|
| 8. Feature selection (train only) | 149,928 | 42 | 0 | 29 redundant / near-constant / near-duplicate / linearly dependent features dropped; 38 kept (max VIF 7.5) |
| 9. Scaling (train only) | 149,928 | 42 | 0 | 4 numeric columns standardised |

---

## §3.7 — Sanity-check bullet

**Current:**
> • A sanity check: after scaling, the training means are exactly 0 while the test means are slightly off (−0.002 for age, −0.010 for comorbidity count, 0.019 for registry incidence).

**Replace with:**
> • A sanity check: after scaling, the training means are exactly 0 while the test means are slightly off (−0.002 for age, 0.006 for PM2.5, −0.001 for the regional smoking rate and 0.019 for registry incidence).

(The rest of the bullet is unchanged.) Also add **"the VIF step"** to the second bullet: "The age median used for filling, the feature-selection rules **and the VIF step**, the scaler and the PCA are all computed on training rows only, then applied unchanged to the test rows."

---

## §4.3–§4.6 — Section headings (new RQ labels)

Change each heading in the body and in the table of contents:

| Current heading | New heading |
|---|---|
| 4.3 RQ2 – Age, and how it interacts with smoking | 4.3 RQ2a – Age, and how it interacts with smoking |
| 4.4 RQ2 – Chronic conditions | 4.4 RQ2b – Chronic conditions |
| 4.5 RQ2 – Is the COPD link only due to smoking? | 4.5 RQ2b – Is the COPD link only due to smoking? |
| 4.6 RQ2 – Socio-demographic and health factors | 4.6 RQ2c – Socio-demographic and health factors |

(4.2 "RQ1 – Smoking" and 4.8 "RQ3 – Regional factors" keep their labels.)

## §4.5 — "How we read it" bullet [A3]

**Current:**
> • How we read it. The crude odds ratio of these 145,897 adults is 14.8 (slightly below the 15.1 of Table 13, which uses everyone with a known COPD answer); …

**Replace with:**
> • How we read it. The crude odds ratio of these 145,897 adults is 14.8 (slightly below the 15.1 of Table 20, which uses everyone with a known COPD answer); …

(The rest of the bullet is unchanged. Table 13 is the missingness table; the 15.1 is in Table 20, the chronic-conditions table.)

### "Age as a second confounder" bullet [A3: two subsets]

**Current:**
> • Age as a second confounder. Adults with COPD are much older (mean age 65.4 vs 52.0). Stratifying by smoking status and age group together (15 strata; Table 22) lowers the odds ratio further, to 5.7 (95% CI 4.8–6.8), still highly significant (χ2 = 501.9, p = 3.7 ×10−111).

**Replace with:**
> • Age as a second confounder. Adults with COPD are much older (mean age 65.4 vs 52.0). On the 145,587 adults whose age is also known (599 cases), the crude odds ratio is 14.7 and the smoking-adjusted one 9.2 – the same picture as above – and stratifying by smoking status and age group together (15 strata; Table 22) lowers it further, to 5.7 (95% CI 4.8–6.8), still highly significant (χ2 = 501.9, p = 3.7 ×10−111).

(The sentence "COPD is therefore strongly associated …" is unchanged.)

---

## §4.7 — Correlations between variables (whole section)

**Current (caption and all four bullets):**
> Figure 12. Spearman correlation matrix of the numeric and binary variables (lower triangle). • Why Spearman. Most columns are 0/1 or ordinal, and age is not normal; Spearman uses ranks and only assumes a monotonic relationship, whereas Pearson assumes linearity (Chapter 4). • The strongest pairs are expected. Hypertension–comorbidity count (0.79), diabetes–comorbidity count (0.49) and asthma–comorbidity count (0.45) are high because the count is built from those columns. Age also goes with hypertension (0.44) and the comorbidity count (0.41), and the regional smoking rate with registry incidence (0.43). • No harmful redundancy. No pair of different personal variables exceeds 0.5, so no personal feature duplicates another. • Nothing correlates strongly with the target (highest: COPD 0.11, age 0.07). For a 0.41% outcome this is normal, and it tells us no single feature will be enough: a model will need to combine them.

**Replace with** (Fig 12 stays, followed by Fig 12b, Fig 12c, two new tables and new text):

> **Figure 12.** Spearman correlation matrix of the numeric and binary variables (lower triangle; all adults, unweighted; each pair uses the adults with both values known, n = 145,664–149,928).
>
> • **Why Spearman.** Most columns are 0/1 or ordinal, and age is not normal; Spearman uses ranks and only assumes a monotonic relationship, whereas Pearson assumes linearity (Chapter 4).
>
> • **The strongest pairs are expected.** Hypertension–comorbidity count (0.79), diabetes–comorbidity count (0.49) and asthma–comorbidity count (0.45) are high because the count is built from those columns. Age also goes with hypertension (0.44) and the comorbidity count (0.41), and the regional smoking rate with registry incidence (0.43).
>
> **Figure 12b.** Association between all 24 analysis variables of the clean dataset and the target (all 149,928 adults, 619 cases; each pair uses the adults with both values known, n = 142,511–149,928; unweighted). Cramér's V for two categorical variables, the correlation ratio η for a categorical and a numeric variable, |Spearman| for two numeric variables; all three run from 0 to 1. person_id and survey_weight are excluded, and CANEV_A is not in the clean dataset. *(file: `fig12b_association_matrix.png`)*
>
> **Figure 12c.** Spearman correlation of the 38 final model features and the target, on the training rows only (n = 119,942 adults, 495 cases; unweighted; lower triangle; values printed where |r| ≥ 0.3). *(file: `fig12c_feature_correlation.png`)*
>
> Fig. 12 only holds the numeric and 0/1 columns, so it cannot show how the categorical variables relate to each other or to the target. Fig. 12b therefore covers **all** 24 analysis variables, with the measure that suits each pair of types, and Fig. 12c covers **all** 38 features handed to the modelling phase. Every pair above 0.5 in either matrix is listed in Table 23a with the reason it is high (`reports/tables/high_correlation_pairs.csv`).
>
> **Table 23a.** Every pair of variables (Fig. 12b) or features (Fig. 12c) with an association above 0.5.
>
> | Matrix | Pair | Measure | Value | Type | Why it is high |
> |---|---|---|---|---|---|
> | Fig 12b | year – smoking_rate | η | 0.67 | built-from | smoking_rate is joined on region + year, and BRFSS smoking fell every year 2019–2023 |
> | Fig 12b | age – age_group | η | 0.96 | built-from | age_group is age cut into bands |
> | Fig 12b | region – pm25_mean | η | 0.71 | built-from | pm25_mean is joined on region + year (20 values) |
> | Fig 12b | region – smoking_rate | η | 0.73 | built-from | smoking_rate is joined on region + year (20 values) |
> | Fig 12b | region – lung_cancer_incidence | η | 0.98 | built-from | joined on region + year; the West is far lower than the other regions |
> | Fig 12b | smoking_status – ever_smoker | Cramér's V | 1.00 | built-from | ever_smoker = Current or Former smoking status |
> | Fig 12b | smoking_status – screening_age_smoker | Cramér's V | 0.69 | built-from | screening_age_smoker = aged 50–80 AND ever smoked |
> | Fig 12b | ever_smoker – screening_age_smoker | Cramér's V | 0.69 | built-from | screening_age_smoker = aged 50–80 AND ever smoked |
> | Fig 12b | hypertension – comorbidity_count | η | 0.73 | built-from | the count is the sum of the 5 condition flags; hypertension is the most common |
> | Fig 12b | diabetes – comorbidity_count | η | 0.57 | built-from | the count is the sum of the 5 condition flags |
> | Fig 12c | screening_age_smoker_unknown – ever_ecigarette_unknown | Spearman | 0.93 | same-concept | both flag the same ~2.8% of adults who skipped the smoking / tobacco questions |
> | Fig 12c | screening_age_smoker – smoking_status_Former | Spearman | 0.55 | built-from | the screening flag is built from smoking status, and most former smokers are aged 50–80 |
>
> • **Redundancy – what is high, and what we did about it.** All twelve pairs above 0.5 are either *built-from* (one column is computed from the other, or both are joined on region + year; 11 pairs) or *same-concept* (two flags for the same missing block; 1 pair). **No two separately measured personal variables are associated above 0.5.** The built-from overlaps are handled in feature selection: ever_smoker and the age bands are dropped by rule 1, and comorbidity_count and the region dummies by the new rule 4 (§5.4). The two high pairs left among the final features are kept on purpose: the two "unknown" flags, which together identify adults with an unknown smoking status (§5.7), and screening_age_smoker with former smoking, because the screening rule needs smoking status and age together. Both have a variance inflation factor below 10 (Table 29a). Pairwise matrices also have a blind spot: a column can be an exact combination of *several* others (the comorbidity count is the sum of five flags; every complete one-hot group sums to 1) without any single pair exceeding 0.5. Hypertension–comorbidity count was the only visible trace of such a dependency in Fig. 12 (Spearman 0.79). This is why feature selection now also checks the VIF (§5.4).
>
> • **Why every correlation with the target is small.** Nothing correlates strongly with the target (Fig. 12: highest COPD 0.11, age 0.07; Fig. 12b: COPD 0.11, general health 0.09). This is a property of a rare outcome, not a sign that the factors are weak. For two 0/1 variables, Spearman, Pearson and φ coincide, and φ can only reach 1 when both have the same prevalence. A feature with prevalence q can reach at most φ_max = √(p(1−q) / (q(1−p))) with a target of prevalence p = 0.41%. COPD (5.6% of adults) can therefore reach at most 0.26, and its observed 0.11 is 43% of that maximum, with an odds ratio of 15.1 (Table 23b). Ever-smoking reaches 71% of its maximum of 0.08. Correlations with this target must therefore be read together with the odds ratios and rates of §4.2–4.6, never on their own. Because no single feature is strongly associated with the target, a model will need to combine them.
>
> **Table 23b.** Observed and largest possible correlation (φ) between each 0/1 factor and the target, with the odds ratio (all adults with the factor known; unweighted; source: `target_correlation_bound.csv`).
>
> | Factor | Adults (cases) | Prevalence | φ observed | φ max | Share of max | Odds ratio |
> |---|---|---|---|---|---|---|
> | COPD | 149,826 (618) | 5.6% | 0.114 | 0.263 | 43% | 15.1 |
> | Heart disease | 149,597 (614) | 6.1% | 0.045 | 0.253 | 18% | 4.6 |
> | Hypertension | 149,766 (619) | 36.6% | 0.038 | 0.085 | 45% | 3.3 |
> | Diabetes | 149,860 (619) | 10.7% | 0.022 | 0.186 | 12% | 2.3 |
> | Asthma | 149,864 (618) | 13.9% | 0.009 | 0.160 | 6% | 1.4 |
> | Ever smoked | 145,975 (601) | 37.7% | 0.059 | 0.083 | 71% | 7.7 |
> | Screening-age smoker | 145,664 (600) | 22.2% | 0.066 | 0.120 | 55% | 6.5 |
> | E-cigarette (ever) | 146,076 (602) | 16.1% | 0.002 | 0.147 | 2% | 1.1 |

---

## §4.8 — RQ3: Regional factors

### Figure 13 caption

**Current:**
> Figure 13. Regional PM2.5, adult smoking rate and registry incidence against the weighted NHIS lung-cancer rate. Each dot is one region-year (n = 20: 4 regions × 5 years).

**Replace with:**
> Figure 13. Regional PM2.5, adult smoking rate and registry incidence against the weighted NHIS lung-cancer rate. Each dot is one region-year (n = 20: 4 regions × 5 years); each region-year rate is the survey-weighted rate of its adults (4,574 to 11,646 adults per region-year; 149,928 adults and 619 cases in total).

### "Why we read it cautiously" bullet: add at the end

**Current (last sentence of the bullet):**
> The two measures also differ in kind – lifetime history vs new cases per year.

**Replace with:**
> The two measures also differ in kind – lifetime history vs new cases per year. For PM2.5 and the regional smoking rate, the between-region correlation in Table 24 is exactly 0.00. This is a real value, not a computation error: both measures rank the four regions differently from the NHIS rate, the squared rank differences add up to exactly 10, and ρ = 1 − 6 × 10 / (4 × 15) = 0. With only four points, ρ can take just a few values (0, ±0.2, ±0.4, …, ±1), so this zero says nothing either way.

### "Do they add information beyond personal factors?" bullet (whole bullet)

**Current:**
> • Do they add information beyond personal factors? Mutual information on training rows (§5.5) puts all three regional columns far below the personal factors: about 0.5–0.6% of the target's information, against 10.2% for age. Their scores are above a shuffled-target noise level, but tiny, so at this resolution they add little.

**Replace with:**
> • Do they add information beyond personal factors? Mutual information on training rows (§5.5) puts all three regional columns far below the personal factors: about 0.5–0.6% of the target's information, against 10.2% for age. Their scores are above a shuffled-target noise level, but tiny. The three scores also measure the same thing. As discrete features, the regional values only say *which region-year* a person lives in: smoking_rate and lung_cancer_incidence take a different value in each of the 20 region-years, so their mutual information is exactly that of the region-year label (0.00017 nats each), and PM2.5, where two pairs of region-years share a value, is slightly lower (0.00014). So the data cannot tell whether pollution, regional smoking or registry incidence matters; at this resolution the three columns together add one small region-year signal.

### Table 24 [A3]

**Current:**

| Regional measure | All 20 region-years | Between 4 regions | Within regions (n = 20) |
|---|---|---|---|
| PM2.5 | -0.12 (p = 0.63) | 0.00 | 0.03 (p = 0.89) |
| Regional smoking rate | 0.09 (p = 0.71) | 0.00 | -0.35 (p = 0.13) |
| Registry incidence | 0.79 (p = 0) | 1.00 | 0.03 (p = 0.89) |

**Replace with** (caption unchanged, plus one sentence):

| Regional measure | All 20 region-years | Between 4 regions | Within regions (n = 20) |
|---|---|---|---|
| PM2.5 | −0.12 (p = 0.63) | 0.00 | 0.03 (p = 0.89) |
| Regional smoking rate | 0.09 (p = 0.71) | 0.00 | −0.35 (p = 0.13) |
| Registry incidence | 0.79 (p = 3.5 × 10⁻⁵) | 1.00 | 0.03 (p = 0.89) |

Caption addition: *"Between-region values use n = 4, so ρ can only take a few values; the 0.00 for PM2.5 and smoking rate is exact (see text)."*

---

## §4.9 — Table 25: RQ column

Only the RQ column changes. The p-values, Holm p-values and "Significant" column stay exactly as they are (source: `all_tests_holm.csv`, column `research_question`).

| Test (rows in current order) | Current RQ | New RQ |
|---|---|---|
| Smoking status (χ²) | RQ1 | RQ1 |
| Age (Mann–Whitney U) | RQ2 | **RQ2a** |
| COPD (χ²) | RQ2 | **RQ2b** |
| COPD adjusted for smoking (CMH) | RQ2 | **RQ2b** |
| Heart disease (χ²) | RQ2 | **RQ2b** |
| Hypertension (χ²) | RQ2 | **RQ2b** |
| Diabetes (χ²) | RQ2 | **RQ2b** |
| Asthma (χ²) | RQ2 | **RQ2b** |
| E-cigarette, ever (χ²) | RQ2 | **RQ2b** |
| Sex (χ²) | RQ2 | **RQ2c** |
| Race / ethnicity (χ²) | RQ2 | **RQ2c** |
| Education (χ²) | RQ2 | **RQ2c** |
| Income vs poverty line (χ²) | RQ2 | **RQ2c** |
| BMI category (χ²) | RQ2 | **RQ2c** |
| Self-rated general health (χ²) | RQ2 | **RQ2c** |
| PM2.5 vs regional rate (Spearman, n = 20) | RQ3 | RQ3 |
| Smoking rate vs regional rate (Spearman, n = 20) | RQ3 | RQ3 |
| Registry incidence vs regional rate (Spearman, n = 20) | RQ3 | RQ3 |

Totals: RQ1 1, RQ2a 1, RQ2b 7, RQ2c 6, RQ3 3 = 18 tests; 14 significant after Holm correction (unchanged).

---

## Section 5 — Feature Engineering (introduction)

**Current:**
> We hand the modelling phase 48 features, every one decided on training rows only, none of which can leak the answer.

**Replace with:**
> We hand the modelling phase 38 features, every one decided on training rows only, none of which can leak the answer, and none of which is an exact or near-exact linear combination of the others.

---

## §5.1 — Table 27, comorbidity_count row ("Why we built it")

**Current:**
> one number for overall health burden; it ranks third in mutual information (Fig. 14) and loads most strongly on the first principal component (§5.6)

**Replace with:**
> one number for overall health burden, used in the EDA (Tables 9–10, Fig. 4, Fig. 12); not a model feature, because it is the exact sum of the five condition flags (dropped by rule 4, §5.4)

---

## §5.2 — Encoding and imputation

### One-hot bullet

**Current:**
> • One-hot encoding for the nominal variables (sex, race/ethnicity, region, smoking status) and for the grouped ordinal ones (education, income, general health, BMI, age group). We chose one-hot over label encoding so the model never assumes a false order or false equal spacing between categories (for example that "Asian" is "greater" than "Black"). The file keeps every category of each group; a linear model will need to drop one reference category per group (§5.6).

**Replace with:**
> • One-hot encoding for the nominal variables (sex, race/ethnicity, region, smoking status) and for the grouped ordinal ones (education, income, general health, BMI, age group). We chose one-hot over label encoding so the model never assumes a false order or false equal spacing between categories (for example that "Asian" is "greater" than "Black"). A complete set of dummies always adds up to 1, which makes a linear model impossible to fit, so the model-ready file drops one **reference category** per group and every other dummy is read relative to it: race/ethnicity *White* (the largest group), education *High school / GED*, income *High* (four times the poverty line or more), BMI *Healthy*, general health *Excellent*, smoking status *Never*; for sex (*Male*) and region (*West*) the reference was already dropped by the near-duplicate rule. The choice is recorded for every dummy in `final_features.csv`.
>
> • Unknown answers and the reference group. *Unknown* categories that survive selection (education, BMI) are kept as their own dummy. For smoking status, the small *Unknown* dummy was dropped as a near-duplicate of `screening_age_smoker_unknown` (r = 0.96), so the 2.6% of adults with an unknown smoking status now fall into the *Never* reference group and are identified only by `screening_age_smoker_unknown`, which is therefore kept (§5.7). The same happens to the 0.04% with unknown general health (reference *Excellent*).

### Age bullet [A3]

**Current:**
> • Age is filled with the training median (54 years) plus an age_missing flag (0.24% of adults).

**Replace with:**
> • Age is filled with the training median (54 years); the age_missing flag created at the same time (0.24% of adults) is later removed by the near-constant filter (§5.4).

---

## §5.3 — Scaling

### Paragraph

**Current:**
> Five numeric columns use completely different units: age in years, a 0–5 count, PM2.5 in µg/m3, smoking prevalence in %, and incidence per 100,000.

**Replace with:**
> Four numeric features use completely different units: age in years, PM2.5 in µg/m³, smoking prevalence in %, and incidence per 100,000. (The 0–5 comorbidity count is no longer a model feature, §5.4.)

(The rest of the paragraph is unchanged.)

### Table 28

**Replace with** (caption unchanged):

| Column | Training mean | Training SD | Scaled test mean |
|---|---|---|---|
| age | 52.81 | 18.33 | −0.002 |
| pm25_mean | 7.289 | 0.763 | 0.006 |
| smoking_rate | 14.475 | 2.078 | −0.001 |
| lung_cancer_incidence | 67.31 | 12.50 | 0.019 |

---

## §5.4 — Feature selection

### Opening sentence

**Current:**
> We went from 67 candidates to 48 features, applying the filter pass from the course [17, 19] on training rows only, in this order:

**Replace with:**
> We went from 67 candidates to 38 features, applying the filter pass from the course [17, 19] on training rows only, in this order:

### Table 29

**Current caption:** "Table 29. Feature-selection rules and the features each rule removed (19 in total)."
**New caption:** "Table 29. Feature-selection rules and the features each rule removed (29 in total)."

Rows 1, 2, 2b, 3 and "Never candidates" are unchanged. **Insert after row 3:**

| Rule | Dropped | Features |
|---|---|---|
| 4a. Exact linear combination of other features (VIF = ∞) | 1 | comorbidity_count (= copd + asthma + hypertension + heart_disease + diabetes; Spearman 0.79 with hypertension); kept in the clean dataset for the EDA |
| 4b. Reference category of each one-hot group | 6 | race_ethnicity_White, education_High school / GED, income_group_High, bmi_category_Healthy, general_health_Excellent, smoking_status_Never (sex_Male and region_West, already dropped by rule 3, serve as the references of sex and region) |
| 4c. Near-collinear block | 3 | region_Midwest, region_Northeast, region_South: the three regional columns already encode region × year, and region dummies + lung_cancer_incidence were near-collinear (VIF 24–39, one near-zero PCA eigenvalue) |

### New paragraph after the existing one ("The near-constant rule removes …")

> **Rule 4: dependencies spread over several columns (VIF).** Rules 1–3 compare columns one pair at a time, so they cannot see a column that is a combination of *several* others. The variance inflation factor can: VIF_j = 1 / (1 − R²_j), where R²_j comes from regressing feature j on all the other features (with an intercept) on the training rows. VIF = 1 means feature j is unrelated to the others, VIF > 10 is the usual alarm level, and VIF = ∞ means an exact linear dependency. After rules 1–3, 28 of the 48 features had an infinite VIF and 42 a VIF above 10 (Table 29a; `reports/tables/vif_before.csv`). The causes were the comorbidity count (the sum of five flags) and the four complete one-hot groups (race, education, income, BMI), each of which summed to 1. General health and smoking status almost did (VIF 100–643, because only their small Unknown dummies had been removed), and the region dummies almost determined the registry incidence (VIF 24–39). Rule 4 removes these dependencies (comorbidity_count, one reference category per group, the three region dummies). After it, the largest VIF is 7.5, nothing exceeds 10, and the PCA has no near-zero component left (§5.6; `vif_after.csv`). The two features closest to the limit are screening_age_smoker_unknown (7.49) and ever_ecigarette_unknown (7.45). They overlap because both flag the adults who skipped the smoking questions, and they are kept because the first is the only marker of an unknown smoking status. The screening_age_smoker flag, built from age and smoking, has a VIF of only 2.2. None of these drops removes information: an exact dependency is a direction with zero variance, and the region dummies added almost nothing beyond the 20 region-year values of the regional columns.
>
> **Table 29a.** Variance inflation factor of every feature after rules 1–3 (48 features) and after rule 4 (38 features), on training rows. ∞ = exact linear dependency.
>
> | Feature | VIF after rules 1–3 | VIF after rule 4 |
> |---|---|---|
> | copd, asthma, hypertension, heart_disease, diabetes | ∞ | 1.18, 1.06, 1.41, 1.14, 1.17 |
> | comorbidity_count | ∞ | dropped (4a) |
> | race_ethnicity_* (7 dummies) | ∞ | 1.01–1.24 (White dropped, 4b) |
> | education_* (6 dummies) | ∞ | 1.02–1.67 (High school / GED dropped, 4b) |
> | income_group_* (4 dummies) | ∞ | 1.32–1.42 (High dropped, 4b) |
> | bmi_category_* (5 dummies) | ∞ | 1.04–1.59 (Healthy dropped, 4b) |
> | general_health_* (5 dummies) | 101–643 | 1.34–1.91 (Excellent dropped, 4b) |
> | smoking_status_* (3 dummies) | 113–259 | Current 1.72, Former 2.03 (Never dropped, 4b) |
> | region_Midwest, region_South, region_Northeast | 39.0, 30.5, 24.0 | dropped (4c) |
> | lung_cancer_incidence | 33.6 | 1.38 |
> | ever_ecigarette_unknown | 15.9 | 7.45 |
> | screening_age_smoker_unknown | 13.1 | 7.49 |
> | smoking_rate, pm25_mean | 2.74, 2.44 | 1.34, 1.14 |
> | screening_age_smoker | 2.18 | 2.18 |
> | age, ever_ecigarette, sex_Female | 1.70, 1.32, 1.07 | 1.70, 1.32, 1.07 |
> | **Maximum (excluding ∞)** | 643 | **7.49** |

---

## §5.5 — Relevance check with mutual information

### Second paragraph (whole paragraph)

**Current:**
> With the correct estimator, the ranking agrees with the EDA: age (0.0027), COPD (0.0026) and comorbidity count (0.0020) lead, followed by the screening-age-smoker flag, the smoking-status dummies and self-rated health. The raw values look small, but MI is bounded by the uncertainty of the target, which is tiny for a 0.41% outcome (an entropy of 0.0268 nats). Expressed as a share of that maximum, age carries 10.2% of the available information and COPD 9.8%, against 0.5–0.6% for each regional column (Table 30). To check that these scores are not noise, we shuffled the target 100 times and recomputed MI: every listed feature scores above the 95th percentile of its shuffled scores, but the regional columns only just – their real information is a small fraction of the personal factors'. The three regional columns do not reach the top 15, consistent with RQ3.

**Replace with:**
> With the correct estimator, the ranking agrees with the EDA: age (0.0027) and COPD (0.0026) lead, followed by the screening-age-smoker flag (0.0016), poor self-rated health, former smoking and hypertension (Fig. 14; 119,942 training rows, 495 cases). The raw values look small, but MI is bounded by the uncertainty of the target, which is tiny for a 0.41% outcome (an entropy of 0.0268 nats). Expressed as a share of that maximum, age carries 10.2% of the available information and COPD 9.8%, against 0.5–0.6% for each regional column (Table 30). To check that these scores are not noise, we shuffled the target 100 times and recomputed MI: every listed feature scores above the 95th percentile of its shuffled scores, but the regional columns only just – their real information is a small fraction of the personal factors'. The three regional columns now occupy the last three places of the top 15 (Fig. 14). Their MI values did not change; they moved up only because ten features left the set. smoking_rate and lung_cancer_incidence have *identical* scores and null percentiles, and PM2.5 is slightly lower, for a structural reason. As discrete features, the regional columns only label the region-year a person lives in: the first two take a different value in each of the 20 region-years, so both equal the mutual information of the region-year label itself (0.000173 nats). PM2.5 has two ties (6.31 µg/m³ in the West in 2019 and 2023; 7.43 in the South in 2022 and the West in 2021), so it merges two pairs of region-years and scores slightly less (0.000145; `regional_mi_identity.csv`). The three regional scores therefore measure one thing, not three independent signals, which is consistent with RQ3.
>
> (Note on the earlier ranking: the tie between screening_age_smoker and smoking_status_Never at 0.00158 in the previous version of Table 30 was a coincidence of rounding, not a duplicate. The two 2×2 tables differ – the screening flag marks 25,794 training adults with 307 cases, never-smoking 72,861 adults with 90 cases – and their MI values only differ in the eighth decimal: 0.00158140 vs 0.00158137 nats, 3 × 10⁻⁸ apart.)

### Table 30

**Replace with** (caption unchanged; comorbidity_count and smoking_status_Never are no longer features, replaced by hypertension and smoking_status_Former):

| Feature | MI (nats) | Share of target entropy | Null, 95th percentile | Permutation p |
|---|---|---|---|---|
| age | 0.00274 | 10.2% | 0.00080 | 0.01 |
| copd | 0.00263 | 9.8% | 0.00002 | 0.01 |
| hypertension | 0.00072 | 2.7% | 0.00002 | 0.01 |
| screening_age_smoker | 0.00158 | 5.9% | 0.00001 | 0.01 |
| smoking_status_Former | 0.00109 | 4.1% | 0.00001 | 0.01 |
| pm25_mean | 0.00014 | 0.5% | 0.00012 | 0.02 |
| smoking_rate | 0.00017 | 0.6% | 0.00013 | 0.01 |
| lung_cancer_incidence | 0.00017 | 0.6% | 0.00013 | 0.01 |

### Figure 14 caption

**Current:**
> Figure 14. Mutual information between each feature and the target on the training set (top 15).

**Replace with:**
> Figure 14. Mutual information between each of the 38 final features and the target on the training rows (top 15; n = 119,942 adults, 495 cases). The three regional columns (orange) take the last three places; their nearly equal values reflect one shared region-year signal (§5.5).

---

## §5.6 — Feature extraction with PCA

### Opening paragraph

**Current:**
> We tested whether Principal Component Analysis could compress the 48 selected features without losing much information.

**Replace with:**
> We tested whether Principal Component Analysis could compress the 38 selected features without losing much information.

### Figure 15 caption

**Current:**
> Figure 15. PCA scree plot: explained variance per component (bars) and cumulative variance (line). 34 of 48 components are needed to keep 95% of the variance.

**Replace with:**
> Figure 15. PCA scree plot (fitted on the 119,942 training rows): explained variance per component (bars) and cumulative variance (line). 32 of 38 components are needed to keep 95% of the variance.

### Table 31

**Replace with** (caption unchanged):

| Quantity | Value |
|---|---|
| Variance explained by PC1 / PC2 / PC3 | 7.3% / 5.4% / 4.9% (2.6% expected per component if all features were unrelated) |
| Components needed for 80% / 90% / 95% of the variance | 24 / 29 / 32 of 38 |
| Kaiser rule (eigenvalue > 1) | 19 components |
| Components with almost zero variance (eigenvalue < 0.01) | 0 (there were 8 with the previous 48 features: the dependencies removed by rule 4, §5.4) |
| Variance of the test rows kept by the first 32 components | 95.8% (the PCA generalises; it never saw the test rows) |
| PC1 – "age and chronic-disease burden" | hypertension +0.59, screening_age_smoker +0.57, age +0.57, diabetes +0.46, COPD +0.45, heart disease +0.42 |
| PC2 – "skipped the smoking questions" | screening_age_smoker_unknown +0.86, ever_ecigarette_unknown +0.86 |
| PC3 – "older age vs current smoking and regional smoking level" (mixed) | age +0.50, smoking_rate −0.42 (next below 0.40: current smoker −0.39, e-cigarette −0.38, registry incidence −0.36, PM2.5 −0.35) |

### Figure 16 caption

**Current:**
> Figure 16. Training adults projected on the first two principal components: all 495 training cases (orange) and a random 8,000 non-cases (grey).

**Replace with:**
> Figure 16. Training adults (n = 119,942) projected on the first two principal components: all 495 training cases (orange) and a random 8,000 non-cases (grey). The separate cloud at the top (high PC2) is the ~2.8% of adults who skipped the smoking questions.

### Paragraph after Figure 16

**Current:**
> Lung-cancer cases sit on the high side of PC1 (mean score +2.89 vs −0.01 for other adults), which is the age-and-disease axis, but the two groups overlap heavily: two components cannot separate a rare outcome.

**Replace with:**
> Lung-cancer cases sit on the high side of PC1 (mean score +2.53 vs −0.01 for other adults), which is the age-and-disease axis, but the two groups overlap heavily: two components cannot separate a rare outcome.

### Decision paragraph (whole paragraph)

**Current:**
> Decision: we keep the original features. Keeping 95% of the variance would still need 34 components – only 29% fewer columns – because the information is spread over many dimensions with no dominant one, which agrees with the weak correlations of Fig. 12. Interpretability also matters for a health question: … The eight near-zero directions are a useful warning: a future linear model must drop one reference category per one-hot group to avoid perfect multicollinearity, and should not include both the region dummies and the regional values without checking their collinearity.

**Replace with:**
> Decision: we keep the original features. Keeping 95% of the variance would still need 32 components – only 16% fewer columns – because the information is spread over many dimensions with no dominant one, which agrees with the weak correlations of Fig. 12c and the low VIFs of Table 29a. Interpretability also matters for a health question: an odds ratio for "COPD" means something to a clinician, a weight on "PC4" does not. Finally, most of our columns are 0/1 indicators, for which PCA, built for continuous data, is less natural (a method for mixed data, such as factor analysis of mixed data, would be the better tool if extraction were needed). The PCA is kept as an option to compare in the modelling phase, especially for distance-based methods such as k-nearest neighbours or clustering, which suffer most from many dimensions. The previous version of the feature set had eight near-zero directions. They are gone because rule 4 removed the dependencies behind them (the comorbidity count, one reference category per one-hot group, the region dummies), and that is also why the count of components dropped: an exact dependency is a zero-variance direction that carries no information.

---

## §5.7 — The final feature set

### Table 32

**Current caption:** "Table 32. The 48 final features, by group (full list in Appendix B)."
**New caption:** "Table 32. The 38 final features, by group (full list in Appendix B)."

| Group | Features | Count |
|---|---|---|
| Demographic | age (scaled); sex_Female (reference: Male); race/ethnicity: 6 dummies (reference: White) | 8 |
| Socio-economic | education: Bachelor, Graduate, Less than high school, Some college, Unknown (reference: High school / GED); income vs poverty line: Below poverty, Low, Middle (reference: High) | 8 |
| Smoking and nicotine | smoking status: Current, Former (reference: Never); screening_age_smoker (+ unknown flag); ever_ecigarette (+ unknown flag) | 6 |
| Health | COPD, asthma, hypertension, heart disease, diabetes; general health: Fair, Good, Poor, Very good (reference: Excellent); BMI: Obese, Overweight, Underweight, Unknown (reference: Healthy) | 13 |
| Regional context | pm25_mean, smoking_rate, lung_cancer_incidence (scaled); region itself is encoded by these three (region dummies dropped, §5.4) | 3 |

### First bullet

**Current:**
> • It carries what the EDA found to matter – age, smoking, COPD and overall health burden – in model-ready form, with no missing values.

**Replace with:**
> • It carries what the EDA found to matter – age, smoking, COPD and the other chronic conditions (whose sum, comorbidity_count, stays in the clean dataset for the EDA only) – in model-ready form, with no missing values and no exact or near-exact multicollinearity (maximum VIF 7.5).

### New paragraph after "Features we are cautious about …" (A2)

> **Reading the dummies.** Every dummy is read against its group's reference category (Table 32). For smoking, the reference group *Never* also contains the 2.6% of adults whose smoking status is unknown. Their own dummy was dropped as a near-duplicate of `screening_age_smoker_unknown` (r = 0.96, rule 3), so that flag is the **only** feature that tells a model these adults are not known never-smokers. It must stay in any model that uses the smoking dummies, and an adjusted odds ratio for *Current* or *Former* smoking should be described as "vs never or unknown, holding the unknown flag fixed".

---

## §6.1 — Table 33, RQ2 row split into RQ2a / RQ2b / RQ2c

**Current RQ2 row:**
> RQ2 – Age, conditions, socio-demographics | Age: median 72 vs 54 years, rank-biserial r = 0.59 (a random case is older than a random non-case 79.5% of the time); age and smoking act together (2.15–2.37% of ever-smokers aged 75+ vs 0.48% of never-smokers the same age). Conditions: COPD is the strongest association (odds ratio 15.1; among adults with known smoking status the crude 14.8 falls to 9.3 after adjusting for smoking, consistently across smoking groups, and to 5.7 after adjusting for smoking and age group); heart disease, hypertension and diabetes are moderate (2.3–4.6), partly through age; asthma is negligible; e-cigarette use shows nothing. Socio-demographics: a clear education and income gradient, partly explained by smoking; the highest rates in underweight adults and those in poor health, most likely reverse causation; no difference by sex; race/ethnicity differences that largely follow age and smoking profiles.

**Replace with three rows** (the existing answer text, split; the RQ1 row is unchanged; the COPD sentence also carries the [A3] subset fix):

| RQ | Answer |
|---|---|
| RQ2a – Age | Median 72 vs 54 years, rank-biserial r = 0.59 (a random case is older than a random non-case 79.5% of the time); age and smoking act together (2.15–2.37% of ever-smokers aged 75+ vs 0.48% of never-smokers the same age). |
| RQ2b – Chronic conditions and e-cigarettes | COPD is the strongest association (odds ratio 15.1; among adults with known smoking status the crude 14.8 falls to 9.3 after adjusting for smoking, consistently across smoking groups, and among those whose age is also known to 5.7 after adjusting for smoking and age group); heart disease, hypertension and diabetes are moderate (2.3–4.6), partly through age; asthma is negligible; e-cigarette use shows nothing. |
| RQ2c – Socio-demographic and health factors | A clear education and income gradient, partly explained by smoking; the highest rates in underweight adults and those in poor health, most likely reverse causation; no difference by sex; race/ethnicity differences that largely follow age and smoking profiles. |

## §6.1 — Table 33, RQ3 row

**Current:**
> The survey rate agrees with the official registry incidence across regions (ρ = 0.79 over 20 region-years), but the agreement comes from the four regional averages (mainly the low West), not from year-to-year changes within regions (ρ = 0.03): consistent with a valid target, not proof of it. PM2.5 and regional smoking rates show no detectable association, and regional features carry far less information than personal ones (0.5–0.6% vs 10.2% of the target's information for age).

**Replace with:**
> The survey rate agrees with the official registry incidence across regions (ρ = 0.79 over 20 region-years, p = 3.5 × 10⁻⁵), but the agreement comes from the four regional averages (mainly the low West), not from year-to-year changes within regions (ρ = 0.03): consistent with a valid target, not proof of it. PM2.5 and regional smoking rates show no detectable association. Regional features carry far less information than personal ones (0.5–0.6% vs 10.2% of the target's information for age), and the three regional columns carry one shared region-year signal, so the data cannot separate the effect of pollution, regional smoking and registry incidence.

---

## §6.2 — Table 34

Row 6, "Observed":
- **Current:** "18 tests: RQ1 (1), RQ2 (14), RQ3 (3)"
- **New:** "18 tests: RQ1 (1), RQ2a (1), RQ2b (7), RQ2c (6), RQ3 (3)" (`success_criteria_check.csv` now checks that each of the five labels has at least one test)

Row 5, "Observed":
- **Current:** "imputation, selection, scaling and PCA fitted on training rows; scaler asserted in code"
- **New:** "imputation, selection (incl. the VIF step), scaling and PCA fitted on training rows; scaler asserted in code"

Row 7, "Observed": **unchanged, "67 of 67 candidates".** The success-criteria check in the code now prints exactly this (`success_criteria_check.csv`: "67 of 67 candidates"); 38 are KEEP and 29 DROP.

---

## §7.1 — Next steps, item 2

**Current:**
> 2. Models. Class-weighted logistic regression – with one reference category dropped per one-hot group – for interpretable adjusted odds ratios, then tree-based models such as random forest and gradient boosting.

**Replace with:**
> 2. Models. Class-weighted logistic regression on the 38 features – the reference categories are already dropped and documented, and no feature has a VIF above 7.5 – for interpretable adjusted odds ratios, then tree-based models such as random forest and gradient boosting.

(The rest of the item is unchanged.)

---

## Appendix A — Reproducibility

### Requirements sentence

**Current:**
> Requirements: Python 3.11, 3.12 or 3.13 and the packages listed in requirements.txt (pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, statsmodels).

**Replace with:**
> Requirements: Python 3.11, 3.12 or 3.13 and the packages listed in requirements.txt (pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, statsmodels, and jupyter, ipykernel and nbconvert for the notebook).

### Table 36

**Replace with** (caption unchanged; one row added):

| Step | Script | Output |
|---|---|---|
| 0 | download_data.py | raw NHIS, EPA, BRFSS and USCS files in data/raw/ (skipped when already present) |
| 1 | build_regional_data.py | data/clean/regional_context.csv (20 region-years) and regional_report.txt |
| 2 | build_clean_dataset.py | clean, model-ready and scaled datasets; data dictionary; feature decisions; VIF tables (vif_before.csv, vif_after.csv); outlier table; before/after summary; cleaning report; Fig. 3 |
| 3 | eda_analysis.py | Figures 1–2, 4–14, 12b and 12c; test, correlation and association tables in reports/tables/; reports/eda_results.txt with every test (with its denominator) and the success-criteria check |
| 4 | feature_extraction_pca.py | Figures 15–16; reports/pca_results.txt; PCA tables |
| 5 | supplementary_analysis.py | dataset composition (Tables 2a–2c), population profile (Tables 5–10), per-year rates, age-adjusted odds ratios, regional within/between check, screening-rule baseline, mutual-information check, survey-design check and the criterion-5 assertion; reports/supplementary_results.txt |
| – | notebooks/lung_cancer_project.ipynb | the whole project section by section; every code cell calls a function of the scripts above (not run by run_all.py) |

### Paragraph after Table 36 (whole paragraph) [A3: hash]

**Current:**
> We tested this on a fresh run: the full pipeline takes about 105 seconds, finishes without errors or warnings, and the regenerated results match the committed ones. The cleaned file data/clean/nhis_lung_ cancer_clean.csv (149,928 × 26) has the SHA-256 prefix 5daf9b80b848efe9, so anyone can check they have the same data. All numbers in this report come from reports/eda_results.txt, reports/ pca_results.txt, reports/supplementary_results.txt, reports/tables/ and data/clean/ cleaning_report.txt; Figure N of this report is the file reports/figures/figNN_*.png with the same number.

**Replace with:**
> We tested this on a fresh run: the full pipeline takes about 100 seconds, finishes without errors or warnings, and the regenerated results match the committed ones. The cleaned file data/clean/nhis_lung_cancer_clean.csv (149,928 × 26) has the SHA-256 prefix 5daf9b80b848efe9 with Unix (LF) line endings, as stored in git; pandas on Windows writes Windows (CRLF) line endings, which give the prefix 2b25cb8142487dd4 for the same data. Every script can also be imported: its steps are functions, and `main()` runs them in order. The notebook notebooks/lung_cancer_project.ipynb uses this to walk through the six CRISP-DM sections by calling those functions, with its outputs saved. By default (REBUILD = False) it reuses the built files; two consecutive runs were checked to leave every output byte-identical to run_all.py's. All numbers in this report come from reports/eda_results.txt, reports/pca_results.txt, reports/supplementary_results.txt, reports/tables/ and data/clean/cleaning_report.txt; Figure N of this report is the file reports/figures/figNN_*.png with the same number (Figures 12b and 12c are fig12b_*.png and fig12c_*.png).

---

## Appendix B — Final feature list

### Introductory sentence

**Current:**
> The 48 features of nhis_lung_cancer_model_ready.csv, in file order. Scaled columns are standardised in nhis_lung_cancer_model_ready_scaled.csv; all other columns are 0/1 indicators.

**Replace with:**
> The 38 features of nhis_lung_cancer_model_ready.csv, in file order. Scaled columns are standardised in nhis_lung_cancer_model_ready_scaled.csv; all other columns are 0/1 indicators. Reference categories (not in the file): race/ethnicity White, education High school / GED, income High, BMI Healthy, general health Excellent, smoking status Never (which also contains unknown smoking status, flagged by feature 3), sex Male; region is represented by features 11–13.

### Table 37 — "The 38 final features."

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | age (scaled) | 20 | race_ethnicity_Other or multiple |
| 2 | screening_age_smoker | 21 | education_Bachelor |
| 3 | screening_age_smoker_unknown | 22 | education_Graduate |
| 4 | ever_ecigarette | 23 | education_Less than high school |
| 5 | ever_ecigarette_unknown | 24 | education_Some college |
| 6 | copd | 25 | education_Unknown |
| 7 | asthma | 26 | income_group_Below poverty |
| 8 | hypertension | 27 | income_group_Low |
| 9 | heart_disease | 28 | income_group_Middle |
| 10 | diabetes | 29 | general_health_Fair |
| 11 | pm25_mean (scaled) | 30 | general_health_Good |
| 12 | smoking_rate (scaled) | 31 | general_health_Poor |
| 13 | lung_cancer_incidence (scaled) | 32 | general_health_Very good |
| 14 | sex_Female | 33 | bmi_category_Obese |
| 15 | race_ethnicity_AIAN | 34 | bmi_category_Overweight |
| 16 | race_ethnicity_AIAN and other | 35 | bmi_category_Underweight |
| 17 | race_ethnicity_Asian | 36 | bmi_category_Unknown |
| 18 | race_ethnicity_Black | 37 | smoking_status_Current |
| 19 | race_ethnicity_Hispanic | 38 | smoking_status_Former |

Removed from the previous Table 37: comorbidity_count, race_ethnicity_White, region_Midwest, region_Northeast, region_South, education_High school / GED, income_group_High, general_health_Excellent, bmi_category_Healthy, smoking_status_Never.

---

## Pre-existing inconsistencies (A3)

We compared every number in the report with `eda_results.txt`, `pca_results.txt`, `supplementary_results.txt`, `reports/tables/*.csv` and `data/clean/*` as they stood before these changes. The list below gives every mismatch beyond rounding. Each one is fixed in place above. Everything else checked matched, including Tables 3–10, 13–26, 33–35, the §2.5 counts, the e-cigarette power figure (OR ≈ 1.34, recomputed) and the smallest expected count of 4.0 before the race groups were merged (3.97, recomputed).

| # | Where | Report says | Correct | Source |
|---|---|---|---|---|
| 1 | §4.5, "How we read it" | "the 15.1 of Table 13" | "the 15.1 of Table 20" (Table 13 is the missingness table) | `conditions_odds_ratios.csv` (Table 20) |
| 2 | §2.5.3, Smoking bullet | former smokers' mean age 59.8 | 59.7 (Table 8 already shows 59.7) | `population_profile.csv`: 59.7463; `supplementary_results.txt` printed 59.75, which was rounded again to 59.8. The script now prints 1 decimal from the raw value |
| 3 | §4.8, Table 24 | registry incidence "0.79 (p = 0)" | 0.79 (p = 3.5 × 10⁻⁵) | `regional_within_between.csv` was written with 4 decimals (3.49e-05 → 0.0000); p-values are now written unrounded |
| 4 | Abstract, paragraph 1 | COPD in "6%" of the other adults | 5% | `cases_vs_noncases.csv`: 5.46% (Table 10 shows 5.5%, rounded a second time to 6%) |
| 5 | Abstract, §4.5, Table 33, README | "the crude 14.8 falls to 9.3 after adjusting for smoking, and to 5.7 after adjusting for smoking and age together" | each number is right, but the chain mixes two subsets: 14.8 → 9.3 is on the 145,897 adults with known COPD and smoking (Table 21); 5.7 is on the 145,587 also with known age, where the matching steps are 14.7 → 9.2 → 5.7 (Table 22) | `copd_stratified_by_smoking.csv`, `copd_adjusted.csv` |
| 6 | Appendix A | SHA-256 prefix 5daf9b80b848efe9 | correct only for LF line endings (the file in git); a run on Windows writes CRLF → 2b25cb8142487dd4 for identical data | `sha256sum` of the committed blob vs the regenerated file |
| 7 | §5.2, age bullet | "plus an age_missing flag" (implies it is a feature) | the flag is created and then dropped by the near-constant rule (99.76% of training rows are 0) | `final_features.csv`, Table 29 |
| 8 | §2.5.1 vs Table 10 | "471 of the 619 cases (76.1%)" vs "Aged 65 or older 76.2%" | both right, different denominators: 471 / 619 (all cases) vs 471 / 618 (cases with a known age). Suggest adding "(76.2% of the 618 cases with a known age)" to §2.5.1 | `population_profile.csv`, `cases_vs_noncases.csv` |

Not a report error but corrected in the repository: README section 7 said "only COPD was adjusted for a confounder"; smoking is also adjusted for age group (report §4.3, §4.10 and §6.3 were already right). One figure in the report could not be re-checked: the §5.5 statement that an earlier version of the code gave the regional columns an MI of 0.0016–0.0022. The current code no longer produces that version.

---

## Figures regenerated

All PNGs below were regenerated (the report reads them from `reports/figures/`). Fig 12b and 12c are new; the others have n (and weighting) added to the title, or a layout fix.

| Figure | What changed |
|---|---|
| Fig 1 | title states n = 149,928; thousands separators on the axis |
| Fig 2 | title states "% of all 149,928 adults" |
| Fig 4, 5 | title states n |
| Fig 6, 7, 8, 9, 10, 11 | title states the subset n (and cases); Fig 11 p-values printed as 0.16, 1.6e-14, … instead of 2e-01, 2e-14 |
| Fig 8, 12 | grid lines removed from the heatmap cells; Fig 12 title states n |
| **Fig 12b (new)** | association matrix of all 24 analysis variables + target |
| **Fig 12c (new)** | Spearman matrix of the 38 final features + target (training rows) |
| Fig 13 | title states n = 20 region-years and the 149,928 adults behind them |
| Fig 14 | 38 final features, n = 119,942 training rows; regional columns now in the top 15 |
| Fig 15 | 32 of 38 components; "95%" label moved off the curve; training n |
| Fig 16 | new projection (38 features); legend moved off the data; training n |
| Fig 3 | unchanged |
