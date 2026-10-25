# Presentation source and speaker notes

## Slide 1: The file’s structure dominates its label associations

- 22,457 rows, 17 columns; source is undocumented.
- After removing the ordered block, smoking OR is 1.00.
- Interpretation: integrity audit and descriptive analysis only.

**Speaker notes:** Open with the single takeaway. This is a review of one supplied file, not a population study. State that no model was fitted and that the checksum mismatch is explicitly handled.

## Slide 2: A smoking checkbox cannot reproduce screening history

- USPSTF 2021 [1] uses ages 50–80.
- It also requires at least 20 pack-years and a quit-time criterion of 15 years.
- This file has no dose, duration, or quit-time fields; no eligibility inference is possible.

**Speaker notes:** Explain the reference criteria and immediately contrast them with the available fields. This is context for limitations, not an assessment of any person.

## Slide 3: The objective is to test whether findings survive integrity checks

- RQ1: characterize ordering, duplicates, and conflicting records.
- RQ2–RQ3: quantify associations and their sensitivity to cleaning.
- RQ4–RQ5: describe representativeness limits and justify candidate features.
- Success means reproducible checks, transparent uncertainty, and traceable claims—not model accuracy.

**Speaker notes:** State that the project is descriptive and integrity-focused. The five research questions are fully listed in the report; summarize the paired questions here rather than reading them verbatim.

## Slide 4: The checksum does not match the project contract

- Observed: 22,457 × 17; 0 null cells.
- Expected SHA-256 prefix: 4ad2333067fcb0a9.
- Observed prefix: 1e0d557990cd7533.
- Recorded-label code 1 share: 82.1%; not population prevalence.

**Speaker notes:** Describe the mismatch plainly. The user approved continuing with this current snapshot under an explicit override; the expected checksum was not changed.

## Slide 5: Several meanings needed for interpretation are unknown

- Dataset source, sampling frame, collection process, and licence: unknown.
- Meaning and generation of the recorded label: unknown.
- GENDER code mapping: unknown.
- The project README only supplied a title; redistribution needs a licence check.

**Speaker notes:** Do not speculate about where the file came from or what the label means. Point to the data card and open questions.

## Slide 6: The leading rows form an all-negative block

- First 1,455 rows are coded negative.
- Smoking YES share: 29.3% in the block vs 56.0% in later negatives.
- The plot shows row order, not a documented timeline.

**Figure references:** Figure 1: row order, Figure 2: YES-share profiles

**Speaker notes:** Figure 1: describe row index, rolling positive-label share, and the boundary. Figure 2: compare YES shares across leading negatives, later negatives, and later positives. Both are descriptive; neither establishes provenance.

## Slide 7: Duplicate patterns and ID order are not harmless

- 2,573 exact duplicate rows; 133 conflicting feature patterns.
- Maximum pattern multiplicity: 86.
- ID-only AUC: 0.67 — an audit statistic, not model performance.
- The source and record-generation process remain unknown.

**Figure references:** Duplicate multiplicity and conflicting patterns

**Speaker notes:** Describe the duplicate-multiplicity distribution and conflicting-pattern counts. Explain why duplicate policies are sensitivity analyses. The ID-only AUC is an integrity statistic; patient_id remains excluded from feature sets.

## Slide 8: Four cleaning views expose policy sensitivity

- A raw: 22,457; B exact-deduplicated: 19,884.
- C no conflicts: 19,618; D no leading block: 19,781.
- Recorded-positive share ranges from 82.1% to 87.1%.
- View D filters B by the original raw row index.

**Figure references:** Figure 5: four cleaning views

**Speaker notes:** Figure 5 shows retained row counts and recorded-label-1 rates for views A through D. Explain that these are sensitivity policies, not a known ground-truth repair.

## Slide 9: After deduplication, pairwise feature relations are weak

- Age: 21–87, mean 62.1.
- Maximum absolute pairwise correlation: 0.022.
- First PCA component: 7.0%; flat reference: 6.7%.
- These are descriptive diagnostics, not proof of independence.

**Figure references:** Figure 3: feature correlations, Figure 4: AGE profiles

**Speaker notes:** Figure 3: describe the feature-pair axes, correlation colour, and clipped range. Figure 4: compare AGE across the leading block, later negatives, and later positives, and show the label-wise and AGE_BAND summaries. These are descriptive views.

## Slide 10: Five pre-specified tests separate effect from significance

- H1 smoking: odds ratio, confidence interval, chi-square, permutation check.
- H2 age: Mann–Whitney and CLES; H3 symptom count: Spearman rho.
- H4 feature family: Benjamini–Hochberg correction; H5: source-shift audit.
- View D H4 minimum raw p = 0.033; minimum q = 0.458.

**Speaker notes:** Describe the pre-specified tests and distinguish p-values, effect sizes, and adjusted q-values. The analysis does not test causal hypotheses.

## Slide 11: The smoking association moves toward the null after cleaning

- View A OR 1.49 (1.39–1.59).
- View D OR 1.00 (0.92–1.08).
- View D H4 minimum q = 0.46.
- The file does not establish a population or causal conclusion.

**Figure references:** Figure 7: smoking odds ratio by view, Figure 6: Cramer's V by feature

**Speaker notes:** Start with Figure 7: describe view, smoking odds ratio, 95% intervals, and the OR=1 reference. Then use Figure 6 to show the feature-level Cramer's V comparison from A to D. These are associations in this file, not population conclusions.

## Slide 12: Feature construction is transparent, not evidence of prediction

- F0: 15 raw features; F1: 19 with selected descriptors.
- F2: 5 domain-prior candidates.
- Age scaling must be fitted inside training folds in any future model.
- The smoking-by-age feature is a proxy, not screening eligibility.

**Speaker notes:** Summarize the three sets and the rationale for retaining F2 as a candidate. State that univariate screening did not select features and no model was fitted.

## Slide 13: The file supports an integrity finding, not a clinical claim

- After removing the leading block, smoking OR is 1.00.
- After BH correction in D, the minimum q is 0.46.
- Provenance, label meaning, representativeness, and external validity are unresolved.
- All conclusions are limited to this supplied file.

**Speaker notes:** Answer the central question directly. Explain that small effects after structural cleaning are about this file only and do not negate external evidence.

## Slide 14: Next steps are planned, not completed

- Verify source, licence, checksum, label definition, and sampling frame.
- If authorized, plan grouped/source-aware validation on independent data.
- Select metrics and calibration checks with the intended use in mind.
- Clean-clone testing and timed presentation rehearsal remain to be done.

**Speaker notes:** Use future tense for all uncompleted work. Do not claim that external validation, model training, USB backup, or rehearsal has occurred.

## Hidden backup 1: Backup: sensitivity estimates by view

- A_raw: OR 1.49, 95% CI 1.39–1.59, p 4.45e-30.
- B_dedup: OR 1.03, 95% CI 0.95–1.12, p 0.521.
- C_dedup_noconflict: OR 1.04, 95% CI 0.96–1.14, p 0.304.
- D_dedup_nolead: OR 1.00, 95% CI 0.92–1.08, p 0.911.

**Speaker notes:** Backup sensitivity values. Keep interpretation descriptive and point to the integrity audit.

## Hidden backup 2: Backup: feature-family BH results

- A_raw: minimum p 2.12e-103, minimum q 0.000, maximum V 0.144.
- B_dedup: minimum p 0.029, minimum q 0.365, maximum V 0.016.
- C_dedup_noconflict: minimum p 0.006, minimum q 0.079, maximum V 0.020.
- D_dedup_nolead: minimum p 0.033, minimum q 0.458, maximum V 0.015.

**Speaker notes:** Backup multiple-testing family summary. The H4 tests are descriptive, not evidence of causation.

## Hidden backup 3: Backup: detectable risk-difference approximation

- View D group sizes: 11,080 smoking-code YES and 8,701 NO.
- Alpha 0.05; target power 0.8.
- Approximate minimum detectable risk difference: 0.0135.
- Planning calculation only; its assumptions are not established by the file.

**Speaker notes:** Explain that this is a planning calculation under assumptions, not evidence that a smaller association is absent.

## Hidden backup 4: Backup: optional leakage model was not run

- The guide proposes a raw-versus-deduplicated cross-validation experiment.
- The project scope excludes model fitting; no AUC from that experiment is reported.
- Current leakage evidence comes from row order, duplicates, and ID-only AUC.

**Speaker notes:** This optional experiment is intentionally not run under the governing no-model-training constraint. Do not quote an unmeasured raw or cleaned model score.
