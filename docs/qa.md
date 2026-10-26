# Presentation Q&A preparation

## Q1. Why not train a model and report accuracy?

The project scope excludes model fitting. An always-positive rule reaches 82.1% accuracy just from the label balance, so it is not a useful success metric. The raw-file associations also change after integrity treatment.

## Q2. What is the clinical claim?

None. This report describes coded associations in one file and is a research prototype, not a diagnostic or screening tool.

## Q3. Why remove duplicates if they could be real people?

We do not know whether they are repeated people or generated/resampled rows. We show sensitivity: the file has 2,573 exact duplicates and view B retains 19,884 unique exact rows. The choice is logged, not presented as proof of provenance.

## Q4. Why keep conflicting rows in the primary deduplicated view?

View B removes exact duplicate rows only, a minimal treatment. View C removes conflicting feature patterns separately; its smoking OR is 1.04 with 95% CI 0.96 to 1.14.

## Q5. How do you know the first block behaves differently?

The first 1,455 rows are all coded negative, with 103 distinct patterns. Feature-shift Cramer's V ranges from 0.18 to 0.54. We say consistent with a different source, not that we know the source.

## Q6. Is patient_id a leak?

It carries order-related information: its AUC against the recorded label is 0.67. We exclude it from feature tables and do not interpret this as model performance.

## Q7. What does the very small raw smoking p-value mean?

In view A the p-value is 4.45e-30, but the OR falls from 1.49 in A to 1.00 in D. The association is sensitive to the integrity artifacts.

## Q8. Why Benjamini-Hochberg instead of Bonferroni?

BH controls the false discovery rate for a family of tests. In D the minimum raw p is 0.033, while the minimum adjusted q is 0.46.

## Q9. What does CLES mean?

CLES compares a randomly selected coded-positive row with a randomly selected coded-negative row. In D it is 0.512, close to 0.5; it is a small rank-based difference in this file.

## Q10. Does a null result mean smoking is not a risk factor?

No. The result only says that this file, after the stated treatments, does not show a robust association. It cannot overturn external evidence or establish absence of association.

## Q11. Why standardize age?

A standardized age column may be useful for methods sensitive to variable scale. Its descriptive scaler was fit on B only; any future model must fit it within the training fold.

## Q12. Why no PCA or log transform?

The later-row age skew is -0.006, and the first PCA component explains 7.0% versus a flat reference of 6.7%. Neither diagnostic motivates a transformation here.

## Q13. What would change the conclusion?

Verified provenance, a documented label definition and collection process, confirmation against an original source, or independent data with a known sampling frame could change the assessment.

## Q14. What models might be considered later?

Only after source validation: baseline and regularized models with group-aware splits, a separate test set, calibration checks, and metrics appropriate to the intended use. That is planned, not done.

## Q15. Why is the recorded positive share not prevalence?

The file has 82.1% of rows coded 1, but the sampling frame and label-generation process are unknown. We cannot infer population prevalence.

## Q16. What does the checksum mismatch mean?

The expected prefix is 4ad2333067fcb0a9; the observed prefix is 1e0d557990cd7533. The user-approved override is explicit; the expected contract has not been changed.

## Q17. Can the source or license be named?

No. The supplied materials do not establish either one. Both remain open questions, and redistribution should wait for a license check.

## Q18. What does the gender code represent?

The mapping is undocumented. We preserve and label it GENDER_CODE without inferring sex or gender identity.

## Q19. What does LUNG_CANCER = 1 mean?

Only that the row contains the recorded code 1. The label definition and generation process are unknown.

## Q20. What is view A?

The original raw data, with 22,457 rows and no row removal.

## Q21. What is view B?

Exact duplicates are removed while rows with conflicting feature patterns remain; B has 19,884 rows.

## Q22. What is view C?

View B plus removal of conflicting feature patterns; C has 19,618 rows.

## Q23. What is view D?

View B filtered by original raw row index to remove the leading all-negative block; D has 19,781 rows.

## Q24. Why is age outlier status not a deletion rule?

The IQR fences are 40.5 and 84.5, flagging 221 observations. That statistical flag does not prove those records invalid.

## Q25. What is the practical meaning of the smoking odds ratio in D?

The estimate is 1.00 with 95% CI 0.92 to 1.08. It is a descriptive association among coded rows, not a causal or individual-risk estimate.

## Q26. What does the power calculation tell you?

Under its planning assumptions, view D's approximate minimum detectable risk difference is 0.0135. It is not proof that smaller effects are absent.

## Q27. Why use a permutation test?

It provides a resampling-based check of the H1 association under label exchangeability. In D the permutation p-value is 0.908; dependence and source structure still limit inference.

## Q28. Why is the heatmap nearly blank?

The maximum absolute pairwise correlation is 0.022 in the inspected later-row deduplicated subset. This does not prove complete independence.

## Q29. Why preserve four cleaning views?

The file's intended source and row validity are unknown. Multiple views expose how substantive-looking results change under reasonable, explicit treatment policies.

## Q30. What is the limitation of the duplicate simulation?

It compares 1,352 observed later repeated patterns with a mean of 614.9 simulated under independent marginal draws. It is a rough diagnostic and not a validated source model.

## Q31. Does the file meet USPSTF eligibility fields?

No. The cited recommendation requires age 50-80 and smoking history of at least 20 pack-years with quit-time criteria; the file has no dose or quit-time variables.

## Q32. What is F0?

F0 contains 15 recoded/raw predictors, excluding patient_id and retaining the label separately.

## Q33. What is F1?

F1 contains 19 raw-plus-engineered predictors; it is an export for later consideration, not a selected or fitted model.

## Q34. What is F2?

F2 contains 5 predictors based on domain priors. It is retained as a candidate because statistical feature selection is inconclusive.

## Q35. What should be checked next?

Establish dataset provenance and licence, confirm the intended checksum, obtain label and code documentation, determine whether the sampling frame supports the intended question, and then obtain independent data if modeling is authorized.

Rehearsal remains a human task: conduct timed runs and practice answering these questions aloud. No rehearsal is represented as completed by this build.
