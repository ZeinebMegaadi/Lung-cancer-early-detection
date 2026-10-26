# Learning notes

## Phase 0 — setup and integrity gate
- Inspecting the raw CSV before any analysis is mandatory; this confirmed a single-file dataset with 22,457 rows and 17 columns.
- The checksum gate is not a formality: the observed SHA-256 prefix `1e0d557990cd7533` does not match the required `4ad2333067fcb0a9`, so the raw file is currently outside the project contract.
- The project scaffold was created in the expected directories, but no further analysis proceeds past this mismatch until the discrepancy is explained or corrected.
- Self-check: What is the exact difference between the expected checksum prefix and the observed raw-file prefix, and why does it block the rest of the pipeline?

## Phase 1 — data understanding and path alignment
- The pipeline expects the raw CSV under the project’s `data/raw/` contract path, so the dataset was copied there to match the code path without altering the original root file.
- Contract-path alignment differs from checksum validity: the raw file can be readable and located correctly while still diverging from the required integrity value.
- The initial integrity tests now pass under the project override, confirming the data-loading and cleaning logic is consistent with the expected schema and view definitions.
- Self-check: Why is a file-path fix not the same as a checksum fix, and what should be recorded in the decision log when the project contract is intentionally overridden?

## Phase 1 — schema and data dictionary
- The notebook now checks shape, dtypes, missing values, unique identifiers, category codes, and age summaries before analysis.
- The data dictionary reports symptom YES shares from code `2` and keeps unknown code meanings explicit instead of inferring them.
- The data card marks source, licence, collection details, and label meaning OPEN because the supplied README contains only a title.
- Self-check: Why should `GENDER` code shares be described without assigning a demographic meaning that the available source does not verify?

## Phase 1 — target balance
- The share of recorded label `1` is also the accuracy of an always-positive rule on this file; this is a baseline arithmetic check, not model training.
- With an imbalanced label, a high accuracy can hide failure to distinguish groups, so accuracy alone is unsuitable as a success criterion.
- Unknown sampling and label meaning prevent treating this file's prevalence as representative of a real screening population.
- Self-check: How can an always-positive rule get a high accuracy while providing no useful separation between the recorded-label groups?

## Phase 2 — row-order and source-shift audit
- The first positive recorded label occurs at zero-based index 1455, and all 1455 preceding rows have label 0.
- Among negatives, SMOKING code 2 occurs in 29.28% of leading-block rows and 56.04% of later rows; this supports a distribution-shift observation, not a verified provenance claim.
- A 300-row rolling plot shows the label rate and feature rates over row position; the table and PNG are generated from the audit function.
- Self-check: Why is comparing feature values against a leading-versus-later group indicator correct for Cramer's V, while cross-tabulating the two groups' feature values against each other is not?

## Phase 2 — duplicate and later-row independence audits
- Exact duplicates are defined over features plus recorded label and exclude patient_id; the audit finds 2,573 exact duplicates and 2,706 repeated feature patterns.
- The data contains 133 conflicting feature patterns (315 raw rows); the maximum feature-pattern multiplicity is 86.
- Under ten independent-marginal simulations, the mean later-row duplicate count is 614.9, compared with 1,352 observed; this is a reference simulation, not proof of provenance.
- In the later deduplicated rows, 4 of 91 binary-feature tests are nominally significant, the maximum absolute feature correlation is 0.0215, and the first PCA component is 6.99% versus 6.67% flat reference.
- Self-check: Which exact columns define an exact duplicate, and why must a feature-only conflict be analyzed separately from a repeated full row?

## Phase 3 — cleaning policy and sensitivity views
- View B removes exact duplicates; C additionally removes conflicting feature patterns; D filters the deduplicated data by the raw first-positive boundary.
- Correct filtering on original row indices gives D = 19,781 rows; slicing B by 1,455 positions would incorrectly remove additional rows because deduplication changes table positions.
- AGE IQR outliers are flagged (221) and retained; cleaning logs, views, and treatments record before/after counts.
- Self-check: Why should D use the original raw index boundary rather than removing 1,455 positions from the deduplicated B table?
