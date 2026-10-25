"""Build the coursework report, references, Q&A notes, and PDF from results.json."""

import json
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import DOCS, ROOT

REPORT = ROOT / "report"
RESULTS = DOCS / "results.json"
DICTIONARY = DOCS / "data_dictionary.md"
FEATURE_SPEC = DOCS / "feature_spec.md"


def number(value, digits=2):
    return f"{value:,.{digits}f}"


def percent(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def p_value(value):
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def source_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = []
    found = False
    for line in lines:
        if line.startswith("|"):
            found = True
            table_lines.append(line)
        elif found:
            break
    return "\n".join(table_lines)


def make_report(data):
    ds, target, age = data["dataset"], data["target"], data["age"]
    audit, views = data["integrity"], data["views"]
    h1, h2, h3, h4 = data["h1"], data["h2"], data["h3"], data["h4"]
    h5 = data["h5"]
    power, spec = data["power"], data["external_context"]["uspstf"]
    refs = data["external_context"]["references"]
    ci_percent = ds["confidence_interval_percent"]
    report = [
        "# Lung Cancer Risk Project: Integrity Audit and Descriptive Analysis",
        "",
        "**Coursework research prototype. Not a diagnostic or screening tool.**",
        "",
        "## Abstract",
        "",
        (
            f"This report examines whether a survey-style file supports stable descriptive "
            f"associations with its recorded label after integrity problems are addressed. "
            f"The supplied file contains {number(ds['n_raw'], 0)} rows and "
            f"{ds['n_columns']} columns; the recorded positive-label share is "
            f"{percent(target['positive_rate'])}. We audited order, duplicates, conflicting "
            f"patterns, source shifts, and sensitivity to four cleaning views, then calculated "
            f"pre-specified effect sizes and multiple-testing-adjusted results. The first "
            f"{number(audit['leading_block_n'], 0)} records are all coded negative, and the "
            f"deduplicated, leading-block-removed view has {number(views['D_dedup_nolead']['n'], 0)} "
            f"rows. In that view, the smoking-label odds ratio is "
            f"{number(h1['D_dedup_nolead']['odds_ratio'], 2)} "
            f"({ds['confidence_interval_percent']}% CI {number(h1['D_dedup_nolead']['ci_low'], 2)} to "
            f"{number(h1['D_dedup_nolead']['ci_high'], 2)}; p="
            f"{p_value(h1['D_dedup_nolead']['p'])}), while the minimum adjusted q-value "
            f"for the feature family is {number(h4['D_dedup_nolead']['min_q'], 2)}. "
            f"These findings describe this file, not a population or an individual. Source, "
            f"licence, label meaning, and gender-code mapping remain unknown, and the supplied "
            f"checksum differs from the project contract. No predictive model was fitted. "
            f"This is a research prototype, not a diagnostic or screening tool."
        ),
        "",
        "## 1. Business understanding",
        "",
        (
            f"Lung-cancer screening guidance is intended for a defined population and requires "
            f"more history than a yes/no smoking item. The {spec['year']} U.S. Preventive "
            f"Services Task Force recommendation applies to adults aged {spec['age_min']} to "
            f"{spec['age_max']} years with at least {spec['pack_years_min']} pack-years who "
            f"currently smoke or stopped within the past {spec['quit_years_max']} years [1]. "
            f"The supplied file has an age field and a coded smoking flag, but no pack-years, "
            f"smoking duration, or quit-time measure. As a result, the file cannot reproduce "
            f"those eligibility criteria. Published work on risk-based screening likewise "
            f"provides context for why a single binary flag is not a complete risk history [2]. "
            f"That context motivates careful data understanding; it does not validate this file."
        ),
        (
            "The project question is whether the observed associations are robust to "
            "integrity-aware treatment, not whether a model can assign clinical risk. The "
            "business objective is therefore to characterize the supplied data, document "
            "uncertainty, and provide reproducible outputs that a reviewer can inspect. A "
            "trivial accuracy benchmark would be misleading: the label is imbalanced, so an "
            f"always-positive rule has accuracy {percent(target['always_positive_accuracy_reference'])} "
            "without learning any structure. That number is an arithmetic baseline, not a "
            "trained model or meaningful success criterion."
        ),
        (
            "Five research questions guide the work. RQ1 asks which integrity problems occur "
            "and how large they are. RQ2 asks what associations with the recorded label remain "
            "after the planned treatments. RQ3 asks how sensitive the results are to cleaning "
            "policy. RQ4 asks what is known about representativeness. RQ5 asks which derived "
            "descriptors can be justified from the available fields. The work succeeds when "
            "the input shape and schema are checked, each treatment is reproducible, estimates "
            "include effect sizes and uncertainty, and all claims are traceable to generated "
            "tables or figures. It does not set a target for predictive performance."
        ),
        (
            "The scope follows a data-understanding and data-preparation workflow. It "
            "includes descriptive statistics, source-shift diagnostics, sensitivity views, "
            "pre-specified tests, and feature specifications. It excludes clinical "
            "interpretation of the label, causal claims, external validation, and model "
            "training. The dataset's provenance is not established, so there is no defensible "
            "sampling frame from which to estimate population representativeness. The positive "
            "label share is high, but the project cannot determine whether this reflects "
            "sampling, synthetic construction, a particular case-control design, or another "
            "process. The report's success criteria concern reproducibility and honest limits, "
            "not whether the file appears clinically plausible."
        ),
        "",
        "## 2. Data understanding",
        "",
        (
            f"The input CSV has {number(ds['n_raw'], 0)} observations and {ds['n_columns']} "
            f"columns. It contains {ds['null_count']} null cells; `patient_id` is unique, and "
            f"the age values span {age['min']} to {age['max']} (mean {number(age['mean'])}, "
            f"standard deviation {number(age['std'])}). The row-level schema checks require "
            "the expected column order and shape, no nulls, unique identifiers, binary label "
            "codes, symptom flags in the supplied 1/2 coding, and a permissive age range. "
            "Observed values are distinguished from undocumented meanings: the file's "
            "`GENDER` code is reported as `GENDER_CODE`, and `LUNG_CANCER` is called the "
            "recorded label, not a verified diagnosis."
        ),
        (
            "The complete column-level dictionary below records type, coding, observed range, "
            "and coded-value share. The source description identifies symptom/lifestyle code "
            "1 as NO and code 2 as YES, but it does not explain the gender mapping or how the "
            "label was generated. Age is treated as a numeric variable in the supplied units; "
            "the unit is presumed to be years based on the values but is not documented. "
            "`patient_id` is a sequential identifier and is excluded from feature construction. "
            "The raw dictionary is also maintained separately in `docs/data_dictionary.md`."
        ),
        "",
        source_table(DICTIONARY),
        "",
        (
            f"The checksum is a separate integrity contract. The expected prefix is "
            f"`{ds['expected_sha256_prefix']}`; the supplied file's observed prefix is "
            f"`{ds['sha256_prefix']}`. The mismatch is documented in `data/raw/CHECKSUMS.txt` "
            "and the decision log. The user explicitly approved analysis of this current "
            "snapshot under an explicit checksum override; the expected checksum has not been "
            "silently changed. Normal loading verifies the contract by default, while the "
            "analysis runner emits a warning and opts out explicitly. This does not establish "
            "that the input is the same file intended by the original project specification."
        ),
        (
            f"The recorded label has {number(target['positive_n'], 0)} rows coded 1 and "
            f"{number(target['negative_n'], 0)} coded 0, for a positive-code share of "
            f"{percent(target['positive_rate'])}. The code meaning and label-generation "
            "process are unknown. Because the file's share is likely not representative of a "
            "screening population, it cannot be interpreted as population prevalence. The "
            "unknown provenance also prevents identification of sampling design, inclusion "
            "criteria, consent, or intended use. These are material limitations rather than "
            "minor documentation gaps."
        ),
        "",
        "## 3. Preprocessing and integrity audit",
        "",
        (
            f"The most visible ordering issue is a leading block: the first "
            f"{number(audit['leading_block_n'], 0)} rows are all coded negative, and the first "
            f"positive appears at zero-based row index {audit['first_positive_index']}. The "
            "rolling label rate stays at zero through that block and rises after the boundary. "
            f"The smoking YES share is {percent(audit['source_shift_smoking_lead_yes'])} in the "
            f"leading block and {percent(audit['source_shift_smoking_later_yes'])} among later "
            "negative rows. Multiple coded features distinguish those groups. This is "
            "consistent with a source or collection-process shift, but the available metadata "
            "cannot prove how the groups were assembled."
        ),
        (
            f"The file contains {number(audit['exact_duplicates'], 0)} exact duplicate rows "
            f"and {number(audit['feature_only_duplicates'], 0)} repeated feature patterns when "
            "the identifier is ignored. There are "
            f"{number(audit['conflicting_patterns'], 0)} feature patterns with conflicting "
            f"labels involving {number(audit['conflicting_rows'], 0)} rows; the most repeated "
            f"pattern occurs {audit['max_pattern_multiplicity']} times. In the later portion, "
            f"{number(audit['later_duplicates_observed'], 0)} repeated feature rows were "
            f"observed, compared with a mean of {number(audit['later_duplicates_simulated_mean'], 1)} "
            "in a seeded independent-marginal simulation. The simulation is a rough diagnostic, "
            "not a valid generative model for the unknown source. Its purpose is to show that "
            "the duplicate burden deserves review, not to establish intent or authenticity."
        ),
        (
            f"The sequential identifier alone has AUC {number(audit['patient_id_auc'], 3)} "
            "for ranking the recorded label. This is an audit statistic demonstrating "
            "ordering-related signal, not a model result. Row order, duplicates, and group "
            "structure can create misleading train/test separation or cross-validation "
            "performance if ignored [3]. Therefore the identifier is excluded from features, "
            "and the project does not fit a predictive model. The raw-file-only association "
            "is retained as a sensitivity result rather than treated as substantive evidence."
        ),
        (
            "Four views make preprocessing choices explicit. View A retains all raw rows. "
            "View B removes exact duplicates while retaining conflicting labels. View C "
            "additionally removes rows belonging to conflicting patterns. View D starts from "
            "the deduplicated B view and removes all rows whose original raw index is before "
            "the first positive label. The last operation is keyed to the raw index, not a "
            "position in a deduplicated table; this avoids accidentally discarding or retaining "
            "the wrong rows. The views are sensitivity analyses, not claims that any single "
            "policy repairs unknown data provenance."
        ),
        "",
        table(
            ["View", "Rows", "Positive-code share", "Treatment"],
            [
                ["A - raw", f"{views['A_raw']['n']:,}", percent(views["A_raw"]["positive_rate"]), "No row removal"],
                ["B - deduplicated", f"{views['B_dedup']['n']:,}", percent(views["B_dedup"]["positive_rate"]), "Keep first exact row"],
                ["C - no conflicts", f"{views['C_dedup_noconflict']['n']:,}", percent(views["C_dedup_noconflict"]["positive_rate"]), "Also drop conflicting patterns"],
                ["D - no leading block", f"{views['D_dedup_nolead']['n']:,}", percent(views["D_dedup_nolead"]["positive_rate"]), "B, filtered by original index"],
            ],
        ),
        "",
        (
            f"Age outlier flags use the descriptive interquartile-range rule: the fences are "
            f"{number(age['iqr_fence_low'], 1)} and {number(age['iqr_fence_high'], 1)}, yielding "
            f"{number(age['outlier_n'], 0)} flagged observations. No age rows were removed "
            "solely for this flag because the observed values satisfy the broad schema bounds "
            "and there is no source documentation establishing measurement errors. This "
            "preserves a transparent distinction between a statistical flag and a confirmed "
            "invalid record. The four analysis views focus on the observed structural "
            "artifacts rather than silently trimming tail values."
        ),
        (
            "The treatment log keeps each decision auditable: preserve a raw baseline; create "
            "an exact-deduplicated view; create a no-conflict sensitivity view; and create a "
            "deduplicated, leading-block-removed view. No missing-value imputation is needed "
            "because the loaded file has no null cells. Identifier removal is part of feature "
            "construction, not raw-data rewriting. Encodings are kept as supplied in the audit "
            "tables and recoded only in derived feature outputs. These rules reduce avoidable "
            "analysis leakage while leaving the original CSV unchanged."
        ),
        "",
        "## 4. Exploratory analysis and statistics",
        "",
        (
            "The analysis pre-specifies five descriptive hypotheses. H1 compares the recorded "
            "label by smoking code using an odds ratio, confidence interval, Pearson "
            "chi-square, and seeded permutation test. H2 compares age distributions using "
            "Mann-Whitney U and the common-language effect size (CLES). H3 relates a four-item "
            "respiratory/chest symptom count to the label using Spearman rho. H4 tests the "
            "binary feature family in each view and adjusts p-values with Benjamini-Hochberg "
            "(BH), which controls the false discovery rate under its assumptions [4]. H5 "
            "compares the leading block with later negative rows as an integrity diagnostic. "
            "P-values are not treated as effect sizes or evidence of causation."
        ),
        "",
        "### Figure 1. Is the file ordered?",
        "",
        f"![Figure 1. Rolling positive-code share and row-order boundary](../reports/figures/{data['figure_files'][0]})",
        "",
        (
            f"The rolling label rate is zero through the first {audit['leading_block_n']:,} "
            "rows, then changes after the first positive label. The plot describes order, not "
            "time; there is no documented temporal variable. The boundary is an observed "
            "sequence feature and supports treating row order as a potential leakage path."
        ),
        "",
        "### Figure 2. How much data survives each cleaning policy?",
        "",
        f"![Figure 2. Record counts by view](../reports/figures/{data['figure_files'][1]})",
        "",
        (
            f"The raw view contains {views['A_raw']['n']:,} rows, whereas the deduplicated "
            f"view contains {views['B_dedup']['n']:,}; the no-conflict and leading-block "
            f"views contain {views['C_dedup_noconflict']['n']:,} and "
            f"{views['D_dedup_nolead']['n']:,} rows. The changes are policy-dependent and "
            "should be read alongside the sensitivity table rather than as a single estimate "
            "of the dataset's true size."
        ),
        "",
        "### Figure 3. Does the leading block behave differently?",
        "",
        f"![Figure 3. Leading-block versus later-negative feature shifts](../reports/figures/{data['figure_files'][2]})",
        "",
        (
            f"Across the tested flags, Cramer's V ranges from "
            f"{number(audit['source_shift_v_min'], 3)} to "
            f"{number(audit['source_shift_v_max'], 3)}; even the largest source-shift p-value "
            f"is {p_value(audit['source_shift_p_max'])}. The distinct age minima are "
            f"{audit['source_shift_age_lead_min']} in the leading block and "
            f"{audit['source_shift_age_later_negative_min']} among later negatives. Together "
            "these observations are consistent with a different source or sampling process. "
            "They do not identify that source."
        ),
        "",
        "### Figure 4. What is the shape of age?",
        "",
        f"![Figure 4. Age distribution and normal reference](../reports/figures/{data['figure_files'][3]})",
        "",
        (
            f"Age ranges from {age['min']} to {age['max']} with mean "
            f"{number(age['mean'], 1)} and skew {number(audit['independence']['age_skew'], 3)} "
            "in the deduplicated later-row subset used for the independence audit. The broad "
            "shape is not strongly skewed, so a log transform is not justified by this "
            "descriptive check. The histogram is a marginal summary and does not establish "
            "representativeness or measurement validity."
        ),
        "",
        "### Figure 5. Are the recorded features strongly related to each other?",
        "",
        f"![Figure 5. Pairwise correlation heatmap](../reports/figures/{data['figure_files'][4]})",
        "",
        (
            f"The maximum absolute pairwise correlation in the later deduplicated subset is "
            f"{number(audit['independence']['max_abs_pairwise_corr'], 3)}. The heatmap's "
            "narrow color scale makes weak pairwise relations visible; the near-blank pattern "
            "is itself an observation. This does not prove mutual independence. The VIF range "
            f"is {number(audit['independence']['vif_min'], 3)} to "
            f"{number(audit['independence']['vif_max'], 3)}, and "
            f"{percent(audit['independence']['share_sig_pairwise_chisq'])} of "
            f"{audit['independence']['pairwise_chisq_tests']} binary-feature pair tests are "
            "nominally significant. These diagnostics depend on the chosen subset and test "
            "assumptions."
        ),
        "",
        "### Figure 6. Do feature associations depend on the cleaning view?",
        "",
        f"![Figure 6. Maximum Cramer's V by view](../reports/figures/{data['figure_files'][5]})",
        "",
        (
            f"The largest Cramer's V falls from {number(h4['A_raw']['max_v'], 3)} in the raw "
            f"view to {number(h4['D_dedup_nolead']['max_v'], 3)} in view D. In D the smallest "
            f"raw feature-family p-value is {p_value(h4['D_dedup_nolead']['min_p'])}, but its "
            f"smallest BH q-value is {number(h4['D_dedup_nolead']['min_q'], 3)}. No feature in "
            "that adjusted family crosses the conventional threshold. The raw view has "
            f"{int(h4['A_raw']['all_q_below_005']) * data['feature_family_size']} of "
            f"{data['feature_family_size']} tests below that threshold, "
            "underscoring that the apparent associations are sensitive to structural "
            "artifacts."
        ),
        "",
        "### Figure 7. What is the smoking association across views?",
        "",
        f"![Figure 7. Smoking odds ratio and {ci_percent}% confidence interval by view](../reports/figures/{data['figure_files'][6]})",
        "",
        (
            f"In view A, the smoking odds ratio is {number(h1['A_raw']['odds_ratio'], 2)} "
            f"({ci_percent}% CI {number(h1['A_raw']['ci_low'], 2)} to "
            f"{number(h1['A_raw']['ci_high'], 2)}; p={p_value(h1['A_raw']['p'])}). In view D "
            f"it is {number(h1['D_dedup_nolead']['odds_ratio'], 2)} "
            f"({ci_percent}% CI {number(h1['D_dedup_nolead']['ci_low'], 2)} to "
            f"{number(h1['D_dedup_nolead']['ci_high'], 2)}; permutation "
            f"p={number(h1['D_dedup_nolead']['permutation_p'], 3)}). The raw association is "
            "not stable to removal of the ordered block and duplicates; this file does not "
            "establish whether smoking has or lacks an association in a valid population."
        ),
        "",
        "### Hypothesis results across views",
        "",
        (
            "For H1-H3, q-values adjust across the sensitivity views within each hypothesis; "
            "H4 q-values adjust the feature family separately in each view; H5 q-values adjust "
            "the raw-file integrity feature family. H5 is not repeated across cleaned views "
            "because view D removes the leading block being compared."
        ),
        "",
        table(
            [
                "View",
                f"H1 OR ({ci_percent}% CI)",
                "H1 p / q",
                f"H2 CLES ({ci_percent}% CI); p / q",
                f"H3 rho ({ci_percent}% CI); p / q",
                "H4 min q",
            ],
            [
                [
                    label,
                    f"{number(h1[label]['odds_ratio'], 2)} ({number(h1[label]['ci_low'], 2)}-{number(h1[label]['ci_high'], 2)})",
                    f"{p_value(h1[label]['p'])} / {number(h1[label]['q_BH'], 3)}",
                    (
                        f"{number(h2[label]['cles'], 3)} "
                        f"({number(h2[label]['ci_low'], 3)}-{number(h2[label]['ci_high'], 3)}); "
                        f"{p_value(h2[label]['p'])} / {number(h2[label]['q_BH'], 3)}"
                    ),
                    (
                        f"{number(h3[label]['rho'], 3)} "
                        f"({number(h3[label]['ci_low'], 3)}-{number(h3[label]['ci_high'], 3)}); "
                        f"{p_value(h3[label]['p'])} / {number(h3[label]['q_BH'], 3)}"
                    ),
                    number(h4[label]["min_q"], 3),
                ]
                for label in ["A_raw", "B_dedup", "C_dedup_noconflict", "D_dedup_nolead"]
            ],
        ),
        "",
        "### H5. Do the leading-block negatives differ from later negatives?",
        "",
        (
            "H5 is an integrity comparison in the raw file, not a cleaned-view effect: view D "
            "removes the leading block and therefore cannot supply this contrast. The following "
            "feature-family results compare the leading-block negatives with later negatives; "
            "Cramer's V intervals use a row-level multinomial bootstrap, and q-values adjust "
            "the 14 feature tests with Benjamini-Hochberg."
        ),
        "",
        table(
            ["Feature", "Chi-square", "p", "q", f"Cramer's V ({ci_percent}% CI)"],
            [
                [
                    feature,
                    number(result["statistic"], 2),
                    p_value(result["p"]),
                    number(result["q_BH"], 3),
                    (
                        f"{number(result['cramers_v'], 3)} "
                        f"({number(result['ci_low'], 3)}-{number(result['ci_high'], 3)})"
                    ),
                ]
                for feature, result in h5.items()
            ],
        ),
        "",
        (
            "In views B and C, age and respiratory-count tests can be nominally significant, "
            f"but their effects are small: CLES is {number(h2['B_dedup']['cles'], 3)} in B and "
            f"{number(h2['C_dedup_noconflict']['cles'], 3)} in C, while respiratory-count rho "
            f"is {number(h3['B_dedup']['rho'], 3)} and "
            f"{number(h3['C_dedup_noconflict']['rho'], 3)}. In view D, H2 has "
            f"p={p_value(h2['D_dedup_nolead']['p'])} and H3 has "
            f"rho={number(h3['D_dedup_nolead']['rho'], 3)} "
            f"(p={p_value(h3['D_dedup_nolead']['p'])}). No association survives the H4 "
            "family correction in D, and observed effects are negligible. This wording "
            "acknowledges the nominal results rather than claiming that no test was ever "
            "significant."
        ),
        (
            f"The planned two-group power approximation in view D uses "
            f"{power['view_D_smokers_n']:,} smoking-code YES rows and "
            f"{power['view_D_nonsmokers_n']:,} NO rows, with alpha "
            f"{number(power['alpha'], 2)} and target power {number(power['power'], 1)}. "
            f"Its approximate minimum detectable risk difference is "
            f"{number(power['minimum_detectable_risk_difference'], 4)} "
            f"({percent(power['minimum_detectable_risk_difference'])}). This is a planning "
            "calculation, not evidence that a smaller association is absent. The unusually "
            "structured data, unknown sampling process, and dependence among duplicate rows "
            "limit the interpretation of nominal inferential quantities."
        ),
        (
            "Across tests, a small p-value indicates incompatibility with a specified null "
            "model under its assumptions; it does not measure effect magnitude, causal impact, "
            "or clinical value. Conversely, a large p-value is not proof of no association. "
            "The goal of the multiple-view analysis is to reveal sensitivity to data handling, "
            "not to select whichever view yields a preferred conclusion. H5 is explicitly an "
            "integrity check, and its statistically small p-values are expected to accompany "
            "the marked differences between the block and later negatives."
        ),
        "",
        "## 5. Feature engineering",
        "",
        (
            "Feature construction is descriptive preparation for a possible later study; it "
            "is not a fitted model. The exported feature tables remove `patient_id`, keep the "
            "recorded label in a separate column, recode symptom/lifestyle fields from "
            "1/2 to 0/1, and rename `GENDER` to `GENDER_CODE` without guessing its meaning. "
            "The feature specification is driven by simple, inspectable transformations."
        ),
        "",
        source_table(FEATURE_SPEC),
        "",
        (
            f"The respiratory count sums four recorded yes/no flags and has support from 0 "
            f"through 4. In view B its distribution is "
            f"{', '.join(f'{key}: {value:,}' for key, value in data['features']['resp_symptom_count_distribution_B'].items())}. "
            f"The broader symptom count spans 0 through 6. `AGE_Z` uses the mean and "
            "population standard deviation fitted on view B for this descriptive export only. "
            "Any future cross-validation must fit the scaler inside each training fold to "
            "avoid leakage [3]. No log transform is justified by the observed age skew, and "
            "the flat PCA spectrum provides no obvious compression target."
        ),
        (
            f"The three candidate feature sets contain {data['features']['F0_raw15']} raw "
            f"features, {data['features']['F1_raw_plus_engineered']} raw-plus-engineered "
            f"features, and {data['features']['F2_prior_driven']} prior-driven features. "
            "Univariate screening is saved as a descriptive table and is not used to select "
            "predictors. Because evidence-based selection is inconclusive after structural "
            "treatment, the domain-prior set is retained as a candidate while the other sets "
            "remain comparison baselines. `SMOKER_IN_SCREEN_AGE` is only a proxy: age and a "
            "binary smoking code do not encode pack-years or time since quitting, so it must "
            "not be described as screening eligibility."
        ),
        "",
        "## 6. Conclusions and next steps",
        "",
        (
            f"RQ1 is answered by the audit: the first {audit['leading_block_n']:,} rows are "
            f"all negative-coded; {audit['exact_duplicates']:,} exact duplicates and "
            f"{audit['conflicting_patterns']:,} conflicting feature patterns occur; the "
            f"identifier AUC is {number(audit['patient_id_auc'], 2)}. RQ2 is answered "
            "cautiously: after removing the block, the observed feature-label effects are "
            "small and the adjusted family results do not cross the chosen threshold. RQ3 is "
            "answered by the changing estimates across A-D; in particular the smoking odds "
            "ratio changes from the raw view toward the null in D. These results show "
            "sensitivity to cleaning, not a universal biological conclusion."
        ),
        (
            "RQ4 cannot be answered as a population-representativeness question because the "
            "source, sampling frame, and collection process are unknown. The high recorded "
            "positive-code share is not a population prevalence estimate. RQ5 yields a "
            "transparent set of candidate descriptors, but no evidence-based feature-selection "
            "claim. The project does not establish causation, diagnostic utility, an individual "
            "risk estimate, or the authenticity of the underlying records."
        ),
        (
            "Material limitations include unknown provenance and licence; undocumented label "
            "meaning and generation; unknown `GENDER` mapping; no smoking dose, duration, or "
            "quit time; a checksum mismatch; possible source shift, duplicates, and row-order "
            "structure; a potentially nonrepresentative label balance; and no external "
            "validation. The analysis can characterize the supplied file only. The source "
            "license must be established before redistribution. The checksum override is "
            "explicit and should be replaced with a verified intended file if one becomes "
            "available."
        ),
        (
            "The next project phase, if authorized, is planned rather than implemented: "
            "confirm provenance and label semantics, obtain a documented sampling frame, "
            "define an appropriate cohort, then design leakage-safe evaluation with grouped "
            "or source-aware splits and clinically meaningful metrics. A later model should "
            "be evaluated on independent, representative data with calibration and uncertainty "
            "analysis. No such model experiment was run here because the coursework scope "
            "excludes model fitting. A model score from the current file would not resolve its "
            "integrity uncertainty."
        ),
        (
            "Reproduction uses the repository environment, raw-file contract, pipeline "
            "runner, result generator, report and slide builders, and tests. From the project "
            "root: create and activate a virtual environment; install `requirements.txt`; "
            "run `python -m src.run_analysis`; run `python -m src.results`; build the report "
            "and slides; then run `python -m pytest -q`. Generated results are intended to "
            "come from the checked-in code, not manual transcription. A clean-clone run and "
            "presentation rehearsal remain external delivery tasks."
        ),
        "",
        "## References",
        "",
        f"[{refs[0]['number']}] {refs[0]['title']}. U.S. Preventive Services Task Force; JAMA, {refs[0]['year']}; {refs[0]['volume']}({refs[0]['issue']}):{refs[0]['page_start']}-{refs[0]['page_end']}. https://doi.org/{refs[0]['doi']}.",
        "",
        f"[{refs[1]['number']}] Tammemagi MC et al. {refs[1]['title']}. New England Journal of Medicine, {refs[1]['year']}; {refs[1]['volume']}({refs[1]['issue']}):{refs[1]['page_start']}-{refs[1]['page_end']}. https://doi.org/{refs[1]['doi']}.",
        "",
        f"[{refs[2]['number']}] Kapoor S, Narayanan A. {refs[2]['title']}. Patterns, {refs[2]['year']}; {refs[2]['volume']}({refs[2]['issue']}):{refs[2]['article_number']}. https://doi.org/{refs[2]['doi']}.",
        "",
        f"[{refs[3]['number']}] Benjamini Y, Hochberg Y. {refs[3]['title']}. Journal of the Royal Statistical Society: Series B, {refs[3]['year']}; {refs[3]['volume']}({refs[3]['issue']}):{refs[3]['page_start']}-{refs[3]['page_end']}. https://doi.org/{refs[3]['doi']}.",
        "",
        "## Reproducibility",
        "",
        "1. Create and activate the project virtual environment.",
        "2. Install the pinned packages from `requirements.txt`.",
        "3. Run `python -m src.run_analysis`.",
        "4. Run `python -m src.results`, `python -m src.build_report`, and `python -m src.build_slides`.",
        "5. Run `python -m pytest -q`.",
        "",
    ]
    return "\n".join(report)


def qa_document(data):
    audit, views = data["integrity"], data["views"]
    h1, h2, h3, h4 = data["h1"], data["h2"], data["h3"], data["h4"]
    options = [
        ("Why not train a model and report accuracy?", f"The project scope excludes model fitting. An always-positive rule reaches {percent(data['target']['always_positive_accuracy_reference'])} accuracy just from the label balance, so it is not a useful success metric. The raw-file associations also change after integrity treatment."),
        ("What is the clinical claim?", "None. This report describes coded associations in one file and is a research prototype, not a diagnostic or screening tool."),
        ("Why remove duplicates if they could be real people?", f"We do not know whether they are repeated people or generated/resampled rows. We show sensitivity: the file has {audit['exact_duplicates']:,} exact duplicates and view B retains {views['B_dedup']['n']:,} unique exact rows. The choice is logged, not presented as proof of provenance."),
        ("Why keep conflicting rows in the primary deduplicated view?", f"View B removes exact duplicate rows only, a minimal treatment. View C removes conflicting feature patterns separately; its smoking OR is {number(h1['C_dedup_noconflict']['odds_ratio'], 2)} with {data['dataset']['confidence_interval_percent']}% CI {number(h1['C_dedup_noconflict']['ci_low'], 2)} to {number(h1['C_dedup_noconflict']['ci_high'], 2)}."),
        ("How do you know the first block behaves differently?", f"The first {audit['leading_block_n']:,} rows are all coded negative, with {audit['leading_block_distinct_patterns']} distinct patterns. Feature-shift Cramer's V ranges from {number(audit['source_shift_v_min'], 2)} to {number(audit['source_shift_v_max'], 2)}. We say consistent with a different source, not that we know the source."),
        ("Is patient_id a leak?", f"It carries order-related information: its AUC against the recorded label is {number(audit['patient_id_auc'], 2)}. We exclude it from feature tables and do not interpret this as model performance."),
        ("What does the very small raw smoking p-value mean?", f"In view A the p-value is {p_value(h1['A_raw']['p'])}, but the OR falls from {number(h1['A_raw']['odds_ratio'], 2)} in A to {number(h1['D_dedup_nolead']['odds_ratio'], 2)} in D. The association is sensitive to the integrity artifacts."),
        ("Why Benjamini-Hochberg instead of Bonferroni?", f"BH controls the false discovery rate for a family of tests. In D the minimum raw p is {p_value(h4['D_dedup_nolead']['min_p'])}, while the minimum adjusted q is {number(h4['D_dedup_nolead']['min_q'], 2)}."),
        ("What does CLES mean?", f"CLES compares a randomly selected coded-positive row with a randomly selected coded-negative row. In D it is {number(h2['D_dedup_nolead']['cles'], 3)}, close to 0.5; it is a small rank-based difference in this file."),
        ("Does a null result mean smoking is not a risk factor?", "No. The result only says that this file, after the stated treatments, does not show a robust association. It cannot overturn external evidence or establish absence of association."),
        ("Why standardize age?", "A standardized age column may be useful for methods sensitive to variable scale. Its descriptive scaler was fit on B only; any future model must fit it within the training fold."),
        ("Why no PCA or log transform?", f"The later-row age skew is {number(audit['independence']['age_skew'], 3)}, and the first PCA component explains {percent(audit['independence']['pca_first_component'])} versus a flat reference of {percent(audit['independence']['pca_flat_reference'])}. Neither diagnostic motivates a transformation here."),
        ("What would change the conclusion?", "Verified provenance, a documented label definition and collection process, confirmation against an original source, or independent data with a known sampling frame could change the assessment."),
        ("What models might be considered later?", "Only after source validation: baseline and regularized models with group-aware splits, a separate test set, calibration checks, and metrics appropriate to the intended use. That is planned, not done."),
        ("Why is the recorded positive share not prevalence?", f"The file has {percent(data['target']['positive_rate'])} of rows coded 1, but the sampling frame and label-generation process are unknown. We cannot infer population prevalence."),
        ("What does the checksum mismatch mean?", f"The expected prefix is {data['dataset']['expected_sha256_prefix']}; the observed prefix is {data['dataset']['sha256_prefix']}. The user-approved override is explicit; the expected contract has not been changed."),
        ("Can the source or license be named?", "No. The supplied materials do not establish either one. Both remain open questions, and redistribution should wait for a license check."),
        ("What does the gender code represent?", "The mapping is undocumented. We preserve and label it GENDER_CODE without inferring sex or gender identity."),
        ("What does LUNG_CANCER = 1 mean?", "Only that the row contains the recorded code 1. The label definition and generation process are unknown."),
        ("What is view A?", f"The original raw data, with {views['A_raw']['n']:,} rows and no row removal."),
        ("What is view B?", f"Exact duplicates are removed while rows with conflicting feature patterns remain; B has {views['B_dedup']['n']:,} rows."),
        ("What is view C?", f"View B plus removal of conflicting feature patterns; C has {views['C_dedup_noconflict']['n']:,} rows."),
        ("What is view D?", f"View B filtered by original raw row index to remove the leading all-negative block; D has {views['D_dedup_nolead']['n']:,} rows."),
        ("Why is age outlier status not a deletion rule?", f"The IQR fences are {number(data['age']['iqr_fence_low'], 1)} and {number(data['age']['iqr_fence_high'], 1)}, flagging {data['age']['outlier_n']:,} observations. That statistical flag does not prove those records invalid."),
        ("What is the practical meaning of the smoking odds ratio in D?", f"The estimate is {number(h1['D_dedup_nolead']['odds_ratio'], 2)} with {data['dataset']['confidence_interval_percent']}% CI {number(h1['D_dedup_nolead']['ci_low'], 2)} to {number(h1['D_dedup_nolead']['ci_high'], 2)}. It is a descriptive association among coded rows, not a causal or individual-risk estimate."),
        ("What does the power calculation tell you?", f"Under its planning assumptions, view D's approximate minimum detectable risk difference is {number(data['power']['minimum_detectable_risk_difference'], 4)}. It is not proof that smaller effects are absent."),
        ("Why use a permutation test?", f"It provides a resampling-based check of the H1 association under label exchangeability. In D the permutation p-value is {number(h1['D_dedup_nolead']['permutation_p'], 3)}; dependence and source structure still limit inference."),
        ("Why is the heatmap nearly blank?", f"The maximum absolute pairwise correlation is {number(audit['independence']['max_abs_pairwise_corr'], 3)} in the inspected later-row deduplicated subset. This does not prove complete independence."),
        ("Why preserve four cleaning views?", "The file's intended source and row validity are unknown. Multiple views expose how substantive-looking results change under reasonable, explicit treatment policies."),
        ("What is the limitation of the duplicate simulation?", f"It compares {audit['later_duplicates_observed']:,} observed later repeated patterns with a mean of {number(audit['later_duplicates_simulated_mean'], 1)} simulated under independent marginal draws. It is a rough diagnostic and not a validated source model."),
        ("Does the file meet USPSTF eligibility fields?", f"No. The cited recommendation requires age {spec_age(data)} and smoking history of at least {data['external_context']['uspstf']['pack_years_min']} pack-years with quit-time criteria; the file has no dose or quit-time variables."),
        ("What is F0?", f"F0 contains {data['features']['F0_raw15']} recoded/raw predictors, excluding patient_id and retaining the label separately."),
        ("What is F1?", f"F1 contains {data['features']['F1_raw_plus_engineered']} raw-plus-engineered predictors; it is an export for later consideration, not a selected or fitted model."),
        ("What is F2?", f"F2 contains {data['features']['F2_prior_driven']} predictors based on domain priors. It is retained as a candidate because statistical feature selection is inconclusive."),
        ("What should be checked next?", "Establish dataset provenance and licence, confirm the intended checksum, obtain label and code documentation, determine whether the sampling frame supports the intended question, and then obtain independent data if modeling is authorized."),
    ]
    lines = ["# Presentation Q&A preparation", ""]
    for index, (question, answer) in enumerate(options, 1):
        lines.extend([f"## Q{index}. {question}", "", answer, ""])
    lines.extend([
        "Rehearsal remains a human task: conduct timed runs and practice answering these questions aloud. No rehearsal is represented as completed by this build.",
        "",
    ])
    return "\n".join(lines)


def spec_age(data):
    spec = data["external_context"]["uspstf"]
    return f"{spec['age_min']}-{spec['age_max']}"


def markdown_to_pdf(markdown, output):
    page_size = (8.27, 11.69)
    margin_left = 0.1
    text_width = 94
    page_number = 0
    pdf = PdfPages(output)

    def new_page():
        nonlocal page_number
        page_number += 1
        fig = plt.figure(figsize=page_size)
        ax = fig.add_axes([margin_left, 0.07, 0.8, 0.86])
        ax.axis("off")
        return fig, ax, 0.995

    fig, ax, y = new_page()

    def save_page():
        ax.text(
            1,
            -0.045,
            f"Page {page_number}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#5b6570",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_text(text, fontsize=9.4, color="#202a33", bold=False, gap=0.008):
        nonlocal fig, ax, y
        plain = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        plain = re.sub(r"`([^`]+)`", r"\1", plain)
        plain = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", plain)
        width = max(45, int(text_width * 9.4 / fontsize))
        wrapped = textwrap.wrap(plain, width=width, break_long_words=True) or [""]
        line_step = fontsize / 72 * 1.35 / page_size[1]
        needed = line_step * len(wrapped) + gap
        if y - needed < 0.015:
            save_page()
            fig, ax, y = new_page()
        ax.text(
            0,
            y,
            "\n".join(wrapped),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=color,
            linespacing=1.3,
        )
        y -= needed

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            y -= 0.006
            index += 1
            continue
        if line.startswith("# "):
            add_text(line[2:], fontsize=19, color="#18334a", bold=True, gap=0.012)
        elif line.startswith("## "):
            add_text(line[3:], fontsize=15, color="#18334a", bold=True, gap=0.012)
        elif line.startswith("### "):
            add_text(line[4:], fontsize=11.5, color="#245577", bold=True)
        elif line.startswith("|"):
            table_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    table_rows.append(cells)
                index += 1
            for row in table_rows:
                add_text("   |   ".join(row), fontsize=8, color="#263746", gap=0.003)
            y -= 0.006
            continue
        elif line.startswith("![") and "](" in line:
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            caption, image_path = match.groups()
            image_path = (REPORT / image_path).resolve()
            if y < 0.46:
                save_page()
                fig, ax, y = new_page()
            image = plt.imread(image_path)
            height_inches = min(3.3, 6.7 * image.shape[0] / image.shape[1])
            normalized_height = height_inches / page_size[1]
            image_ax = fig.add_axes([0.12, y - normalized_height, 0.76, normalized_height])
            image_ax.imshow(image)
            image_ax.axis("off")
            y -= normalized_height + 0.005
            add_text(caption, fontsize=8, color="#34495a", gap=0.01)
        elif re.match(r"^\d+\.\s", line):
            add_text(line, fontsize=9)
        else:
            add_text(line, fontsize=10.5)
        index += 1
    save_page()
    pdf.close()


def build():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Run `python -m src.results` first: {RESULTS}")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    REPORT.mkdir(parents=True, exist_ok=True)
    report = make_report(data)
    (REPORT / "report.md").write_text(report, encoding="utf-8")
    (DOCS / "qa.md").write_text(qa_document(data), encoding="utf-8")
    pdf_path = REPORT / "report.pdf"
    markdown_to_pdf(report, pdf_path)
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_path.read_bytes()))
    minimum = data["deliverable_spec"]["report_pages_min"]
    maximum = data["deliverable_spec"]["report_pages_max"]
    if not minimum <= page_count <= maximum:
        raise AssertionError(
            f"Report PDF must have {minimum}-{maximum} pages; generated {page_count}"
        )
    print(f"Built {REPORT / 'report.pdf'}")
    print(f"Report PDF pages: {page_count}")
    report_word_count = len(re.findall(r"\b[\w-]+\b", report))
    qa_count = len(re.findall(r"^## Q\d+\.", qa_document(data), re.M))
    print(f"Report words: {report_word_count}")
    print(f"Q&A answers: {qa_count}")


if __name__ == "__main__":
    build()
