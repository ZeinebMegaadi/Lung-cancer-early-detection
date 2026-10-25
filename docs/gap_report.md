# Phase 1 compliance audit and prioritized gaps

**Scope:** Read-only evaluation against `docs/handbook.txt` (Chapters 2.2,
4, 6, 10, 12-13, and Appendices A-B). No code or raw-data changes were made.
This file records the evaluation only; implementation awaits user approval.

## Compliance matrix

Statuses reflect current repository evidence, not planned work. Rubric allocations
are copied from Handbook Chapter 2.2; data findings below are from the generated
`docs/results.json`.

| Handbook rubric line | Evidence in repository | Status | Phase 1 assessment |
|---|---|---|---|
| Data loading and organization (5 pts) | `src/data_io.py`, `tests/test_pipeline.py`, `docs/data_dictionary.md`, `docs/data_card.md`, `data/raw/CHECKSUMS.txt` | PARTIAL | Schema, dictionary, and data card are present. The observed raw-file SHA-256 prefix is `1e0d557990cd7533`, not the handbook contract `4ad2333067fcb0a9`; the approved override is documented. The handbook lineage match task has no match-count evidence. |
| Cleaning and preprocessing (12 pts) | `src/audit.py`, `src/clean.py`, `reports/tables/cleaning_log.csv`, `reports/tables/treatments.csv`, `data/interim/` | PARTIAL | Expected view sizes and treatment outputs exist. G-01 now makes the audit and cleaning scripts directly executable; other gaps remain. |
| EDA and statistics (13 pts) | `notebooks/01_explore.ipynb`, `src/figures.py`, `src/hypotheses.py`, `reports/figures/`, `reports/tables/hypotheses.csv` | PARTIAL | G-02 completes the H1-H5 inference table; G-03 generates and registers the requested non-model figure inventory; G-04 aligns report and slide figure mapping with the handbook. Notebook and human-deliverable gaps remain. |
| Feature engineering (7 pts) | `src/features.py`, `docs/feature_spec.md`, `data/processed/`, feature tests in `tests/test_pipeline.py` | OK | F0/F1/F2 are generated with 15/19/5 predictors, patient ID excluded, and verification tests passing. No predictive model is fitted. |
| Code quality and reproducibility (3 pts) | `Makefile`, pinned `requirements.txt`, `pytest.ini`, `tests/test_pipeline.py` | PARTIAL | Local Make targets and direct handbook scripts pass focused tests; package versions are pinned. A clean-clone run has not been demonstrated, and the consistency parser has permissive exceptions. |
| Structure and clarity (6 pts) | `slides/presentation.pptx`, `slides/slides.md`, `report/report.md` | PARTIAL | Main slide storyline and speaker notes exist, and G-04 aligns the seven numbered report figures and slide references with the handbook; no timing sheet is present. |
| Methodological explanation (9 pts) | `slides/presentation.pptx` speaker notes, `notebooks/01_explore.ipynb`, `report/report.md` | PARTIAL | G-02 adds the complete supported hypothesis results and explains family adjustments and H5 scope; the figure sequence is now mapped, while notebook and delivery gaps remain. |
| Results and interpretation (9 pts) | `docs/results.json`, `report/report.md`, `reports/tables/hypotheses.csv`, `reports/figures/captions.json` | PARTIAL | G-02 adds all H1-H5 feature/view results, intervals, and adjusted values to JSON/CSV and the report; G-04 aligns the canonical seven-figure sequence. Remaining report and visualization gaps are tracked below. |
| Technical understanding and Q&A (8 pts) | `docs/qa.md` | PARTIAL | The required question bank exists with 35 entries. Mock-round scorecards and cross-training evidence are absent. |
| Delivery and time (3 pts) | No rehearsal log or timing sheet found | MISSING | The two timed rehearsals and delivery-time evidence are human tasks and are not documented. |
| Report: six sections (25 pts) | `report/report.md`, `report/report.pdf` | PARTIAL | A six-section PDF exists and is 10 pages. Current section word counts are 392, 643, 552, 989, 453, and 386; Sections 1, 3, and 6 are below their handbook minima. Tables are not consistently numbered/captioned. |

## CRISP-DM phase matrix (Handbook Chapter 6)

| Phase | Evidence | Status | Assessment |
|---|---|---|---|
| Business understanding | `report/report.md` Section 1; `slides/slides.md` | PARTIAL | RQs and scope are described, but the report section is below its word target and does not present the complete success-criteria set in the handbook. |
| Data understanding | `src/data_io.py`, `docs/data_dictionary.md`, `docs/data_card.md`, `notebooks/01_explore.ipynb` | PARTIAL | The core artifacts exist; checksum differs from the expected snapshot and lineage/source matching is undocumented. |
| Preprocessing | `src/clean.py`, `src/audit.py`, `reports/tables/cleaning_log.csv`, `reports/tables/treatments.csv` | PARTIAL | Four views and logs exist, and direct audit/clean script commands now execute; remaining preprocessing documentation gaps remain. |
| EDA and statistics | `src/figures.py`, `src/hypotheses.py`, `reports/figures/`, `reports/tables/hypotheses.csv` | PARTIAL | G-02 completes the hypothesis table, G-03 adds the requested group-profile visualizations, and G-04 aligns the numbered figures; broader notebook deliverables remain. |
| Feature engineering | `src/features.py`, `docs/feature_spec.md`, `tests/test_pipeline.py` | OK | Feature sets, recoding, leakage exclusions, and verification are implemented and tested. |
| Conclusions and next steps | `report/report.md` Section 6; `slides/slides.md` | PARTIAL | Limits and planned work are stated, but Section 6 is below its word target; human delivery activities remain open. |

## Prioritized gaps

| ID | Severity | Description and evidence | Rubric points at risk | Proposed fix | Test/evidence that will prove it |
|---|---|---|---|---|---|
| G-01 | major | **DONE.** Added package-safe direct entry points for the audit, cleaning, and full-analysis scripts. The audit writes `docs/audit_summary.json` and `reports/tables/source_shift.csv`; cleaning writes its view and treatment outputs. | Cleaning/preprocessing; code quality/reproducibility | Bootstrap package context when invoked as a script; implement functional audit and cleaning `main()` entry points with an explicit warning for the approved checksum override. | **PASS:** `.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -k handbook_direct_script_entrypoints -q` → `3 passed`; subprocess checks invoke all three `python src/...py` commands and assert successful exit plus expected output. |
| G-02 | major | **DONE.** The 82-row table now has statistic, p, q, effect size, and 95% interval for every supported comparison. H1-H3 q-values adjust across views within each hypothesis; H4 and H5 adjust their feature families. H5 is the single raw-file integrity contrast (14 features), not a cleaned-view analysis: view D removes the leading block, so fabricating H5 rows for B-D would change its meaning. The report includes H1-H4 by view and the full H5 feature table. | EDA/statistics; results/interpretation; report | Add family-specific BH values and row-level multinomial bootstrap Cramer's V intervals; register all 82 records in `results.json`; explain H5's non-view scope in the report. | **PASS:** `.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -k "h1_smoking_across_views or benjamini_hochberg_and_view_d_feature_family or h2_age_and_h3_respiratory_count_across_views or hypothesis_suite_has_complete_effects_intervals_and_adjusted_p_values" -q` → `4 passed`. `python -m src.results` regenerated 82 CSV/JSON records with zero missing statistic/p/q/effect/CI fields; `python -m src.build_report` generated a 10-page report with H5 table. |
| G-03 | major | **DONE.** The analysis now generates 19 200-dpi PNGs covering the required V1-V8 and S1-S8 visualizations: class balance, schema/value sets, order/views, duplicates, block-group YES shares, age profiles, correlation/PCA, Cramer's V, smoking and feature odds ratios, permutation null, p/q comparison, power sensitivity, respiratory-count rates, and AGE scaling/feature sets. S9 remains excluded by the no-model-training rule. | EDA/statistics; results/interpretation; report; presentation | Add the requested plots with purpose-specific takeaways; register all filenames/captions in `docs/results.json`; display the gallery in the notebook; test non-empty PNGs, 200-dpi metadata, captions, and registration. | **PASS:** `.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -k requested_visualizations_have_registered_200_dpi_artifacts -q` → `1 passed`; checks all 19 images/captions, image DPI, schema missingness and 1/2 flag values, results registration, and notebook gallery. `python -m src.run_analysis` → `Figures generated: 19`; `python -m src.build_report` → 10-page PDF. Notebook executed in memory: 45 cells, no errors. |
| G-04 | major | **DONE.** The report's seven figure numbers now follow the handbook mapping. Figure 2 compares the 13 symptom/lifestyle flags across leading negatives, later negatives, and later positives (excluding `GENDER`, whose code meaning is unknown); Figures 3–7 now reference the correlation heatmap, AGE profiles, cleaning views, Cramer's V comparison, and smoking odds-ratio plot respectively. Slide references and speaker notes use the same mapping, including side-by-side figures on slides 6, 9, and 11. | EDA/statistics; structure/clarity; report; presentation | Map generated figure content, captions, report citations, slide references, and JSON registration to the handbook inventory. | **PASS:** `.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -k "handbook_figure_mapping_matches_report_and_slides or requested_visualizations_have_registered_200_dpi_artifacts" -q` → `2 passed`; validates report filenames by numbered section, slide figure filenames/references, the 13-flag Figure 2 source table, all 19 generated artifacts and their metadata. `python -m src.run_analysis`, `python -m src.build_report`, and `python -m src.build_slides` completed; report PDF has 10 pages, deck has 14 main slides and a 14-page PDF. |
| G-05 | major | Only `notebooks/01_explore.ipynb` exists. It has 43 cells (15 code cells) and is a long, multi-step walkthrough; Handbook Chapter 2.2 names `notebooks/03_eda.ipynb` and Phase 4 expects thin notebooks that import project modules. | EDA/statistics; code quality/reproducibility | Retain the introductory notebook and add a thin EDA notebook that imports the shared `src/` outputs; add other notebooks only where the handbook requires them. | Execute all submitted notebooks from a fresh kernel with no errors; assert key figures/tables are displayed and saved outputs are cleared or intentionally reproducible. |
| G-06 | major | `docs/decision_log.md` records D-01 and Phase 1 notes only. The handbook requires D-01 through D-12; the actual B/C nominal H2/H3 findings also need an explicit discrepancy decision rather than an unqualified “all associations vanish” statement. | Cleaning/preprocessing; EDA/statistics; methodological explanation; report | Complete the decision log from handbook decisions and record the measured B/C nuance without changing generated results or editing `data/raw`. | Test/document review verifies each required decision ID and an explicit handbook-versus-run discrepancy entry. |
| G-07 | major | Report section counts from the current Markdown are: Section 1 392 vs minimum 400; Section 3 552 vs minimum 700; Section 6 386 vs minimum 400. Figure/table numbering and table captions are inconsistent; the data source is not listed as a reference. | Report (25 pts) | Expand only the deficient sections using results-backed claims and limitations; number/caption tables; correct figure mapping; add the course-provided dataset citation with provenance explicitly unknown. | Automated section word-count gate, figure/table numbering check, references check, and rebuild to a 10-14 page PDF. |
| G-08 | major | Local `make all` passed earlier, but a clean-clone run under the handbook's time limit is not evidenced. Direct script commands fail as recorded in G-01. | Code quality/reproducibility; delivery | Validate the pinned requirements and documented commands in a fresh folder/environment; record elapsed time and outputs. Do not modify the raw CSV. | Fresh environment installs `requirements.txt`, runs all targets, passes tests and consistency checks within the handbook limit; save a concise run log. |
| G-09 | major | Handbook task B-4 calls for row-match/lineage checks against the referenced source files. `docs/data_card.md` appropriately marks provenance unknown, but no attempted match counts or source lineage result is present. | Data loading/organization; data understanding | Record whether the source files are available and, if so, perform the documented exact-match check; otherwise retain OPEN with evidence of unavailable source access. | A lineage table/report has explicit inputs, matching rules, counts, and a test on the matching function; if sources are unavailable, document that limitation without inventing results. |
| G-10 | major | `src/results.py::check_consistency` is invoked by Make, but only scans `report.md` and `slides.md`, exempts values from 0 to 7, and does not validate other numeric documentation or PDF/PPTX content. It currently reports PASS, but that is weaker than the handbook's all-numbers-from-results requirement. | Code quality/reproducibility; report; presentation | Make the consistency validation cover every numeric claim in report/slides source, remove broad numeric exemptions, and test known mismatch cases. Keep static rubric/slide-number metadata distinct from measured findings. | Unit tests inject an unsupported number and require failure; generated report/slide sources pass; output identifies unmatched values and source locations. |
| G-11 | minor | The reproducible figure test checks that the seven current PNG paths exist, but does not assert non-empty bytes, 200-dpi metadata, caption/takeaway presence, notebook display, or full requested inventory. | EDA/statistics; results/interpretation | Extend figure tests alongside G-03 so all required quality properties are checked. | Focused figure test verifies each applicable file's size, DPI, registration, caption, and notebook reference. |
| G-12 | major | `docs/qa.md` contains 35 answers, but mock-round scorecards, cross-training evidence, two timed rehearsal records, and timing sheet are absent. | Technical understanding/Q&A; delivery/time | Keep the existing Q&A bank; complete and log the human rehearsal, timing, and mock-round activities separately. | Rehearsal/timing log records two runs at or below the target and mock scorecards record the required coverage. |

## Run results and discrepancies

| Check | Observed output | Status |
|---|---|---|
| Direct audit command | Phase 1 reproduced the import error; now package-safe and writes audit summary plus source-shift table | RESOLVED by G-01; focused test passes |
| Direct clean command | Phase 1 reproduced the import error; now package-safe and writes view/treatment outputs | RESOLVED by G-01; focused test passes |
| Direct analysis command | Phase 1 reproduced the import error; now package-safe and emits analysis summary | RESOLVED by G-01; focused test passes |
| Package analysis command | `python -m src.run_analysis` -> completed; 22,457 x 17, zero nulls; first positive index 1,455; 103 leading-block patterns; 2,573 exact duplicates; 133 conflicting patterns; ID AUC 0.6703 | PASS with checksum warning |
| Cleaning views | A/B/C/D rows: 22,457 / 19,884 / 19,618 / 19,781 | MATCH |
| Smoking odds ratio | A 1.4867; B 1.0273; C 1.0450; D 0.9952 | MATCH |
| BH feature-family minimum q | A 2.97e-102; B 0.3645; C 0.0788; D 0.4583 | MATCH |
| H2 age | B CLES 0.5168, p 0.00520; C CLES 0.5186, p 0.00244; D CLES 0.5119, p 0.0511 | B/C small nominal findings; D not nominally significant |
| H3 respiratory count | B rho 0.01484, p 0.03634; C rho 0.02068, p 0.00378; D rho -0.00205, p 0.77281 | B/C small nominal findings; D not nominally significant |
| Tests | `python -m pytest -q` -> 16 passed | PASS |

The B/C H2 and H3 findings are not silently reconciled with the handbook headline:
they are small nominal associations in those views, while the effects attenuate or
disappear in D. The report and the completed G-02 table retain this nuance. The
actual run is retained. The checksum mismatch is also retained as a documented
deviation; `data/raw` was not changed.

## Scope constraint

The optional S9 request for raw-versus-cleaned model AUC conflicts with the hard
no-model-training rule. It is not included in the gaps to implement while that
rule remains active. Existing ID-only AUC is an integrity statistic, not a model
result.

## Phase 2 progress

G-01 through G-04 are complete and committed independently. G-03 passes its
figure acceptance test and notebook execution check; G-04's focused mapping and
visualization tests pass. Further implementation is paused at G-05 for review,
as requested by the one-gap-at-a-time workflow.
