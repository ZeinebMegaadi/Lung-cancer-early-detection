"""
Step 6 (optional) - Build the LaTeX report from the pipeline outputs.

    python scripts/build_report.py          (after python run_all.py)

Reads : results/tables/*.csv, data/clean/*.csv, results/*.txt (main.tex reads the figures from results/figures/)
Writes: report/generated/numbers.tex   \\newcommand macros for the key numbers used in the prose
        report/generated/tab_*.tex     the body (tabular) of every table that comes from a CSV
        report/build/main.pdf          pdflatex run twice (build/ is git-ignored)

Captions, labels and prose live in report/sections/*.tex; this script never touches them.
Not part of run_all.py: the report is built after the analysis, not as part of it.
"""
import re
import shutil
import subprocess

import pandas as pd

from build_clean_dataset import REFERENCE                      # reference category of every group
from common import CLEAN, NON_FEATURES, ROOT, TAB
from eda_analysis import RESEARCH_QUESTIONS
from supplementary_analysis import COLUMN_TYPES, FEATURE_GROUPS

REPORT = ROOT / "report"
GEN = REPORT / "generated"


# ---------------------------------------------------------------- formatting helpers
def tex(s):
    """Escape LaTeX special characters in plain text from a CSV."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"$\sim$"), ("^", r"\^{}")]:  # "~2.8%" in the CSVs means "about"
        s = s.replace(a, b)
    s = re.sub(r"""(?<![\w\\])(["'])([^"']+)\1(?!\w)""", r"``\2''", s)   # "word" / 'word' -> ``word''
    return s.replace(" x ", r" $\times$ ").replace("->", r"$\rightarrow$")


def n(x):
    """Integer with thousands separator."""
    return f"{int(round(x)):,}"


def pct(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}\\%"


def num(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


def sci(p, digits=2):
    """p-value in LaTeX: 0.63, 2.9 x 10^-3, < 10^-300."""
    if p == 0:
        return r"$<10^{-300}$"
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.{digits - 1}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def ci(o, lo, hi, d=1):
    return f"{o:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write(name, body):
    (GEN / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def tabular(spec, header, rows, env="tabular", width=None):
    """booktabs table body. header: list of cells; rows: list of lists, or the string 'MIDRULE'."""
    head = rf"\begin{{{env}}}" + (f"{{{width}}}" if width else "") + f"{{{spec}}}\n\\toprule\n"
    # a "|" in a header cell is a line break (two-line header via \makecell); {} stops "\\[95\% CI]" being read as \\[skip]
    head += " & ".join(r"\makecell[b]{" + c.replace("|", r"\\{}") + "}" if "|" in c else c for c in header)
    head += r" \\" + "\n\\midrule\n"
    body = "".join(r"\midrule" + "\n" if r == "MIDRULE" else " & ".join(r) + r" \\" + "\n" for r in rows)
    return head + body + f"\\bottomrule\n\\end{{{env}}}"


# ---------------------------------------------------------------- data
def csv(name, folder=TAB, **kw):
    return pd.read_csv(folder / name, **kw)


def text_value(path, pattern):
    m = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"Pattern not found in {path}: {pattern}")
    return m.groups()


VAR_LABEL = {"sex": "Sex", "age_group": "Age group (years)", "race_ethnicity": "Race / ethnicity",
             "education": "Education", "income_group": "Family income vs poverty line", "region": "Census region",
             "year": "Survey year", "smoking_status": "Cigarette smoking", "ever_ecigarette": "Ever used an e-cigarette",
             "bmi_category": "BMI category", "general_health": "Self-rated general health", "copd": "COPD",
             "heart_disease": "Heart disease", "hypertension": "Hypertension", "diabetes": "Diabetes",
             "asthma": "Asthma", "comorbidity_count": "Number of the 5 conditions"}
CAT_LABEL = {"Below poverty": "Below poverty ($<$1)", "Low": "Low (1--2)", "Middle": "Middle (2--4)",
             "High": "High (4+)", "18-44": "18--44", "45-54": "45--54", "55-64": "55--64", "65-74": "65--74"}
COND_LABEL = {"copd": "COPD", "heart_disease": "Heart disease", "hypertension": "Hypertension",
              "diabetes": "Diabetes", "asthma": "Asthma", "ever_ecigarette": "E-cigarette (ever)"}
DEMO_LABEL = {"sex": "Sex", "race_ethnicity": "Race / ethnicity", "education": "Education",
              "income_group": "Income vs poverty line", "bmi_category": "BMI category",
              "general_health": "Self-rated health"}
GROUP_LABEL = {"sex": "sex", "race_ethnicity": "race/ethnicity", "education": "education",
               "income_group": "income vs poverty line", "bmi_category": "BMI category",
               "general_health": "general health", "smoking_status": "smoking status", "region": "region"}
REGIONAL_LABEL = {"pm25_mean": "PM2.5", "smoking_rate": "Regional smoking rate",
                  "lung_cancer_incidence": "Registry incidence"}


# ---------------------------------------------------------------- tables
def composition_tables():
    c = csv("dataset_composition.csv", dtype={"adults": "Int64", "cases": "Int64", "n_columns": "Int64"})
    rows = []
    for _, r in c[c.block.str.startswith("1")].iterrows():
        label = r"\textbf{All years}" if r["item"] == "all years" else r["item"]
        cells = [n(r.adults), str(r.cases), pct(r.rate_pct)]
        if r["item"] == "all years":
            cells = [rf"\textbf{{{x}}}" for x in cells]
        rows.append([label] + cells)
    rows.append("MIDRULE")
    for _, r in c[c.block.str.startswith("4") & (c["item"] != "all")].iterrows():
        rows.append([{"train": "Training rows", "test": "Test rows"}[r["item"]], n(r.adults), str(r.cases),
                     pct(r.rate_pct)])
    write("tab_composition_years.tex", tabular("lrrr", ["", "Adults", "Cases", "Rate (sample)"], rows))

    names = {"identifier": "Identifier", "weight": "Survey weight", "target": "Target", "numeric": "Numeric",
             "ordinal": "Ordinal", "nominal": "Nominal", "binary": "Binary", "engineered": "Engineered",
             "regional": "Regional (joined)"}
    rows = [[names[r["item"]], str(r.n_columns), tex(r["columns"])]
            for _, r in c[c.block.str.startswith("2") & (c["item"] != "all columns")].iterrows()]
    total = int(c.loc[c["item"] == "all columns", "n_columns"].iloc[0])
    rows += ["MIDRULE", [r"\textbf{Total}", rf"\textbf{{{total}}}", ""]]
    write("tab_composition_columns.tex",
          tabular(r"l r >{\raggedright\arraybackslash}X", ["Type", "Columns", "Which"], rows, "tabularx", r"\textwidth"))
    write("tab_feature_groups.tex", feature_group_table())


def feature_group_table():
    feats = [x for x in csv("nhis_lung_cancer_model_ready.csv", CLEAN, nrows=0).columns
             if x not in NON_FEATURES]
    scaled = set(csv("scaler_parameters.csv", CLEAN).column)
    group_of = {f: next(g for g, pre in FEATURE_GROUPS.items() if f.startswith(pre)) for f in feats}
    labels = {"demographic": "Demographic", "socio-economic": "Socio-economic", "smoking": "Smoking and nicotine",
              "health": "Health", "regional": "Regional context"}
    rows = []
    for g in FEATURE_GROUPS:
        members = [f for f in feats if group_of[f] == g]
        parts, done = [], set()
        for f in members:
            prefix = next((p for p in REFERENCE if f.startswith(p + "_")), None)
            if prefix and prefix not in done:
                done.add(prefix)
                levels = [m[len(prefix) + 1:] for m in members if m.startswith(prefix + "_")]
                parts.append(f"{GROUP_LABEL[prefix]}: {tex(', '.join(levels))} "
                             f"(reference: {tex(REFERENCE[prefix])})")
            elif not prefix:
                parts.append(tex(f) + (" (scaled)" if f in scaled else ""))
        rows.append([labels[g], "; ".join(parts), str(len(members))])
    rows += ["MIDRULE", [r"\textbf{Total}", "", rf"\textbf{{{len(feats)}}}"]]
    return tabular(r"l >{\raggedright\arraybackslash}X r", ["Group", "Features", "Count"], rows, "tabularx", r"\textwidth")


def dictionary_table():
    d = csv("data_dictionary.csv", CLEAN)
    kind = {c: k for k, cols in COLUMN_TYPES.items() for c in cols}
    rows = [[tex(r.column), tex(r.meaning), kind[r.column], tex(r.built_from)] for r in d.itertuples()]
    write("tab_dictionary.tex", tabular(r">{\raggedright\arraybackslash}p{3.8cm} >{\raggedright\arraybackslash}X l "
                                        r">{\raggedright\arraybackslash}p{3.4cm}",
                                        ["Column", "Meaning", "Type", "Built from"], rows, "tabularx", r"\textwidth"))


def profile_tables():
    p = csv("population_profile.csv")
    blocks = {"tab_profile_demo.tex": ["sex", "age_group", "race_ethnicity"],
              "tab_profile_socio.tex": ["education", "income_group", "region", "year"],
              "tab_profile_smoking.tex": ["smoking_status", "ever_ecigarette", "bmi_category", "general_health"],
              "tab_profile_conditions.tex": ["copd", "heart_disease", "hypertension", "diabetes", "asthma",
                                             "comorbidity_count"]}
    header = ["Group", "Adults", r"\% of|sample", r"\% of US|adults (wtd)", "Lung|cancer", "No lung|cancer",
              "Rate|(sample)", "Rate|(wtd)", "Mean|age"]
    for name, variables in blocks.items():
        rows = []
        for v in variables:
            if rows:
                rows.append("MIDRULE")
            rows.append([rf"\textbf{{{VAR_LABEL[v]}}}"] + [""] * 8)
            for r in p[p.variable == v].itertuples():
                cat = str(r.category)
                rows.append([r"\quad " + CAT_LABEL.get(cat, tex(cat)), n(r.adults), num(r.pct_of_sample, 1),
                             num(r.pct_of_population_weighted, 1), n(r.cases), n(r.non_cases),
                             pct(r.rate_sample_pct), pct(r.rate_weighted_pct), num(r.mean_age, 1)])
        write(name, tabular("l" + "r" * 8, header, rows, "tabular*", r"\textwidth").replace(
            r"{l" + "r" * 8 + "}", r"{@{\extracolsep{\fill}}l" + "r" * 8 + "}", 1))

    g = csv("group_profile.csv")
    rows = []
    for v in ["sex", "race_ethnicity"]:
        if rows:
            rows.append("MIDRULE")
        rows.append([rf"\textbf{{{VAR_LABEL[v]}}}"] + [""] * 9)
        for r in g[g.variable == v].itertuples():
            women = "--" if v == "sex" else num(r.pct_female, 1)
            rows.append([r"\quad " + tex(r.group), n(r.adults), n(r.cases), num(r.mean_age, 1), f"{r.median_age:g}",
                         num(r.pct_65_plus, 1), women, num(r.pct_ever_smoked, 1), num(r.pct_copd, 1),
                         pct(r.rate_weighted_pct)])
    header = ["Group", "Adults", "Cases", "Mean|age", "Median|age", r"\% aged|65+", r"\%|women", r"\% ever|smoked",
              r"\%|COPD", "Rate|(wtd)"]
    write("tab_group_profile.tex", tabular(r"@{\extracolsep{\fill}}l" + "r" * 9, header, rows, "tabular*", r"\textwidth"))


def cases_table():
    c = csv("cases_vs_noncases.csv").set_index("characteristic")
    def fmt(x, unit):
        return {"count": n, "years": lambda v: f"{v:.1f}", "%": lambda v: f"{v:.1f}\\%",
                "conditions": lambda v: f"{v:.1f}"}[unit](x)
    rows = []
    for ch, r in c.iterrows():
        if ch == "Age, Q3":
            continue
        if ch == "Age, Q1":
            q3 = c.loc["Age, Q3"]
            rows.append(["Age, interquartile range (years)", f"{r.lung_cancer:.0f}--{q3.lung_cancer:.0f}",
                         f"{r.no_lung_cancer:.0f}--{q3.no_lung_cancer:.0f}"])
            continue
        label = tex(ch) + (" (years)" if r.unit == "years" else "")
        label = label.replace("Race/ethnicity: ", "Race/ethnicity -- ").replace("Region: ", "Region -- ")
        rows.append([label, fmt(r.lung_cancer, r.unit), fmt(r.no_lung_cancer, r.unit)])
    write("tab_cases_vs_noncases.tex", tabular("lrr", ["Characteristic", "Lung cancer", "No lung cancer"], rows))


def missingness_table():
    m = csv("missingness_vs_target.csv")
    rows, prev = [], None
    for _, r in m.iterrows():
        key = (r.pct_missing, r.chi2, r.p)
        if rows and key == prev:                       # smoking_status and ever_smoker: same gaps, same test
            rows[-1][0] += " / " + tex(r.column)
            continue
        prev = key
        rows.append([tex(r.column), num(r.pct_missing), pct(r["lc_rate_if_missing_%"]),
                     pct(r["lc_rate_if_present_%"]), f"{r.chi2:.2f}", f"{r.p:.2f}", f"{r.p_holm:.2f}"])
    write("tab_missingness.tex", tabular("lrrrrrr", [r"Column", r"\% missing", "Rate if missing", "Rate if present",
                                                     r"$\chi^2$", "$p$", "Holm $p$"], rows))


def outlier_table():
    o = csv("outlier_check.csv", CLEAN)
    rows = [[tex(r.column), f"[{num(r.lower_bound, 1)}, {num(r.upper_bound, 1)}]",
             f"{n(r.n_outliers)} ({r.pct_outliers:g}\\%)",
             tex(r.decision).replace("KEEP - ", "keep -- ").replace(" - ", " -- ")] for r in o.itertuples()]
    write("tab_outliers.tex", tabular(r"l l r >{\raggedright\arraybackslash}X",
                                      ["Column", "Bounds", "Flagged", "Decision and reason"], rows, "tabularx", r"\textwidth"))


def before_after_table():
    s = csv("before_after_summary.csv", CLEAN)
    rows = [[tex(r.step), n(r.rows), str(r.columns), n(r.missing_cells), tex(r.what_changed)] for r in s.itertuples()]
    write("tab_before_after.tex", tabular(r">{\raggedright\arraybackslash}p{3.6cm} r r r >{\raggedright\arraybackslash}X",
                                          ["Step", "Rows", "Columns", "Missing cells", "What changed"], rows,
                                          "tabularx", r"\textwidth"))


def numeric_shape_table():
    s = csv("numeric_shape_statistics.csv", index_col=0)
    unit = {"age": "age (years)", "comorbidity_count": "comorbidity\\_count", "pm25_mean": r"pm25\_mean (\textmu g/m$^3$)",
            "smoking_rate": r"smoking\_rate (\%)", "lung_cancer_incidence": r"lung\_cancer\_incidence (/100k)"}
    dec = {"age": 1, "comorbidity_count": 2, "pm25_mean": 2, "smoking_rate": 2, "lung_cancer_incidence": 1}
    rows = [[unit[c], *(num(r[k], dec[c]) for k in ["mean", "median", "std", "IQR", "min", "max"]),
             num(r.skewness), num(r.kurtosis_excess)] for c, r in s.iterrows()]
    write("tab_numeric_shape.tex", tabular("lrrrrrrrr", ["Column", "Mean", "Median", "SD", "IQR", "Min", "Max",
                                                         "Skewness", "Excess|kurtosis"], rows))


def smoking_table():
    s = csv("rate_by_smoking.csv").set_index("smoking_status")
    never = s.loc["Never", "weighted_rate_%"]
    rows = [[k, n(r.n), str(int(r.cases)), pct(r["rate_%"]), pct(r["weighted_rate_%"]),  # iterrows makes it float
             "1 (reference)" if k == "Never" else f"{r['weighted_rate_%'] / never:.1f}"] for k, r in s.iterrows()]
    write("tab_rate_smoking.tex", tabular("lrrrrr", ["Smoking status", "Adults", "Cases", "Sample|rate",
                                                     "Weighted|(population) rate", "Risk ratio|(weighted)"], rows))


def smoking_age_table():
    s = csv("smoking_by_age_group.csv")
    rows = []
    for r in s.itertuples():
        grp = CAT_LABEL.get(r.group, r.group)
        if r.group.startswith("Crude") or r.group.startswith("Mantel"):
            if r.group.startswith("Crude"):
                rows.append("MIDRULE")
            rows.append([grp.replace("Mantel-Haenszel", "Mantel--Haenszel"), n(r.adults), "", "",
                         ci(r.odds_ratio, r.ci_low, r.ci_high)])
        else:
            rows.append([grp, n(r.adults), f"{r.cases_ever_smokers:.0f}", f"{r.cases_never_smokers:.0f}",
                         ci(r.odds_ratio, r.ci_low, r.ci_high)])
    write("tab_smoking_age.tex", tabular("lrrrr", ["Age group", "Adults", "Cases,|ever-smokers", "Cases,|never-smokers",
                                                   r"Odds ratio [95\% CI]"], rows))


def holm_lookup():
    t = csv("all_tests_holm.csv")
    return dict(zip(t.test, t.p_holm)), t


def conditions_table():
    holm, _ = holm_lookup()
    c = csv("conditions_odds_ratios.csv")
    rows = [[COND_LABEL[r.condition], pct(r["rate_if_yes_%"]), pct(r["rate_if_no_%"]),
             ci(r.odds_ratio, r.ci_low, r.ci_high), f"{r.cramers_v:.3f}",
             sci(holm[f"{r.condition} vs lung cancer (chi-square)"])] for _, r in c.iterrows()]
    write("tab_conditions.tex", tabular("lrrrrr", ["Condition", "Rate|if yes", "Rate|if no", r"Odds ratio|[95\% CI]",
                                                   r"Cram\'er's $V$", "Holm $p$"], rows))


def copd_tables():
    s = csv("copd_stratified_by_smoking.csv")
    rows = []
    for r in s.itertuples():
        if r.group.startswith("Crude"):
            rows.append("MIDRULE")
        rows.append([r.group.replace("Mantel-Haenszel", "Mantel--Haenszel"), n(r.n), str(r.cases),
                     ci(r.odds_ratio, r.ci_low, r.ci_high)])
    write("tab_copd_smoking.tex", tabular("lrrr", ["Group", "Adults", "Cases", r"Odds ratio [95\% CI]"], rows))
    a = csv("copd_adjusted.csv")
    label = {"none (crude)": "None (crude)", "smoking status": "Smoking status ({} strata)",
             "smoking status x age group": r"Smoking status $\times$ age group ({} strata)"}
    rows = [[label[r.adjustment].format(r.strata), ci(r.odds_ratio, r.ci_low, r.ci_high)] for r in a.itertuples()]
    write("tab_copd_adjusted.tex", tabular("lr", ["Adjusted for", r"Odds ratio [95\% CI]"], rows))


def demographics_table():
    holm, _ = holm_lookup()
    d = csv("demographics_chi_square.csv")
    rows = [[DEMO_LABEL[r.variable],
             f"{tex(r.highest_rate_group)} {r['highest_weighted_rate_%']:.2f}\\% $\\rightarrow$ "
             f"{tex(r.lowest_rate_group)} {r['lowest_weighted_rate_%']:.2f}\\%",
             f"{r.chi2:,.1f} ({r.dof})", sci(holm[f"{r.variable} vs lung cancer (chi-square)"]), f"{r.cramers_v:.3f}"]
            for _, r in d.iterrows()]
    write("tab_demographics.tex", tabular(r"l >{\raggedright\arraybackslash}X r r r",
                                          ["Factor", r"Weighted rate: highest $\rightarrow$ lowest group",
                                           r"$\chi^2$ (df)", "Holm $p$", r"Cram\'er's $V$"], rows, "tabularx", r"\textwidth"))


def correlation_tables():
    h = csv("high_correlation_pairs.csv")
    measure = {"eta": r"$\eta$", "Cramer's V": r"Cram\'er's $V$", "Spearman": "Spearman"}
    pair = lambda a, b: (tex(a) + " -- " + tex(b)).replace(r"\_", r"\_\allowbreak{}")  # long names break after _
    rows = [[r.matrix.split(" (")[0].replace("Fig ", "Fig.~"), pair(r.var_a, r.var_b), measure[r.measure],
             f"{r.value:.2f}", r.type, tex(r.explanation).replace("2019-2023", "2019--2023").replace("50-80", "50--80")]
            for r in h.itertuples()]
    write("tab_high_pairs.tex", tabular(r"l >{\raggedright\arraybackslash}p{4.2cm} l r l >{\raggedright\arraybackslash}X",
                                        ["Matrix", "Pair", "Measure", "Value", "Type", "Why it is high"], rows,
                                        "tabularx", r"\textwidth"))
    b = csv("target_correlation_bound.csv")
    label = {"copd": "COPD", "heart_disease": "Heart disease", "hypertension": "Hypertension", "diabetes": "Diabetes",
             "asthma": "Asthma", "ever_smoker": "Ever smoked", "screening_age_smoker": "Screening-age smoker",
             "ever_ecigarette": "E-cigarette (ever)"}
    rows = [[label[r.feature], f"{n(r.adults)} ({r.cases})", pct(r["prevalence_%"], 1), f"{r.phi:.3f}",
             f"{r.phi_max:.3f}", f"{r['phi_share_of_max_%']:.0f}\\%", f"{r.odds_ratio:.1f}"] for _, r in b.iterrows()]
    write("tab_target_bound.tex", tabular("lrrrrrr", ["Factor", "Adults (cases)", "Prevalence", r"$\varphi$|observed",
                                                      r"$\varphi_{\max}$", "Share|of max", "Odds|ratio"], rows))


def regional_table():
    r = csv("regional_within_between.csv")
    rows = [[REGIONAL_LABEL[x.regional_measure], f"{num(x.rho_all_20)} ($p$ = {sci(x.p_all_20)})",
             num(x.rho_between_4_regions), f"{num(x.rho_within_region)} ($p$ = {sci(x.p_within_region)})"]
            for x in r.itertuples()]
    write("tab_regional.tex", tabular("lrrr", ["Regional measure", "All 20|region-years", "Between|4 regions",
                                               "Within regions|($n$ = 20)"], rows))


TEST_ORDER = [("Smoking status vs lung cancer (chi-square)", r"Smoking status ($\chi^2$)"),
              ("Age vs lung cancer (Mann-Whitney)", "Age (Mann--Whitney $U$)"),
              ("copd vs lung cancer (chi-square)", r"COPD ($\chi^2$)"),
              ("COPD vs lung cancer adjusted for smoking (CMH)", "COPD adjusted for smoking (CMH)"),
              ("heart_disease vs lung cancer (chi-square)", r"Heart disease ($\chi^2$)"),
              ("hypertension vs lung cancer (chi-square)", r"Hypertension ($\chi^2$)"),
              ("diabetes vs lung cancer (chi-square)", r"Diabetes ($\chi^2$)"),
              ("asthma vs lung cancer (chi-square)", r"Asthma ($\chi^2$)"),
              ("ever_ecigarette vs lung cancer (chi-square)", r"E-cigarette, ever ($\chi^2$)"),
              ("sex vs lung cancer (chi-square)", r"Sex ($\chi^2$)"),
              ("race_ethnicity vs lung cancer (chi-square)", r"Race / ethnicity ($\chi^2$)"),
              ("education vs lung cancer (chi-square)", r"Education ($\chi^2$)"),
              ("income_group vs lung cancer (chi-square)", r"Income vs poverty line ($\chi^2$)"),
              ("bmi_category vs lung cancer (chi-square)", r"BMI category ($\chi^2$)"),
              ("general_health vs lung cancer (chi-square)", r"Self-rated general health ($\chi^2$)"),
              ("pm25_mean vs NHIS regional rate (Spearman, 20 region-years)", "PM2.5 vs regional rate (Spearman, $n$ = 20)"),
              ("smoking_rate vs NHIS regional rate (Spearman, 20 region-years)",
               "Smoking rate vs regional rate (Spearman, $n$ = 20)"),
              ("lung_cancer_incidence vs NHIS regional rate (Spearman, 20 region-years)",
               r"Registry incidence vs regional rate (Spearman, $n$ = 20)$^{a}$")]


def tests_table():
    _, t = holm_lookup()
    t = t.set_index("test")
    assert set(t.index) == {k for k, _ in TEST_ORDER}, "all_tests_holm.csv has tests the report does not list"
    rows = [[t.loc[k, "research_question"], label, sci(t.loc[k, "p"]), sci(t.loc[k, "p_holm"]),
             "yes" if t.loc[k, "significant_after_correction"] else "no"] for k, label in TEST_ORDER]
    write("tab_all_tests.tex", tabular(r"l >{\raggedright\arraybackslash}X r r l", ["RQ", "Test", "$p$", "Holm $p$",
                                                                                     "Significant"], rows,
                                       "tabularx", r"\textwidth"))


def design_table():
    d = csv("design_based_odds_ratios.csv")
    rows = [[tex(r.exposure), n(r.adults), ci(r.or_unweighted, r.ci_low_unweighted, r.ci_high_unweighted, 2),
             ci(r.or_design, r.ci_low_design, r.ci_high_design, 2)] for r in d.itertuples()]
    write("tab_design.tex", tabular("lrrr", ["Exposure", "Adults", r"Unweighted OR|[95\% CI]",
                                             r"Weighted, clustered|OR [95\% CI]"], rows))


def scaler_table():
    sp = csv("scaler_parameters.csv", CLEAN)
    ms = csv("nhis_lung_cancer_model_ready_scaled.csv", CLEAN, usecols=["split"] + list(sp.column))
    test_mean = ms[ms.split == "test"][list(sp.column)].mean()
    dec = {"age": 2, "pm25_mean": 3, "smoking_rate": 3, "lung_cancer_incidence": 2}
    rows = [[tex(r.column), f"{r.train_mean:.{dec[r.column]}f}", f"{r.train_std:.{dec[r.column]}f}",
             num(test_mean[r.column], 3)] for r in sp.itertuples()]
    write("tab_scaler.tex", tabular("lrrr", ["Column", "Training|mean", "Training|SD", "Scaled|test mean"], rows))


RULES = [("1. Redundant by construction", lambda r: r.startswith("redundant")),
         ("2. Constant in training rows", lambda r: r.startswith("constant")),
         (r"2b. Near-constant: more than 99.5\% of training rows share one value", lambda r: r.startswith("near-constant")),
         (r"3. Near-duplicates: $|r| > 0.95$ with a column already kept", lambda r: r.startswith("near-duplicate")),
         (r"4a. Exact linear combination of other features (VIF = $\infty$)", lambda r: r.startswith("exact linear")),
         ("4b. Reference category of each one-hot group", lambda r: r.startswith("reference category")),
         ("4c. Near-collinear block (region dummies vs regional columns)", lambda r: r.startswith("the 3 regional"))]


def selection_table():
    ff = csv("final_features.csv", CLEAN)
    drop = ff[ff.decision == "DROP"]
    rows, used = [], set()
    for name, rule in RULES:
        hit = drop[drop.reason.map(rule)]
        used |= set(hit.column)
        rows.append([name, str(len(hit)), tex(", ".join(hit.column)) if len(hit) else "--"])
    assert used == set(drop.column), "a DROP reason does not match any rule of Table 29"
    rows += ["MIDRULE", ["Total dropped", str(len(drop)), f"{len(ff[ff.decision == 'KEEP'])} features kept"]]
    write("tab_selection.tex", tabular(r">{\raggedright\arraybackslash}p{5.2cm} r >{\raggedright\arraybackslash}X",
                                       ["Rule", "Dropped", "Features"], rows, "tabularx", r"\textwidth"))


def vif_table():
    vb = csv("vif_before.csv").set_index("feature")["vif"]
    va = csv("vif_after.csv").set_index("feature")["vif"]
    def fmt(v):
        return r"$\infty$" if v == float("inf") else f"{v:.2f}"
    def rng(vals):
        vals = list(vals)
        if not vals:
            return "dropped"
        if any(v == float("inf") for v in vals):
            return r"$\infty$"
        lo, hi = min(vals), max(vals)
        return fmt(lo) if lo == hi else f"{lo:.2f}--{hi:.2f}"
    rows = []
    groups = [p for p in ["race_ethnicity", "education", "income_group", "bmi_category", "general_health",
                          "smoking_status", "region"]]
    singles = [f for f in vb.index if not any(f.startswith(g + "_") for g in groups)]
    for f in singles:
        after = fmt(va[f]) if f in va.index else "dropped (4a)"
        rows.append([tex(f), fmt(vb[f]), after])
    for g in groups:
        members = [f for f in vb.index if f.startswith(g + "_")]
        kept = [f for f in members if f in va.index]
        gone = [f[len(g) + 1:] for f in members if f not in va.index]
        note = "4c" if g == "region" else "4b"
        after = (rng(va[kept]) if kept else "dropped") + (f" ({tex(', '.join(gone))} dropped, {note})" if gone else "")
        rows.append([tex(g) + rf"\_* ({len(members)} dummies)", rng(vb[members]), after])
    rows.sort(key=lambda r: (r[1] != r"$\infty$", r[0]))
    rows += ["MIDRULE", [r"\textbf{Maximum (finite)}", f"{vb[vb != float('inf')].max():.2f}", rf"\textbf{{{va.max():.2f}}}"]]
    write("tab_vif.tex", tabular(r">{\raggedright\arraybackslash}p{5.5cm} r >{\raggedright\arraybackslash}X",
                                 ["Feature", "VIF after rules 1--3", "VIF after rule 4"], rows, "tabularx", r"\textwidth"))


def mi_table():
    m = csv("mutual_information_checked.csv").sort_values("mi", ascending=False)
    rows = [[tex(r.feature), f"{r.mi:.5f}", pct(r.mi_pct_of_target_entropy, 1), f"{r.null_95th_percentile:.5f}",
             f"{r.permutation_p:.2f}"] for r in m.itertuples()]
    write("tab_mi.tex", tabular("lrrrr", ["Feature", "MI (nats)", "Share of target|entropy", "Null, 95th|percentile",
                                          "Permutation|$p$"], rows))


def pca_table(num_macros):
    ev = csv("pca_explained_variance.csv")
    ld = csv("pca_loadings.csv", index_col=0)
    nice = {"screening_age_smoker_unknown": "screening\\_age\\_smoker\\_unknown"}
    def strong(pc, k=0.40):
        s = ld[pc][ld[pc].abs() >= k].sort_values(key=abs, ascending=False)
        return ", ".join(f"{nice.get(i, tex(i))} {v:+.2f}".replace("-", "$-$") for i, v in s.items())
    p3_next = ld["PC3"][(ld["PC3"].abs() < 0.40)].sort_values(key=abs, ascending=False).head(4)
    m = num_macros
    rows = [["Variance explained by PC1 / PC2 / PC3",
             rf"{ev.iloc[0, 2]:.1f}\% / {ev.iloc[1, 2]:.1f}\% / {ev.iloc[2, 2]:.1f}\% ({100 / len(ld):.1f}\% expected per "
             "component if all features were unrelated)"],
            [r"Components needed for 80\% / 90\% / 95\% of the variance",
             f"{m['pcaEighty']} / {m['pcaNinety']} / {m['pcaNinetyFive']} of {len(ld)}"],
            ["Kaiser rule (eigenvalue $>$ 1)", f"{m['pcaKaiser']} components"],
            ["Components with almost zero variance (eigenvalue $<$ 0.01)",
             f"{m['pcaNearZero']} (there were 8 with the previous 48 features: the dependencies removed by rule 4, "
             r"\S\ref{sec:selection})"],
            [f"Variance of the test rows kept by the first {m['pcaNinetyFive']} components",
             rf"{m['pcaTestKept']}\% (the PCA generalises; it never saw the test rows)"],
            [r"PC1 -- ``age and chronic-disease burden''", strong("PC1")],
            [r"PC2 -- ``skipped the smoking questions''", strong("PC2")],
            [r"PC3 -- ``older age vs current smoking and regional smoking level'' (mixed)",
             strong("PC3") + " (next below 0.40: " + ", ".join(f"{tex(i)} {v:+.2f}".replace("-", "$-$")
                                                              for i, v in p3_next.items()) + ")"]]
    write("tab_pca.tex", tabular(r">{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}X",
                                 ["Quantity", "Value"], rows, "tabularx", r"\textwidth"))


CRITERIA = ["No duplicate person IDs", "No missing values in the model-ready file", "Regional join adds no rows",
            "Same case rate in train and test", "No test information in fitted steps",
            "Every RQ tested, with effect sizes and Holm correction", "Every feature has a KEEP / DROP reason",
            "CANEV\\_A never a feature"]


def criteria_table():
    sc = csv("success_criteria_check.csv")
    _, t = holm_lookup()
    counts = t.research_question.value_counts()
    obs = list(sc.observed.astype(str))
    obs[2] = obs[2].replace("->", r"$\rightarrow$")
    obs[3] = obs[3].replace("%", r"\%")
    obs[4] = "imputation, selection (incl.\\ the VIF step), scaling and PCA fitted on training rows; scaler asserted in code"
    obs[5] = f"{len(t)} tests: " + ", ".join(f"{rq} ({counts.get(rq, 0)})" for rq in RESEARCH_QUESTIONS)
    obs[0] = obs[0] if obs[0] != "0" else "0"
    obs[1] = "0 missing cells"
    obs[7] = "absent" if obs[7] == "False" else "PRESENT"
    met = list(sc.met)
    met.insert(4, True)                                # criterion 5 is asserted in supplementary_analysis.py
    rows = [[str(i + 1), CRITERIA[i], obs[i], "yes" if met[i] else "no"] for i in range(8)]
    write("tab_criteria.tex", tabular(r"r >{\raggedright\arraybackslash}p{5.6cm} >{\raggedright\arraybackslash}X l",
                                      ["\\#", "Criterion", "Observed", "Met"], rows, "tabularx", r"\textwidth"))


def screening_table():
    b = csv("screening_rule_baseline.csv")
    rows = [[{"all adults": "All adults", "test rows": "Test rows"}[r.rows], n(r.adults),
             f"{n(r.flagged)} ({r.pct_flagged:.1f}\\%)", f"{r.cases_flagged} of {r.cases}", pct(r.recall_pct, 1),
             pct(r.precision_pct), f"{r.lift_vs_base_rate:.1f}"] for r in b.itertuples()]
    write("tab_screening.tex", tabular("lrrrrrr", ["Rows", "Adults", "Flagged", "Cases|flagged", "Recall", "Precision",
                                                   "Lift"], rows))


def feature_list_table():
    feats = [x for x in csv("nhis_lung_cancer_model_ready.csv", CLEAN, nrows=0).columns
             if x not in NON_FEATURES]
    scaled = set(csv("scaler_parameters.csv", CLEAN).column)
    cells = [(str(i + 1), tex(f) + (" (scaled)" if f in scaled else "")) for i, f in enumerate(feats)]
    half = (len(cells) + 1) // 2
    rows = []
    for i in range(half):
        right = cells[i + half] if i + half < len(cells) else ("", "")
        rows.append([*cells[i], *right])
    write("tab_feature_list.tex", tabular("rl rl", ["\\#", "Feature", "\\#", "Feature"], rows))


# ---------------------------------------------------------------- numbers.tex
def numbers():
    df = csv("nhis_lung_cancer_clean.csv", CLEAN, usecols=["region", "year", "lung_cancer"])
    ff = csv("final_features.csv", CLEAN)
    cand = ff[ff.decision != "NOT A FEATURE"]
    comp = csv("dataset_composition.csv", dtype={"adults": "Int64", "cases": "Int64"}).set_index("item")
    ev = csv("pca_explained_variance.csv")
    cum = ev["cumulative_%"].values
    va = csv("vif_after.csv")["vif"]
    vb = csv("vif_before.csv")
    mi = csv("feature_relevance_mutual_information.csv", index_col=0)["mutual_information"]
    holm, t = holm_lookup()
    rw = csv("regional_within_between.csv").set_index("regional_measure")
    cells = df.groupby(["region", "year"]).size()
    test_kept, = text_value("results/pca_results.txt", r"keep ([\d.]+)% of the TEST variance")
    case_mean, = text_value("results/pca_results.txt", r"mean PC1 score = ([+-][\d.]+) for lung cancer cases")
    m = {
        "nAdults": n(len(df)), "nCases": str(int(df.lung_cancer.sum())),
        "sampleRate": f"{100 * df.lung_cancer.mean():.2f}",
        "nRawRows": n(csv("before_after_summary.csv", CLEAN).rows.iloc[0]),
        "nTrain": n(comp.loc["train", "adults"]), "nTest": n(comp.loc["test", "adults"]),
        "nTrainCases": str(comp.loc["train", "cases"]), "nTestCases": str(comp.loc["test", "cases"]),
        "nCandidates": str(len(cand)), "nFeatures": str((cand.decision == "KEEP").sum()),
        "nDropped": str((cand.decision == "DROP").sum()), "nModelColumns": str((cand.decision == "KEEP").sum() + 4),
        "nScaled": str(len(csv("scaler_parameters.csv", CLEAN))),
        "maxVIF": f"{va.max():.2f}", "maxVIFshort": f"{va.max():.1f}", "nInfVIF": str(int(vb.exact_dependency.sum())),
        "nVIFaboveTenBefore": str(int((vb.vif > 10).sum())), "nFeaturesBefore": str(len(vb)),
        "pcaEighty": str(int((cum < 80).sum() + 1)), "pcaNinety": str(int((cum < 90).sum() + 1)),
        "pcaNinetyFive": str(int((cum < 95).sum() + 1)), "pcaKaiser": str(int((ev.eigenvalue > 1).sum())),
        "pcaNearZero": str(int((ev.eigenvalue < 0.01).sum())), "pcaPCone": f"{ev.iloc[0, 2]:.1f}",
        "pcaTestKept": test_kept, "pcaCaseMean": case_mean,
        "pcaReduction": f"{100 * (1 - int((cum < 95).sum() + 1) / len(ev)):.0f}",
        "nTests": str(len(t)), "nSignificant": str(int(t.significant_after_correction.sum())),
        "rhoRegistry": f"{rw.loc['lung_cancer_incidence', 'rho_all_20']:.2f}",
        "pRegistry": sci(rw.loc["lung_cancer_incidence", "p_all_20"]),
        "holmRegistry": sci(holm["lung_cancer_incidence vs NHIS regional rate (Spearman, 20 region-years)"]),
        "regionYearMin": n(cells.min()), "regionYearMax": n(cells.max()),
        "miAge": f"{mi['age']:.4f}", "miCOPD": f"{mi['copd']:.4f}", "miScreening": f"{mi['screening_age_smoker']:.4f}",
        "formerSmokerAge": f"{csv('population_profile.csv').query('variable == \"smoking_status\" and category == \"Former\"').mean_age.iloc[0]:.1f}",
    }
    or_s = csv("rate_by_smoking.csv")
    copd = csv("copd_stratified_by_smoking.csv").set_index("group")
    m["orCOPDcrude"] = f"{copd.loc['Crude (smoking ignored)', 'odds_ratio']:.1f}"
    m["orCOPDsmoking"] = f"{copd.loc['Mantel-Haenszel (adjusted for smoking)', 'odds_ratio']:.1f}"
    m["orCOPDsmokingAge"] = f"{csv('copd_adjusted.csv').odds_ratio.iloc[-1]:.1f}"
    m["orCOPD"] = f"{csv('conditions_odds_ratios.csv').set_index('condition').loc['copd', 'odds_ratio']:.1f}"
    sa = csv("smoking_by_age_group.csv").set_index("group")
    m["orSmokingMH"] = f"{sa.loc['Mantel-Haenszel (adjusted for age group)', 'odds_ratio']:.1f}"
    o, lo, hi = map(float, text_value("results/eda_results.txt",
                                      r"ever vs never smokers: ([\d.]+) \(95% CI ([\d.]+) - ([\d.]+)\)"))
    m.update(orSmoking=f"{o:.1f}", orSmokingLo=f"{lo:.1f}", orSmokingHi=f"{hi:.1f}", orSmokingExact=f"{o:.2f}")
    assert len(or_s) == 3
    lines = ["% Generated by scripts/build_report.py from the pipeline outputs - do not edit by hand.",
             "% Every macro is followed by \\xspace, so it can be used inside a sentence."]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return m


# ---------------------------------------------------------------- main
def compile_pdf():
    # ponytail: plain pdflatex x2, not latexmk (MiKTeX's latexmk needs Perl); no bibtex, references are hand-written
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise SystemExit("pdflatex not found: install MiKTeX or TeX Live. Generated files were written.")
    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"]
    (REPORT / "build").mkdir(exist_ok=True)
    for _ in range(2):
        if subprocess.run(cmd, cwd=REPORT, stdout=subprocess.DEVNULL).returncode:
            raise SystemExit("LaTeX failed - see report/build/main.log")
    print("Compiled report/build/main.pdf")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    m = numbers()
    composition_tables(); dictionary_table(); profile_tables(); cases_table(); missingness_table()
    outlier_table(); before_after_table(); numeric_shape_table(); smoking_table(); smoking_age_table()
    conditions_table(); copd_tables(); demographics_table(); correlation_tables(); regional_table()
    tests_table(); design_table(); scaler_table(); selection_table(); vif_table(); mi_table(); pca_table(m)
    write("tab_feature_groups_final.tex", feature_group_table())
    criteria_table(); screening_table(); feature_list_table()
    print(f"Wrote {len(list(GEN.glob('*.tex')))} files to report/generated/.")
    compile_pdf()


if __name__ == "__main__":
    main()
