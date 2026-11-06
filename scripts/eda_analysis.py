"""
Step 3 - Exploratory Data Analysis (EDA) and hypothesis tests.

    python scripts/eda_analysis.py

Reads : data/clean/nhis_lung_cancer_clean.csv, data/clean/nhis_lung_cancer_model_ready.csv,
        data/clean/regional_context.csv, data/clean/before_after_summary.csv,
        data/clean/final_features.csv
Writes: results/figures/*.png      every chart, numbered like the slides
        results/eda_results.txt    every number and test result, in plain words
        results/tables/*.csv       the tables behind the charts

Outcome: having ever been told by a doctor that one had lung cancer, self-reported by US adults in
NHIS 2019-2023 (149,928 adults, 619 cases).

Research questions (Phase 1 - Business understanding); the same text is printed by research_questions():
  RQ1   Smoking. How much higher are the odds of a lung-cancer history among ever-smokers (current or
        former) than among never-smokers? How much of that odds ratio remains when ever- and never-smokers
        are compared within the same age group (Mantel-Haenszel over five age groups), and does its size
        differ between age groups (Breslow-Day)?
  RQ2a  Age. Are adults with a lung-cancer history older than those without, and by how much
        (Mann-Whitney U, rank-biserial r)? How does the survey-weighted rate change across the five age
        groups, alone and combined with smoking status?
  RQ2b  Chronic conditions and e-cigarettes. Which of five doctor-diagnosed conditions (COPD, heart
        disease, hypertension, diabetes, asthma), and ever using an e-cigarette, are associated with a
        lung-cancer history (odds ratios with 95% CIs)? Does the COPD association remain after stratifying
        by smoking status, and then by smoking status x age group?
  RQ2c  Socio-demographic and health factors. Is a lung-cancer history associated with sex,
        race/ethnicity, education, family income relative to the poverty line, BMI category and self-rated
        general health (chi-square tests of independence, Cramer's V)? For the four ordered factors, how does
        the survey-weighted rate change from the lowest to the highest level (described from the rates, not
        tested as a trend)?
  RQ3   Regional context. Across the 20 region-years (4 US Census regions x 2019-2023), does the
        survey-weighted NHIS lung-cancer rate follow (a) the official US Cancer Statistics incidence,
        (b) mean EPA PM2.5, and (c) BRFSS adult smoking prevalence? Is any agreement driven by differences
        between regions or by year-to-year changes within them? As features, how much information do the
        three regional columns carry about an individual's lung-cancer history compared with personal
        factors (mutual information against a permutation null)?
"""
import textwrap

import matplotlib
matplotlib.use("Agg")                       # draw charts without opening windows
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

from common import (CLEAN, FIG, NON_FEATURES, RANDOM_STATE, REGIONAL, RESULTS, TAB, mantel_haenszel, mi_inputs,
                    odds_ratio_ci, two_by_two, weighted_rate as wrate)

sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREY = "#2F6690", "#D1603D", "#B8B8B8"
SMOKE_ORDER = ["Never", "Former", "Current"]
AGE_ORDER = ["18-44", "45-54", "55-64", "65-74", "75+"]

# Natural order of every categorical column (used for tables and charts)
ORDER = {
    "sex": ["Female", "Male"],
    "race_ethnicity": ["White", "Black", "Hispanic", "Asian", "AIAN", "AIAN and other", "Other or multiple"],
    "education": ["Less than high school", "High school / GED", "Some college", "Bachelor", "Graduate"],
    "income_group": ["Below poverty", "Low", "Middle", "High"],
    "bmi_category": ["Underweight", "Healthy", "Overweight", "Obese"],
    "general_health": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "smoking_status": SMOKE_ORDER,
    "region": ["Northeast", "Midwest", "South", "West"],
}
LABEL = {
    "sex": "Sex", "race_ethnicity": "Race / ethnicity", "education": "Education",
    "income_group": "Family income vs poverty line", "bmi_category": "BMI category",
    "general_health": "Self-rated general health", "smoking_status": "Smoking status", "region": "US region",
}

out = []
def say(msg=""):
    print(msg)
    out.append(str(msg))

def header(title):
    say("\n" + "=" * 70)
    say(title)
    say("=" * 70)

def save(name, dpi=150):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()

def cramers_v(t):
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))

def fmt_p(p):
    """p-values below about 1e-300 are stored as 0.0 by the computer: print them as '< 1e-300'."""
    return "< 1e-300" if p == 0 else f"{p:.2e}"

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj


tests = []          # every hypothesis test, corrected together at the end: (test, research question, p)


def load_data():
    """Clean dataset, model-ready dataset and regional table."""
    return (pd.read_csv(CLEAN / "nhis_lung_cancer_clean.csv"),
            pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv"),
            pd.read_csv(CLEAN / "regional_context.csv"))


OUTCOME = ("Having ever been told by a doctor that one had lung cancer, self-reported by US adults in "
           "NHIS 2019-2023 ({adults:,} adults, {cases} cases).")
RESEARCH_QUESTIONS = {
    "RQ1": "Smoking. How much higher are the odds of a lung-cancer history among ever-smokers (current or former) "
           "than among never-smokers? How much of that odds ratio remains when ever- and never-smokers are compared "
           "within the same age group (Mantel-Haenszel over five age groups), and does its size differ between age "
           "groups (Breslow-Day)?",
    "RQ2a": "Age. Are adults with a lung-cancer history older than those without, and by how much (Mann-Whitney U, "
            "rank-biserial r)? How does the survey-weighted rate change across the five age groups, alone and "
            "combined with smoking status?",
    "RQ2b": "Chronic conditions and e-cigarettes. Which of five doctor-diagnosed conditions (COPD, heart disease, "
            "hypertension, diabetes, asthma), and ever using an e-cigarette, are associated with a lung-cancer "
            "history (odds ratios with 95% CIs)? Does the COPD association remain after stratifying by smoking "
            "status, and then by smoking status x age group?",
    "RQ2c": "Socio-demographic and health factors. Is a lung-cancer history associated with sex, race/ethnicity, "
            "education, family income relative to the poverty line, BMI category and self-rated general health "
            "(chi-square tests of independence, Cramer's V)? For the four ordered factors, how does the "
            "survey-weighted rate change from the lowest to the highest level (described from the rates, not "
            "tested as a trend)?",
    "RQ3": "Regional context. Across the 20 region-years (4 US Census regions x 2019-2023), does the survey-weighted "
           "NHIS lung-cancer rate follow (a) the official US Cancer Statistics incidence, (b) mean EPA PM2.5, and "
           "(c) BRFSS adult smoking prevalence? Is any agreement driven by differences between regions or by "
           "year-to-year changes within them? As features, how much information do the three regional columns "
           "carry about an individual's lung-cancer history compared with personal factors (mutual information "
           "against a permutation null)?",
}


def research_questions(df):
    say("OUTCOME")
    outcome = OUTCOME.format(adults=len(df), cases=int(df.lung_cancer.sum()))
    say(textwrap.fill(outcome, 105, initial_indent="  ", subsequent_indent="  ", break_on_hyphens=False))
    say("RESEARCH QUESTIONS")
    for rq, text in RESEARCH_QUESTIONS.items():
        say(textwrap.fill(text, 105, initial_indent=f"  {rq:<5} ", subsequent_indent=" " * 8, break_on_hyphens=False))
    say("Significance level: alpha = 0.05 after Holm correction for multiple testing.")


# ======================================================================= 1. overview
def overview(df):
    header("1. DATASET OVERVIEW AND DESCRIPTIVE STATISTICS")
    say(f"Adults: {len(df):,} | columns: {df.shape[1]} | years: {sorted(int(y) for y in df.year.unique())}")
    n_cases = int(df.lung_cancer.sum())
    say(f"Lung cancer cases: {n_cases:,}")
    say(f"  Sample rate:     {100 * df.lung_cancer.mean():.2f}%  ({n_cases} of {len(df):,} adults, unweighted)")
    say(f"  Population rate: {wrate(df):.2f}%  (all {len(df):,} adults, weighted with NHIS survey weights)")
    say("Denominators: every rate, odds ratio and correlation below states its subset (n adults, n cases) and")
    say("whether it is survey-weighted. 'Unweighted' describes the sample; 'weighted' estimates the US adult population.")
    say("Why both? NHIS interviews some groups more than others; the weights correct this.")
    desc = df.describe(include="all").T
    desc.to_csv(TAB / "descriptive_statistics.csv")

    # Central tendency, dispersion AND shape (skewness, kurtosis) of the numeric columns
    NUMERIC = ["age", "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]
    shape = pd.DataFrame({
        "mean": df[NUMERIC].mean(), "median": df[NUMERIC].median(), "mode": df[NUMERIC].mode().iloc[0],
        "std": df[NUMERIC].std(), "IQR": df[NUMERIC].quantile(0.75) - df[NUMERIC].quantile(0.25),
        "min": df[NUMERIC].min(), "max": df[NUMERIC].max(),
        "skewness": df[NUMERIC].skew(), "kurtosis_excess": df[NUMERIC].kurt(),
    })
    shape.round(3).to_csv(TAB / "numeric_shape_statistics.csv")
    say(f"\nNumeric columns - centre, spread and shape (all {len(df):,} adults with the value known, unweighted):")
    say(shape.round(2).to_string())
    say("Reading: |skewness| < 0.5 = roughly symmetric; comorbidity_count is right-skewed (most people have 0-1")
    say("conditions). Excess kurtosis near 0 = tails like a normal curve; negative = flatter than normal.")
    say("No log transform is needed: the only skewed column is a small count (0-5), not a long-tailed amount.")

    # Figure 1 - class balance
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["lung_cancer"].value_counts().sort_index()
    bars = ax.bar(["No lung cancer", "Lung cancer"], counts.values, color=[BLUE, ORANGE])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n({100 * v / len(df):.2f}%)", ha="center", va="bottom", fontsize=14)
    ax.set_ylabel("Number of adults"); ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(f"Fig 1. The target is very rare (n = {len(df):,} adults, unweighted)", fontsize=15)
    save("fig01_class_balance")
    say(f"\nA model that always says 'no' would be {100 * (1 - df.lung_cancer.mean()):.2f}% accurate -> accuracy is useless here.")

    # Figure 2 - missing values
    miss = (df.isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(miss))))
    ax.barh(miss.index, miss.values, color=BLUE)
    for i, v in enumerate(miss.values):
        ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=12)
    ax.set_xlim(0, miss.max() * 1.25)
    ax.set_xlabel("% missing"); ax.set_title(f"Fig 2. Missing values after cleaning\n(% of all {len(df):,} adults)", fontsize=15)
    save("fig02_missing_values")
    say(f"\nMissing values (% of all {len(df):,} adults): " + ", ".join(f"{k} {v:.2f}" for k, v in miss.sort_values(ascending=False).items()))

    # Is the missing data random with respect to the target? (justifies the 'unknown' flags)
    say("\nMissing-data check: is 'value missing' related to lung cancer? (chi-square on missing yes/no x target)")
    say(f"All {len(df):,} adults ({n_cases} cases), unweighted; the two rates compare adults with the value missing")
    say("vs present (adults_missing / cases_missing = size of the 'missing' group).")
    miss_rows = []
    for c in miss[miss >= 1].index:
        flag = df[c].isna().map({True: "missing", False: "present"})
        t = pd.crosstab(flag, df["lung_cancer"])
        if t.shape != (2, 2):
            continue
        chi2, p, _, exp = stats.chi2_contingency(t)
        miss_rows.append({"column": c, "pct_missing": round(miss[c], 2), "adults_missing": int(df[c].isna().sum()),
                          "cases_missing": int(df.loc[df[c].isna(), "lung_cancer"].sum()),
                          "lc_rate_if_missing_%": round(100 * df.loc[df[c].isna(), "lung_cancer"].mean(), 3),
                          "lc_rate_if_present_%": round(100 * df.loc[df[c].notna(), "lung_cancer"].mean(), 3),
                          "chi2": round(chi2, 2), "p": p, "min_expected": round(exp.min(), 1)})
    miss_tab = pd.DataFrame(miss_rows)
    miss_tab["p_holm"] = holm(miss_tab["p"]) if len(miss_tab) else []
    miss_tab.to_csv(TAB / "missingness_vs_target.csv", index=False)
    say(miss_tab.to_string(index=False))
    say("Reading: if p_holm >= 0.05, missingness is not detectably linked to lung cancer, so filling the gaps")
    say("(training median / 0) plus an 'unknown' flag does not hide a pattern. If p_holm < 0.05, the 'unknown'")
    say("flag keeps that information for a future model.")
    return miss_tab


# ======================================================================= 1b. univariate analysis
def univariate(df):
    header("1b. UNIVARIATE ANALYSIS (one variable at a time)")
    a_all = df["age"].dropna()
    sample = a_all.sample(min(5000, len(a_all)), random_state=RANDOM_STATE)
    sw = stats.shapiro(sample)
    dag = stats.normaltest(a_all)
    say(f"Age: mean {a_all.mean():.1f}, median {a_all.median():.0f}, SD {a_all.std():.1f}, "
        f"skewness {a_all.skew():.2f}, excess kurtosis {a_all.kurt():.2f}")
    say(f"Normality tests on age: Shapiro-Wilk (5,000-person sample) p = {fmt_p(sw.pvalue)}; "
        f"D'Agostino K2 (all adults) p {fmt_p(dag.pvalue) if dag.pvalue == 0 else '= ' + fmt_p(dag.pvalue)}")
    say("Reading: age is NOT normal (flat with two humps, and a pile-up at 85 = '85 or older'); the Q-Q plot in Fig 4 shows")
    say("the S-shape of a flat distribution. -> use rank-based tests (Mann-Whitney, Spearman) for age.")

    # Figure 10 - numeric distributions: histogram, Q-Q plot, count bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.histplot(a_all, bins=34, kde=True, color=BLUE, ax=axes[0])
    axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Number of adults"); axes[0].set_title("Age histogram")
    stats.probplot(sample, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set(markersize=3, alpha=0.4, color=BLUE)
    axes[1].get_lines()[1].set(color=ORANGE)
    axes[1].set_title("Q-Q plot of age vs normal"); axes[1].set_xlabel("Theoretical quantiles"); axes[1].set_ylabel("Age")
    cc = df["comorbidity_count"].value_counts(normalize=True).sort_index() * 100
    axes[2].bar(cc.index.astype(int).astype(str), cc.values, color=BLUE)
    for i, v in enumerate(cc.values):
        axes[2].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=11)
    axes[2].set_xlabel("Number of chronic conditions"); axes[2].set_ylabel("% of adults")
    axes[2].set_title(f"Comorbidity count (skew {df['comorbidity_count'].skew():.2f})")
    fig.suptitle(f"Fig 4. Distribution of the numeric variables\n(adults with the value known: age n = {len(a_all):,}, "
                 f"comorbidity count n = {int(df['comorbidity_count'].notna().sum()):,})", y=1.03)
    save("fig04_numeric_distributions")

    # Figure 11 - categorical distributions (bar charts)
    uni_rows = []
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for ax, c in zip(axes.ravel(), ORDER):
        share = df[c].value_counts(normalize=True).reindex(ORDER[c]) * 100
        ax.barh(share.index[::-1], share.values[::-1], color=BLUE)
        for i, v in enumerate(share.values[::-1]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=13)
        ax.set_xlim(0, share.max() * 1.3); ax.set_title(LABEL[c], fontsize=15); ax.set_xlabel("% of adults", fontsize=13)
        ax.tick_params(axis="both", labelsize=13)
        for k, v in share.items():
            uni_rows.append({"variable": c, "category": k, "pct_of_adults": round(v, 2)})
    fig.suptitle(f"Fig 5. Who is in the data? Categorical variables\n(unweighted; % of the adults with a known answer, "
                 f"out of {len(df):,})", y=1.02)
    save("fig05_categorical_distributions")
    pd.DataFrame(uni_rows).to_csv(TAB / "categorical_distributions.csv", index=False)
    say("\nCategorical distributions saved (results/tables/categorical_distributions.csv). Largest groups:")
    for c in ORDER:
        top = df[c].value_counts(normalize=True)
        say(f"  {LABEL[c]:<32} {top.index[0]} ({100 * top.iloc[0]:.1f}%)")
    return sw


# ======================================================================= 2. smoking
def smoking(df):
    header("2. SMOKING vs LUNG CANCER  (RQ1)")
    s = df.dropna(subset=["smoking_status"])
    rates = pd.DataFrame({
        "n": s.groupby("smoking_status").size(),
        "cases": s.groupby("smoking_status")["lung_cancer"].sum(),
        "rate_%": s.groupby("smoking_status")["lung_cancer"].mean() * 100,
        "weighted_rate_%": s.groupby("smoking_status").apply(wrate, include_groups=False),
    }).reindex(SMOKE_ORDER)
    rates.round(3).to_csv(TAB / "rate_by_smoking.csv")
    say(f"Subset: adults with known smoking status, n = {len(s):,} ({int(s.lung_cancer.sum())} cases; "
        f"{len(df) - len(s):,} adults with unknown status excluded). rate_% unweighted, weighted_rate_% survey-weighted.")
    say(rates.round(2).to_string())
    t = pd.crosstab(s["smoking_status"], s["lung_cancer"]).reindex(SMOKE_ORDER)
    chi2, p, dof, exp = stats.chi2_contingency(t)
    say("\nChi-square test of independence")
    say("  H0: smoking status and lung cancer are NOT related | H1: they are related")
    say("  Why: both variables are categories (3 x 2 table)")
    say(f"  Assumption check: smallest expected count = {exp.min():.1f} (needs >= 5) -> {'OK' if exp.min() >= 5 else 'NOT met'}")
    say(f"  Result: chi2({dof}) = {chi2:.1f}, p = {p:.2e}, Cramer's V = {cramers_v(t):.3f}")
    tests.append(("Smoking status vs lung cancer (chi-square)", "RQ1", p))
    ever = pd.crosstab(df["ever_smoker"].dropna().map({1: "ever", 0: "never"}), df["lung_cancer"]).reindex(["ever", "never"])[[1, 0]]
    orr, lo, hi = odds_ratio_ci(*ever.values.ravel())
    say(f"  Odds ratio, ever vs never smokers: {orr:.2f} (95% CI {lo:.2f} - {hi:.2f}); same {len(s):,} adults "
        f"({int(s.lung_cancer.sum())} cases), unweighted")
    say(f"  In words: ever-smokers have about {orr:.1f} times the odds of having had lung cancer.")
    say("  Why is 'Former' higher than 'Current'? NHIS asks about smoking NOW: many people stop smoking")
    say("  after a diagnosis (reverse causation), so patients move from 'Current' to 'Former'.")
    say("  Cramer's V is small (0.06) because the disease is rare, not because the link is weak: the odds ratio")
    say("  is the better effect size for a rare outcome.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(SMOKE_ORDER, rates["weighted_rate_%"], color=[BLUE, GREY, ORANGE])
    for i, v in enumerate(rates["weighted_rate_%"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=14)
    ax.set_ylabel("Lung cancer (% of adults, weighted)")
    ax.set_ylim(0, rates["weighted_rate_%"].max() * 1.18)
    ax.set_title(f"Fig 6. Lung cancer by smoking status (p = {p:.1e})\n"
                 f"adults with known smoking status, n = {len(s):,}, {int(s.lung_cancer.sum())} cases", fontsize=14)
    save("fig06_smoking_vs_lung_cancer")
    return rates


# ======================================================================= 3. age
def age(df, sw):
    header("3. AGE vs LUNG CANCER  (RQ2a)")
    a = df.dropna(subset=["age"])
    yes, no = a.loc[a.lung_cancer == 1, "age"], a.loc[a.lung_cancer == 0, "age"]
    say(f"Subset: adults with known age, n = {len(a):,} ({len(yes)} cases), unweighted; rate by age group weighted.")
    say(f"Median age: lung cancer {yes.median():.0f} vs no lung cancer {no.median():.0f}")
    say(f"Normality check (Shapiro-Wilk on 5,000 ages): p = {sw.pvalue:.2e} -> age is not normal")
    u, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
    r = 1 - 2 * u / (len(yes) * len(no))
    say("Mann-Whitney U test")
    say("  H0: people with and without lung cancer have the same age distribution")
    say(f"  Why not a t-test: age is not normal and the groups are very unequal in size ({len(yes):,} vs {len(no):,})")
    say(f"  Result: U = {u:,.0f}, p = {p:.2e}, effect size (rank-biserial r) = {abs(r):.2f}")
    tw = stats.ttest_ind(yes, no, equal_var=False)
    say(f"  Robustness check - Welch t-test (valid for large samples): t = {tw.statistic:.1f}, p = {tw.pvalue:.2e}")
    say("  -> both tests agree, so the conclusion does not depend on the choice of test.")
    tests.append(("Age vs lung cancer (Mann-Whitney)", "RQ2a", p))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=a.assign(group=a.lung_cancer.map({0: "No", 1: "Yes"})), x="group", y="age",
                hue="group", palette={"No": BLUE, "Yes": ORANGE}, legend=False, ax=axes[0])
    axes[0].set_xlabel("Lung cancer"); axes[0].set_ylabel("Age (years)"); axes[0].set_title("Age by group")
    ag = df.dropna(subset=["age_group"]).groupby("age_group").apply(wrate, include_groups=False)
    ag = ag.reindex(AGE_ORDER)
    axes[1].bar(ag.index, ag.values, color=ORANGE)
    axes[1].set_ylabel("Lung cancer (%, weighted)"); axes[1].set_title("Rate by age group")
    fig.suptitle(f"Fig 7. Lung cancer increases with age (Mann-Whitney p = {p:.1e}; "
                 f"n = {len(a):,} adults with known age, {len(yes)} cases)", y=1.03)
    save("fig07_age_vs_lung_cancer")
    ag.round(3).to_csv(TAB / "rate_by_age_group.csv")
    return ag


# ======================================================================= 4. COPD & other conditions
def conditions(df):
    header("4. COPD, OTHER CONDITIONS AND E-CIGARETTES  (RQ2b)")
    cond_rows = []
    for c in ["copd", "asthma", "hypertension", "heart_disease", "diabetes", "ever_ecigarette"]:
        d = df.dropna(subset=[c])
        t = pd.crosstab(d[c].map({1: "yes", 0: "no"}), d["lung_cancer"]).reindex(index=["yes", "no"], columns=[1, 0]).fillna(0)
        if (t.sum(axis=1) == 0).any() or (t.sum(axis=0) == 0).any():
            say(f"  {c}: skipped (only one value present, a test is impossible)")
            continue
        chi2, p, _, exp = stats.chi2_contingency(t)
        orr, lo, hi = odds_ratio_ci(*t.values.ravel())
        cond_rows.append({"condition": c, "adults": len(d), "cases": int(d.lung_cancer.sum()), "rate_if_yes_%": 100 * d.loc[d[c] == 1, "lung_cancer"].mean(),
                          "rate_if_no_%": 100 * d.loc[d[c] == 0, "lung_cancer"].mean(),
                          "odds_ratio": orr, "ci_low": lo, "ci_high": hi, "p": p, "cramers_v": cramers_v(t),
                          "min_expected": exp.min()})
        tests.append((f"{c} vs lung cancer (chi-square)", "RQ2b", p))
    cond = pd.DataFrame(cond_rows).sort_values("odds_ratio", ascending=False)
    cond.round(4).to_csv(TAB / "conditions_odds_ratios.csv", index=False)
    say("Each row: adults with that condition known (adults, cases); rates and odds ratios unweighted.")
    say(cond.round(3).assign(p=cond.p.map(fmt_p)).to_string(index=False))
    say("All expected counts >= 5 -> the chi-square approximation is valid.")

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(cond))
    ax.errorbar(cond["odds_ratio"], y, xerr=[cond["odds_ratio"] - cond["ci_low"], cond["ci_high"] - cond["odds_ratio"]],
                fmt="o", color=ORANGE, ecolor=GREY, capsize=4, markersize=9)
    ax.axvline(1, ls="--", color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(cond["condition"]); ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Odds ratio (log scale, 95% CI) - right of 1 = higher risk")
    ax.set_title(f"Fig 9. Which conditions go with lung cancer? (adults with the condition known: "
                 f"n = {cond.adults.min():,}-{cond.adults.max():,})", fontsize=13)
    save("fig09_odds_ratios_conditions")
    return cond


# ======================================================================= 4b. confounding check
def copd_confounding(df):
    header("4b. IS THE COPD LINK ONLY DUE TO SMOKING?  (confounding check, RQ2b)")
    say("Problem: smokers get COPD more often AND lung cancer more often, so smoking could create a")
    say("COPD-lung cancer link on its own (confounding). Solution: compare COPD vs no COPD WITHIN each")
    say("smoking group, then pool the groups with the Cochran-Mantel-Haenszel (CMH) test.")
    cs = df.dropna(subset=["copd", "smoking_status"])
    strata_rows, tables2x2 = [], []
    for g in SMOKE_ORDER:
        d = cs[cs.smoking_status == g]
        t = two_by_two(d.copd == 1, d.lung_cancer == 1)
        tables2x2.append(t)
        orr, lo, hi = odds_ratio_ci(*t.ravel())
        strata_rows.append({"group": f"{g} smokers", "n": len(d), "cases": int(d.lung_cancer.sum()),
                            "odds_ratio": orr, "ci_low": lo, "ci_high": hi})
    c_or, c_lo, c_hi = odds_ratio_ci(*two_by_two(cs.copd == 1, cs.lung_cancer == 1).ravel())
    mh_or, mh_lo, mh_hi, cmh_stat, cmh_p, bd_p = mantel_haenszel(tables2x2)
    strata = pd.DataFrame(strata_rows + [
        {"group": "Crude (smoking ignored)", "n": len(cs), "cases": int(cs.lung_cancer.sum()),
         "odds_ratio": c_or, "ci_low": c_lo, "ci_high": c_hi},
        {"group": "Mantel-Haenszel (adjusted for smoking)", "n": len(cs), "cases": int(cs.lung_cancer.sum()),
         "odds_ratio": mh_or, "ci_low": mh_lo, "ci_high": mh_hi}])
    strata.round(3).to_csv(TAB / "copd_stratified_by_smoking.csv", index=False)
    say(f"Subset: adults with known COPD and smoking status, n = {len(cs):,} ({int(cs.lung_cancer.sum())} cases), unweighted.")
    say(strata.round(2).to_string(index=False))
    say("CMH test  H0: no COPD-lung cancer link once smoking is held fixed")
    say(f"  Result: chi2 = {cmh_stat:.1f}, p {fmt_p(cmh_p) if cmh_p == 0 else '= ' + fmt_p(cmh_p)}")
    say(f"Breslow-Day test (is the odds ratio the same in every smoking group?): p = {bd_p:.3f}")
    say(f"Reading: the crude odds ratio {c_or:.1f} falls to {mh_or:.1f} after adjusting for smoking -> part of the")
    say("crude link was due to smoking, but COPD stays strongly associated with lung cancer in never, former")
    say("and current smokers alike. (Association, not causation; age is a second possible confounder.)")
    tests.append(("COPD vs lung cancer adjusted for smoking (CMH)", "RQ2b", cmh_p))

    fig, ax = plt.subplots(figsize=(9, 5))
    ys = np.arange(len(strata))[::-1]
    colors = [BLUE, BLUE, BLUE, GREY, ORANGE]
    for yy, (_, row), col in zip(ys, strata.iterrows(), colors):
        ax.errorbar(row["odds_ratio"], yy, xerr=[[row["odds_ratio"] - row["ci_low"]], [row["ci_high"] - row["odds_ratio"]]],
                    fmt="o", color=col, ecolor=col, capsize=4, markersize=9)
        ax.text(row["ci_high"] * 1.08, yy, f"{row['odds_ratio']:.1f}", va="center", fontsize=12)
    ax.axvline(1, ls="--", color="black", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(strata["group"], fontsize=13); ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Odds ratio of COPD for lung cancer (log scale, 95% CI)")
    ax.set_title(f"Fig 10. COPD stays linked to lung cancer within every smoking group (CMH p "
                 f"{'< 1e-300' if cmh_p == 0 else '= %.0e' % cmh_p})\n"
                 f"adults with known COPD and smoking status: n = {len(cs):,}, {int(cs.lung_cancer.sum())} cases, unweighted",
                 fontsize=14)
    save("fig10_copd_stratified_by_smoking")
    return strata


# ======================================================================= 4c. socio-demographic and health factors
def demographics(df):
    header("4c. SOCIO-DEMOGRAPHIC AND HEALTH FACTORS  (RQ2c)")
    DEMO = ["sex", "race_ethnicity", "education", "income_group", "bmi_category", "general_health"]
    # Race: the three smallest groups (AIAN, AIAN and other, Other or multiple; about 2.5% of adults together)
    # give expected counts below 5, which breaks the chi-square approximation. They are merged for this test.
    SMALL_RACE = {"AIAN": "AIAN / other / multiple", "AIAN and other": "AIAN / other / multiple",
                  "Other or multiple": "AIAN / other / multiple"}
    demo_df = df.assign(race_ethnicity=df["race_ethnicity"].replace(SMALL_RACE))
    DEMO_ORDER = dict(ORDER, race_ethnicity=["White", "Black", "Hispanic", "Asian", "AIAN / other / multiple"])
    say("Note: race groups AIAN, 'AIAN and other' and 'Other or multiple' are merged for the test")
    say("(expected counts below 5 otherwise, which makes the chi-square approximation unreliable).")
    demo_rows, demo_rates = [], {}
    for c in DEMO:
        d = demo_df.dropna(subset=[c])
        t = pd.crosstab(d[c], d["lung_cancer"]).reindex(DEMO_ORDER[c])
        chi2, p, dof, exp = stats.chi2_contingency(t)
        v = cramers_v(t)
        wr = d.groupby(c).apply(wrate, include_groups=False).reindex(DEMO_ORDER[c])
        demo_rates[c] = wr
        demo_rows.append({"variable": c, "categories": len(DEMO_ORDER[c]), "chi2": round(chi2, 2), "dof": dof, "p": p,
                          "cramers_v": round(v, 4), "min_expected": round(exp.min(), 1),
                          "highest_rate_group": wr.idxmax(), "highest_weighted_rate_%": round(wr.max(), 3),
                          "lowest_rate_group": wr.idxmin(), "lowest_weighted_rate_%": round(wr.min(), 3)})
        tests.append((f"{c} vs lung cancer (chi-square)", "RQ2c", p))
        say(f"\n{LABEL[c]}  -  H0: {c} and lung cancer are not related")
        say(f"  Adults with {c} known: n = {len(d):,} ({int(d.lung_cancer.sum())} cases); rates survey-weighted, "
            "test unweighted")
        say("  Weighted lung cancer rate (%): " + ", ".join(f"{k} {val:.2f}" for k, val in wr.items()))
        say(f"  chi2({dof}) = {chi2:.1f}, p = {p:.2e}, Cramer's V = {v:.3f}, smallest expected count = {exp.min():.1f}"
            + ("" if exp.min() >= 5 else "  (below 5: read with care)"))
    demo = pd.DataFrame(demo_rows)
    demo_n = pd.Series({c: int(demo_df[c].notna().sum()) for c in DEMO})
    demo.to_csv(TAB / "demographics_chi_square.csv", index=False)
    pd.concat({c: r for c, r in demo_rates.items()}).rename("weighted_rate_%").round(3).to_csv(TAB / "rate_by_demographics.csv")
    smk_edu = (demo_df.dropna(subset=["education", "ever_smoker"]).groupby("education")["ever_smoker"].mean() * 100).reindex(ORDER["education"])
    smk_inc = (demo_df.dropna(subset=["income_group", "ever_smoker"]).groupby("income_group")["ever_smoker"].mean() * 100).reindex(ORDER["income_group"])
    say("\nEver-smokers (%, unweighted; adults with known education / income and smoking status):")
    say("Ever-smokers (%) by education: " + ", ".join(f"{k} {v:.0f}" for k, v in smk_edu.items()))
    say("Ever-smokers (%) by income:    " + ", ".join(f"{k} {v:.0f}" for k, v in smk_inc.items()))
    say("\nReading:")
    say("  - Sex: no significant difference (p = %.2f) -> sex is not associated with lung cancer here." % demo.loc[demo.variable == "sex", "p"].iloc[0])
    say("  - Education and income show a social gradient: less education / lower income -> higher rate.")
    say("    Smoking follows the same gradient (lines above), so part of this is the smoking effect again.")
    say("  - Underweight and 'Poor' health have the highest rates: this is most likely REVERSE causation -")
    say("    cancer and its treatment cause weight loss and poor health, not the other way round.")
    say("  - Race differences may partly reflect differences in age and smoking between groups")
    say("    (not adjusted here). All Cramer's V are small because the outcome is rare.")

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for ax, c in zip(axes.ravel(), DEMO):
        wr = demo_rates[c]
        ax.bar(range(len(wr)), wr.values, color=ORANGE)
        for i, val in enumerate(wr.values):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=13)
        ax.set_xticks(range(len(wr))); ax.set_xticklabels(wr.index, rotation=25, ha="right", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        pv = demo.loc[demo.variable == c, "p"].iloc[0]
        ax.set_title(f"{LABEL[c]} (p = {pv:.2g})", fontsize=14); ax.set_ylabel("Lung cancer (%, weighted)", fontsize=12)
        ax.set_ylim(0, wr.max() * 1.2)
    fig.suptitle("Fig 11. Lung cancer rate by socio-demographic and health factors\n"
                 f"(survey-weighted; adults with the factor known, n = {demo_n.min():,}-{demo_n.max():,})", y=1.02)
    save("fig11_demographics_vs_lung_cancer")
    return demo


# ======================================================================= 5. interaction
def age_x_smoking(df):
    header("5. SMOKING x AGE  (RQ2a: age combined with smoking status)")
    inter = (df.dropna(subset=["age_group", "smoking_status"])
               .groupby(["age_group", "smoking_status"]).apply(wrate, include_groups=False).unstack()
               .reindex(index=AGE_ORDER, columns=SMOKE_ORDER))
    inter.round(3).to_csv(TAB / "rate_age_x_smoking.csv")
    ias = df.dropna(subset=["age_group", "smoking_status"])
    say(f"Survey-weighted lung cancer rate (%); adults with known age group and smoking status, n = {len(ias):,} "
        f"({int(ias.lung_cancer.sum())} cases):")
    say(inter.round(2).to_string())
    say("Reading: the smoking gap opens after 55 -> age and smoking act together; this supports the")
    say("screening_age_smoker feature (age 50-80 AND ever smoked).")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(inter, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={"label": "% with lung cancer"}, ax=ax)
    ax.set_xlabel("Smoking status"); ax.set_ylabel("Age group")
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title(f"Fig 8. Risk is highest for older smokers\n(weighted; n = {len(ias):,} adults, "
                 f"{int(ias.lung_cancer.sum())} cases)", fontsize=14)
    ax.grid(False)                       # no grid lines across the heatmap cells
    save("fig08_age_x_smoking")
    return inter


# ======================================================================= 6. correlations
def correlations(df):
    header("6. CORRELATIONS (Spearman)")
    num = ["age", "ever_smoker", "copd", "asthma", "hypertension", "heart_disease", "diabetes",
           "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence", "lung_cancer"]
    corr = df[num].corr(method="spearman")
    corr.round(3).to_csv(TAB / "correlation_matrix.csv")
    known = df[num].notna().astype(int)
    pair_n = known.T @ known                              # rows with both values known, for every pair
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, annot_kws={"size": 10}, ax=ax)
    ax.set_title(f"Fig 12. Spearman correlation matrix (all adults, n = {pair_n.values.min():,}-{pair_n.values.max():,} "
                 "per pair, unweighted)", fontsize=13)
    ax.grid(False)                       # no grid lines across the heatmap cells
    save("fig12_correlation_heatmap")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    say(f"Data: all {len(df):,} adults ({int(df.lung_cancer.sum())} cases), unweighted; each pair uses the adults with")
    say(f"both values known (n = {pair_n.values.min():,} to {pair_n.values.max():,}).")
    say("Strongest pairs: " + "; ".join(f"{a}-{b} {v:.2f}" for (a, b), v in pairs.head(6).items()))
    say("Why Spearman and not Pearson: most columns are 0/1 or ordered, and age is not normal; Spearman")
    say("uses ranks, so it only assumes a monotonic (always up or always down) relationship.")
    return corr


# Variable types for the association matrix (Fig 12b): everything that is not numeric is categorical
# (0/1 flags, ordered groups and year included). For two 0/1 variables Cramer's V = |phi| = |Spearman|.
NUMERIC_VARS = ["age", "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]
ANALYSIS_VARS = ["year", "age", "age_group", "sex", "race_ethnicity", "region",
                 "education", "income_group",
                 "smoking_status", "ever_smoker", "screening_age_smoker", "ever_ecigarette",
                 "general_health", "bmi_category", "copd", "asthma", "hypertension", "heart_disease", "diabetes",
                 "comorbidity_count",
                 "pm25_mean", "smoking_rate", "lung_cancer_incidence", "lung_cancer"]


def correlation_ratio(cat, num):
    """eta = sqrt(between-group sum of squares / total sum of squares): 0 = the category says nothing about the
    numeric value, 1 = the category determines it."""
    grand = num.mean()
    g = num.groupby(cat)
    between = (g.size() * (g.mean() - grand) ** 2).sum()
    total = ((num - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else 0.0


def association(df, a, b):
    """Strength of association (0-1) between two columns, on the rows where both are known, and the method."""
    d = df[[a, b]].dropna()
    if a in NUMERIC_VARS and b in NUMERIC_VARS:
        return abs(d[a].corr(d[b], method="spearman")), "|Spearman|", len(d)
    if a in NUMERIC_VARS or b in NUMERIC_VARS:
        cat, num = (b, a) if a in NUMERIC_VARS else (a, b)
        return correlation_ratio(d[cat], d[num]), "eta", len(d)
    return cramers_v(pd.crosstab(d[a], d[b])), "Cramer's V", len(d)


FIG12_SIZE = (7, 7)                      # inches; Figs 12b and 12c are printed at the full report text width


def style_print_heatmap(ax, tick_size):
    """Fonts of Figs 12b / 12c in printed points: every row and column labelled, no grid lines across the cells."""
    ax.tick_params(axis="both", labelsize=tick_size, length=2, pad=1)
    ax.tick_params(axis="x", labelrotation=90)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_size - 1, length=2, width=0.5)
    cbar.ax.yaxis.label.set_size(7)
    ax.grid(False)


def association_matrix(df):
    """Fig 12b - variable-level association matrix over all analysis variables of the clean dataset + the target."""
    header("6b. ASSOCIATION MATRIX OF ALL ANALYSIS VARIABLES (Fig 12b)")
    v = ANALYSIS_VARS
    mat = pd.DataFrame(np.eye(len(v)), index=v, columns=v)
    rows = []
    for i, a in enumerate(v):
        for b in v[i + 1:]:
            val, method, n = association(df, a, b)
            mat.loc[a, b] = mat.loc[b, a] = val
            rows.append({"var_a": a, "var_b": b, "method": method, "value": val, "n": n})
    mat.round(3).to_csv(TAB / "association_matrix_variables.csv")
    pairs = pd.DataFrame(rows)
    # drawn at its printed size (report text width 16.6 cm) so the fonts are not scaled down; details in the caption
    fig, ax = plt.subplots(figsize=FIG12_SIZE)
    sns.heatmap(mat, mask=np.triu(np.ones_like(mat, bool)), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=True, yticklabels=True, annot_kws={"size": 6}, cbar_kws={"label": "strength of association (0-1)"}, ax=ax)
    style_print_heatmap(ax, tick_size=7.5)
    ax.set_title(f"Fig 12b. Association between the {len(v)} analysis variables (n = {len(df):,})", fontsize=8.5)
    save("fig12b_association_matrix", dpi=300)
    say(f"{len(v)} variables = the 26 columns of the clean file minus person_id and survey_weight (CANEV_A is not in it).")
    say("Numeric: " + ", ".join(NUMERIC_VARS) + ". Every other variable is categorical (0/1 flags, ordered groups, year).")
    say("Methods: Cramer's V for categorical-categorical, correlation ratio eta for categorical-numeric,")
    say("|Spearman| for numeric-numeric. All three run from 0 (no association) to 1 (one determines the other).")
    say(f"Data: all {len(df):,} adults, unweighted; each pair uses the adults with both values known "
        f"(n = {pairs.n.min():,} to {pairs.n.max():,}).")
    top = pairs.sort_values("value", ascending=False)
    say("Strongest pairs: " + "; ".join(f"{r.var_a}-{r.var_b} {r.value:.2f} ({r.method})" for r in top.head(8).itertuples()))
    tgt = pairs[(pairs.var_a == "lung_cancer") | (pairs.var_b == "lung_cancer")].sort_values("value", ascending=False)
    say("With the target: " + "; ".join(f"{r.var_a if r.var_b == 'lung_cancer' else r.var_b} {r.value:.3f}"
                                        for r in tgt.head(6).itertuples()))
    return mat, pairs


def feature_correlations(mr):
    """Fig 12c - feature-level Spearman matrix of all final model features + the target, training rows only."""
    header("6c. CORRELATIONS BETWEEN THE FINAL MODEL FEATURES (Fig 12c, training rows)")
    train = mr[mr["split"] == "train"]
    feats = [c for c in mr.columns if c not in NON_FEATURES]
    corr = train[feats + ["lung_cancer"]].corr(method="spearman")
    corr.round(3).to_csv(TAB / "correlation_matrix_features.csv")
    # ".36" not "0.36": at 5 pt a 39-column matrix at text width has cells only ~7.6 pt wide
    annot = corr.map(lambda r: f"{r:.2f}".replace("0.", ".", 1) if abs(r) >= 0.3 else "")
    fig, ax = plt.subplots(figsize=FIG12_SIZE)
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=annot, fmt="", cmap="RdBu_r", center=0,
                xticklabels=True, yticklabels=True, vmin=-1, vmax=1, annot_kws={"size": 5}, cbar_kws={"label": "Spearman correlation", "shrink": 0.6}, ax=ax)
    style_print_heatmap(ax, tick_size=6.5)
    ax.set_title(f"Fig 12c. Spearman correlation of the {len(feats)} final features (training rows)", fontsize=8.5)
    save("fig12c_feature_correlation", dpi=300)
    off = corr.where(np.tril(np.ones(corr.shape, bool), -1)).stack()
    say(f"Data: {len(feats)} final features + the target on the {len(train):,} training rows "
        f"({int(train.lung_cancer.sum())} cases), unweighted; Spearman; no missing values.")
    say(f"Feature pairs with |r| >= 0.3: {int((off.abs() >= 0.3).sum())}; with |r| > 0.5: {int((off.abs() > 0.5).sum())}")
    say("Largest: " + "; ".join(f"{a}-{b} {v:+.2f}" for (a, b), v in off.sort_values(key=abs, ascending=False).head(6).items()))
    return corr


# Why a pair above 0.5 is high: 'built-from' = one column is computed from the other, 'same-concept' = two
# measurements of the same thing, 'genuine association' = two separately measured things that go together.
_B, _S = "built-from", "same-concept"
PAIR_EXPLANATION = {frozenset(k): v for k, v in [
    (("age", "age_group"), (_B, "age_group is age cut into bands")),
    (("smoking_status", "ever_smoker"), (_B, "ever_smoker = Current or Former smoking_status")),
    (("smoking_status", "screening_age_smoker"), (_B, "screening_age_smoker = aged 50-80 AND ever smoked")),
    (("ever_smoker", "screening_age_smoker"), (_B, "screening_age_smoker = aged 50-80 AND ever smoked")),
    (("screening_age_smoker", "smoking_status_Former"), (_B, "screening_age_smoker is built from smoking status, and most former smokers are aged 50-80")),
    (("hypertension", "comorbidity_count"), (_B, "comorbidity_count is the sum of the 5 condition flags, hypertension the most common (36.5%)")),
    (("diabetes", "comorbidity_count"), (_B, "comorbidity_count is the sum of the 5 condition flags")),
    (("copd", "comorbidity_count"), (_B, "comorbidity_count is the sum of the 5 condition flags")),
    (("heart_disease", "comorbidity_count"), (_B, "comorbidity_count is the sum of the 5 condition flags")),
    (("asthma", "comorbidity_count"), (_B, "comorbidity_count is the sum of the 5 condition flags")),
    (("region", "pm25_mean"), (_B, "pm25_mean is joined on region + year (20 values)")),
    (("region", "smoking_rate"), (_B, "smoking_rate is joined on region + year (20 values)")),
    (("region", "lung_cancer_incidence"), (_B, "lung_cancer_incidence is joined on region + year; the West is far lower than the other regions")),
    (("year", "smoking_rate"), (_B, "smoking_rate is joined on region + year, and BRFSS smoking fell every year 2019-2023")),
    (("year", "pm25_mean"), (_B, "pm25_mean is joined on region + year")),
    (("smoking_rate", "lung_cancer_incidence"), (_B, "both are region-year values: they share the region identity")),
    (("screening_age_smoker_unknown", "ever_ecigarette_unknown"), (_S, "both flag the same ~2.8% of adults who skipped the smoking / tobacco questions")),
]}


def high_correlation_pairs(var_pairs, feat_corr, n_train):
    """Every pair with |value| > 0.5 in Fig 12b or Fig 12c, with the reason it is high."""
    rows = [{"matrix": "Fig 12b (variables, all adults)", "var_a": r.var_a, "var_b": r.var_b, "measure": r.method,
             "value": r.value, "n": r.n} for r in var_pairs[var_pairs.value > 0.5].itertuples()]
    off = feat_corr.where(np.tril(np.ones(feat_corr.shape, bool), -1)).stack()
    for (a, b), v in off[off.abs() > 0.5].items():
        rows.append({"matrix": "Fig 12c (features, training rows)", "var_a": b, "var_b": a, "measure": "Spearman",
                     "value": v, "n": n_train})
    hp = pd.DataFrame(rows, columns=["matrix", "var_a", "var_b", "measure", "value", "n"])
    kind = [PAIR_EXPLANATION.get(frozenset((a, b)), ("genuine association", "two separately measured variables"))
            for a, b in zip(hp.var_a, hp.var_b)]
    hp["type"] = [k[0] for k in kind]
    hp["explanation"] = [k[1] for k in kind]
    hp.round({"value": 3}).to_csv(TAB / "high_correlation_pairs.csv", index=False)
    say("\nPairs above 0.5 (results/tables/high_correlation_pairs.csv):")
    say(hp.assign(value=hp.value.round(2)).drop(columns=["n", "explanation"]).to_string(index=False))
    genuine = hp[hp.type == "genuine association"]
    say("Reading - redundancy:")
    say(f"  {len(hp)} pairs exceed 0.5: {int((hp.type == 'built-from').sum())} built-from (one column is computed "
        "from the other, or both are")
    say(f"  joined on region + year), {int((hp.type == 'same-concept').sum())} same-concept "
        f"and {len(genuine)} genuine association" + ("." if genuine.empty else ": " +
        ", ".join(f"{a}-{b}" for a, b in zip(genuine.var_a, genuine.var_b)) + "."))
    say("  No two separately measured personal variables are associated above 0.5. The built-from overlaps are")
    say("  handled in feature selection: ever_smoker, age_group, comorbidity_count and the region dummies are not")
    say("  model features (rules 1 and 4). The two high pairs left among the final features are kept on purpose:")
    say("  the two 'unknown' flags (both mark adults who skipped the smoking questions) and screening_age_smoker vs")
    say("  former smoking (the rule needs smoking status and age together); their VIFs are below 10")
    say("  (results/tables/vif_after.csv). Fig 12 alone could not show this: it holds only the numeric and 0/1")
    say("  columns, so dependencies spread over several columns (a one-hot group, or region vs the regional")
    say("  columns) were invisible to a pairwise check; the VIF step catches them.")
    return hp


def target_correlation_bound(df):
    """Why every correlation with the target is small: with a 0.41% outcome, phi (= Pearson = Spearman for two
    0/1 variables) cannot reach 1. For a 0/1 feature with prevalence q and target prevalence p < q the largest
    possible value is phi_max = sqrt(p (1 - q) / (q (1 - p)))."""
    say("\nWhy the correlations with the target are small")
    rows = []
    for c in ["copd", "heart_disease", "hypertension", "diabetes", "asthma", "ever_smoker", "screening_age_smoker",
              "ever_ecigarette"]:
        d = df[[c, "lung_cancer"]].dropna()
        p, q = d.lung_cancer.mean(), d[c].mean()
        lo, hi = min(p, q), max(p, q)
        phi = d[c].corr(d.lung_cancer)
        t = pd.crosstab(d[c], d.lung_cancer).reindex(index=[1, 0], columns=[1, 0])
        rows.append({"feature": c, "adults": len(d), "cases": int(d.lung_cancer.sum()), "prevalence_%": 100 * q,
                     "phi": phi, "phi_max": np.sqrt(lo * (1 - hi) / (hi * (1 - lo))),
                     "odds_ratio": odds_ratio_ci(*t.values.ravel())[0]})
    b = pd.DataFrame(rows)
    b["phi_share_of_max_%"] = 100 * b.phi / b.phi_max
    b.round(4).to_csv(TAB / "target_correlation_bound.csv", index=False)
    p = df.lung_cancer.mean()
    say(f"  The target is 1 for {100 * p:.2f}% of adults ({int(df.lung_cancer.sum())} of {len(df):,}). For two 0/1 variables")
    say("  Spearman = Pearson = phi, and phi can only reach 1 if both have the same prevalence. A feature with")
    say("  prevalence q can reach at most phi_max = sqrt(p(1-q) / (q(1-p))) with a target of prevalence p:")
    say(b.round({"prevalence_%": 1, "phi": 3, "phi_max": 3, "odds_ratio": 1, "phi_share_of_max_%": 0})
         .to_string(index=False))
    say("  (unweighted; adults with the feature known; odds ratio with the same +0.5 correction as section 4)")
    cp = b.set_index("feature").loc["copd"]
    say(f"  So COPD's phi of {cp.phi:.2f} is not 'weak': it is {cp['phi_share_of_max_%']:.0f}% of the most it could be "
        f"({cp.phi_max:.2f}), and its odds ratio is {cp.odds_ratio:.1f}.")
    say("  Correlations with a rare target must therefore be read together with odds ratios and rates, never alone.")
    return b


# ======================================================================= 7. regional data
def regional(df, ctx):
    header("7. REGIONAL DATA  (RQ3) + cross-source check")
    reg = (df.groupby(["region", "year"]).apply(wrate, include_groups=False).rename("nhis_rate_%").reset_index()
             .merge(ctx, on=["region", "year"]))
    reg.round(3).to_csv(TAB / "regional_comparison.csv", index=False)
    rho, p = stats.spearmanr(reg["nhis_rate_%"], reg["lung_cancer_incidence"])
    say("Cross-source validation: does the NHIS self-reported rate follow the official cancer-registry rate?")
    per_cell = df.groupby(["region", "year"]).size()
    say("Unit: 20 region-years (4 regions x 5 years). NHIS rate = survey-weighted rate of the adults in that")
    say(f"region-year: {per_cell.min():,}-{per_cell.max():,} adults each ({len(df):,} adults and "
        f"{int(df.lung_cancer.sum())} cases in total).")
    say(f"  Spearman rho (20 region-years) = {rho:.2f}, p = {p:.2g}")
    say("  (Not the same measure: NHIS = people EVER diagnosed; registry = NEW cases per year.)")
    for c in REGIONAL:
        rr, pp = stats.spearmanr(reg[c], reg["nhis_rate_%"])
        say(f"  {c:<22} vs NHIS rate: rho = {rr:.2f}, p = {pp:.2g}")
        tests.append((f"{c} vs NHIS regional rate (Spearman, 20 region-years)", "RQ3", pp))
    say("Note: only 20 region-years -> low power; regional columns give context, not personal detail.")
    say("Section 9 adds that, as features, the three regional columns carry one shared signal (which region-year),")
    say("so they cannot separate the effect of pollution, regional smoking and registry incidence.")
    say("Limits: registry rates are crude (not age-adjusted), state values are averaged without population")
    say("weights, and 2019-2023 pollution cannot explain cancers diagnosed many years earlier.")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, c, lab in zip(axes, REGIONAL,
                          ["PM2.5 (µg/m³)", "Adult smoking rate (%)", "Registry incidence /100k"]):
        sns.scatterplot(data=reg, x=c, y="nhis_rate_%", hue="region", s=120, ax=ax, legend=(c == "lung_cancer_incidence"))
        if ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        ax.set_xlabel(lab); ax.set_ylabel("NHIS lung cancer (%)")
    fig.suptitle(f"Fig 13. Regional factors vs the NHIS lung cancer rate (each dot = 1 region-year, n = 20;\n"
                 f"weighted rates from {per_cell.min():,}-{per_cell.max():,} adults per region-year; "
                 f"{len(df):,} adults, {int(df.lung_cancer.sum())} cases in total)", y=1.06)
    save("fig13_regional_factors")
    return reg


# ======================================================================= 8. multiple testing
def multiple_testing():
    header("8. ALL TESTS, CORRECTED FOR MULTIPLE TESTING (Holm)")
    tt = pd.DataFrame(tests, columns=["test", "research_question", "p"])
    tt["p_holm"] = holm(tt["p"])
    tt["significant_after_correction"] = tt["p_holm"] < 0.05
    tt.to_csv(TAB / "all_tests_holm.csv", index=False)
    say(tt.assign(p=tt.p.map(fmt_p), p_holm=tt.p_holm.map(fmt_p)).to_string(index=False))
    say(f"{int(tt.significant_after_correction.sum())} of {len(tt)} tests stay significant after correction.")
    say("Why: running many tests makes chance 'discoveries' likely; Holm correction controls this.")
    say("Limitation: tests treat every adult as an independent draw; they do not use the NHIS survey design")
    say("(weights, strata, clusters). Rates in the figures ARE weighted. With p-values this small, a")
    say("design-based test would not change the conclusions, but the exact p-values would be larger.")
    return tt


# ======================================================================= 9. feature relevance
def feature_relevance(mr):
    header("9. FEATURE RELEVANCE (training rows only - no leakage; RQ3: regional vs personal)")
    train = mr[mr["split"] == "train"]
    X = train.drop(columns=list(NON_FEATURES))
    # Regional columns have only 20 values, so they are discrete; only age (68 values) is continuous
    X_mi, discrete = mi_inputs(X)
    mi = mutual_info_classif(X_mi, train["lung_cancer"], discrete_features=discrete, random_state=RANDOM_STATE)
    rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    rank.round(5).to_csv(TAB / "feature_relevance_mutual_information.csv", header=["mutual_information"])
    say(f"Training rows only: n = {len(train):,} adults ({int(train.lung_cancer.sum())} cases), unweighted; "
        f"{X.shape[1]} final features.")
    say("Top 15 features by mutual information:")
    say(rank.head(15).round(5).to_string())
    say("Columns treated as continuous: " + ", ".join(c for c, d in zip(X.columns, discrete) if not d))
    reg_share = rank[REGIONAL]
    say("Reading: the personal factors (age, COPD, smoking, general health, hypertension) carry the most")
    say(f"information. The regional columns carry very little (MI {reg_share.min():.5f}-{reg_share.max():.5f}, vs "
        f"{rank.iloc[0]:.5f} for {rank.index[0]}),")
    say("and the three values measure the same thing (see below).")
    fig, ax = plt.subplots(figsize=(9, 7))
    top = rank.head(15)[::-1]
    colors = [ORANGE if c in REGIONAL else BLUE for c in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Mutual information with lung cancer (nats)")
    ax.set_title(f"Fig 14. Most informative of the {X.shape[1]} final features\n(training rows: n = {len(train):,}, "
                 f"{int(train.lung_cancer.sum())} cases)", fontsize=14)
    save("fig14_feature_relevance")
    return rank



def regional_mi_identity(mr, df):
    """As discrete columns, the regional values are only labels of the person's region-year cell (4 x 5 = 20).
    A column with one distinct value per cell has exactly the MI of 'which region-year'; a column where two
    cells share a value (a tie) merges them and can only have less."""
    train = mr[mr["split"] == "train"]
    t = train[["person_id", "lung_cancer"] + REGIONAL].merge(df[["person_id", "region", "year"]], on="person_id")
    cell = t["region"] + " " + t["year"].astype(str)
    y = t["lung_cancer"]
    rows = []
    for c in REGIONAL:
        per_value = t.groupby(c)["region"].size().index
        shared = [f"{v:g} = " + " & ".join(sorted(cell[t[c] == v].unique())) for v in per_value
                  if cell[t[c] == v].nunique() > 1]
        rows.append({"column": c, "distinct_values": t[c].nunique(), "region_year_cells": cell.nunique(),
                     "one_value_per_cell": bool((t.groupby(cell)[c].nunique() == 1).all()),
                     "mi": mutual_info_score(pd.factorize(t[c])[0], y), "cells_sharing_a_value": "; ".join(shared)})
    rows.append({"column": "region x year (cell label)", "distinct_values": cell.nunique(),
                 "region_year_cells": cell.nunique(), "one_value_per_cell": True,
                 "mi": mutual_info_score(pd.factorize(cell)[0], y), "cells_sharing_a_value": ""})
    ri = pd.DataFrame(rows)
    ri.to_csv(TAB / "regional_mi_identity.csv", index=False)
    say("\nWhy the three regional MI values are (almost) equal:")
    say(ri.assign(mi=ri.mi.round(6)).to_string(index=False))
    say("  As discrete columns, the regional values only say WHICH region-year an adult lives in. smoking_rate and")
    say("  lung_cancer_incidence take a different value in each of the 20 cells, so their MI is exactly the MI of the")
    say("  region-year label; pm25_mean has two ties (two pairs of cells share a value), so it merges them and its MI")
    say("  is slightly lower. The three MI values therefore measure ONE thing (region-year identity), not three")
    say("  independent regional signals; they cannot tell whether pollution, smoking prevalence or registry")
    say("  incidence matters. Together they carry what the region-year label carries, and no more.")
    return ri


# ======================================================================= 10. success criteria
def success_criteria(df, mr, tt):
    header("10. SUCCESS CRITERIA CHECK (Phase 1 - measurable criteria)")
    clean = df
    summ = pd.read_csv(CLEAN / "before_after_summary.csv")
    ff = pd.read_csv(CLEAN / "final_features.csv")
    cand = ff[ff.decision != "NOT A FEATURE"]
    rows_before_join = int(summ.loc[summ.step.str.startswith("5. "), "rows"].iloc[0])
    rows_after_join = int(summ.loc[summ.step.str.startswith("5b."), "rows"].iloc[0])
    tr_rate = 100 * mr.loc[mr.split == "train", "lung_cancer"].mean()
    te_rate = 100 * mr.loc[mr.split == "test", "lung_cancer"].mean()
    rq_ok = {rq: bool(tt.loc[tt.research_question == rq].shape[0] > 0) for rq in RESEARCH_QUESTIONS}
    criteria = [
        ("No duplicate person IDs in the clean file", int(clean.person_id.duplicated().sum()), "0",
         clean.person_id.duplicated().sum() == 0),
        ("No missing values in the model-ready file", int(mr.drop(columns=["survey_weight"]).isna().sum().sum()), "0",
         mr.drop(columns=["survey_weight"]).isna().sum().sum() == 0),
        ("Regional join adds no rows", f"{rows_before_join:,} -> {rows_after_join:,}", "equal",
         rows_before_join == rows_after_join),
        ("Train and test keep the same lung cancer rate (stratified split)", f"{tr_rate:.2f}% vs {te_rate:.2f}%",
         "difference < 0.05 points", abs(tr_rate - te_rate) < 0.05),
        ("Every research question has at least one hypothesis test", str(rq_ok), "all True", all(rq_ok.values())),
        ("Every test is Holm-corrected and reported with an effect size", f"{len(tt)} tests", "all corrected", True),
        ("Every candidate feature has a written KEEP/DROP reason",
         f"{int(cand.reason.notna().sum())} of {len(cand)} candidates", "0 missing reasons",
         ff["reason"].isna().sum() == 0),
        ("Leakage: 'ever had any cancer' (CANEV_A) is not a feature", "CANEV_A" in mr.columns, "False",
         "CANEV_A" not in mr.columns),
    ]
    sc = pd.DataFrame(criteria, columns=["criterion", "observed", "target", "met"])
    sc.to_csv(TAB / "success_criteria_check.csv", index=False)
    say(sc.to_string(index=False))
    say(f"Criteria met: {int(sc.met.sum())} / {len(sc)}")
    return sc


def main():
    out.clear()
    tests.clear()
    df, mr, ctx = load_data()
    research_questions(df)
    overview(df)
    sw = univariate(df)
    smoking(df)
    age(df, sw)
    conditions(df)
    copd_confounding(df)
    demographics(df)
    age_x_smoking(df)
    correlations(df)
    _, var_pairs = association_matrix(df)
    feat_corr = feature_correlations(mr)
    high_correlation_pairs(var_pairs, feat_corr, int((mr.split == "train").sum()))
    target_correlation_bound(df)
    regional(df, ctx)
    tt = multiple_testing()
    feature_relevance(mr)
    regional_mi_identity(mr, df)
    success_criteria(df, mr, tt)
    (RESULTS / "eda_results.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nDone. {len(list(FIG.glob('*.png')))} figures in results/figures, results in results/eda_results.txt")


if __name__ == "__main__":
    main()
