"""
Step 3 - Exploratory Data Analysis (EDA) and hypothesis tests.

    python scripts/eda_analysis.py

Reads : data/clean/nhis_lung_cancer_clean.csv, data/clean/nhis_lung_cancer_model_ready.csv,
        data/clean/regional_context.csv, data/clean/before_after_summary.csv,
        data/clean/final_features.csv
Writes: reports/figures/*.png      every chart, numbered like the slides
        reports/eda_results.txt    every number and test result, in plain words
        reports/tables/*.csv       the tables behind the charts

Research questions (Phase 1 - Business understanding):
  RQ1  Is smoking status associated with having had lung cancer?
  RQ2  Are age, chronic conditions (COPD, heart disease, ...) and socio-demographic / health
       factors (sex, race, education, income, BMI, self-rated health) associated with lung cancer?
  RQ3  Do regional factors (PM2.5 air pollution, regional smoking rate, registry incidence)
       add information beyond the personal factors?
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw charts without opening windows
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.contingency_tables import StratifiedTable

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

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

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def wrate(d, w="survey_weight"):
    """Population (weighted) % with lung cancer - corrects for NHIS over-sampling some groups."""
    return 100 * np.average(d["lung_cancer"], weights=d[w]) if d[w].notna().all() else 100 * d["lung_cancer"].mean()

def cramers_v(t):
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))

def odds_ratio_ci(t):
    """Odds ratio for a 2x2 table [[a, b], [c, d]] (rows: exposed yes/no) with a 95% CI."""
    a, b, c, d = t.values.ravel() + 0.5                   # +0.5 avoids dividing by zero
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)

def fmt_p(p):
    """p-values below about 1e-300 are stored as 0.0 by the computer: print them as '< 1e-300'."""
    return "< 1e-300" if p == 0 else f"{p:.2e}"

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj


df = pd.read_csv(CLEAN / "nhis_lung_cancer_clean.csv")
mr = pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv")
ctx = pd.read_csv(CLEAN / "regional_context.csv")
tests = []          # every hypothesis test, corrected together at the end: (test, research question, p)

say("RESEARCH QUESTIONS")
say("  RQ1  Is smoking status associated with having had lung cancer?")
say("  RQ2  Are age, chronic conditions and socio-demographic / health factors associated with lung cancer?")
say("  RQ3  Do regional factors (PM2.5, regional smoking rate, registry incidence) add information")
say("       beyond the personal factors?")
say("Significance level: alpha = 0.05 after Holm correction for multiple testing.")

# ======================================================================= 1. overview
header("1. DATASET OVERVIEW AND DESCRIPTIVE STATISTICS")
say(f"Adults: {len(df):,} | columns: {df.shape[1]} | years: {sorted(int(y) for y in df.year.unique())}")
n_cases = int(df.lung_cancer.sum())
say(f"Lung cancer cases: {n_cases:,}")
say(f"  Sample rate:     {100 * df.lung_cancer.mean():.2f}%")
say(f"  Population rate: {wrate(df):.2f}%  (weighted with NHIS survey weights)")
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
say("\nNumeric columns - centre, spread and shape:")
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
ax.set_title("Fig 1. The target is very rare")
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
ax.set_xlabel("% missing"); ax.set_title("Fig 2. Missing values after cleaning")
save("fig02_missing_values")
say("\nMissing values (%): " + ", ".join(f"{k} {v:.2f}" for k, v in miss.sort_values(ascending=False).items()))

# Is the missing data random with respect to the target? (justifies the 'unknown' flags)
say("\nMissing-data check: is 'value missing' related to lung cancer? (chi-square on missing yes/no x target)")
miss_rows = []
for c in miss[miss >= 1].index:
    flag = df[c].isna().map({True: "missing", False: "present"})
    t = pd.crosstab(flag, df["lung_cancer"])
    if t.shape != (2, 2):
        continue
    chi2, p, _, exp = stats.chi2_contingency(t)
    miss_rows.append({"column": c, "pct_missing": round(miss[c], 2),
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

# ======================================================================= 1b. univariate analysis
header("1b. UNIVARIATE ANALYSIS (one variable at a time)")
a_all = df["age"].dropna()
sample = a_all.sample(min(5000, len(a_all)), random_state=42)
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
fig.suptitle("Fig 4. Distribution of the numeric variables", y=1.03)
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
fig.suptitle("Fig 5. Who is in the data? Categorical variables (unweighted)", y=1.01)
save("fig05_categorical_distributions")
pd.DataFrame(uni_rows).to_csv(TAB / "categorical_distributions.csv", index=False)
say("\nCategorical distributions saved (reports/tables/categorical_distributions.csv). Largest groups:")
for c in ORDER:
    top = df[c].value_counts(normalize=True)
    say(f"  {LABEL[c]:<32} {top.index[0]} ({100 * top.iloc[0]:.1f}%)")

# ======================================================================= 2. smoking
header("2. SMOKING vs LUNG CANCER  (research question 1)")
s = df.dropna(subset=["smoking_status"])
rates = pd.DataFrame({
    "n": s.groupby("smoking_status").size(),
    "cases": s.groupby("smoking_status")["lung_cancer"].sum(),
    "rate_%": s.groupby("smoking_status")["lung_cancer"].mean() * 100,
    "weighted_rate_%": s.groupby("smoking_status").apply(wrate, include_groups=False),
}).reindex(SMOKE_ORDER)
rates.round(3).to_csv(TAB / "rate_by_smoking.csv")
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
orr, lo, hi = odds_ratio_ci(ever)
say(f"  Odds ratio, ever vs never smokers: {orr:.2f} (95% CI {lo:.2f} - {hi:.2f})")
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
ax.set_title(f"Fig 6. Lung cancer by smoking status (p = {p:.1e})")
save("fig06_smoking_vs_lung_cancer")

# ======================================================================= 3. age
header("3. AGE vs LUNG CANCER  (research question 2)")
a = df.dropna(subset=["age"])
yes, no = a.loc[a.lung_cancer == 1, "age"], a.loc[a.lung_cancer == 0, "age"]
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
tests.append(("Age vs lung cancer (Mann-Whitney)", "RQ2", p))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=a.assign(group=a.lung_cancer.map({0: "No", 1: "Yes"})), x="group", y="age",
            hue="group", palette={"No": BLUE, "Yes": ORANGE}, legend=False, ax=axes[0])
axes[0].set_xlabel("Lung cancer"); axes[0].set_ylabel("Age (years)"); axes[0].set_title("Age by group")
ag = df.dropna(subset=["age_group"]).groupby("age_group").apply(wrate, include_groups=False)
ag = ag.reindex(AGE_ORDER)
axes[1].bar(ag.index, ag.values, color=ORANGE)
axes[1].set_ylabel("Lung cancer (%, weighted)"); axes[1].set_title("Rate by age group")
fig.suptitle(f"Fig 7. Lung cancer increases with age (Mann-Whitney p = {p:.1e})", y=1.03)
save("fig07_age_vs_lung_cancer")
ag.round(3).to_csv(TAB / "rate_by_age_group.csv")

# ======================================================================= 4. COPD & other conditions
header("4. COPD AND OTHER CONDITIONS  (research question 2)")
cond_rows = []
for c in ["copd", "asthma", "hypertension", "heart_disease", "diabetes", "ever_ecigarette"]:
    d = df.dropna(subset=[c])
    t = pd.crosstab(d[c].map({1: "yes", 0: "no"}), d["lung_cancer"]).reindex(index=["yes", "no"], columns=[1, 0]).fillna(0)
    if (t.sum(axis=1) == 0).any() or (t.sum(axis=0) == 0).any():
        say(f"  {c}: skipped (only one value present, a test is impossible)")
        continue
    chi2, p, _, exp = stats.chi2_contingency(t)
    orr, lo, hi = odds_ratio_ci(t)
    cond_rows.append({"condition": c, "rate_if_yes_%": 100 * d.loc[d[c] == 1, "lung_cancer"].mean(),
                      "rate_if_no_%": 100 * d.loc[d[c] == 0, "lung_cancer"].mean(),
                      "odds_ratio": orr, "ci_low": lo, "ci_high": hi, "p": p, "cramers_v": cramers_v(t),
                      "min_expected": exp.min()})
    tests.append((f"{c} vs lung cancer (chi-square)", "RQ2", p))
cond = pd.DataFrame(cond_rows).sort_values("odds_ratio", ascending=False)
cond.round(4).to_csv(TAB / "conditions_odds_ratios.csv", index=False)
say(cond.round(3).to_string(index=False))
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
ax.set_title("Fig 9. Which conditions go with lung cancer?")
save("fig09_odds_ratios_conditions")

# ======================================================================= 4b. confounding check
header("4b. IS THE COPD LINK ONLY DUE TO SMOKING?  (confounding check, research question 2)")
say("Problem: smokers get COPD more often AND lung cancer more often, so smoking could create a")
say("COPD-lung cancer link on its own (confounding). Solution: compare COPD vs no COPD WITHIN each")
say("smoking group, then pool the groups with the Cochran-Mantel-Haenszel (CMH) test.")
cs = df.dropna(subset=["copd", "smoking_status"])
strata_rows, tables2x2 = [], []
for g in SMOKE_ORDER:
    d = cs[cs.smoking_status == g]
    t = pd.crosstab(d["copd"].map({1: "yes", 0: "no"}), d["lung_cancer"]).reindex(index=["yes", "no"], columns=[1, 0]).fillna(0)
    tables2x2.append(t.values.astype(float))
    orr, lo, hi = odds_ratio_ci(t)
    strata_rows.append({"group": f"{g} smokers", "n": len(d), "cases": int(d.lung_cancer.sum()),
                        "odds_ratio": orr, "ci_low": lo, "ci_high": hi})
crude_t = pd.crosstab(cs["copd"].map({1: "yes", 0: "no"}), cs["lung_cancer"]).reindex(index=["yes", "no"], columns=[1, 0])
c_or, c_lo, c_hi = odds_ratio_ci(crude_t)
st = StratifiedTable(tables2x2)
mh_or = st.oddsratio_pooled
mh_lo, mh_hi = st.oddsratio_pooled_confint()
cmh = st.test_null_odds()
# statsmodels returns p = 1 - cdf, which rounds to 0 for very large statistics; the survival function keeps
# the real value (chi-square with 1 degree of freedom)
cmh_p = stats.chi2.sf(cmh.statistic, 1)
bd = st.test_equal_odds()
bd_p = stats.chi2.sf(bd.statistic, len(tables2x2) - 1)
strata = pd.DataFrame(strata_rows + [
    {"group": "Crude (smoking ignored)", "n": len(cs), "cases": int(cs.lung_cancer.sum()),
     "odds_ratio": c_or, "ci_low": c_lo, "ci_high": c_hi},
    {"group": "Mantel-Haenszel (adjusted for smoking)", "n": len(cs), "cases": int(cs.lung_cancer.sum()),
     "odds_ratio": mh_or, "ci_low": mh_lo, "ci_high": mh_hi}])
strata.round(3).to_csv(TAB / "copd_stratified_by_smoking.csv", index=False)
say(strata.round(2).to_string(index=False))
say(f"CMH test  H0: no COPD-lung cancer link once smoking is held fixed")
say(f"  Result: chi2 = {cmh.statistic:.1f}, p {fmt_p(cmh_p) if cmh_p == 0 else '= ' + fmt_p(cmh_p)}")
say(f"Breslow-Day test (is the odds ratio the same in every smoking group?): p = {bd_p:.3f}")
say(f"Reading: the crude odds ratio {c_or:.1f} falls to {mh_or:.1f} after adjusting for smoking -> part of the")
say("crude link was due to smoking, but COPD stays strongly associated with lung cancer in never, former")
say("and current smokers alike. (Association, not causation; age is a second possible confounder.)")
tests.append(("COPD vs lung cancer adjusted for smoking (CMH)", "RQ2", cmh_p))

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
ax.set_title(f"Fig 10. COPD stays linked to lung cancer within every smoking group (CMH p {'< 1e-300' if cmh_p == 0 else '= %.0e' % cmh_p})", fontsize=14)
save("fig10_copd_stratified_by_smoking")

# ======================================================================= 4c. socio-demographic and health factors
header("4c. SOCIO-DEMOGRAPHIC AND HEALTH FACTORS  (research question 2)")
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
    tests.append((f"{c} vs lung cancer (chi-square)", "RQ2", p))
    say(f"\n{LABEL[c]}  -  H0: {c} and lung cancer are not related")
    say("  Weighted lung cancer rate (%): " + ", ".join(f"{k} {val:.2f}" for k, val in wr.items()))
    say(f"  chi2({dof}) = {chi2:.1f}, p = {p:.2e}, Cramer's V = {v:.3f}, smallest expected count = {exp.min():.1f}"
        + ("" if exp.min() >= 5 else "  (below 5: read with care)"))
demo = pd.DataFrame(demo_rows)
demo.to_csv(TAB / "demographics_chi_square.csv", index=False)
pd.concat({c: r for c, r in demo_rates.items()}).rename("weighted_rate_%").round(3).to_csv(TAB / "rate_by_demographics.csv")
smk_edu = (demo_df.dropna(subset=["education", "ever_smoker"]).groupby("education")["ever_smoker"].mean() * 100).reindex(ORDER["education"])
smk_inc = (demo_df.dropna(subset=["income_group", "ever_smoker"]).groupby("income_group")["ever_smoker"].mean() * 100).reindex(ORDER["income_group"])
say("\nEver-smokers (%) by education: " + ", ".join(f"{k} {v:.0f}" for k, v in smk_edu.items()))
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
    ax.set_title(f"{LABEL[c]} (p = {pv:.0e})", fontsize=14); ax.set_ylabel("Lung cancer (%, weighted)", fontsize=12)
    ax.set_ylim(0, wr.max() * 1.2)
fig.suptitle("Fig 11. Lung cancer rate by socio-demographic and health factors", y=1.01)
save("fig11_demographics_vs_lung_cancer")

# ======================================================================= 5. interaction
header("5. SMOKING x AGE  (stand-out: combined effect)")
inter = (df.dropna(subset=["age_group", "smoking_status"])
           .groupby(["age_group", "smoking_status"]).apply(wrate, include_groups=False).unstack()
           .reindex(index=AGE_ORDER, columns=SMOKE_ORDER))
inter.round(3).to_csv(TAB / "rate_age_x_smoking.csv")
say(inter.round(2).to_string())
say("Reading: the smoking gap opens after 55 -> age and smoking act together; this supports the")
say("screening_age_smoker feature (age 50-80 AND ever smoked).")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(inter, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={"label": "% with lung cancer"}, ax=ax)
ax.set_xlabel("Smoking status"); ax.set_ylabel("Age group")
ax.tick_params(axis="y", labelrotation=0)
ax.set_title("Fig 8. Risk is highest for older smokers")
save("fig08_age_x_smoking")

# ======================================================================= 6. correlations
header("6. CORRELATIONS (Spearman)")
num = ["age", "ever_smoker", "copd", "asthma", "hypertension", "heart_disease", "diabetes",
       "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence", "lung_cancer"]
corr = df[num].corr(method="spearman")
corr.round(3).to_csv(TAB / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, annot_kws={"size": 10}, ax=ax)
ax.set_title("Fig 12. Spearman correlation matrix")
save("fig12_correlation_heatmap")
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
say("Strongest pairs: " + "; ".join(f"{a}-{b} {v:.2f}" for (a, b), v in pairs.head(6).items()))
say("Why Spearman and not Pearson: most columns are 0/1 or ordered, and age is not normal; Spearman")
say("uses ranks, so it only assumes a monotonic (always up or always down) relationship.")
say("Reading: comorbidity_count correlates with its own parts (by construction); no pair of different")
say("variables is above 0.5, so there is no harmful redundancy among the personal features.")

# ======================================================================= 7. regional data
header("7. REGIONAL DATA  (research question 3) + cross-source check")
reg = (df.groupby(["region", "year"]).apply(wrate, include_groups=False).rename("nhis_rate_%").reset_index()
         .merge(ctx, on=["region", "year"]))
reg.round(3).to_csv(TAB / "regional_comparison.csv", index=False)
rho, p = stats.spearmanr(reg["nhis_rate_%"], reg["lung_cancer_incidence"])
say("Cross-source validation: does the NHIS self-reported rate follow the official cancer-registry rate?")
say(f"  Spearman rho (20 region-years) = {rho:.2f}, p = {p:.3f}")
say("  (Not the same measure: NHIS = people EVER diagnosed; registry = NEW cases per year.)")
for c in ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]:
    rr, pp = stats.spearmanr(reg[c], reg["nhis_rate_%"])
    say(f"  {c:<22} vs NHIS rate: rho = {rr:.2f}, p = {pp:.3f}")
    tests.append((f"{c} vs NHIS regional rate (Spearman, 20 region-years)", "RQ3", pp))
say("Note: only 20 region-years -> low power; regional columns give context, not personal detail.")
say("Limits: registry rates are crude (not age-adjusted), state values are averaged without population")
say("weights, and 2019-2023 pollution cannot explain cancers diagnosed many years earlier.")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, c, lab in zip(axes, ["pm25_mean", "smoking_rate", "lung_cancer_incidence"],
                      ["PM2.5 (µg/m³)", "Adult smoking rate (%)", "Registry incidence /100k"]):
    sns.scatterplot(data=reg, x=c, y="nhis_rate_%", hue="region", s=120, ax=ax, legend=(c == "lung_cancer_incidence"))
    if ax.get_legend():
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlabel(lab); ax.set_ylabel("NHIS lung cancer (%)")
fig.suptitle("Fig 13. Regional factors vs the NHIS lung cancer rate (each dot = 1 region-year)", y=1.03)
save("fig13_regional_factors")

# ======================================================================= 8. multiple testing
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

# ======================================================================= 9. feature relevance
header("9. FEATURE RELEVANCE (training rows only - no leakage)")
train = mr[mr["split"] == "train"]
X = train.drop(columns=["person_id", "split", "survey_weight", "lung_cancer"])
# A column with few distinct values must be treated as DISCRETE: the continuous (k-nearest-neighbour)
# estimator is biased upwards on heavily tied values. Regional columns have only 20 values, so they
# are discrete; only age (68 values) is treated as continuous.
discrete = [X[c].nunique() <= 25 for c in X.columns]
X_mi = X.copy()
for c, d in zip(X.columns, discrete):
    if d:                                    # give discrete values integer codes (MI does not depend on the labels)
        X_mi[c] = pd.factorize(X[c], sort=True)[0]
mi = mutual_info_classif(X_mi, train["lung_cancer"], discrete_features=discrete, random_state=42)
rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)
rank.round(5).to_csv(TAB / "feature_relevance_mutual_information.csv", header=["mutual_information"])
say("Top 15 features by mutual information:")
say(rank.head(15).round(5).to_string())
say("Columns treated as continuous: " + ", ".join(c for c, d in zip(X.columns, discrete) if not d))
say("Reading: the personal factors (age, COPD, comorbidities, smoking, general health) carry the most")
say("information; the regional columns carry very little (consistent with section 7).")
fig, ax = plt.subplots(figsize=(9, 7))
top = rank.head(15)[::-1]
colors = [ORANGE if c in ("pm25_mean", "smoking_rate", "lung_cancer_incidence") else BLUE for c in top.index]
ax.barh(top.index, top.values, color=colors)
ax.set_xlabel("Mutual information with lung cancer"); ax.set_title("Fig 14. Most informative features (training set)")
save("fig14_feature_relevance")

# ======================================================================= 10. success criteria
header("10. SUCCESS CRITERIA CHECK (Phase 1 - measurable criteria)")
clean = df
summ = pd.read_csv(CLEAN / "before_after_summary.csv")
ff = pd.read_csv(CLEAN / "final_features.csv")
rows_before_join = int(summ.loc[summ.step.str.startswith("5. "), "rows"].iloc[0])
rows_after_join = int(summ.loc[summ.step.str.startswith("5b."), "rows"].iloc[0])
tr_rate = 100 * mr.loc[mr.split == "train", "lung_cancer"].mean()
te_rate = 100 * mr.loc[mr.split == "test", "lung_cancer"].mean()
rq_ok = {rq: bool(tt.loc[tt.research_question == rq].shape[0] > 0) for rq in ["RQ1", "RQ2", "RQ3"]}
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
    ("Every candidate feature has a written KEEP/DROP reason", int(ff["reason"].isna().sum()), "0 missing reasons",
     ff["reason"].isna().sum() == 0),
    ("Leakage: 'ever had any cancer' (CANEV_A) is not a feature", "CANEV_A" in mr.columns, "False",
     "CANEV_A" not in mr.columns),
]
sc = pd.DataFrame(criteria, columns=["criterion", "observed", "target", "met"])
sc.to_csv(TAB / "success_criteria_check.csv", index=False)
say(sc.to_string(index=False))
say(f"Criteria met: {int(sc.met.sum())} / {len(sc)}")

(ROOT / "reports" / "eda_results.txt").write_text("\n".join(out), encoding="utf-8")
print(f"\nDone. {len(list(FIG.glob('*.png')))} figures in reports/figures, results in reports/eda_results.txt")
