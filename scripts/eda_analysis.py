"""
Step 3 - Exploratory Data Analysis (EDA) and hypothesis tests.

    python scripts/eda_analysis.py

Reads : data/clean/nhis_lung_cancer_clean.csv, data/clean/nhis_lung_cancer_model_ready.csv,
        data/clean/regional_context.csv
Writes: reports/figures/*.png      every chart, numbered like the slides
        reports/eda_results.txt    every number and test result, in plain words
        reports/tables/*.csv       the tables behind the charts
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

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREY = "#2F6690", "#D1603D", "#B8B8B8"
SMOKE_ORDER = ["Never", "Former", "Current"]

out = []
def say(msg=""):
    print(msg)
    out.append(str(msg))

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def wrate(d, w="survey_weight"):
    """Population (weighted) % with lung cancer - corrects for NHIS over-sampling some groups."""
    return 100 * np.average(d["lung_cancer"], weights=d[w]) if d[w].notna().all() else 100 * d["lung_cancer"].mean()

def cramers_v(t):
    chi2 = stats.chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))

def odds_ratio_ci(t):
    """Odds ratio for a 2x2 table [[a, b], [c, d]] (rows: exposed yes/no) with a 95% CI."""
    a, b, c, d = t.values.ravel() + 0.5                   # +0.5 avoids dividing by zero
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj


df = pd.read_csv(CLEAN / "nhis_lung_cancer_clean.csv")
mr = pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv")
ctx = pd.read_csv(CLEAN / "regional_context.csv")
tests = []          # every hypothesis test, corrected together at the end

# ======================================================================= 1. overview
say("=" * 70)
say("1. DATASET OVERVIEW")
say("=" * 70)
say(f"Adults: {len(df):,} | columns: {df.shape[1]} | years: {sorted(df.year.unique())}")
n_cases = int(df.lung_cancer.sum())
say(f"Lung cancer cases: {n_cases:,}")
say(f"  Sample rate:     {100 * df.lung_cancer.mean():.2f}%")
say(f"  Population rate: {wrate(df):.2f}%  (weighted with NHIS survey weights)")
say("Why both? NHIS interviews some groups more than others; the weights correct this.")
desc = df.describe(include="all").T
desc.to_csv(TAB / "descriptive_statistics.csv")
say("\nAge: " + df["age"].describe().round(1).to_string().replace("\n", " | "))

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
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=12)
ax.set_xlabel("% missing"); ax.set_title("Fig 2. Missing values after cleaning")
save("fig02_missing_values")
say("\nMissing values (%): " + ", ".join(f"{k} {v:.1f}" for k, v in miss.sort_values(ascending=False).items()))

# ======================================================================= 2. smoking
say("\n" + "=" * 70)
say("2. SMOKING vs LUNG CANCER  (research question 1)")
say("=" * 70)
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
say("  H0: smoking status and lung cancer are NOT related")
say("  Why: both variables are categories (3 x 2 table)")
say(f"  Assumption check: smallest expected count = {exp.min():.1f} (needs >= 5) -> {'OK' if exp.min() >= 5 else 'NOT met'}")
say(f"  Result: chi2({dof}) = {chi2:.1f}, p = {p:.2e}, Cramer's V = {cramers_v(t):.3f}")
tests.append(("Smoking status vs lung cancer (chi-square)", p))
ever = pd.crosstab(df["ever_smoker"].dropna().map({1: "ever", 0: "never"}), df["lung_cancer"]).reindex(["ever", "never"])[[1, 0]]
orr, lo, hi = odds_ratio_ci(ever)
say(f"  Odds ratio, ever vs never smokers: {orr:.2f} (95% CI {lo:.2f} - {hi:.2f})")
say(f"  In words: ever-smokers have about {orr:.1f} times the odds of having had lung cancer.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(SMOKE_ORDER, rates["weighted_rate_%"], color=[BLUE, GREY, ORANGE])
for i, v in enumerate(rates["weighted_rate_%"]):
    ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=14)
ax.set_ylabel("Lung cancer (% of adults, weighted)")
ax.set_title(f"Fig 3. Lung cancer by smoking status (p = {p:.1e})")
save("fig03_smoking_vs_lung_cancer")

# ======================================================================= 3. age
say("\n" + "=" * 70)
say("3. AGE vs LUNG CANCER  (research question 2)")
say("=" * 70)
a = df.dropna(subset=["age"])
yes, no = a.loc[a.lung_cancer == 1, "age"], a.loc[a.lung_cancer == 0, "age"]
sample = a["age"].sample(min(5000, len(a)), random_state=42)
say(f"Median age: lung cancer {yes.median():.0f} vs no lung cancer {no.median():.0f}")
say(f"Normality check (Shapiro-Wilk on 5,000 ages): p = {stats.shapiro(sample).pvalue:.2e} -> age is not normal")
u, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
r = 1 - 2 * u / (len(yes) * len(no))
say("Mann-Whitney U test")
say("  H0: people with and without lung cancer have the same age distribution")
say("  Why not a t-test: age is not normal and the groups are very unequal in size")
say(f"  Result: U = {u:,.0f}, p = {p:.2e}, effect size (rank-biserial r) = {abs(r):.2f}")
tests.append(("Age vs lung cancer (Mann-Whitney)", p))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=a.assign(group=a.lung_cancer.map({0: "No", 1: "Yes"})), x="group", y="age",
            hue="group", palette={"No": BLUE, "Yes": ORANGE}, legend=False, ax=axes[0])
axes[0].set_xlabel("Lung cancer"); axes[0].set_ylabel("Age (years)"); axes[0].set_title("Age by group")
ag = df.dropna(subset=["age_group"]).groupby("age_group").apply(wrate, include_groups=False)
ag = ag.reindex(["18-44", "45-54", "55-64", "65-74", "75+"])
axes[1].bar(ag.index, ag.values, color=ORANGE)
axes[1].set_ylabel("Lung cancer (%, weighted)"); axes[1].set_title("Rate by age group")
fig.suptitle(f"Fig 4. Lung cancer increases with age (Mann-Whitney p = {p:.1e})", y=1.03)
save("fig04_age_vs_lung_cancer")
ag.round(3).to_csv(TAB / "rate_by_age_group.csv")

# ======================================================================= 4. COPD & other conditions
say("\n" + "=" * 70)
say("4. COPD AND OTHER CONDITIONS  (research question 2)")
say("=" * 70)
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
                      "odds_ratio": orr, "ci_low": lo, "ci_high": hi, "p": p, "cramers_v": cramers_v(t)})
    tests.append((f"{c} vs lung cancer (chi-square)", p))
cond = pd.DataFrame(cond_rows).sort_values("odds_ratio", ascending=False)
cond.round(4).to_csv(TAB / "conditions_odds_ratios.csv", index=False)
say(cond.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(cond))
ax.errorbar(cond["odds_ratio"], y, xerr=[cond["odds_ratio"] - cond["ci_low"], cond["ci_high"] - cond["odds_ratio"]],
            fmt="o", color=ORANGE, ecolor=GREY, capsize=4, markersize=9)
ax.axvline(1, ls="--", color="black", lw=1)
ax.set_yticks(y); ax.set_yticklabels(cond["condition"]); ax.set_xscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Odds ratio (log scale, 95% CI) - right of 1 = higher risk")
ax.set_title("Fig 5. Which conditions go with lung cancer?")
save("fig05_odds_ratios_conditions")

# ======================================================================= 5. interaction
say("\n" + "=" * 70)
say("5. SMOKING x AGE  (stand-out: combined effect)")
say("=" * 70)
inter = (df.dropna(subset=["age_group", "smoking_status"])
           .groupby(["age_group", "smoking_status"]).apply(wrate, include_groups=False).unstack()
           .reindex(index=["18-44", "45-54", "55-64", "65-74", "75+"], columns=SMOKE_ORDER))
inter.round(3).to_csv(TAB / "rate_age_x_smoking.csv")
say(inter.round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(inter, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={"label": "% with lung cancer"}, ax=ax)
ax.set_xlabel("Smoking status"); ax.set_ylabel("Age group")
ax.set_title("Fig 6. Risk is highest for older smokers")
save("fig06_age_x_smoking")

# ======================================================================= 6. correlations
say("\n" + "=" * 70)
say("6. CORRELATIONS (Spearman)")
say("=" * 70)
num = ["age", "ever_smoker", "copd", "asthma", "hypertension", "heart_disease", "diabetes",
       "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence", "lung_cancer"]
corr = df[num].corr(method="spearman")
corr.round(3).to_csv(TAB / "correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, annot_kws={"size": 10}, ax=ax)
ax.set_title("Fig 7. Spearman correlation matrix")
save("fig07_correlation_heatmap")
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
say("Strongest pairs: " + "; ".join(f"{a}-{b} {v:.2f}" for (a, b), v in pairs.head(6).items()))
say("Why Spearman: most columns are 0/1 or ordered, not normally distributed.")

# ======================================================================= 7. regional data
say("\n" + "=" * 70)
say("7. REGIONAL DATA  (research question 3) + cross-source check")
say("=" * 70)
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
say("Note: only 20 region-years -> low power; regional columns give context, not personal detail.")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c, lab in zip(axes, ["pm25_mean", "smoking_rate", "lung_cancer_incidence"],
                      ["PM2.5 (µg/m³)", "Adult smoking rate (%)", "Registry incidence /100k"]):
    sns.scatterplot(data=reg, x=c, y="nhis_rate_%", hue="region", s=120, ax=ax, legend=(c == "lung_cancer_incidence"))
    if ax.get_legend():
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlabel(lab); ax.set_ylabel("NHIS lung cancer (%)")
fig.suptitle("Fig 8. Regional factors vs the NHIS lung cancer rate (each dot = 1 region-year)", y=1.03)
save("fig08_regional_factors")

# ======================================================================= 8. multiple testing
say("\n" + "=" * 70)
say("8. ALL TESTS, CORRECTED FOR MULTIPLE TESTING (Holm)")
say("=" * 70)
tt = pd.DataFrame(tests, columns=["test", "p"])
tt["p_holm"] = holm(tt["p"])
tt["significant_after_correction"] = tt["p_holm"] < 0.05
tt.to_csv(TAB / "all_tests_holm.csv", index=False)
say(tt.to_string(index=False))
say("Why: running many tests makes chance 'discoveries' likely; Holm correction controls this.")

# ======================================================================= 9. feature relevance
say("\n" + "=" * 70)
say("9. FEATURE RELEVANCE (training rows only - no leakage)")
say("=" * 70)
train = mr[mr["split"] == "train"]
X = train.drop(columns=["person_id", "split", "survey_weight", "lung_cancer"])
mi = mutual_info_classif(X, train["lung_cancer"], discrete_features=[X[c].nunique() <= 2 for c in X.columns],
                         random_state=42)
rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)
rank.round(5).to_csv(TAB / "feature_relevance_mutual_information.csv", header=["mutual_information"])
say("Top 15 features by mutual information:")
say(rank.head(15).round(5).to_string())
fig, ax = plt.subplots(figsize=(9, 7))
top = rank.head(15)[::-1]
ax.barh(top.index, top.values, color=BLUE)
ax.set_xlabel("Mutual information with lung cancer"); ax.set_title("Fig 9. Most informative features (training set)")
save("fig09_feature_relevance")

(ROOT / "reports" / "eda_results.txt").write_text("\n".join(out), encoding="utf-8")
print(f"\nDone. {len(list(FIG.glob('*.png')))} figures in reports/figures, results in reports/eda_results.txt")
