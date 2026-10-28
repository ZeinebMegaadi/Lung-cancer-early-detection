"""
Step 5 - Detailed population profile and supplementary checks.

    python scripts/supplementary_analysis.py

Reads : data/clean/nhis_lung_cancer_clean.csv, data/clean/nhis_lung_cancer_model_ready.csv,
        data/clean/scaler_parameters.csv, reports/tables/regional_comparison.csv,
        reports/tables/feature_relevance_mutual_information.csv,
        data/raw/nhis/adultYYcsv.zip (only for the survey-design variables PSTRAT / PPSU)
Writes: reports/tables/population_profile.csv          every category: adults, cases, rates
        reports/tables/cases_vs_noncases.csv           characteristics of cases vs other adults
        reports/tables/group_profile.csv               age, sex and smoking inside each sex / race group
        reports/tables/rate_by_year.csv                cases and rates per survey year
        reports/tables/smoking_by_age_group.csv        smoking odds ratio inside each age group (+ CMH)
        reports/tables/copd_adjusted.csv               COPD odds ratio adjusted for smoking, then smoking x age
        reports/tables/regional_within_between.csv     RQ3 correlations split into between- and within-region
        reports/tables/screening_rule_baseline.csv     the USPSTF-style rule as a benchmark
        reports/tables/mutual_information_checked.csv  normalised MI and a permutation null
        reports/tables/design_based_odds_ratios.csv    survey-weighted, cluster-robust odds ratios
        reports/supplementary_results.txt              everything above in plain words

Nothing here changes the clean or model-ready files. Descriptive tables use all adults (they describe the
population and fit nothing); the mutual-information check uses training rows only.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import StratifiedTable

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw" / "nhis"
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
YEARS = [2019, 2020, 2021, 2022, 2023]

out = []
def say(msg=""):
    print(msg)
    out.append(str(msg))

def header(title):
    say("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

def wmean(x, w):
    ok = x.notna()
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan

def odds_ratio_ci(a, b, c, d):
    """2x2 table [[a, b], [c, d]] = exposed cases, exposed non-cases, unexposed cases, unexposed non-cases.
    Same convention as eda_analysis.py: +0.5 in every cell (Haldane-Anscombe), so the numbers match."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)

df = pd.read_csv(CLEAN / "nhis_lung_cancer_clean.csv")
w = df["survey_weight"]
N, CASES = len(df), int(df["lung_cancer"].sum())

# ======================================================================= 1. population profile
header("1. WHO IS IN THE DATA? Every category: adults, share, cases, rates")
say(f"{N:,} adults, {CASES} with a lung-cancer history. 'Weighted' = NHIS survey weights (US adult population).")
say("Unknown = the person did not give a usable answer (kept as its own row so the counts add up).")

ORDER = {
    "sex": ["Female", "Male"],
    "age_group": ["18-44", "45-54", "55-64", "65-74", "75+"],
    "race_ethnicity": ["White", "Hispanic", "Black", "Asian", "AIAN", "AIAN and other", "Other or multiple"],
    "region": ["Northeast", "Midwest", "South", "West"],
    "year": YEARS,
    "education": ["Less than high school", "High school / GED", "Some college", "Bachelor", "Graduate"],
    "income_group": ["Below poverty", "Low", "Middle", "High"],
    "smoking_status": ["Never", "Former", "Current"],
    "ever_ecigarette": [0.0, 1.0],
    "bmi_category": ["Underweight", "Healthy", "Overweight", "Obese"],
    "general_health": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "copd": [0.0, 1.0], "asthma": [0.0, 1.0], "hypertension": [0.0, 1.0],
    "heart_disease": [0.0, 1.0], "diabetes": [0.0, 1.0],
    "comorbidity_count": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}
LABEL = {0.0: "No", 1.0: "Yes"}
rows = []
for var, levels in ORDER.items():
    col = df[var]
    groups = [(lv, col == lv) for lv in levels]
    if col.isna().any():
        groups.append(("Unknown", col.isna()))
    for lv, mask in groups:
        sub = df[mask]
        n, cases = len(sub), int(sub["lung_cancer"].sum())
        name = LABEL.get(lv, lv) if var not in ("year", "comorbidity_count") else (int(lv) if lv != "Unknown" else lv)
        rows.append({"variable": var, "category": name, "adults": n,
                     "pct_of_sample": 100 * n / N,
                     "pct_of_population_weighted": 100 * w[mask].sum() / w.sum(),
                     "cases": cases, "non_cases": n - cases,
                     "pct_of_all_cases": 100 * cases / CASES,
                     "rate_sample_pct": 100 * cases / n if n else np.nan,
                     "rate_weighted_pct": 100 * wmean(sub["lung_cancer"], sub["survey_weight"]) if n else np.nan,
                     "mean_age": sub["age"].mean()})
prof = pd.DataFrame(rows)
prof.round(4).to_csv(TAB / "population_profile.csv", index=False)
for var in ORDER:
    t = prof[prof.variable == var]
    say(f"\n{var}")
    say(t[["category", "adults", "pct_of_sample", "pct_of_population_weighted", "cases", "non_cases",
           "rate_sample_pct", "rate_weighted_pct", "mean_age"]].round(2).to_string(index=False))

# ======================================================================= 2. cases vs non-cases
header("2. CHARACTERISTICS OF ADULTS WITH AND WITHOUT A LUNG-CANCER HISTORY")
yes, no = df[df.lung_cancer == 1], df[df.lung_cancer == 0]
def pct(s, cond):
    s = s.dropna()
    return 100 * cond(s).mean()
char = []
def add(label, f_yes, f_no, kind):
    char.append({"characteristic": label, "lung_cancer": f_yes, "no_lung_cancer": f_no, "unit": kind})
add("Adults", len(yes), len(no), "count")
add("Age, mean", yes.age.mean(), no.age.mean(), "years")
add("Age, standard deviation", yes.age.std(), no.age.std(), "years")
add("Age, median", yes.age.median(), no.age.median(), "years")
add("Age, Q1", yes.age.quantile(.25), no.age.quantile(.25), "years")
add("Age, Q3", yes.age.quantile(.75), no.age.quantile(.75), "years")
add("Aged 65 or older", pct(yes.age, lambda s: s >= 65), pct(no.age, lambda s: s >= 65), "%")
add("Female", pct(yes.sex, lambda s: s == "Female"), pct(no.sex, lambda s: s == "Female"), "%")
for r in ["White", "Hispanic", "Black", "Asian"]:
    add(f"Race/ethnicity: {r}", pct(yes.race_ethnicity, lambda s: s == r), pct(no.race_ethnicity, lambda s: s == r), "%")
other = ["AIAN", "AIAN and other", "Other or multiple"]
add("Race/ethnicity: AIAN / other / multiple", pct(yes.race_ethnicity, lambda s: s.isin(other)),
    pct(no.race_ethnicity, lambda s: s.isin(other)), "%")
add("Ever smoked (former or current)", pct(yes.ever_smoker, lambda s: s == 1), pct(no.ever_smoker, lambda s: s == 1), "%")
add("Current smoker", pct(yes.smoking_status, lambda s: s == "Current"), pct(no.smoking_status, lambda s: s == "Current"), "%")
add("Ever used an e-cigarette", pct(yes.ever_ecigarette, lambda s: s == 1), pct(no.ever_ecigarette, lambda s: s == 1), "%")
for c, lab in [("copd", "COPD"), ("heart_disease", "Heart disease"), ("hypertension", "Hypertension"),
               ("diabetes", "Diabetes"), ("asthma", "Asthma")]:
    add(lab, pct(yes[c], lambda s: s == 1), pct(no[c], lambda s: s == 1), "%")
add("Comorbidity count, mean", yes.comorbidity_count.mean(), no.comorbidity_count.mean(), "conditions")
add("Fair or poor self-rated health", pct(yes.general_health, lambda s: s.isin(["Fair", "Poor"])),
    pct(no.general_health, lambda s: s.isin(["Fair", "Poor"])), "%")
add("Underweight", pct(yes.bmi_category, lambda s: s == "Underweight"), pct(no.bmi_category, lambda s: s == "Underweight"), "%")
add("Obese", pct(yes.bmi_category, lambda s: s == "Obese"), pct(no.bmi_category, lambda s: s == "Obese"), "%")
add("Less than high school", pct(yes.education, lambda s: s == "Less than high school"),
    pct(no.education, lambda s: s == "Less than high school"), "%")
add("Bachelor or graduate degree", pct(yes.education, lambda s: s.isin(["Bachelor", "Graduate"])),
    pct(no.education, lambda s: s.isin(["Bachelor", "Graduate"])), "%")
add("Family income below the poverty line", pct(yes.income_group, lambda s: s == "Below poverty"),
    pct(no.income_group, lambda s: s == "Below poverty"), "%")
for r in ["Northeast", "Midwest", "South", "West"]:
    add(f"Region: {r}", pct(yes.region, lambda s: s == r), pct(no.region, lambda s: s == r), "%")
chart = pd.DataFrame(char)
chart.round(4).to_csv(TAB / "cases_vs_noncases.csv", index=False)
say("Percentages are among adults with a known answer (unweighted, i.e. describing our sample).")
say(chart.round(1).to_string(index=False))

# ======================================================================= 3. inside each group
header("3. AGE, SEX AND SMOKING INSIDE EACH SEX AND RACE/ETHNICITY GROUP")
grp = []
for var in ["sex", "race_ethnicity"]:
    for lv in ORDER[var]:
        s = df[df[var] == lv]
        grp.append({"variable": var, "group": lv, "adults": len(s), "cases": int(s.lung_cancer.sum()),
                    "mean_age": s.age.mean(), "median_age": s.age.median(),
                    "pct_65_plus": 100 * (s.age.dropna() >= 65).mean(),
                    "pct_female": 100 * (s.sex.dropna() == "Female").mean(),
                    "pct_ever_smoked": 100 * s.ever_smoker.dropna().mean(),
                    "pct_current_smoker": 100 * (s.smoking_status.dropna() == "Current").mean(),
                    "pct_copd": 100 * s.copd.dropna().mean(),
                    "rate_sample_pct": 100 * s.lung_cancer.mean(),
                    "rate_weighted_pct": 100 * wmean(s.lung_cancer, s.survey_weight)})
grp = pd.DataFrame(grp)
grp.round(4).to_csv(TAB / "group_profile.csv", index=False)
say(grp.round(1).to_string(index=False))
sm_age = df.groupby("smoking_status")["age"].median()
say(f"\nMedian age by smoking status: " + ", ".join(f"{k} {v:.0f}" for k, v in sm_age.items()))
say(f"Aged 65+: {100 * (df.age.dropna() >= 65).mean():.1f}% of the sample vs "
    f"{100 * wmean((df.age >= 65).where(df.age.notna()).astype(float), w):.1f}% of the population (weighted)")

# ======================================================================= 4. per year
header("4. CASES AND RATES PER SURVEY YEAR (comparability check)")
yr = []
for y in YEARS:
    s = df[df.year == y]
    yr.append({"year": y, "adults": len(s), "cases": int(s.lung_cancer.sum()),
               "rate_sample_pct": 100 * s.lung_cancer.mean(),
               "rate_weighted_pct": 100 * wmean(s.lung_cancer, s.survey_weight),
               "mean_age": s.age.mean(), "pct_ever_smoked": 100 * s.ever_smoker.dropna().mean(),
               "pct_missing_smoking": 100 * s.smoking_status.isna().mean()})
yr = pd.DataFrame(yr)
yr.round(4).to_csv(TAB / "rate_by_year.csv", index=False)
say(yr.round(2).to_string(index=False))
ct = pd.crosstab(df.year, df.lung_cancer)
chi2, p_year, dof, _ = stats.chi2_contingency(ct)
say(f"Chi-square test, survey year x lung cancer: chi2({dof}) = {chi2:.1f}, p = {p_year:.3f}")
say("Note: 2019 was the first year of the redesigned NHIS questionnaire, and 2020 data collection was")
say("disrupted by COVID-19 (telephone interviews). Rates are compared across years for that reason.")

# ======================================================================= 5. smoking adjusted for age
header("5. IS THE SMOKING EFFECT JUST AGE? Smoking odds ratio inside each age group (CMH)")
s5 = df.dropna(subset=["ever_smoker", "age_group"])
strata, tabs = [], []
for g in ORDER["age_group"]:
    t = s5[s5.age_group == g]
    a = ((t.ever_smoker == 1) & (t.lung_cancer == 1)).sum(); b = ((t.ever_smoker == 1) & (t.lung_cancer == 0)).sum()
    c = ((t.ever_smoker == 0) & (t.lung_cancer == 1)).sum(); d = ((t.ever_smoker == 0) & (t.lung_cancer == 0)).sum()
    tabs.append(np.array([[a, b], [c, d]], float))
    orr, lo, hi = odds_ratio_ci(a, b, c, d)
    strata.append({"group": g, "adults": len(t), "cases_ever_smokers": int(a), "cases_never_smokers": int(c),
                   "odds_ratio": orr, "ci_low": lo, "ci_high": hi})
a, b = sum(t[0, 0] for t in tabs), sum(t[0, 1] for t in tabs)
c, d = sum(t[1, 0] for t in tabs), sum(t[1, 1] for t in tabs)
crude = odds_ratio_ci(a, b, c, d)
st = StratifiedTable(tabs)
mh, (mh_lo, mh_hi) = st.oddsratio_pooled, st.oddsratio_pooled_confint()
cmh, bd = st.test_null_odds(), st.test_equal_odds()
strata += [{"group": "Crude (age ignored)", "adults": len(s5), "odds_ratio": crude[0], "ci_low": crude[1], "ci_high": crude[2]},
           {"group": "Mantel-Haenszel (adjusted for age group)", "adults": len(s5), "odds_ratio": mh, "ci_low": mh_lo, "ci_high": mh_hi}]
sa = pd.DataFrame(strata)
sa.round(4).to_csv(TAB / "smoking_by_age_group.csv", index=False)
say(sa.round(2).to_string(index=False))
say(f"CMH test (H0: no smoking-lung cancer link once age group is fixed): chi2 = {cmh.statistic:.1f}, "
    f"p = {stats.chi2.sf(cmh.statistic, 1):.1e}")
say(f"Breslow-Day test (same odds ratio in every age group?): p = {bd.pvalue:.4f}")
say("Reading: the odds ratio falls from the crude value to the age-adjusted value, so part of the crude smoking")
say("effect was age, but a strong association remains. Breslow-Day < 0.05 means the size of the effect differs")
say("between age groups, so the pooled value is an average (the 18-44 stratum has very few cases).")

# ======================================================================= 6. COPD adjusted for smoking and age
header("6. COPD ADJUSTED FOR SMOKING, THEN FOR SMOKING x AGE GROUP")
s6 = df.dropna(subset=["copd", "smoking_status", "age_group"])
def mh_by(data, keys):
    tb = []
    for _, t in data.groupby(keys):
        a = ((t.copd == 1) & (t.lung_cancer == 1)).sum(); b = ((t.copd == 1) & (t.lung_cancer == 0)).sum()
        c = ((t.copd == 0) & (t.lung_cancer == 1)).sum(); d = ((t.copd == 0) & (t.lung_cancer == 0)).sum()
        if (a + c) > 0 and (b + d) > 0 and (a + b) > 0 and (c + d) > 0:
            tb.append(np.array([[a, b], [c, d]], float))
    st = StratifiedTable(tb)
    lo, hi = st.oddsratio_pooled_confint()
    stat = st.test_null_odds().statistic
    return st.oddsratio_pooled, lo, hi, stats.chi2.sf(stat, 1), len(tb), stat
a = ((s6.copd == 1) & (s6.lung_cancer == 1)).sum(); b = ((s6.copd == 1) & (s6.lung_cancer == 0)).sum()
c = ((s6.copd == 0) & (s6.lung_cancer == 1)).sum(); d = ((s6.copd == 0) & (s6.lung_cancer == 0)).sum()
cr = odds_ratio_ci(a, b, c, d)
r1, r2 = mh_by(s6, ["smoking_status"]), mh_by(s6, ["smoking_status", "age_group"])
cop = pd.DataFrame([{"adjustment": "none (crude)", "odds_ratio": cr[0], "ci_low": cr[1], "ci_high": cr[2], "strata": 1},
                    {"adjustment": "smoking status", "odds_ratio": r1[0], "ci_low": r1[1], "ci_high": r1[2], "strata": r1[4]},
                    {"adjustment": "smoking status x age group", "odds_ratio": r2[0], "ci_low": r2[1], "ci_high": r2[2], "strata": r2[4]}])
cop.round(4).to_csv(TAB / "copd_adjusted.csv", index=False)
say(f"Adults with known COPD, smoking status and age: {len(s6):,} ({int(s6.lung_cancer.sum())} cases)")
say(cop.round(2).to_string(index=False))
say(f"CMH test with smoking x age strata: chi2 = {r2[5]:.1f}, p = {r2[3]:.1e}")

# ======================================================================= 7. RQ3 between vs within regions
header("7. RQ3: BETWEEN-REGION AND WITHIN-REGION CORRELATIONS")
r = pd.read_csv(TAB / "regional_comparison.csv")
say("The 20 region-years are 4 regions x 5 years, so they are not 20 independent observations.")
reg_w = df.groupby("region").apply(lambda t: 100 * wmean(t.lung_cancer, t.survey_weight), include_groups=False)
reg_s = df.groupby("region")["lung_cancer"].mean() * 100
means = r.groupby("region")[["nhis_rate_%", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]].mean()
means["pooled_weighted_rate_%"] = reg_w
means["pooled_sample_rate_%"] = reg_s
say("Region means over 2019-2023:")
say(means.round(3).to_string())
rw = []
for cvar in ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]:
    rho20, p20 = stats.spearmanr(r[cvar], r["nhis_rate_%"])
    rho4, _ = stats.spearmanr(means[cvar], means["nhis_rate_%"])
    x = r[cvar] - r.groupby("region")[cvar].transform("mean")
    y = r["nhis_rate_%"] - r.groupby("region")["nhis_rate_%"].transform("mean")
    rhow, pw = stats.spearmanr(x, y)
    rw.append({"regional_measure": cvar, "rho_all_20": rho20, "p_all_20": p20, "rho_between_4_regions": rho4,
               "rho_within_region": rhow, "p_within_region": pw})
rw = pd.DataFrame(rw)
rw.round(4).to_csv(TAB / "regional_within_between.csv", index=False)
say(rw.round(3).to_string(index=False))
say("Reading: the registry agreement comes from differences BETWEEN regions (the West is low on both measures);")
say("year-to-year changes within a region do not move together. With effectively 4 independent regions, the")
say("agreement is descriptive support for the target, not a formal validation, and its p-value is optimistic.")

# ======================================================================= 8. screening rule as a benchmark
header("8. THE USPSTF-STYLE RULE AS A BENCHMARK FOR THE FUTURE MODEL")
mr = pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv")
base = []
for name, part in [("all adults", mr), ("test rows", mr[mr.split == "test"])]:
    flag = part["screening_age_smoker"] == 1
    tp = int((flag & (part.lung_cancer == 1)).sum())
    pos, cases = int(flag.sum()), int(part.lung_cancer.sum())
    base.append({"rows": name, "adults": len(part), "flagged": pos, "pct_flagged": 100 * pos / len(part),
                 "cases": cases, "cases_flagged": tp, "recall_pct": 100 * tp / cases,
                 "precision_pct": 100 * tp / pos, "lift_vs_base_rate": (tp / pos) / (cases / len(part))})
base = pd.DataFrame(base)
base.round(4).to_csv(TAB / "screening_rule_baseline.csv", index=False)
say("Rule: flag every adult aged 50-80 who has ever smoked (no pack-years in NHIS, so a proxy).")
say(base.round(2).to_string(index=False))

# ======================================================================= 9. mutual information, normalised + null
header("9. MUTUAL INFORMATION: NORMALISED, AND COMPARED WITH A PERMUTATION NULL (training rows)")
train = mr[mr.split == "train"]
y_tr = train["lung_cancer"].values
p1 = y_tr.mean()
H = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
mi = pd.read_csv(TAB / "feature_relevance_mutual_information.csv", index_col=0)["mutual_information"]
check = ["age", "copd", "comorbidity_count", "screening_age_smoker", "smoking_status_Never",
         "pm25_mean", "smoking_rate", "lung_cancer_incidence"]
X = train[check].copy()
discrete = [X[c].nunique() <= 25 for c in check]
for c, dsc in zip(check, discrete):
    if dsc:
        X[c] = pd.factorize(X[c], sort=True)[0]
rng = np.random.default_rng(RANDOM_STATE)
N_PERM = 100
null = np.array([mutual_info_classif(X, rng.permutation(y_tr), discrete_features=discrete,
                                     random_state=RANDOM_STATE) for _ in range(N_PERM)])
mic = pd.DataFrame({"feature": check, "mi": [mi[c] for c in check],
                    "mi_pct_of_target_entropy": [100 * mi[c] / H for c in check],
                    "null_mean": null.mean(axis=0), "null_95th_percentile": np.percentile(null, 95, axis=0),
                    "permutation_p": [(1 + (null[:, j] >= mi[c]).sum()) / (N_PERM + 1) for j, c in enumerate(check)]})
mic.to_csv(TAB / "mutual_information_checked.csv", index=False)
say(f"Entropy of the target (0.41% positive): {H:.4f} nats. MI / entropy = share of the available information.")
say(f"Null: the target shuffled {N_PERM} times (same features), MI recomputed each time.")
say(mic.round(5).to_string(index=False))

# ======================================================================= 10. survey design
header("10. SURVEY-DESIGN SENSITIVITY: weighted odds ratios with cluster-robust confidence intervals")
design = []
for y in YEARS:
    with zipfile.ZipFile(RAW / f"adult{str(y)[-2:]}csv.zip") as z:
        member = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(member) as f:
            dd = pd.read_csv(f, usecols=lambda c: c.upper() in ("HHX", "PSTRAT", "PPSU"), low_memory=False)
    dd.columns = dd.columns.str.upper()
    dd["person_id"] = f"{y}_" + dd["HHX"].astype(str)
    dd["cluster"] = f"{y}_" + dd["PSTRAT"].astype(str) + "_" + dd["PPSU"].astype(str)
    design.append(dd[["person_id", "cluster"]])
design = pd.concat(design)
dsg = df.merge(design, on="person_id", how="left", validate="one_to_one")
say(f"Design variables matched for {dsg.cluster.notna().sum():,} of {len(dsg):,} adults; "
    f"{dsg.cluster.nunique()} clusters (year x stratum x PSU).")
def design_or(data, xcol, label):
    d = data.dropna(subset=[xcol, "cluster"])
    X = sm.add_constant(d[[xcol]].astype(float))
    wt = d["survey_weight"] / d["survey_weight"].mean()
    naive = sm.GLM(d["lung_cancer"], X, family=sm.families.Binomial()).fit()
    des = sm.GLM(d["lung_cancer"], X, family=sm.families.Binomial(), var_weights=wt).fit()
    # cluster-robust (sandwich) covariance for the weighted fit, computed explicitly:
    # bread = (X' diag(w mu (1 - mu)) X)^-1, meat = sum over clusters of the summed weighted scores
    Xv, mu = X.values, des.fittedvalues.values
    bread = np.linalg.inv(Xv.T @ (Xv * (wt.values * mu * (1 - mu))[:, None]))
    score = Xv * (wt.values * (d["lung_cancer"].values - mu))[:, None]
    g = pd.factorize(d["cluster"])[0]
    S = np.zeros((g.max() + 1, Xv.shape[1]))
    np.add.at(S, g, score)
    G = len(S)
    cov = bread @ (S.T @ S) @ bread * G / (G - 1)
    j = list(X.columns).index(xcol)
    beta, se = des.params[xcol], np.sqrt(cov[j, j])
    lo_n, hi_n = np.exp(naive.conf_int().loc[xcol])
    lo_d, hi_d = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
    return {"exposure": label, "adults": len(d), "or_unweighted": np.exp(naive.params[xcol]), "ci_low_unweighted": lo_n,
            "ci_high_unweighted": hi_n, "or_design": np.exp(beta), "ci_low_design": lo_d, "ci_high_design": hi_d,
            "ci_width_ratio": (np.log(hi_d) - np.log(lo_d)) / (np.log(hi_n) - np.log(lo_n))}
dsg["age_10y"] = dsg["age"] / 10
res = pd.DataFrame([design_or(dsg, "ever_smoker", "Ever smoked (vs never)"),
                    design_or(dsg, "copd", "COPD (vs none)"),
                    design_or(dsg, "age_10y", "Age, per 10 years")])
res.round(4).to_csv(TAB / "design_based_odds_ratios.csv", index=False)
say(res.round(2).to_string(index=False))
say("Unweighted = simple logistic regression (exact odds ratio; the 2x2 tables elsewhere add 0.5 to every cell,")
say("e.g. 7.66 instead of 7.69 for smoking). Design = survey-weighted logistic regression with standard errors")
say("clustered on the NHIS sampling units (year x stratum x PSU). The weighted point estimates are somewhat")
say("larger, the interval widths change by the factor ci_width_ratio, and every interval stays far from 1.")

# ======================================================================= 11. automated leakage check
header("11. AUTOMATED CHECK OF SUCCESS CRITERION 5 (scaler fitted on training rows only)")
sp = pd.read_csv(CLEAN / "scaler_parameters.csv")
cols = ["age", "comorbidity_count", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]
refit = StandardScaler().fit(train[cols])
saved_mean = sp.set_index(sp.columns[0]).loc[cols].iloc[:, 0].values
same = np.array_equal(np.round(refit.mean_, 4), np.round(saved_mean, 4))   # file stores 4 decimals
allfit = StandardScaler().fit(mr[cols])
differs = not np.array_equal(np.round(allfit.mean_, 4), np.round(saved_mean, 4))
say(f"Saved scaler means equal a refit on training rows only: {same}")
say(f"Saved scaler means differ from a fit on all rows: {differs}")
assert same and differs, "Criterion 5 failed: the scaler was not fitted on training rows only"
say("Criterion 5: PASSED (checked, not assumed).")

(ROOT / "reports" / "supplementary_results.txt").write_text("\n".join(out), encoding="utf-8")
print("\nDone. Results in reports/supplementary_results.txt")
