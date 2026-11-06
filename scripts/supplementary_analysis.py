"""
Step 5 - Detailed population profile and supplementary checks.

    python scripts/supplementary_analysis.py

Reads : data/clean/nhis_lung_cancer_clean.csv, data/clean/nhis_lung_cancer_model_ready.csv,
        data/clean/scaler_parameters.csv, results/tables/regional_comparison.csv,
        results/tables/feature_relevance_mutual_information.csv,
        data/raw/nhis/adultYYcsv.zip (only for the survey-design variables PSTRAT / PPSU)
Writes: results/tables/dataset_composition.csv         rows per year, column types, feature groups, train/test
        results/tables/population_profile.csv          every category: adults, cases, rates
        results/tables/cases_vs_noncases.csv           characteristics of cases vs other adults
        results/tables/group_profile.csv               age, sex and smoking inside each sex / race group
        results/tables/rate_by_year.csv                cases and rates per survey year
        results/tables/smoking_by_age_group.csv        RQ1: smoking odds ratio inside each age group (MH + Breslow-Day)
        results/tables/copd_adjusted.csv               RQ2b: COPD odds ratio adjusted for smoking, then smoking x age
        results/tables/regional_within_between.csv     RQ3: correlations split into between- and within-region
        results/tables/screening_rule_baseline.csv     the USPSTF-style rule as a benchmark
        results/tables/mutual_information_checked.csv  RQ3: normalised MI and a permutation null
        results/tables/design_based_odds_ratios.csv    survey-weighted, cluster-robust odds ratios
        results/supplementary_results.txt              everything above in plain words

Outcome and research questions: see eda_analysis.py (OUTCOME, RESEARCH_QUESTIONS). This script adds the
age-adjusted smoking odds ratio (RQ1), the COPD odds ratio adjusted for smoking x age group (RQ2b), the
between- / within-region split and the regional mutual-information check (RQ3); it runs no new hypothesis
test that enters the 18 Holm-corrected tests.

Nothing here changes the clean or model-ready files. Descriptive tables use all adults (they describe the
population and fit nothing); the mutual-information check uses training rows only.
"""
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from common import (CLEAN, COLUMN_TYPES, FEATURE_GROUPS, NON_FEATURES, RANDOM_STATE, RAW, REGIONAL, RESULTS,
                    TAB, YEARS, mantel_haenszel, mi_inputs, odds_ratio_ci, two_by_two, weighted_mean as wmean,
                    weighted_rate, text_log)

out, say, header = text_log()

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



# ======================================================================= 0. dataset composition
def dataset_composition(df, mr):
    """What the EDA numbers are computed on: rows per survey year, column types of the clean file, the final
    feature matrix by group, and the train / test split."""
    header("0. DATASET COMPOSITION AT A GLANCE")
    rows = []
    def add(block, item, part=None, columns=None):
        r = {"block": block, "item": item}
        if part is not None:
            r.update(adults=len(part), cases=int(part.lung_cancer.sum()), rate_pct=100 * part.lung_cancer.mean())
        if columns is not None:
            r.update(n_columns=len(columns), columns=", ".join(columns))
        rows.append(r)
    for y in YEARS:
        add("1. adults per survey year", str(y), df[df.year == y])
    add("1. adults per survey year", "all years", df)
    assert sorted(sum(COLUMN_TYPES.values(), [])) == sorted(df.columns), "COLUMN_TYPES must cover the 26 columns"
    for kind, cols in COLUMN_TYPES.items():
        add("2. columns of the clean file", kind, columns=cols)
    add("2. columns of the clean file", "all columns", columns=list(df.columns))
    feats = [c for c in mr.columns if c not in NON_FEATURES]
    group = {c: next(g for g, pre in FEATURE_GROUPS.items() if c.startswith(pre)) for c in feats}
    for g in FEATURE_GROUPS:
        add("3. final features by group", g, columns=[c for c in feats if group[c] == g])
    add("3. final features by group", "all final features", columns=feats)
    for part in ["train", "test"]:
        add("4. train / test split", part, mr[mr.split == part])
    add("4. train / test split", "all", mr)
    comp = pd.DataFrame(rows, columns=["block", "item", "adults", "cases", "rate_pct", "n_columns", "columns"])
    comp[["adults", "cases", "n_columns"]] = comp[["adults", "cases", "n_columns"]].astype("Int64")
    comp.to_csv(TAB / "dataset_composition.csv", index=False)
    for b, t in comp.groupby("block", sort=False):
        say(f"\n{b}")
        t = t.drop(columns="block").dropna(axis=1, how="all")
        say(t.round({"rate_pct": 2}).to_string(index=False, max_colwidth=90))
    say("\nRates are unweighted sample rates (cases / adults). Every later table states its own subset.")
    return comp


# ======================================================================= 1. population profile
def population_profile(df):
    w = df["survey_weight"]
    N, CASES = len(df), int(df["lung_cancer"].sum())
    header("1. WHO IS IN THE DATA? Every category: adults, share, cases, rates")
    say(f"{N:,} adults, {CASES} with a lung-cancer history. 'Weighted' = NHIS survey weights (US adult population).")
    say("Unknown = the person did not give a usable answer (kept as its own row so the counts add up).")

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
                         "rate_weighted_pct": weighted_rate(sub) if n else np.nan,
                         "mean_age": sub["age"].mean()})
    prof = pd.DataFrame(rows)
    prof.round(4).to_csv(TAB / "population_profile.csv", index=False)
    for var in ORDER:
        t = prof[prof.variable == var]
        say(f"\n{var}")
        say(t[["category", "adults", "pct_of_sample", "pct_of_population_weighted", "cases", "non_cases",
               "rate_sample_pct", "rate_weighted_pct", "mean_age"]].round(2)
            .assign(mean_age=t["mean_age"].round(1)).to_string(index=False))   # 1 decimal from the raw value
    return prof


# ======================================================================= 2. cases vs non-cases
def cases_vs_noncases(df):
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
    return chart


# ======================================================================= 3. inside each group
def group_profile(df):
    w = df["survey_weight"]
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
                        "rate_weighted_pct": weighted_rate(s)})
    grp = pd.DataFrame(grp)
    grp.round(4).to_csv(TAB / "group_profile.csv", index=False)
    say(grp.round(1).to_string(index=False))
    sm_age = df.groupby("smoking_status")["age"].median()
    say("\nMedian age by smoking status: " +", ".join(f"{k} {v:.0f}" for k, v in sm_age.items()))
    say(f"Aged 65+: {100 * (df.age.dropna() >= 65).mean():.1f}% of the sample vs "
        f"{100 * wmean((df.age >= 65).where(df.age.notna()).astype(float), w):.1f}% of the population (weighted)")
    return grp


# ======================================================================= 4. per year
def rate_by_year(df):
    header("4. CASES AND RATES PER SURVEY YEAR (comparability check)")
    yr = []
    for y in YEARS:
        s = df[df.year == y]
        yr.append({"year": y, "adults": len(s), "cases": int(s.lung_cancer.sum()),
                   "rate_sample_pct": 100 * s.lung_cancer.mean(),
                   "rate_weighted_pct": weighted_rate(s),
                   "mean_age": s.age.mean(), "pct_ever_smoked": 100 * s.ever_smoker.dropna().mean(),
                   "pct_missing_smoking": 100 * s.smoking_status.isna().mean()})
    yr = pd.DataFrame(yr)
    yr.round(4).to_csv(TAB / "rate_by_year.csv", index=False)
    say(yr.round(2).assign(mean_age=yr["mean_age"].round(1)).to_string(index=False))
    ct = pd.crosstab(df.year, df.lung_cancer)
    chi2, p_year, dof, _ = stats.chi2_contingency(ct)
    say(f"Chi-square test, survey year x lung cancer: chi2({dof}) = {chi2:.1f}, p = {p_year:.3f}")
    say("Note: 2019 was the first year of the redesigned NHIS questionnaire, and 2020 data collection was")
    say("disrupted by COVID-19 (telephone interviews). Rates are compared across years for that reason.")
    return yr


# ======================================================================= 5. smoking adjusted for age
def smoking_by_age(df):
    header("5. RQ1: IS THE SMOKING EFFECT JUST AGE? Smoking odds ratio inside each age group (CMH)")
    s5 = df.dropna(subset=["ever_smoker", "age_group"])
    strata, tabs = [], []
    for g in ORDER["age_group"]:
        t = s5[s5.age_group == g]
        tabs.append(two_by_two(t.ever_smoker == 1, t.lung_cancer == 1))
        (a, b), (c, d) = tabs[-1]
        orr, lo, hi = odds_ratio_ci(a, b, c, d)
        strata.append({"group": g, "adults": len(t), "cases_ever_smokers": int(a), "cases_never_smokers": int(c),
                       "odds_ratio": orr, "ci_low": lo, "ci_high": hi})
    a, b = sum(t[0, 0] for t in tabs), sum(t[0, 1] for t in tabs)
    c, d = sum(t[1, 0] for t in tabs), sum(t[1, 1] for t in tabs)
    crude = odds_ratio_ci(a, b, c, d)
    mh, mh_lo, mh_hi, cmh_stat, cmh_p, bd_p = mantel_haenszel(tabs)
    strata += [{"group": "Crude (age ignored)", "adults": len(s5), "odds_ratio": crude[0], "ci_low": crude[1], "ci_high": crude[2]},
               {"group": "Mantel-Haenszel (adjusted for age group)", "adults": len(s5), "odds_ratio": mh, "ci_low": mh_lo, "ci_high": mh_hi}]
    sa = pd.DataFrame(strata)
    sa.round(4).to_csv(TAB / "smoking_by_age_group.csv", index=False)
    say(sa.round(2).to_string(index=False))
    say(f"CMH test (H0: no smoking-lung cancer link once age group is fixed): chi2 = {cmh_stat:.1f}, "
        f"p = {cmh_p:.1e}")
    say(f"Breslow-Day test (same odds ratio in every age group?): p = {bd_p:.4f}")
    say("Reading: the odds ratio falls from the crude value to the age-adjusted value, so part of the crude smoking")
    say("effect was age, but a strong association remains. Breslow-Day < 0.05 means the size of the effect differs")
    say("between age groups, so the pooled value is an average (the 18-44 stratum has very few cases).")
    return sa


# ======================================================================= 6. COPD adjusted for smoking and age
def copd_adjusted(df):
    header("6. RQ2b: COPD ADJUSTED FOR SMOKING, THEN FOR SMOKING x AGE GROUP")
    s6 = df.dropna(subset=["copd", "smoking_status", "age_group"])
    def mh_by(data, keys):
        tb = []
        for _, t in data.groupby(keys):
            t = two_by_two(t.copd == 1, t.lung_cancer == 1)
            if (t.sum(axis=0) > 0).all() and (t.sum(axis=1) > 0).all():     # skip strata with an empty margin
                tb.append(t)
        orr, lo, hi, stat, p, _ = mantel_haenszel(tb)
        return orr, lo, hi, p, len(tb), stat
    cr = odds_ratio_ci(*two_by_two(s6.copd == 1, s6.lung_cancer == 1).ravel())
    r1, r2 = mh_by(s6, ["smoking_status"]), mh_by(s6, ["smoking_status", "age_group"])
    cop = pd.DataFrame([{"adjustment": "none (crude)", "odds_ratio": cr[0], "ci_low": cr[1], "ci_high": cr[2], "strata": 1},
                        {"adjustment": "smoking status", "odds_ratio": r1[0], "ci_low": r1[1], "ci_high": r1[2], "strata": r1[4]},
                        {"adjustment": "smoking status x age group", "odds_ratio": r2[0], "ci_low": r2[1], "ci_high": r2[2], "strata": r2[4]}])
    cop.round(4).to_csv(TAB / "copd_adjusted.csv", index=False)
    say(f"Adults with known COPD, smoking status and age: {len(s6):,} ({int(s6.lung_cancer.sum())} cases)")
    say(cop.round(2).to_string(index=False))
    say(f"CMH test with smoking x age strata: chi2 = {r2[5]:.1f}, p = {r2[3]:.1e}")
    return cop


# ======================================================================= 7. RQ3 between vs within regions
def regional_within_between(df):
    header("7. RQ3: BETWEEN-REGION AND WITHIN-REGION CORRELATIONS")
    r = pd.read_csv(TAB / "regional_comparison.csv")
    say("The 20 region-years are 4 regions x 5 years, so they are not 20 independent observations.")
    reg_w = df.groupby("region").apply(weighted_rate, include_groups=False)
    reg_s = df.groupby("region")["lung_cancer"].mean() * 100
    means = r.groupby("region")[["nhis_rate_%", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]].mean()
    means["pooled_weighted_rate_%"] = reg_w
    means["pooled_sample_rate_%"] = reg_s
    say("Region means over 2019-2023:")
    say(means.round(3).to_string())
    rw = []
    for cvar in REGIONAL:
        rho20, p20 = stats.spearmanr(r[cvar], r["nhis_rate_%"])
        rho4, _ = stats.spearmanr(means[cvar], means["nhis_rate_%"])
        x = r[cvar] - r.groupby("region")[cvar].transform("mean")
        y = r["nhis_rate_%"] - r.groupby("region")["nhis_rate_%"].transform("mean")
        rhow, pw = stats.spearmanr(x, y)
        rw.append({"regional_measure": cvar, "rho_all_20": rho20, "p_all_20": p20, "rho_between_4_regions": rho4,
                   "rho_within_region": rhow, "p_within_region": pw})
    rw = pd.DataFrame(rw)
    p_cols = ["p_all_20", "p_within_region"]
    # p-values are written unrounded: rounding to 4 decimals turned p = 3.5e-05 into "0"
    rw.round({c: 4 for c in rw.columns if c not in p_cols}).to_csv(TAB / "regional_within_between.csv", index=False)
    say(rw.round(3).assign(**{c: rw[c].map(lambda v: f"{v:.2g}") for c in p_cols}).to_string(index=False))
    ranks = means[["nhis_rate_%", "pm25_mean", "smoking_rate", "lung_cancer_incidence"]].rank().astype(int)
    say("\nRanks of the 4 region means (1 = lowest), used by rho_between_4_regions:")
    say(ranks.to_string())
    for cvar in REGIONAL:
        d2 = int(((ranks[cvar] - ranks["nhis_rate_%"]) ** 2).sum())
        say(f"  {cvar:<22} sum of squared rank differences = {d2:>2} -> rho = 1 - 6 x {d2} / (4 x 15) = {1 - d2 / 10:.1f}")
    say("With n = 4, Spearman rho can only take the values 0, +-0.2, +-0.4, ..., +-1, so a between-region rho of")
    say("exactly 0 is a real value (not a computation error); with 4 points it is not evidence either way.")
    say("Reading: the registry agreement comes from differences BETWEEN regions (the West is low on both measures);")
    say("year-to-year changes within a region do not move together. With effectively 4 independent regions, the")
    say("agreement is descriptive support for the target, not a formal validation, and its p-value is optimistic.")
    return rw


# ======================================================================= 8. screening rule as a benchmark
def screening_baseline(mr):
    header("8. THE USPSTF-STYLE RULE AS A BENCHMARK FOR THE FUTURE MODEL")
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
    return base


# ======================================================================= 9. mutual information, normalised + null
def mi_checked(mr):
    header("9. RQ3: MUTUAL INFORMATION, NORMALISED AND COMPARED WITH A PERMUTATION NULL (training rows)")
    train = mr[mr.split == "train"]
    y_tr = train["lung_cancer"].values
    p1 = y_tr.mean()
    H = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    mi = pd.read_csv(TAB / "feature_relevance_mutual_information.csv", index_col=0)["mutual_information"]
    check = ["age", "copd", "hypertension", "screening_age_smoker", "smoking_status_Former",
             "pm25_mean", "smoking_rate", "lung_cancer_incidence"]
    X, discrete = mi_inputs(train[check])                 # same coding as eda_analysis.feature_relevance
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
    ri = pd.read_csv(TAB / "regional_mi_identity.csv").set_index("column")
    say("Regional rows: smoking_rate and lung_cancer_incidence have identical MI and null percentiles because each")
    say(f"takes one value per region-year cell ({int(ri.loc['smoking_rate', 'distinct_values'])} and "
        f"{int(ri.loc['lung_cancer_incidence', 'distinct_values'])} values for "
        f"{int(ri.loc['smoking_rate', 'region_year_cells'])} cells): as discrete columns they split the adults")
    say(f"into the same groups. pm25_mean has {int(ri.loc['pm25_mean', 'distinct_values'])} values (ties: "
        f"{ri.loc['pm25_mean', 'cells_sharing_a_value']}),")
    say("so its MI is slightly lower. The three regional MI values measure one thing - region-year identity - not")
    say("three independent signals (results/tables/regional_mi_identity.csv). They are above the shuffled-target")
    say("null, but tiny next to the personal factors.")
    return mic


# ======================================================================= 10. survey design
def survey_design(df):
    header("10. SURVEY-DESIGN SENSITIVITY: weighted odds ratios with cluster-robust confidence intervals")
    design = []
    for y in YEARS:
        with zipfile.ZipFile(RAW / "nhis" / f"adult{str(y)[-2:]}csv.zip") as z:
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
    return res


# ======================================================================= 11. automated leakage check
def criterion5(mr):
    train = mr[mr.split == "train"]
    header("11. AUTOMATED CHECK OF SUCCESS CRITERION 5 (scaler fitted on training rows only)")
    sp = pd.read_csv(CLEAN / "scaler_parameters.csv")
    cols = sp["column"].tolist()             # the scaled columns, as saved by build_clean_dataset.py
    refit = StandardScaler().fit(train[cols])
    saved_mean = sp.set_index(sp.columns[0]).loc[cols].iloc[:, 0].values
    same = np.array_equal(np.round(refit.mean_, 4), np.round(saved_mean, 4))   # file stores 4 decimals
    allfit = StandardScaler().fit(mr[cols])
    differs = not np.array_equal(np.round(allfit.mean_, 4), np.round(saved_mean, 4))
    say(f"Saved scaler means equal a refit on training rows only: {same}")
    say(f"Saved scaler means differ from a fit on all rows: {differs}")
    assert same and differs, "Criterion 5 failed: the scaler was not fitted on training rows only"
    say("Criterion 5: PASSED (checked, not assumed).")


def main():
    out.clear()
    df = pd.read_csv(CLEAN / "nhis_lung_cancer_clean.csv")
    mr = pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv")
    dataset_composition(df, mr)
    population_profile(df)
    cases_vs_noncases(df)
    group_profile(df)
    rate_by_year(df)
    smoking_by_age(df)
    copd_adjusted(df)
    regional_within_between(df)
    screening_baseline(mr)
    mi_checked(mr)
    survey_design(df)
    criterion5(mr)
    (RESULTS / "supplementary_results.txt").write_text("\n".join(out), encoding="utf-8")
    print("\nDone. Results in results/supplementary_results.txt")


if __name__ == "__main__":
    main()
