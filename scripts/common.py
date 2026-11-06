"""
Shared paths and constants used by every pipeline script, build_report.py and the notebook.

    from common import ROOT, RAW, CLEAN, FIG, TAB, YEARS, RANDOM_STATE
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw charts without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                 # original files: nhis/, epa/, brfss/, uscs/, other/
CLEAN = ROOT / "data" / "clean"             # clean and model-ready datasets, cleaning logs
RESULTS = ROOT / "results"                  # pipeline outputs: text results, figures/, tables/
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
for _folder in (CLEAN, FIG, TAB):
    _folder.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023]      # NHIS survey years
RANDOM_STATE = 42                           # every random step (split, sampling, MI, permutations)

# Chart colours: blue = personal factors / main series, orange = regional or highlighted, grey = reference
BLUE, ORANGE, GREY = "#2F6690", "#D1603D", "#B8B8B8"

# Columns of the model-ready file that are not features
NON_FEATURES = ("person_id", "split", "survey_weight", "lung_cancer")
# The three regional columns joined on region + year
REGIONAL = ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]

# Reference category of every one-hot group: dropped from the model-ready file (Rule 4), so the other
# dummies of the group are read relative to it and the group no longer sums to 1 (no exact collinearity)
REFERENCE = {"race_ethnicity": "White", "education": "High school / GED", "income_group": "High",
             "bmi_category": "Healthy", "general_health": "Excellent", "smoking_status": "Never",
             "sex": "Male", "region": "West"}

# Column type of every column of the 26-column clean file
COLUMN_TYPES = {
    "identifier": ["person_id"], "weight": ["survey_weight"], "target": ["lung_cancer"],
    "numeric": ["age", "year"],
    "ordinal": ["education", "income_group", "general_health", "bmi_category"],
    "nominal": ["race_ethnicity", "region", "smoking_status"],
    "binary": ["sex", "ever_ecigarette", "copd", "asthma", "hypertension", "heart_disease", "diabetes"],
    "engineered": ["age_group", "ever_smoker", "screening_age_smoker", "comorbidity_count"],
    "regional": REGIONAL,
}
# Group of every final feature, by column-name prefix
FEATURE_GROUPS = {
    "demographic": ("age", "sex_", "race_ethnicity_"),
    "socio-economic": ("education_", "income_group_"),
    "smoking": ("smoking_status_", "screening_age_smoker", "ever_ecigarette"),
    "health": ("copd", "asthma", "hypertension", "heart_disease", "diabetes", "general_health_", "bmi_category_"),
    "regional": tuple(REGIONAL),
}


# ---------------------------------------------------------------- text results and figures
def text_log():
    """(lines, say, header) for a script's plain-text results: say() prints a line and keeps it,
    header() prints a section title between two rules; the caller writes "\n".join(lines) to a file."""
    lines = []

    def say(msg=""):
        print(msg)
        lines.append(str(msg))

    def header(title):
        say("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

    return lines, say, header


def set_style():
    """Chart style of the EDA and PCA figures (fig03 deliberately keeps matplotlib's defaults)."""
    sns.set_theme(style="whitegrid", context="talk")


def save_fig(name, dpi=150):
    """Save the current figure as results/figures/<name>.png and close it."""
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- statistics shared by the EDA scripts
def weighted_mean(x, w):
    """Weighted mean of x, ignoring missing x (NaN if nothing is left)."""
    ok = x.notna()
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def weighted_rate(d):
    """Population (survey-weighted) % with lung cancer - corrects for NHIS over-sampling some groups."""
    return 100 * weighted_mean(d["lung_cancer"], d["survey_weight"])


def two_by_two(exposed, case):
    """2x2 table [[a, b], [c, d]] = exposed cases, exposed non-cases, unexposed cases, unexposed non-cases.
    exposed and case are boolean Series over the same rows."""
    return np.array([[(exposed & case).sum(), (exposed & ~case).sum()],
                     [(~exposed & case).sum(), (~exposed & ~case).sum()]], float)


def odds_ratio_ci(a, b, c, d):
    """Odds ratio of a 2x2 table [[a, b], [c, d]] with a 95% CI; +0.5 in every cell (Haldane-Anscombe)
    avoids dividing by zero."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)


def mantel_haenszel(tables):
    """Pooled odds ratio over 2x2 strata: (OR, CI low, CI high, CMH statistic, CMH p, Breslow-Day p).
    p-values use the chi-square survival function: statsmodels returns 1 - cdf, which rounds to 0 for
    very large statistics."""
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    cmh, bd = st.test_null_odds().statistic, st.test_equal_odds().statistic
    return st.oddsratio_pooled, lo, hi, cmh, stats.chi2.sf(cmh, 1), stats.chi2.sf(bd, len(tables) - 1)


def mi_inputs(X):
    """Prepare X for mutual_info_classif: a column with 25 or fewer distinct values is DISCRETE (the
    continuous k-nearest-neighbour estimator is biased upwards on heavily tied values) and gets integer
    codes (MI does not depend on the labels). Returns (coded X, discrete mask)."""
    discrete = [X[c].nunique() <= 25 for c in X.columns]
    X = X.copy()
    for c, d in zip(X.columns, discrete):
        if d:
            X[c] = pd.factorize(X[c], sort=True)[0]
    return X, discrete
