import math

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(table):
    table = np.asarray(table)
    if table.size == 0:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, c = table.shape
    phi = math.sqrt(chi2 / n)
    return float(phi / math.sqrt(min(r - 1, c - 1)))


def odds_ratio_ci(x, y):
    a = ((x == 1) & (y == 1)).sum()
    b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum()
    d = ((x == 0) & (y == 0)).sum()
    or_value = (a * d) / (b * c)
    log_or = math.log(or_value)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low = math.exp(log_or - 1.96 * se)
    ci_high = math.exp(log_or + 1.96 * se)
    return or_value, ci_low, ci_high


def permutation_p(x, y, n_perms=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    observed = stats.chi2_contingency(pd.crosstab(x, y), correction=False)[0]
    count = 0
    binary = np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all()
    if binary:
        n = len(y)
        row_one = int(np.sum(x == 1))
        row_zero = n - row_one
        col_one = int(np.sum(y == 1))
        col_zero = n - col_one
        denominator = row_one * row_zero * col_one * col_zero
        if denominator == 0:
            return 1.0
        x_one = x == 1
        for _ in range(n_perms):
            intersection = int(np.sum(x_one & (rng.permutation(y) == 1)))
            numerator = (n * intersection - row_one * col_one) ** 2 * n
            if numerator / denominator >= observed:
                count += 1
        return count / n_perms

    for _ in range(n_perms):
        perm = rng.permutation(len(y))
        permuted = pd.crosstab(x, y[perm])
        if stats.chi2_contingency(permuted, correction=False)[0] >= observed:
            count += 1
    return count / n_perms


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values in original input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be finite and between 0 and 1")
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(m, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def feature_family(df, feature_cols, target_col="LUNG_CANCER"):
    rows = []
    for col in feature_cols:
        table = pd.crosstab(df[col], df[target_col])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        rows.append({
            "feature": col,
            "chi2": float(chi2),
            "p": float(p),
            "v": float(cramers_v(table)),
        })
    results = pd.DataFrame(rows)
    results["q_BH"] = benjamini_hochberg(results["p"].to_numpy())
    return results


def mann_whitney_cles(x, y):
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return stat, p, float(np.mean(x > y) + 0.5 * np.mean(x == y))


def risk_difference_ci(x, y):
    p1 = y[x == 1].mean()
    p0 = y[x == 0].mean()
    diff = p1 - p0
    se = math.sqrt((p1 * (1 - p1)) / (x == 1).sum() + (p0 * (1 - p0)) / (x == 0).sum())
    return diff, diff - 1.96 * se, diff + 1.96 * se


def min_detectable_difference(p, n1, n0, power=0.8, alpha=0.05):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    standard_error = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return float((z_alpha + z_beta) * standard_error)


def global_lr_test(df, features, target_col="LUNG_CANCER"):
    from statsmodels.discrete.discrete_model import Logit
    X = df[features].copy()
    X.insert(0, "intercept", 1.0)
    y = df[target_col].astype(int)
    model = Logit(y, X).fit(disp=False)
    return model.llr, model.df_model, model.llr_pvalue, model.prsquared


def fisher_exact_sig(table):
    return stats.fisher_exact(table)[1]
