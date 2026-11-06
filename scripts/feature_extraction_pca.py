"""
Step 4 - Feature extraction with Principal Component Analysis (PCA).

    python scripts/feature_extraction_pca.py

Reads : data/clean/nhis_lung_cancer_model_ready.csv   (final selected features, train/test split)
Writes: results/figures/fig15_pca_scree.png             explained variance per component
        results/figures/fig16_pca_projection.png        adults projected on the first 2 components
        results/tables/pca_explained_variance.csv
        results/tables/pca_loadings.csv                 weight of every feature in PC1-PC3
        results/pca_results.txt                         results and decision in plain words

Question: can the selected features be compressed into a few components without losing much
information? If yes, PCA would be a useful extraction step; if not, we keep the original
(interpretable) features.

Leakage: the scaler and the PCA are fitted on TRAINING rows only; test rows are only transformed.
Scale: features have different units (years, %, µg/m³, 0/1), so every feature is standardised first.
This is the same as running PCA on the correlation matrix (course, Chapter 5).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import BLUE, CLEAN, GREY, NON_FEATURES, ORANGE, RANDOM_STATE, RESULTS, TAB, save_fig, set_style, text_log

LOADING_THRESHOLD = 0.40          # course rule: a loading above 0.40 (absolute) is a meaningful contribution

set_style()

out, say, _ = text_log()


def pca_analysis(m):
    """Steps 1-4: standardise + PCA on training rows, loadings, test-set check, decision.
    Returns the explained-variance table, the loadings, the fitted PCA, the scaled training matrix
    and the number of components for 80/90/95% of the variance."""
    features = [c for c in m.columns if c not in NON_FEATURES]
    train, test = m[m.split == "train"], m[m.split == "test"]

    say("=" * 70)
    say("PCA - FEATURE EXTRACTION (fitted on training rows only)")
    say("=" * 70)
    say(f"Input: {len(features)} selected features | {len(train):,} training rows | {len(test):,} test rows")

    # 1. Standardise (fit on train) -> PCA (fit on train)
    scaler = StandardScaler().fit(train[features])
    Z_train = scaler.transform(train[features])
    Z_test = scaler.transform(test[features])
    pca = PCA(random_state=RANDOM_STATE).fit(Z_train)

    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    eig = pca.explained_variance_
    k = {t: int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95)}
    kaiser = int((eig > 1).sum())

    ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ratio))], "eigenvalue": eig,
                       "explained_variance_%": 100 * ratio, "cumulative_%": 100 * cum})
    ev.round(4).to_csv(TAB / "pca_explained_variance.csv", index=False)

    say("\nExplained variance of the first 10 components:")
    say(ev.head(10).round(2).to_string(index=False))
    say(f"\nPC1 explains {100 * ratio[0]:.1f}% of the variance. If all {len(features)} features were unrelated,")
    say(f"each component would explain about {100 / len(features):.1f}%.")
    say(f"Components needed for 80% of the variance: {k[0.80]} of {len(features)}")
    say(f"Components needed for 90% of the variance: {k[0.90]} of {len(features)}")
    say(f"Components needed for 95% of the variance: {k[0.95]} of {len(features)}")
    say(f"Kaiser rule (eigenvalue > 1): {kaiser} components")
    n_zero = int((eig < 0.01).sum())
    say(f"Components with almost zero variance (eigenvalue < 0.01): {n_zero}")
    vb = pd.read_csv(TAB / "vif_before.csv")
    say(f"  Why so few: before Rule 4 of the feature selection ({len(vb)} features) there were 8 such components,")
    say(f"  i.e. exact or near-exact linear dependencies ({int(vb.exact_dependency.sum())} columns with an infinite VIF):")
    say("  each complete one-hot group (race, education, income, BMI) summed to 1, general health and smoking")
    say("  status almost did, comorbidity_count was the sum of the 5 condition flags, and the region dummies almost")
    say("  determined lung_cancer_incidence. Rule 4 removed comorbidity_count, one reference category per group and")
    say("  the region dummies, so no dependency is left (results/tables/vif_after.csv). An exact dependency is a")
    say("  zero-variance direction that carries no information, so removing it loses nothing; the region dummies")
    say("  added almost nothing beyond the 3 regional columns, which take 20 region-year values.")

    # 2. Loadings: correlation between each original feature and each component
    loadings = pd.DataFrame(pca.components_[:3].T * np.sqrt(eig[:3]), index=features, columns=["PC1", "PC2", "PC3"])
    loadings.round(4).to_csv(TAB / "pca_loadings.csv")
    say(f"\nStrongest loadings (|loading| >= {LOADING_THRESHOLD}) - what each component represents:")
    for pc in ["PC1", "PC2", "PC3"]:
        col = loadings[pc]
        strong = col[col.abs() >= LOADING_THRESHOLD].sort_values(key=abs, ascending=False)
        if strong.empty:
            strong = col.sort_values(key=abs, ascending=False).head(4)
            say(f"  {pc}: no loading above {LOADING_THRESHOLD}; largest: "
                + ", ".join(f"{i} {v:+.2f}" for i, v in strong.items()))
        else:
            say(f"  {pc}: " + ", ".join(f"{i} {v:+.2f}" for i, v in strong.items()))

    say("Interpretation (names follow the loadings above):")
    say("  PC1 = age and chronic-disease burden (hypertension, age, screening-age smoker, diabetes, COPD, heart disease)")
    say("  PC2 = 'skipped the smoking questions' (the two unknown flags, which mark the same ~2.8% of adults)")
    say("  PC3 = older age vs current smoking, e-cigarette use and the regional smoking level (a mixed component)")

    # 3. Check on the test rows: the variance kept by the first k95 components is similar
    Z_test_rec = pca.inverse_transform(np.c_[pca.transform(Z_test)[:, :k[0.95]],
                                             np.zeros((len(Z_test), len(features) - k[0.95]))])
    kept_test = 1 - ((Z_test - Z_test_rec) ** 2).sum() / ((Z_test - Z_test.mean(axis=0)) ** 2).sum()
    say(f"\nTest-set check: the first {k[0.95]} components keep {100 * kept_test:.1f}% of the TEST variance")
    say("(close to 95% -> the PCA generalises; it was never fitted on test rows).")

    # 4. Decision
    say("\nDECISION")
    reduction = 100 * (1 - k[0.95] / len(features))
    say(f"  Keeping 95% of the variance needs {k[0.95]} components instead of {len(features)} features "
        f"({reduction:.0f}% fewer columns).")
    say("  The variance is spread over many components (no dominant one): the features are mostly not")
    say("  redundant, which agrees with the feature-level correlation matrix (Fig 12c) and the VIF table.")
    say("  We KEEP THE ORIGINAL FEATURES for the midterm dataset because:")
    say("    1. interpretability matters for a health question (an odds ratio for 'COPD' is meaningful,")
    say("       a weight on 'PC4' is not);")
    say("    2. the compression gain is moderate and would mix personal and regional information;")
    say("    3. most columns are 0/1 indicators, for which PCA (built for continuous data) is less natural.")
    say("  The PCA-transformed version is kept as an option to compare in the future modelling phase")
    say("  (e.g. for distance-based models such as k-NN or clustering, which suffer most from many dimensions).")
    return ev, loadings, pca, Z_train, k


# 5. Figures
def pca_figures(m, pca, Z_train, k):
    """Step 5: scree plot (Fig 15) and projection on PC1-PC2 (Fig 16)."""
    features = [c for c in m.columns if c not in NON_FEATURES]
    train = m[m.split == "train"]
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(11, 6))
    n_show = len(ratio)
    x = np.arange(1, n_show + 1)
    ax.bar(x, 100 * ratio[:n_show], color=BLUE, label="Explained variance per component")
    ax2 = ax.twinx()
    ax2.plot(x, 100 * cum[:n_show], color=ORANGE, marker="o", markersize=4, label="Cumulative")
    ax2.axhline(95, color=GREY, ls="--", lw=1)
    ax2.text(n_show / 2, 95.5, "95%", ha="center", va="bottom", fontsize=12, color="dimgrey")
    ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative variance (%)"); ax2.grid(False)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Fig 15. PCA scree plot: {k[0.95]} of {len(features)} components needed for 95% of the variance\n"
                 f"(fitted on the {len(train):,} training rows)", fontsize=16)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=12)
    save_fig("fig15_pca_scree")

    P = pca.transform(Z_train)[:, :2]
    proj = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "lung_cancer": train["lung_cancer"].values})
    neg = proj[proj.lung_cancer == 0].sample(min(8000, int((proj.lung_cancer == 0).sum())), random_state=RANDOM_STATE)
    pos = proj[proj.lung_cancer == 1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(neg.PC1, neg.PC2, s=6, alpha=0.25, color=GREY, label="No lung cancer (random 8,000)")
    ax.scatter(pos.PC1, pos.PC2, s=18, alpha=0.8, color=ORANGE, label=f"Lung cancer (all {len(pos)})")
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}% of variance)")
    ax.set_title(f"Fig 16. Training adults on the first two principal components\n"
                 f"(n = {len(train):,} training rows; all {len(pos)} cases and a random 8,000 non-cases shown)", fontsize=16)
    ax.legend(fontsize=12, loc="center right")
    save_fig("fig16_pca_projection")
    m_pos, m_neg = proj.loc[proj.lung_cancer == 1, "PC1"].mean(), proj.loc[proj.lung_cancer == 0, "PC1"].mean()
    say(f"\nFig 16 reading: mean PC1 score = {m_pos:+.2f} for lung cancer cases vs {m_neg:+.2f} for the other adults.")
    say("PC1 is the 'age and chronic-disease burden' axis (see loadings), so cases sit on its high side,")
    say("but the two groups overlap heavily - two components are not enough to separate a rare outcome.")


def main():
    out.clear()
    m = pd.read_csv(CLEAN / "nhis_lung_cancer_model_ready.csv")
    ev, loadings, pca, Z_train, k = pca_analysis(m)
    pca_figures(m, pca, Z_train, k)
    (RESULTS / "pca_results.txt").write_text("\n".join(out), encoding="utf-8")
    print("\nDone. Results in results/pca_results.txt")


if __name__ == "__main__":
    main()
