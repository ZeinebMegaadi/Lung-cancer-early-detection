# Lung-Cancer Risk Project: Step-by-Step Build Guide (Step 0 to Submission)

A solo build guide for the midterm on Wednesday 28 October 2026. Every number below was reproduced by running the code on your actual CSV. The finished, tested project is in the attached file lung-cancer-ds.zip: use it to check your own work, not to skip the learning.

## How to use this guide

For every step: (1) read the idea, (2) predict what you will see, (3) type the code into a Jupyter notebook cell, (4) compare with the expected output, (5) explain it out loud in one sentence. Only then move on. Each step names the file in the zip that holds the clean version.

The story of the project in one paragraph: the raw file shows a smoking odds ratio of 1.49, but that comes from a duplicated, all-negative block of rows at the top of the file plus exact duplicates. After cleaning, the odds ratio is 1.00. You diagnose, clean, test, engineer features and explain honestly. You do not train a model (the rubric excludes it).

## Step 0: Prerequisites and rules

- Python 3.11 or newer, a code editor (VS Code) and Jupyter. Git is optional but useful.
- Rule 1: the raw CSV is never edited. All changes happen in code.
- Rule 2: every number in your report and slides must come from code output (docs/results.json). Never type a statistic from memory.
- Rule 3: use the words recorded label and risk indication, never diagnosis, detection or accurate model. This is a research prototype with no clinical claim.
- Time budget: Sun = steps 1-9, Mon = steps 10-15, Tue = steps 16-19, Wed = present.

## Step 1: Create the project and environment

Idea: a virtual environment keeps your packages separate so the project runs on any machine.

```
mkdir lung-cancer-ds && cd lung-cancer-ds
mkdir -p data/raw data/interim data/processed src tests notebooks reports/figures reports/tables docs
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv/Scripts/Activate.ps1
pip install pandas numpy scipy matplotlib seaborn scikit-learn pytest jupyter
pip freeze > requirements.txt
```

Unzip your Lung-cancer-early-detection-main.zip and copy lung\_cancer\_combined\_22457.csv into data/raw/. Make it read-only (chmod 444 on Mac/Linux). Start Jupyter with `jupyter notebook` and create notebooks/01\_explore.ipynb.

## Step 2: Verify the file (checksum)

Idea: a checksum is a fingerprint. If it matches, you and the grader have the same data.

```python
import hashlib, pandas as pd, numpy as np
path = 'data/raw/lung_cancer_combined_22457.csv'
print(hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16])   # expect 4ad2333067fcb0a9
df = pd.read_csv(path)
print(df.shape)                                                    # expect (22457, 17)
```

Save the fingerprint in data/raw/CHECKSUMS.txt. Clean version: src/data\_io.py (load\_raw refuses to run if the fingerprint changed).

## Step 3: Look at the data (rubric: data loading and organization, 5 pts)

Idea: before analysing anything, know what every column is. Run each line and write down what you see.

```python
print(df.dtypes)                      # all int64
print(df.isna().sum().sum())          # 0 missing values
print(df['patient_id'].is_unique)     # True
for c in df.columns[3:-1]:            # the 13 yes/no flags
    print(c, sorted(df[c].unique()))  # [1, 2]  (1 = NO, 2 = YES)
print(df['GENDER'].unique(), df['AGE'].describe())   # GENDER is 0/1; AGE 21 to 87, mean 62.08
```

Write docs/data\_dictionary.md with one row per column: type, coding, range, share of YES, and unknown wherever you do not know. Two honest unknowns: what GENDER = 1 means (call it GENDER\_CODE) and what LUNG\_CANCER = 1 means (call it the recorded label). The README in the zip contains only a title, so source and licence are also unknown: say so in a data card.

Add schema asserts so a silent file change is caught: assert shape, no nulls, flags in {1, 2}, ids unique, AGE between 18 and 110. Clean version: validate\_schema in src/data\_io.py.

## Step 4: The target and why accuracy is meaningless

```python
print(df['LUNG_CANCER'].value_counts())            # 1: 18,434   0: 4,023
print(df['LUNG_CANCER'].mean())                    # 0.8209
```

Idea (class imbalance): a model that always answers positive scores 82.1% accuracy without learning anything. So accuracy can never be your success metric here. Real screening populations are mostly cancer-free, so this prevalence is also unrepresentative: put that in your limitations.

## Step 5: Audit 1: the file is ordered (rubric: cleaning, 12 pts)

Idea: row order can leak the answer. If the first rows are all one class, then position predicts the label.

```python
fp = int(np.argmax(df['LUNG_CANCER'].values == 1))
print(fp)                                           # 1455: first positive label
print(df.iloc[:fp]['LUNG_CANCER'].eq(0).all())      # True: rows 0-1454 are ALL negative
import matplotlib.pyplot as plt
plt.plot(df['LUNG_CANCER'].rolling(300).mean()); plt.axvline(fp, color='r'); plt.show()
```

The rolling mean sits at 0, then jumps to about 1 at row 1455 and settles near 0.87. Also compare smokers in the block versus after it:

```python
smk = lambda d: (d['SMOKING'] == 2).mean()
print(smk(df.iloc[:fp]), smk(df.iloc[fp:]))        # about 0.29 vs 0.56
```

The leading block has a different distribution from the rest: it behaves like a different source. Clean version: leading\_block\_end in src/audit.py, figure fig1\_order.png.

## Step 6: Audit 2: duplicates and label conflicts

Idea: exact copies of a row count the same person twice (pseudo-replication) and can land in both training and test sets (duplicate leakage).

```python
X = [c for c in df.columns if c not in ('patient_id', 'LUNG_CANCER')]   # 15 features
T = 'LUNG_CANCER'
print(df.duplicated(subset=X + [T]).sum())          # 2573 exact duplicates
print(df.duplicated(subset=X).sum())                # 2706 (same features, label ignored)
print(len(df.iloc[:fp][X].drop_duplicates()))       # 103: the 1455-row block has only 103 patterns
g = df.groupby(X)[T].agg(['nunique', 'size'])
print((g['nunique'] > 1).sum(), g.loc[g['nunique'] > 1, 'size'].sum())   # 133 patterns, 315 rows with BOTH labels
print(g['size'].max())                              # 86: one pattern repeated 86 times
```

Is 2,573 more than chance? With 14 binary answers plus age there are about a million possible patterns, so accidental copies should be rare. Simulate it: shuffle each column independently and count duplicates (about 594 expected in rows from 1455 onward, versus 1,352 observed):

```python
rng = np.random.default_rng(42)
rest = df.iloc[fp:]
sim = np.mean([pd.DataFrame({c: rng.choice(rest[c].values, len(rest)) for c in X}).duplicated().sum() for _ in range(10)])
print(round(sim), rest.duplicated(subset=X).sum())   # about 594 vs 1352
```

Conflicting labels (same inputs, different label) mean the label is not a clean function of these features. Clean version: duplicate\_report in src/audit.py.

## Step 7: Audit 3: source shift and the identifier leak

Idea (Cramer's V): an effect size from 0 (no association) to 1 for a chi-square table. 0.1 is small, 0.3 medium, 0.5 large.

```python
from scipy import stats
lead, later_neg = df.iloc[:fp], df.iloc[fp:][df.iloc[fp:][T] == 0]
vs = []
for c in ['GENDER'] + X[2:]:                        # the 14 binary features
    grp = np.r_[np.zeros(len(lead)), np.ones(len(later_neg))]
    val = np.r_[lead[c], later_neg[c]]
    chi2 = stats.chi2_contingency(pd.crosstab(grp, val), correction=False)[0]
    vs.append(np.sqrt(chi2 / len(grp)))
print(round(min(vs), 2), round(max(vs), 2))         # 0.18 to 0.54
```

These are large effects: the leading block and the later negatives are not interchangeable. Now the identifier. patient\_id is just 1 to 22,457 in order, which should carry no information, yet:

```python
pos, neg = df.loc[df[T] == 1, 'patient_id'], df.loc[df[T] == 0, 'patient_id']
print(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))   # 0.67
```

AUC 0.5 means no information; 0.67 means the id alone predicts the label, only because of the ordering. So patient\_id must never be a feature (decision D-04). Clean version: source\_shift and auc\_of\_id in src/audit.py.

## Step 8: Audit 4: do the later rows behave like independent coin flips?

```python
bulk = df.iloc[fp:].drop_duplicates(subset=X + [T])
corr = bulk[X].corr().values
print(np.abs(corr[~np.eye(len(X), dtype=bool)]).max())   # 0.0215: essentially no correlation
print(stats.skew(bulk['AGE']), stats.kurtosis(bulk['AGE']))   # about -0.006 and -0.06: almost perfectly bell-shaped
```

In real self-reported data, smoking, age, chronic disease and symptoms are correlated. Here they are not: 4.4% of 91 pairwise chi-square tests are significant at 5%, which is exactly the chance rate, and a PCA first component explains 7.0% against 6.7% for a flat spectrum. How to say it on stage: the structure is consistent with programmatic generation or resampling. Do not say the data is fake: you have shown statistical behaviour, not provenance.

## Step 9: Cleaning policy, four views (rubric: cleaning, 12 pts)

Idea: cleaning is a set of choices. Make the choices explicit, log them, and test whether your conclusion depends on them (sensitivity analysis).

```python
is_dup = df.duplicated(subset=X + [T], keep='first')
B = df[~is_dup]                                           # exact duplicates removed (PRIMARY view)
n_labels = df.groupby(X)[T].transform('nunique')
C = B[n_labels[B.index] == 1]                             # B without label-conflict rows
D = B[B.index >= fp]                                      # B without the leading block
A = df                                                    # raw, shown only to quantify the artefact
print([len(v) for v in (A, B, C, D)])                     # [22457, 19884, 19618, 19781]
print([round(v[T].mean(), 4) for v in (A, B, C, D)])      # [0.8209, 0.8661, 0.8711, 0.8706]
```

Treatment table for your report:

| Issue | Diagnostic | Decision | Result |
| --- | --- | --- | --- |
| Missing values | isna().sum() | none needed | 0 nulls |
| Exact duplicates | duplicated(features + label) | remove, keep first (view B) | 22,457 to 19,884 |
| Label conflicts | groupby nunique > 1 | keep in B, remove in view C | 133 patterns, 315 rows |
| Order and source block | first positive index | keep in B, remove in view D | block of 1,455 rows |
| AGE outliers | IQR fences 40.5 to 84.5 | keep and flag (ages are plausible) | 221 flagged |
| Identifier leakage | AUC of patient\_id | drop from all features | AUC 0.67 |

Leakage rules to state in the report: deduplicate and group before any split; shuffle with seed 42 and never use contiguous folds on this file; fit every learned transformation on training data only; no identifier or row position as a feature; log every before/after count. Clean version: make\_views and cleaning\_log in src/clean.py, output reports/tables/cleaning\_log.csv.

## Step 10: Hypothesis H1, the primary test (rubric: EDA and statistics, 13 pts)

Idea: state the hypothesis before running it. H1 (null): the share of positive labels is the same for smokers and non-smokers. Smoking is the dominant known risk factor, so it is the natural primary hypothesis.

Key concepts: the p-value is the probability of data at least this extreme if the null were true (it is not the effect size and not the chance the null is true). The odds ratio is ad/bc for the 2x2 table; the confidence interval tells you how precisely you know it; an OR of 1 means no association.

```python
def h1(d):
    a, b = d[d.SMOKING == 2][T], d[d.SMOKING == 1][T]
    tab = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    orr = tab[0, 0] * tab[1, 1] / (tab[0, 1] * tab[1, 0])
    se = np.sqrt((1 / tab).sum())
    lo, hi = np.exp(np.log(orr) + np.array([-1.96, 1.96]) * se)
    return round(orr, 2), round(lo, 2), round(hi, 2), p, np.sqrt(chi2 / len(d))
for name, d in zip('ABCD', (A, B, C, D)):
    print(name, h1(d))
```

Expected: A raw OR 1.49 \[1.39, 1.59\], p = 4.5e-30; B 1.03 \[0.95, 1.12\], p = 0.52; C 1.04 \[0.96, 1.14\], p = 0.30; D 1.00 \[0.92, 1.08\], p = 0.91. The effect exists only in the raw file.

Permutation check (distribution-free): shuffle the labels many times and see how often a gap as large as the observed one appears by chance. Use the permutation\_p function in src/stats\_utils.py: expect about 0.51 on view B and about 0.91 on D, and about 0 on A.

Power: with 11,080 smokers and 8,701 non-smokers in D, the smallest risk difference detectable with 80% power at alpha 0.05 is about 0.0135 (1.3 percentage points). So the null is informative: any smoking effect larger than about 1.3 points would very likely have been seen in this file. This is a statement about the file, not about real smokers.

Careful reading: the absence of a smoking effect is itself evidence that the label is not a faithful record of lung-cancer status (construct-validity problem), because in real populations smoking is a strong cause of lung cancer. Say that about the file, never about smoking.

## Step 11: The family of 14 tests and Benjamini-Hochberg correction

Idea (multiple comparisons): run 14 tests at 5% and the chance of at least one false positive is 1 - 0.95^14 = 51%. Benjamini-Hochberg (BH) controls the false discovery rate: sort the p-values, multiply the i-th by m/i, then take a running minimum from the largest down.

```python
def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    s = p[o] * m / np.arange(1, m + 1)
    q = np.minimum(np.minimum.accumulate(s[::-1])[::-1], 1)
    out = np.empty(m); out[o] = q; return out
print(bh([0.01, 0.04, 0.03, 0.005]))     # [0.02 0.04 0.04 0.02]  (check by hand!)

bin14 = ['GENDER'] + X[2:]
rows = []
for c in bin14:
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(D[c], D[T]), correction=False)
    rows.append((c, np.sqrt(chi2 / len(D)), p))
fam = pd.DataFrame(rows, columns=['feature', 'V', 'p']); fam['q'] = bh(fam.p)
print(fam.sort_values('p').round(4))
```

Expected on view D: smallest p = 0.033 (GENDER code), V = 0.015, q = 0.46; largest V is 0.015, nothing survives. In the raw view A all 14 features have q below 0.05 and V up to 0.144. Clean version: benjamini\_hochberg and feature\_family in src/stats\_utils.py (written without statsmodels so it runs anywhere).

## Step 12: The other hypotheses, and an honest nuance

- H2 (AGE): Mann-Whitney U on positives vs negatives, effect size CLES = U / (n1 n0) (0.5 = no effect). Expected: A p = 6.7e-9, CLES 0.529; B p = 0.005, CLES 0.517; C p = 0.002, CLES 0.519; D p = 0.051, CLES 0.512.
- H3 (respiratory symptom count 0-4 vs label): Spearman correlation. Expected: A rho = 0.19; B 0.015 (p = 0.036); C 0.021 (p = 0.004); D -0.002 (p = 0.77).
- H4: the 14-feature family above, run on all four views. Smallest BH q: A about 3e-102; B 0.36; C 0.079; D 0.46.
- H5 (integrity): leading block vs later negatives, 14 chi-square tests, V from 0.18 to 0.54, all p below 1e-25.

Honest nuance you must mention: in views B and C a few tests are nominally significant (AGE, the respiratory count), but the effects are tiny (CLES about 0.52, rho about 0.02) and they vanish in view D, which removes the leftover young all-negative block that is still present in B and C. So say: the associations are explained by the block, not that nothing was ever nominally significant. Your clean sentence is: after removing the block, no association survives correction and all effect sizes are negligible. Run the whole thing with `python src/run_analysis.py` to get reports/tables/hypotheses.csv.

## Step 13: The seven figures

Each figure answers one written question. Use the clean versions in src/figures.py (200 dpi PNGs in reports/figures):

1. fig1\_order: Is the file ordered? (rolling positive rate and smoker share)
2. fig5\_views: How much data survives each cleaning policy?
3. fig2\_source\_shift: Is the leading block a different source?
4. fig4\_age: What is the shape of AGE? (histogram with normal overlay)
5. fig3\_correlation: Are features related to each other? (heatmap, colour scale clipped to plus or minus 0.10; the near-blank heatmap is the finding)
6. fig6\_cramers\_v: Does any feature relate to the label, and does it depend on cleaning? (raw vs cleaned)
7. fig7\_smoking\_forest: odds ratio per view with 95% CI.

Rule: every figure gets a number, labelled axes with units, and one takeaway sentence in the caption. Describe the axes before stating the message when you present.

## Step 14: Feature engineering (rubric: 7 pts)

Idea: prepare the data the way a later model would need it, with a written justification for each feature, and without leaking.

```python
from config import FLAGS
fe = B.copy()
for c in FLAGS: fe[c] = (fe[c] == 2).astype('int8')                 # 1/2 coding to 0/1 (2 = YES)
fe = fe.rename(columns={'GENDER': 'GENDER_CODE'})                   # never claim this is sex
mu, sd = B['AGE'].mean(), B['AGE'].std(ddof=0)                      # learned parameters: must come from TRAINING data in modelling
fe['AGE_Z'] = (fe['AGE'] - mu) / sd
fe['IN_SCREEN_AGE'] = fe['AGE'].between(50, 80).astype('int8')      # USPSTF 2021 age range
resp = ['COUGHING', 'WHEEZING', 'SHORTNESS_OF_BREATH', 'CHEST_PAIN']
fe['RESP_SYMPTOM_COUNT'] = fe[resp].sum(axis=1)                     # 0-4
fe['SYMPTOM_COUNT_ALL'] = fe[resp + ['FATIGUE', 'SWALLOWING_DIFFICULTY']].sum(axis=1)   # 0-6
fe['SMOKER_IN_SCREEN_AGE'] = fe['SMOKING'] & fe['IN_SCREEN_AGE']    # PROXY for USPSTF eligibility only
fe = fe.drop(columns=['patient_id'])
print(fe['RESP_SYMPTOM_COUNT'].value_counts().sort_index().to_dict())   # {0: 663, 1: 3416, 2: 7110, 3: 6426, 4: 2269}
```

Three feature sets: F0\_raw15 (15 raw features, the baseline), F1\_raw\_plus\_engineered (19 columns), F2\_prior\_driven (5 columns: AGE\_Z, SMOKING, IN\_SCREEN\_AGE, RESP\_SYMPTOM\_COUNT, SMOKER\_IN\_SCREEN\_AGE). Decisions you must be able to defend:

- Scaling (z-score) matters for distance or gradient-based models, not trees; the mean and sd are learned parameters, so in modelling they come from the training fold only (leakage rule L3).
- No log transform: AGE skewness is about -0.006, so there is nothing to fix.
- No PCA: the spectrum is flat (7.0% vs 6.7%), so there is nothing to compress.
- No feature dropping: VIF is 1.00 to 1.03 and no flag is near-constant.
- SMOKER\_IN\_SCREEN\_AGE is a proxy, not eligibility, because the file has no pack-years or quit time.
- Honest conclusion: selection by evidence is inconclusive because no feature carries detectable information, so the final set is chosen by domain priors (F2), with F0 and F1 kept for comparison.

Add a verify function with asserts (no id column, flags in {0, 1}, counts in range, every set's columns exist). Clean version: src/features.py.

## Step 15: Optional experiment: see leakage with your own eyes

This is a sanity check to understand the risk, not a result to present as modelling (the rubric excludes models). It takes two minutes and teaches leakage better than any lecture:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(5, shuffle=True, random_state=42)
auc = lambda d, cols: cross_val_score(LogisticRegression(max_iter=1000), d[cols], d[T], cv=cv, scoring='roc_auc').mean()
print(auc(df, X))      # 0.672  raw file: inflated by duplicates and the block
print(auc(B, X))       # 0.509  de-duplicated: chance level
```

The raw-file score is an artefact. On the cleaned data AUC is about 0.50, which is what a correct pipeline should return when there is no signal. Keep this as a backup slide, labelled internal check.

## Step 16: Turn your notebook into a reproducible project (rubric: 3 pts, plus credibility for all 37 others)

1. Copy the structure from the zip: src/config.py, data\_io.py, audit.py, clean.py, stats\_utils.py, features.py, figures.py, run\_analysis.py, tests/test\_pipeline.py, Makefile, README.md.
2. Put your own typed versions in place of any file you can improve, and compare outputs with the zip. Never submit code you cannot explain line by line.
3. Run the whole pipeline: `make all`. No make installed? Run `python src/audit.py`, `python src/clean.py`, `python src/run_analysis.py` (with the environment variable PYTHONPATH=src) and then `python -m pytest -q`. The test file also runs without pytest: `python tests/test_pipeline.py` (8 tests pass).
4. Check docs/results.json: it holds every number quoted in the report and slides.
5. Clean-clone test on Tuesday: copy the project to a fresh folder, create a new venv, install requirements, run everything, and time it (it runs in well under 5 minutes).
6. Keep notebooks thin: they import from src and show outputs; run Restart and Run All before saving.

## Step 17: The written report (25 pts)

Target 10-14 pages as PDF, six numbered sections matching the rubric (confirm the format with your instructor):

| Section | Pts | Words | Must contain |
| --- | --- | --- | --- |
| Abstract |  | 150 | question, data, method, key numbers, one limitation, research-prototype statement |
| 1 Business understanding | 4 | 400-500 | clinical context with citations, problem statement, five research questions, measurable success criteria, scope |
| 2 Data understanding | 4 | 500-700 | source (with unknowns), dimensions, dictionary table, types, units, first quality issues |
| 3 Preprocessing | 5 | 700-900 | audit findings, treatment table, views A-D, leakage rules, before/after table |
| 4 EDA and statistics | 6 | 900-1,200 | descriptives, figures 1-7 each interpreted, hypothesis table H1-H5 across views, effect sizes with CIs, power, what the statistics do not show |
| 5 Feature engineering | 3 | 400-600 | feature specification table, encoding and scaling decisions, honest selection outcome |
| 6 Conclusions | 3 | 400-600 | answer each research question, specific limitations, planned modelling (not implemented), ethics note |

Suggested research questions: RQ1 what integrity problems does the file contain and how large are they; RQ2 after treatment which features are associated with the label, with what effect size; RQ3 how sensitive are conclusions to the cleaning policy; RQ4 how representative is the sample; RQ5 which engineered features are justified.

Writing rules: each paragraph states a claim, cites a figure or table number with the key number, and says what the evidence does not show. Interpret, never paste raw output. Format counts with thousands separators, percentages with one decimal, ORs with 95% CI.

Do not write: detect cancer, early detection, diagnose, accurate model, predicts cancer, proves, causes, clinically validated, the data is fake. Write instead: recorded label, risk indication in a survey file, association in this file, consistent with, we found no evidence of, research prototype.

Specific limitations to state: provenance and licence unknown; label meaning unverified; GENDER mapping unverified; no smoking dose, duration or quit time, so USPSTF eligibility cannot be reproduced; structure consistent with generation or resampling; findings concern this file only; no external validation possible.

Citations to use (verify each before citing): USPSTF lung cancer screening recommendation (JAMA 2021); Tammemagi et al., PLCOm2012 (NEJM 2013); Kapoor and Narayanan on data leakage (Patterns 2023); Chapman et al., CRISP-DM 1.0 (2000); Benjamini and Hochberg (1995).

## Step 18: The presentation (35 pts) as a solo presenter

20 minutes, aim for 18:30 to 19:30 in rehearsal. Twelve to fourteen slides:

1. Title and one-line takeaway (0:30)
2. Why it matters: USPSTF criteria use age, pack-years and quit time; our file has a yes/no smoking flag (1:00)
3. Problem, research questions, success criteria (1:00)
4. The data: 22,457 x 17, dictionary, checksum (1:30)
5. What we do not know: source, licence, label meaning, GENDER coding (1:00)
6. Audit I: ordered file, figures 1 and 2 (1:30)
7. Audit II: 2,573 duplicates, 133 conflicting patterns, id AUC 0.67 (1:00)
8. Cleaning policy: four views, before/after table, leakage rules (2:00)
9. EDA of the cleaned data: heatmap and AGE (2:00)
10. Hypotheses and method: H1-H5 fixed in advance, effect sizes, BH, power (0:45)
11. Results: OR 1.49 to 1.00, V 0.144 to 0.015, BH q at least 0.46 (2:15)
12. Feature engineering: spec table, three sets, honest selection outcome (2:30)
13. Conclusions and limitations (1:00)
14. Next steps: planned leakage-safe evaluation, labelled PLANNED (1:00)

Design: one message per slide as the headline, at least 20 pt body text, figures fill most of the slide, footer on every slide: Research prototype - not a diagnostic tool. Every number must appear in docs/results.json. Add hidden backup slides: sensitivity table, family-test table, power calculation, the leakage experiment from Step 15.

Delivery: describe a figure's axes before its message; say consistent with, not is; if you do not know, say what you would check and how.

## Step 19: Q and A preparation (8 pts) and rehearsal

As a solo presenter you answer everything, so prepare these in 60-90 seconds each (direct answer, one number, one limitation):

- Why not train a model and report accuracy? (Rubric excludes it; always-positive scores 82%; the raw signal is an artefact.)
- What is your clinical claim? (None; research prototype.)
- Why remove duplicates, they could be real people? (1,352 observed vs about 594 expected by chance; we keep one copy and show B, C, D agree.)
- Why keep conflicting rows in the primary view? (Minimal intervention; view C gives OR 1.04 \[0.96, 1.14\].)
- How do you know the first 1,455 rows are a different source? (All negative, 103 patterns, V 0.18-0.54, ages down to 21; we say behaves like a separate source.)
- Is patient\_id a leak? (Yes, AUC 0.67 from the ordering.)
- What does p = 4.5e-30 mean? (Surprising if no association, but the effect arises from the block and is small.)
- Why BH and not Bonferroni? (Controls false discovery rate and is less conservative; smallest p 0.033 becomes q 0.46.)
- What does CLES = 0.512 mean? (A random positive is older than a random negative 51.2% of the time versus 50% by chance.)
- Does a null result mean smoking is not a risk factor? (No; the file does not contain that relationship, which questions the file's validity.)
- Why z-score AGE, why no PCA, why no log transform? (See Step 14.)
- What would change your conclusion? (Provenance showing the label is clinical and the file genuine; a match to the original survey; correlated structure in the later rows.)
- What models would you build later? (Planned: baselines, regularised logistic regression, gradient-boosted trees, grouped stratified nested CV, PR-AUC, calibration; on this file we expect chance-level performance as a negative control.)

Rehearsal plan: two full timed runs with a stopwatch, two Q and A rounds (record yourself, or ask a friend to ask random questions), and one explain-it-to-a-friend session on leakage and BH.

## Step 20: Freeze and submission checklist

- [ ] make all (or the manual commands) runs from a clean clone; tests pass; time logged
- [ ] results.json regenerated; every number in report and slides found in it
- [ ] Report exported to PDF: six sections, numbered figures and tables with captions, references, reproducibility paragraph (5 commands)
- [ ] Slides exported to PDF and PPTX, 12-14 slides plus backups, footer disclaimer, slide numbers
- [ ] Banned phrases searched in report and slides: zero hits
- [ ] Data access: CHECKSUMS.txt plus source note (or the CSV if the licence allows)
- [ ] Backups: slides on a USB stick and by email, laptop charger, exported notebook HTML
- [ ] Two timed rehearsals at or under 19:30; Q and A practised

Suggested schedule: Sunday steps 1-9 and tests; Monday steps 10-15 and the report skeleton; Tuesday morning finish the report and build slides; Tuesday afternoon rehearsals and the clean-clone test; Tuesday 23:59 freeze (typo fixes only afterwards); Wednesday morning a 20-minute slide run-through, then present.

## Questions to ask your instructor (email early)

1. Where did this dataset come from, and is there a dictionary or licence?
2. What is the deadline and required format for the report and code?
3. Is a data-integrity finding acceptable as the main result?
4. Will the later project require modelling on this file?
5. How long is the Q and A, and is it individual?

This is a research prototype for coursework. Nothing in it is a validated diagnostic or screening tool.
