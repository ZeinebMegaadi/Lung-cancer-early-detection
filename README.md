# Lung Cancer Risk Project

This coursework repository audits the integrity and statistical behavior of a
single survey-style CSV containing age, coded symptom/lifestyle flags, and a
recorded label. The work emphasizes duplicate and row-order artifacts, cleaning
sensitivity, descriptive statistics, and feature construction. It does not fit
predictive models or make clinical claims.

## Setup and run

Use Python 3.11 or newer. From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_analysis
python -m pytest -q
```

In Git Bash, activate the same Windows environment with
`source .venv/Scripts/activate`. In WSL/Linux, create a separate environment and
activate it with `source .venv/bin/activate`.

The root `Makefile` exposes `audit`, `clean`, `eda`, `hypotheses`, `features`,
`results`, `report`, `slides`, `test`, and `all` targets. If GNU Make is not
installed, run the Python commands above and then
`python -m src.results`, `python -m src.build_report`, and
`python -m src.build_slides`.

## Data provenance and checksum

The supplied repository README did not identify the source, collection method,
or licence. These remain **OPEN**; do not redistribute the raw CSV unless its
licence is confirmed. The dataset must be supplied at
`data/raw/lung_cancer_combined_22457.csv`.

The project contract expects checksum prefix `4ad2333067fcb0a9`; the supplied
file's observed prefix is `1e0d557990cd7533`. This mismatch is recorded in
`data/raw/CHECKSUMS.txt`. The analysis pipeline continues only under the
explicitly approved override and shows a warning. `load_raw()` checks the
contract checksum by default; the explicit `verify=False` override is used only
for the approved current snapshot.

## Repository map

- `src/`: data loading, integrity audit, cleaning, statistical tests, figures,
  feature engineering, and report/slides builders.
- `notebooks/01_explore.ipynb`: executed data-understanding and audit walkthrough
  with saved outputs cleared.
- `data/interim/`: generated A-D cleaning views.
- `data/processed/`: feature tables for the B view.
- `reports/tables/`: audit, cleaning, statistical, and figure backing tables.
- `reports/figures/`: generated PNG figures and captions.
- `docs/`: data dictionary/card, decision log, feature specification, results,
  and learning notes.
- `report/` and `slides/`: generated coursework deliverables.

## Python environment

Use a virtual environment so this project's Python packages stay isolated from other
projects and the system Python. This also makes the pinned dependencies in
`requirements.txt` reproducible on another machine.

In Windows PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

When activated, `python` and `pip` refer to `.venv`. Confirm this with
`python -c "import sys; print(sys.executable)"`; the path should end in
`.venv\Scripts\python.exe`. Activation applies only to the current terminal session.
