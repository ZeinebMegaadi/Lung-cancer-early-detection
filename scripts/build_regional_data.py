"""
Step 1 - Build one small table with 3 regional measures per US region and year.

    python scripts/build_regional_data.py

Output: data/clean/regional_context.csv   (4 regions x 5 years = 20 rows)

  region | year | pm25_mean | smoking_rate | lung_cancer_incidence

Each source is reported by STATE. NHIS only tells us the person's REGION, so we
group states into the 4 US Census regions (the same regions NHIS uses).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)
YEARS = [2019, 2020, 2021, 2022, 2023]

report = []
def log(msg=""):
    print(msg)
    report.append(str(msg))

# US Census regions (the same 4 regions used by NHIS)
REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN",
              "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}
STATE_NAMES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District Of Columbia": "DC",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def add_region(df, state_col, by_name):
    """Map each state to its region; drop territories (Puerto Rico, Guam...) and national rows."""
    abbr = df[state_col].map(STATE_NAMES) if by_name else df[state_col].str.upper().str.strip()
    df = df.assign(state=abbr, region=abbr.map(STATE_TO_REGION))
    dropped = df["region"].isna().sum()
    return df[df["region"].notna()].copy(), dropped


# ----------------------------------------------------------------- A. EPA PM2.5
log("A. EPA air pollution (fine particles, PM2.5)")
frames = []
for y in YEARS:
    with zipfile.ZipFile(RAW / "epa" / f"annual_conc_by_monitor_{y}.zip") as z:
        with z.open([n for n in z.namelist() if n.lower().endswith(".csv")][0]) as f:
            d = pd.read_csv(f, low_memory=False)
    n0 = len(d)
    d = d[d["Parameter Name"] == "PM2.5 - Local Conditions"]
    # One annual value per monitor: the official annual standard, excluding wildfire-type events
    if "Pollutant Standard" in d.columns:
        d = d[d["Pollutant Standard"].astype(str).str.contains("Annual", case=False)]
    if "Event Type" in d.columns:
        d = d[d["Event Type"].isin(["No Events", "Events Excluded"])]
    d = d[d["Arithmetic Mean"].between(0, 100)]                      # remove impossible values
    d = d.drop_duplicates(subset=[c for c in ["State Name", "State Code", "County Code", "Site Num", "POC"] if c in d.columns])
    log(f"  {y}: {n0:,} monitor rows -> {len(d):,} PM2.5 annual values")
    frames.append(d.assign(year=y))
epa = pd.concat(frames, ignore_index=True)
epa, dropped = add_region(epa, "State Name", by_name=True)
log(f"  Rows outside the 50 states + DC dropped: {dropped}")
# state average first (so states with many monitors do not dominate), then region average
pm25 = (epa.groupby(["region", "state", "year"])["Arithmetic Mean"].mean()
           .groupby(["region", "year"]).mean().rename("pm25_mean").reset_index())

# ----------------------------------------------------------------- B. BRFSS smoking
log("\nB. BRFSS adult current-smoking rate")
b = pd.read_csv(RAW / "brfss" / "brfss_current_smoker_by_state_2019_2023.csv", low_memory=False)
b.columns = b.columns.str.lower()
n0 = len(b)
b = b[b["response"].astype(str).str.strip().str.lower() == "yes"]
b["data_value"] = pd.to_numeric(b["data_value"], errors="coerce")
b["year"] = pd.to_numeric(b["year"], errors="coerce")
b = b[b["data_value"].between(0, 100) & b["year"].isin(YEARS)]
b = b.drop_duplicates(subset=["locationabbr", "year"])
b, dropped = add_region(b, "locationabbr", by_name=False)
log(f"  {n0:,} rows -> {len(b):,} state-year smoking rates (territories/national rows dropped: {dropped})")
missing_years = sorted(set(YEARS) - set(b["year"].astype(int)))
if missing_years:
    log(f"  Note: no BRFSS smoking data for {missing_years} (filled from the nearest year below)")
smoke = b.groupby(["region", "year"])["data_value"].mean().rename("smoking_rate").reset_index()

# ----------------------------------------------------------------- C. US Cancer Statistics
log("\nC. US Cancer Statistics - new lung cancer cases per 100,000 people")
u = pd.read_csv(RAW / "uscs" / "uscs_lung_incidence_by_area.csv", dtype=str)
u.columns = u.columns.str.upper()
n0 = len(u)
u = u[(u["SEX"].str.lower() == "male and female") & (u["RACE"].str.lower() == "all races")]
u["year"] = pd.to_numeric(u["YEAR"], errors="coerce")               # drops '2017-2021' style rows
for c in ["COUNT", "POPULATION"]:
    u[c] = pd.to_numeric(u[c], errors="coerce")                      # '~' or '.' = suppressed -> missing
u = u[u["year"].isin(YEARS) & u["COUNT"].notna() & u["POPULATION"].gt(0)]
u, dropped = add_region(u, "AREA", by_name=True)
u = u.drop_duplicates(subset=["state", "year"])
log(f"  {n0:,} rows -> {len(u):,} state-year rows (other areas dropped: {dropped})")
# region rate = all cases / all people x 100,000 (crude rate, built from counts so it is exact)
inc = (u.groupby(["region", "year"])[["COUNT", "POPULATION"]].sum()
         .assign(lung_cancer_incidence=lambda t: 100000 * t["COUNT"] / t["POPULATION"])
         [["lung_cancer_incidence"]].reset_index())

# ----------------------------------------------------------------- D. combine
log("\nD. Combine the 3 regional tables (one row per region and year)")
grid = pd.DataFrame([(r, y) for r in REGIONS for y in YEARS], columns=["region", "year"])
ctx = grid.merge(pm25, on=["region", "year"], how="left", validate="one_to_one") \
          .merge(smoke, on=["region", "year"], how="left", validate="one_to_one") \
          .merge(inc, on=["region", "year"], how="left", validate="one_to_one")
assert len(ctx) == 20

# A missing year (e.g. a survey year not collected) is filled with the same region's nearest year
for c in ["pm25_mean", "smoking_rate", "lung_cancer_incidence"]:
    n_miss = ctx[c].isna().sum()
    if n_miss:
        ctx[c] = ctx.groupby("region")[c].transform(lambda s: s.interpolate(limit_direction="both"))
        log(f"  {c}: {n_miss} missing region-year value(s) filled from neighbouring years")
ctx = ctx.round({"pm25_mean": 2, "smoking_rate": 2, "lung_cancer_incidence": 1})
log(ctx.to_string(index=False))
assert ctx.isna().sum().sum() == 0, "A regional value is still missing - check the raw files"

ctx.to_csv(CLEAN / "regional_context.csv", index=False)
(CLEAN / "regional_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nSaved data/clean/regional_context.csv")
