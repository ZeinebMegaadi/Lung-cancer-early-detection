"""
Step 0 - Download every raw dataset into data/raw/.

    python scripts/download_data.py

Sources (all free, public US government data):
  1. NHIS 2019-2023 adult files (CDC)                 -> data/raw/nhis/
  2. EPA annual air-pollution summaries 2019-2023     -> data/raw/epa/
  3. BRFSS smoking prevalence by state (CDC)          -> data/raw/brfss/
  4. US Cancer Statistics, lung cancer by state (CDC) -> data/raw/uscs/

Big files are reduced to the rows we need, so every file stays under GitHub's 100 MB limit.
"""
import io
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
YEARS = [2019, 2020, 2021, 2022, 2023]
HEADERS = {"User-Agent": "Mozilla/5.0 (student data project)"}


def download(url, dest):
    """Download url to dest (skip if it is already there)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already here: {dest.relative_to(ROOT)}")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  saved {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# 1. NHIS ------------------------------------------------------------------
print("1. NHIS adult files")
for y in YEARS:
    yy = str(y)[-2:]
    download(f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/{y}/adult{yy}csv.zip",
             RAW / "nhis" / f"adult{yy}csv.zip")

# 2. EPA air quality ---------------------------------------------------------
print("\n2. EPA annual concentration by monitor")
for y in YEARS:
    download(f"https://aqs.epa.gov/aqsweb/airdata/annual_conc_by_monitor_{y}.zip",
             RAW / "epa" / f"annual_conc_by_monitor_{y}.zip")

# 3. BRFSS smoking prevalence --------------------------------------------------
print("\n3. BRFSS current-smoker prevalence by state")
brfss_out = RAW / "brfss" / "brfss_current_smoker_by_state_2019_2023.csv"
if not brfss_out.exists():
    brfss_out.parent.mkdir(parents=True, exist_ok=True)
    # Ask the CDC API for only the rows we need (small file)
    where = "topic like '%Smoker Status%' AND break_out='Overall' AND year between '2019' and '2023'"
    url = ("https://data.cdc.gov/resource/dttw-5yxu.csv?$limit=50000&$where="
           + urllib.parse.quote(where))
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=300) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        if df.empty:
            raise ValueError("API returned no rows")
    except Exception as e:  # fallback: full export, then keep only smoking rows
        print(f"  filtered API failed ({e}); downloading the full file instead")
        full = download("https://data.cdc.gov/api/views/dttw-5yxu/rows.csv?accessType=DOWNLOAD",
                        RAW / "brfss" / "_full_brfss_prevalence.csv")
        df = pd.read_csv(full, low_memory=False)
        df.columns = df.columns.str.lower()
        df = df[df["topic"].astype(str).str.contains("Smoker Status", case=False)
                & (df["break_out"].astype(str).str.lower() == "overall")
                & df["year"].between(2019, 2023)]
        full.unlink()
    df.to_csv(brfss_out, index=False)
    print(f"  saved {brfss_out.relative_to(ROOT)} ({len(df):,} rows)")
else:
    print(f"  already here: {brfss_out.relative_to(ROOT)}")

# 4. US Cancer Statistics (lung cancer incidence) -------------------------------
print("\n4. US Cancer Statistics - lung cancer incidence by state")
uscs_out = RAW / "uscs" / "uscs_lung_incidence_by_area.csv"
if not uscs_out.exists():
    z = download("https://www.cdc.gov/cancer/uscs/USCS-1999-2023-ASCII.zip", RAW / "uscs" / "_uscs.zip")
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.upper().endswith("BYAREA.TXT")][0]
        with zf.open(name) as f:
            df = pd.read_csv(f, sep="|", dtype=str, encoding="latin-1")
    df.columns = df.columns.str.upper()
    df = df[df["SITE"].str.contains("Lung", case=False, na=False)
            & (df["EVENT_TYPE"].str.lower() == "incidence")]
    df.to_csv(uscs_out, index=False)
    z.unlink()                                    # the full zip is large; keep only the lung rows
    print(f"  saved {uscs_out.relative_to(ROOT)} ({len(df):,} rows)")
else:
    print(f"  already here: {uscs_out.relative_to(ROOT)}")

print("\nAll raw data downloaded.")
