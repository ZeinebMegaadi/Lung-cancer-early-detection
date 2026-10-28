"""Run the whole project with one command:  python run_all.py"""
import subprocess
import sys
from pathlib import Path

STEPS = ["download_data.py", "build_regional_data.py", "build_clean_dataset.py", "eda_analysis.py", "feature_extraction_pca.py",
         "supplementary_analysis.py"]
for step in STEPS:
    print(f"\n========== {step} ==========")
    result = subprocess.run([sys.executable, str(Path(__file__).parent / "scripts" / step)])
    if result.returncode != 0:
        sys.exit(f"Stopped: {step} failed. Read the error above.")
print("\nAll steps finished. See data/clean/ and reports/.")
