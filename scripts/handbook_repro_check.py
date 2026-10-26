"""Record a handbook-style reproducibility run (G-08) with elapsed times."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "docs" / "reproducibility_run_log.md"
HANDOOK_LIMIT_SECONDS = 5 * 60


def run_step(label: str, command: list[str]) -> tuple[bool, float, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, elapsed, output.strip()


def main() -> int:
    python = sys.executable
    steps = [
        ("Lineage check (B-4)", [python, "-m", "src.lineage"]),
        ("Analysis pipeline", [python, "-m", "src.run_analysis"]),
        ("Results manifest", [python, "-m", "src.results"]),
        ("Report build", [python, "-m", "src.build_report"]),
        ("Slides build", [python, "-m", "src.build_slides"]),
        ("Pytest suite", [python, "-m", "pytest", "-q"]),
        (
            "Numeric consistency",
            [
                python,
                "-c",
                "from src.results import check_consistency; "
                "raise SystemExit(0 if check_consistency() else 1)",
            ],
        ),
    ]
    records = []
    total = 0.0
    all_ok = True
    for label, command in steps:
        ok, elapsed, output = run_step(label, command)
        total += elapsed
        all_ok = all_ok and ok
        tail = "\n".join(output.splitlines()[-4:]) if output else "(no output)"
        records.append((label, ok, elapsed, tail))

    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# Reproducibility run log (handbook G-08)",
        "",
        f"- **Recorded (UTC):** {stamp}",
        f"- **Python:** `{sys.executable}`",
        f"- **Working directory:** `{ROOT}`",
        f"- **Total elapsed:** {total:.1f}s (handbook target: ≤ {HANDOOK_LIMIT_SECONDS}s for `make all` equivalent)",
        f"- **Overall:** {'PASS' if all_ok else 'FAIL'}",
        "",
        "| Step | Status | Seconds | Tail output |",
        "| --- | --- | ---: | --- |",
    ]
    for label, ok, elapsed, tail in records:
        safe_tail = tail.replace("|", "\\|").replace("\n", "<br>")
        lines.append(
            f"| {label} | {'PASS' if ok else 'FAIL'} | {elapsed:.1f} | {safe_tail} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This log records the documented command sequence on the current machine.",
            "- A true clean-clone check still requires a fresh directory clone plus a new virtual environment.",
            "- Raw data under `data/raw/` is read-only and was not modified.",
            "",
        ]
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {LOG_PATH}")
    print(f"Total elapsed: {total:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
