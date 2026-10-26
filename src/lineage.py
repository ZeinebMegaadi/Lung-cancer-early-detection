"""Handbook task B-4: row-level lineage checks against optional external source CSVs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DOCS, FEATURE_COLUMNS, RAW_FILE, TAB, TARGET

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"

LINEAGE_SOURCES = (
    {
        "id": "kaggle_mysarahmadbhat",
        "label": "Kaggle Lung Cancer survey (handbook ~309 rows)",
        "handbook_ref": "https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer",
        "candidate_files": (
            "kaggle_mysarahmadbhat_lung_cancer.csv",
            "lung_cancer_survey.csv",
            "survey_lung_cancer.csv",
        ),
    },
    {
        "id": "kaggle_akashnath29",
        "label": "Kaggle Lung Cancer Dataset (handbook ~3,000 rows)",
        "handbook_ref": "https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset",
        "candidate_files": (
            "kaggle_akashnath29_lung_cancer.csv",
            "lung_cancer_dataset.csv",
            "lung_cancer_data.csv",
        ),
    },
)

GENDER_MAPS = {
    "numeric_01": {"M": 0, "F": 1, "m": 0, "f": 1},
    "numeric_10": {"M": 1, "F": 0, "m": 1, "f": 0},
}

COLUMN_ALIASES = {
    "gender": "GENDER",
    "Gender": "GENDER",
    "GENDER": "GENDER",
    "age": "AGE",
    "Age": "AGE",
    "AGE": "AGE",
    "lung_cancer": TARGET,
    "LUNG_CANCER": TARGET,
    "lung cancer": TARGET,
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    for column in out.columns:
        key = str(column).strip()
        if key in COLUMN_ALIASES:
            mapping[column] = COLUMN_ALIASES[key]
        elif key.upper() in {name.upper() for name in FEATURE_COLUMNS + [TARGET]}:
            mapping[column] = key.upper()
    return out.rename(columns=mapping)


def _symptom_to_12(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"Y", "YES", "2"}:
            return 2
        if text in {"N", "NO", "1"}:
            return 1
        if text in {"0"}:
            return 1
        if text in {"1", "2"}:
            return int(text)
        return None
    if value in (0, 1):
        return 1 if value == 0 else 2
    if value in (1, 2):
        return int(value)
    return None


def _normalize_gender(series: pd.Series, mapping_name: str) -> pd.Series | None:
    mapping = GENDER_MAPS[mapping_name]
    if series.dtype == object or series.map(type).eq(str).any():
        mapped = series.map(lambda value: mapping.get(str(value).strip(), None))
        if mapped.isna().any():
            return None
        return mapped.astype(int)
    if set(series.dropna().unique()).issubset({0, 1}):
        return series.astype(int)
    return None


def normalize_source_frame(
    df: pd.DataFrame, gender_mapping: str = "numeric_01"
) -> pd.DataFrame | None:
    """Return 15 feature columns in project coding, or None if normalization fails."""
    renamed = _rename_columns(df)
    missing = [column for column in FEATURE_COLUMNS if column not in renamed.columns]
    if missing:
        return None
    out = renamed[FEATURE_COLUMNS].copy()
    gender = _normalize_gender(out["GENDER"], gender_mapping)
    if gender is None:
        return None
    out["GENDER"] = gender
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce").astype("Int64")
    if out["AGE"].isna().any():
        return None
    for column in FEATURE_COLUMNS:
        if column in {"GENDER", "AGE"}:
            continue
        converted = out[column].map(_symptom_to_12)
        if converted.isna().any():
            return None
        out[column] = converted.astype(int)
    return out.astype({column: int for column in FEATURE_COLUMNS})


def row_fingerprints(df: pd.DataFrame) -> set[tuple[int, ...]]:
    """Multiset keys for exact 15-feature matches."""
    return {tuple(row) for row in df[FEATURE_COLUMNS].itertuples(index=False, name=None)}


def count_row_matches(source: pd.DataFrame, combined: pd.DataFrame) -> dict:
    """Count how many source rows appear in the combined file for each gender map."""
    combined_sets = {
        mapping: row_fingerprints(normalized)
        for mapping, normalized in (
            (name, normalize_source_frame(combined, gender_mapping=name))
            for name in GENDER_MAPS
        )
        if normalized is not None
    }
    best = {
        "gender_mapping": None,
        "source_rows": int(len(source)),
        "matches_in_combined": 0,
        "match_rate": 0.0,
        "unique_source_patterns": 0,
        "unique_matched_patterns": 0,
    }
    for mapping_name in GENDER_MAPS:
        normalized_source = normalize_source_frame(source, gender_mapping=mapping_name)
        if normalized_source is None:
            continue
        target_set = combined_sets.get(mapping_name)
        if target_set is None:
            continue
        source_keys = [tuple(row) for row in normalized_source.itertuples(index=False, name=None)]
        matches = sum(1 for key in source_keys if key in target_set)
        unique_source = len(set(source_keys))
        unique_matched = len({key for key in source_keys if key in target_set})
        rate = matches / len(source_keys) if source_keys else 0.0
        if matches > best["matches_in_combined"]:
            best = {
                "gender_mapping": mapping_name,
                "source_rows": len(source_keys),
                "matches_in_combined": matches,
                "match_rate": rate,
                "unique_source_patterns": unique_source,
                "unique_matched_patterns": unique_matched,
            }
    return best


def resolve_source_path(source: dict) -> Path | None:
    for name in source["candidate_files"]:
        path = EXTERNAL / name
        if path.is_file():
            return path
    return None


def run_lineage_check(combined_path: Path = RAW_FILE) -> dict:
    """Run B-4 checks when external files exist; otherwise record unavailability."""
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(combined_path)
    records = []
    any_available = False
    for source in LINEAGE_SOURCES:
        path = resolve_source_path(source)
        if path is None:
            records.append(
                {
                    "source_id": source["id"],
                    "label": source["label"],
                    "handbook_ref": source["handbook_ref"],
                    "path": None,
                    "available": False,
                    "gender_mapping": None,
                    "source_rows": None,
                    "matches_in_combined": None,
                    "match_rate": None,
                    "unique_source_patterns": None,
                    "unique_matched_patterns": None,
                }
            )
            continue
        any_available = True
        frame = pd.read_csv(path)
        stats = count_row_matches(frame, combined)
        records.append(
            {
                "source_id": source["id"],
                "label": source["label"],
                "handbook_ref": source["handbook_ref"],
                "path": str(path.relative_to(path.parents[2])),
                "available": True,
                **stats,
            }
        )

    if any_available:
        status = "completed"
        interpretation = (
            "External source files were found under data/external/. Match counts use exact "
            "15-feature equality after symptom recoding to the project 1/2 scheme and the "
            "best-performing GENDER map (M/F to 0/1, both orientations tried)."
        )
    else:
        status = "sources_unavailable"
        interpretation = (
            "The handbook B-4 Kaggle source CSVs were not present in data/external/. No match "
            "counts are invented. Place downloaded files using data/external/README.md, then "
            "re-run python -m src.lineage."
        )

    summary = {
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "combined_file": str(combined_path.relative_to(combined_path.parents[2])),
        "feature_match_columns": FEATURE_COLUMNS,
        "gender_mappings_tried": list(GENDER_MAPS),
        "sources": records,
        "interpretation": interpretation,
    }
    return summary


def write_lineage_outputs(summary: dict) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary_path = DOCS / "lineage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for record in summary["sources"]:
        rows.append(
            {
                "source_id": record["source_id"],
                "available": record["available"],
                "source_rows": record["source_rows"],
                "matches_in_combined": record["matches_in_combined"],
                "match_rate": record["match_rate"],
                "gender_mapping": record["gender_mapping"],
                "path": record["path"],
            }
        )
    pd.DataFrame(rows).to_csv(TAB / "lineage_match.csv", index=False)

    lines = [
        "# Lineage report (handbook task B-4)",
        "",
        f"- **Status:** `{summary['status']}`",
        f"- **Checked (UTC):** {summary['checked_at_utc']}",
        f"- **Combined file:** `{summary['combined_file']}`",
        "",
        "## Matching rules",
        "",
        "Rows match when all 15 handbook features "
        f"({', '.join(f'`{c}`' for c in FEATURE_COLUMNS)}) are equal after:",
        "",
        "- Symptom/lifestyle fields recoded to the project `1 = NO`, `2 = YES` scheme.",
        "- `GENDER` mapped from `M`/`F` to `0`/`1` with both orientations attempted.",
        "",
        "## Results",
        "",
        "| Source | Available | Rows | Matches in combined | Match rate | Gender map | File |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for record in summary["sources"]:
        rows_display = "" if record["source_rows"] is None else f"{record['source_rows']:,}"
        matches_display = (
            "" if record["matches_in_combined"] is None else f"{record['matches_in_combined']:,}"
        )
        rate_display = (
            ""
            if record["match_rate"] is None
            else f"{100 * record['match_rate']:.2f}%"
        )
        lines.append(
            "| {label} | {available} | {rows} | {matches} | {rate} | {gender} | {path} |".format(
                label=record["label"],
                available="yes" if record["available"] else "no",
                rows=rows_display,
                matches=matches_display,
                rate=rate_display,
                gender=record["gender_mapping"] or "—",
                path=record["path"] or "—",
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    report_path = DOCS / "lineage_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    summary = run_lineage_check()
    path = write_lineage_outputs(summary)
    print(f"Lineage status: {summary['status']}")
    print(f"Wrote {path}")
    print(f"Wrote {DOCS / 'lineage_report.md'}")
    print(f"Wrote {TAB / 'lineage_match.csv'}")


if __name__ == "__main__":
    main()
