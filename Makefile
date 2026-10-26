.PHONY: audit clean eda hypotheses features analysis lineage results report slides test consistency repro-log all

all: analysis lineage results report slides test consistency

lineage:
	python -m src.lineage

repro-log:
	python scripts/handbook_repro_check.py

analysis:
	python -m src.run_analysis

audit clean eda hypotheses features: analysis

results: analysis
	python -m src.results

report: results
	python -m src.build_report

slides: results
	python -m src.build_slides

test:
	python -m pytest -q

consistency: report slides
	python -c "from src.results import check_consistency; raise SystemExit(0 if check_consistency() else 1)"
