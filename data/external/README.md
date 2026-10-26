# External lineage sources (handbook task B-4)

Download the Kaggle files referenced in the team handbook and place them here **without**
committing credentials. Expected filenames (any one name per source is enough):

| Handbook source | Suggested filename |
| --- | --- |
| [mysarahmadbhat/lung-cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) | `kaggle_mysarahmadbhat_lung_cancer.csv` |
| [akashnath29/lung-cancer-dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset) | `kaggle_akashnath29_lung_cancer.csv` |

Then run:

```bash
python -m src.lineage
```

Outputs:

- `docs/lineage_summary.json`
- `docs/lineage_report.md`
- `reports/tables/lineage_match.csv`

If the files are absent, the pipeline records `sources_unavailable` and does **not**
invent match counts.
