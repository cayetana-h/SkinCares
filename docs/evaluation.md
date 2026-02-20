# Evaluation Harness

A minimal evaluation harness that runs one or more scenarios and produces an
`artifacts/evaluation_report.json` file with simple metrics.

## Metrics
- **constraint_compliance**: fraction of recommendations that satisfy budget/category constraints
- **category_diversity**: unique categories / top_n
- **avg_similarity**: mean similarity score (if available)

## Run locally
```bash
python scripts/run_evaluation.py
```

## Output
The report is written to:
- `artifacts/evaluation_report.json`
