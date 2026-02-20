# Artifacts Pipeline

This pipeline builds vector artifacts and writes a manifest with hashes for data, code, and outputs.

## What it produces
- `artifacts/product_vectors.npy`
- `artifacts/product_index.json`
- `artifacts/feature_schema.json`
- `artifacts/tfidf.joblib`
- `artifacts/manifest.json`

## Run locally
```bash
python scripts/build_artifacts.py --schema-version v1
```

## Manifest contents
The manifest stores:
- `data_hashes`: hashes for dataset inputs
- `code_hashes`: hashes for code used to build artifacts
- `artifact_hashes`: hashes for outputs
- `schema_version`: version label you pass on the command line
