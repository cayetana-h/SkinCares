# Deployment Skeleton

This folder captures deployment placeholders and run instructions for the project.

## Docker (local)
Build artifacts, then run the evaluation container with Compose:

```bash
python -m pip install -e .
python scripts/build_artifacts.py --schema-version v1
docker compose up --build
```

If you want a one-off run without Compose:

```bash
docker build -t skincares:latest .
docker run --rm skincares:latest
```

## Notes
- This is a lightweight skeleton for midterm readiness.
- Add an API server and logging before production deployment.
