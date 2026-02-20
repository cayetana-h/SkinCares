FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY setup.py /app/
COPY miguellib /app/miguellib
COPY features /app/features
COPY artifacts /app/artifacts
COPY scripts /app/scripts
COPY docs /app/docs

RUN python -m pip install --upgrade pip \
    && pip install -e .

CMD ["python", "scripts/run_evaluation.py"]
