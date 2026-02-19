import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent.parent

VECTORS_PATH = ROOT / "artifacts" / "product_vectors.npy"
INDEX_PATH = ROOT / "artifacts" / "product_index.json"
METADATA_PATH = ROOT / "miguellib" / "datasets" / "datasets" / "products_clean.csv"


# Load everything once

def load_artifacts():
    vectors = np.load(VECTORS_PATH)

    with open(INDEX_PATH) as f:
        product_index = json.load(f)

    metadata = pd.read_csv(METADATA_PATH, dtype={"product_id": str})

    needed = {"product_id", "brand", "category", "price"}
    missing = needed - set(metadata.columns)
    if missing:
        raise ValueError(f"products_clean.csv missing columns: {missing}")

    metadata["price"] = pd.to_numeric(metadata["price"], errors="coerce")

    return vectors, product_index, metadata


VECTORS, PRODUCT_INDEX, METADATA = load_artifacts()

# quick reverse lookup: row index -> product_id
INDEX_TO_ID = {v: k for k, v in PRODUCT_INDEX.items()}


# Main dupe finder logic

def find_dupes(product_id, top_n=5, max_price=None):
    # find the most similar cheaper products in the same category.

    if product_id not in PRODUCT_INDEX:
        raise ValueError(f"Unknown product_id: {product_id}")

    source_idx = PRODUCT_INDEX[product_id]
    source_vec = VECTORS[source_idx].reshape(1, -1)

    source_row = METADATA[METADATA["product_id"] == product_id].iloc[0]
    source_category = source_row["category"]
    source_price = source_row["price"]

    # cosine similarity against all products
    sims = cosine_similarity(source_vec, VECTORS).flatten()

    # build candidate table aligned with vector rows
    candidates = pd.DataFrame({
        "product_id": [INDEX_TO_ID[i] for i in range(len(sims))],
        "similarity": sims,
    })

    # attach metadata
    candidates = candidates.merge(METADATA, on="product_id", how="left")

    # basic filters
    candidates = candidates[candidates["product_id"] != product_id]
    candidates = candidates[candidates["category"] == source_category]
    candidates = candidates[candidates["price"] < source_price]

    if max_price is not None:
        candidates = candidates[candidates["price"] <= max_price]

    # rank by similarity
    candidates = (
        candidates
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return candidates[["product_id", "brand", "category", "price", "similarity"]]


# Simple demo run

if __name__ == "__main__":
    # grab any product from the index as a quick test
    demo_id = next(iter(PRODUCT_INDEX))
    demo_row = METADATA[METADATA["product_id"] == demo_id].iloc[0]

    print("Finding dupes for:")
    print(f"  {demo_row['brand']} | {demo_row['category']} | ${demo_row['price']}")
    print()

    results = find_dupes(demo_id)

    if results.empty:
        print("No cheaper dupes found.")
    else:
        print(results.to_string(index=False, float_format="%.4f"))