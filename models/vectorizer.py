import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent

DATA_PRODUCTS = ROOT / "data" / "processed" / "products_clean.csv"
DATA_TOKENS = ROOT / "data" / "processed" / "products_tokens.csv"
GROUPS_PATH = ROOT / "features" / "ingredient_groups.json"

ARTIFACT_DIR = ROOT / "artifacts"


# Data loading

def load_data():
    """Load cleaned product metadata and ingredient tokens, then merge on product_id."""
    products = pd.read_csv(DATA_PRODUCTS, dtype={"product_id": str})
    tokens = pd.read_csv(DATA_TOKENS, dtype={"product_id": str})

    df = products.merge(tokens, on="product_id", how="inner")

    # basic schema check so downstream code doesn't silently break
    required = ["product_id", "category", "price", "ingredient_tokens"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df.reset_index(drop=True)


def load_groups():
    """Load ingredient -> group mapping created during feature engineering."""
    with open(GROUPS_PATH) as f:
        return json.load(f)


# Feature blocks

def build_tfidf(token_series):
    """TF-IDF representation of ingredient text."""
    text = (
        token_series.fillna("")
        .str.lower()
        .str.replace(r"\s*,\s*", " ", regex=True)  # treat commas as token separators
        .str.strip()
    )

    vectorizer = TfidfVectorizer(
        max_features=512,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(text)
    return X, vectorizer


def build_group_features(token_series, group_map):
    # for each product, count how many ingredients belong to each group (humectant, exfoliant, irritant, etc.)

    groups = sorted(set(group_map.values()))
    group_idx = {g: i for i, g in enumerate(groups)}

    rows, cols, data = [], [], []

    for i, row in enumerate(token_series.fillna("")):
        tokens = [t.strip().lower() for t in row.split(",") if t.strip()]
        counts = {}

        for tok in tokens:
            if tok in group_map:
                idx = group_idx[group_map[tok]]
                counts[idx] = counts.get(idx, 0) + 1

        for c, v in counts.items():
            rows.append(i)
            cols.append(c)
            data.append(v)

    X = csr_matrix((data, (rows, cols)), shape=(len(token_series), len(groups)))
    names = [f"group_{g}" for g in groups]

    return X, names


def build_category_features(series):
    # one-hot encode product category (cleanser, moisturizer, serum, etc.)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X = encoder.fit_transform(series.fillna("unknown").to_frame())

    names = [f"cat_{c}" for c in encoder.categories_[0]]
    return X, names


def build_price_feature(series):
    # normalize price into [0,1] so it scales nicely with other features

    values = (
        pd.to_numeric(series, errors="coerce")
        .fillna(series.median())
        .to_numpy()
        .reshape(-1, 1)
    )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    return csr_matrix(scaled)



# Combine and save

def stack_all(X_tfidf, X_groups, X_cat, X_price):
    # horizontally combine all feature blocks into one matrix
    return hstack([X_tfidf, X_groups, X_cat, X_price], format="csr")


def build_schema(tfidf_vec, group_names, cat_names):
    # save where each feature block lives in the final vector.

    tfidf_names = tfidf_vec.get_feature_names_out().tolist()

    group_start = len(tfidf_names)
    cat_start = group_start + len(group_names)
    price_index = cat_start + len(cat_names)

    return {
        "tfidf": tfidf_names,
        "groups": group_names,
        "categories": cat_names,
        "price_index": price_index,
        "total_features": price_index + 1,
    }


def save_outputs(X, df, schema, tfidf_vec):
    # persist vectors and metadata used by other models

    ARTIFACT_DIR.mkdir(exist_ok=True)

    np.save(ARTIFACT_DIR / "product_vectors.npy", X.toarray().astype(np.float32))

    product_index = {pid: i for i, pid in enumerate(df["product_id"])}
    with open(ARTIFACT_DIR / "product_index.json", "w") as f:
        json.dump(product_index, f, indent=2)

    with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    joblib.dump(tfidf_vec, ARTIFACT_DIR / "tfidf.joblib")



# Main pipeline

def run():
    df = load_data()
    groups = load_groups()

    # build each feature block
    X_tfidf, tfidf_vec = build_tfidf(df["ingredient_tokens"])
    X_groups, group_names = build_group_features(df["ingredient_tokens"], groups)
    X_cat, cat_names = build_category_features(df["category"])
    X_price = build_price_feature(df["price"])

    # combine into single vector space
    X = stack_all(X_tfidf, X_groups, X_cat, X_price)

    schema = build_schema(tfidf_vec, group_names, cat_names)

    save_outputs(X, df, schema, tfidf_vec)

    print("Vectorization finished:", X.shape)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("Pipeline failed:", e)
        sys.exit(1)