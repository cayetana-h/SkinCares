import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.similarity import score_similarity
from models.user_profile import build_user_vector

ROOT = Path(__file__).resolve().parent.parent

VECTORS_PATH = ROOT / "artifacts" / "product_vectors.npy"
INDEX_PATH = ROOT / "artifacts" / "product_index.json"
METADATA_PATH = ROOT / "data" / "processed" / "products_clean.csv"
TOKENS_PATH = ROOT / "data" / "processed" / "products_tokens.csv"

_EMPTY_RECS = pd.DataFrame(
    columns=["product_id", "brand", "category", "price", "similarity"]
)


def load_artifacts():
    """Load product vectors, index, and metadata from disk."""
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


def rank_products(
    user_vector,
    product_vectors,
    metadata_df,
    constraints,
    top_n=10,
    tokens_df=None,
    product_index=None,
):
    """
    Filter the product catalog by constraints and rank by similarity to the user vector.

    Args:
        user_vector:     np.ndarray shape (534,)
        product_vectors: np.ndarray shape (N, 534)
        metadata_df:     DataFrame with columns [product_id, brand, category, price]
        constraints:     dict with optional keys:
                           budget (float): max price in dollars (inclusive)
                           categories (list[str]): allowed category names
                           banned_ingredients (list[str]): ingredient tokens to exclude
                           liked_product_ids (list[str]): already-liked products to exclude
        top_n:           int, number of recommendations to return
        tokens_df:       DataFrame with columns [product_id, ingredient_tokens] (for banned filtering)
        product_index:   dict mapping product_id (str) -> row index (int)

    Returns:
        DataFrame with columns [product_id, brand, category, price, similarity],
        sorted by similarity descending (or price descending if cold start).
    """
    candidates = metadata_df.copy()

    # --- Filter 1: Budget (hard cap on raw price) ---
    budget = constraints.get("budget")
    if budget is not None:
        candidates = candidates[candidates["price"] <= float(budget)]

    # --- Filter 2: Category / routine step ---
    allowed_categories = constraints.get("categories")
    if allowed_categories:
        candidates = candidates[candidates["category"].isin(allowed_categories)]

    # --- Filter 3: Banned ingredients ---
    banned_ingredients = constraints.get("banned_ingredients") or []
    if banned_ingredients and tokens_df is not None:
        banned_set = {ing.lower().strip() for ing in banned_ingredients}

        def has_no_banned(token_string):
            if not isinstance(token_string, str) or not token_string.strip():
                return True  # missing data -> pass through safely
            tokens = {t.strip().lower() for t in token_string.split(",")}
            return tokens.isdisjoint(banned_set)

        merged = candidates.merge(
            tokens_df[["product_id", "ingredient_tokens"]],
            on="product_id",
            how="left",
        )
        mask = merged["ingredient_tokens"].apply(has_no_banned)
        candidates = merged[mask][["product_id", "brand", "category", "price"]]

    # --- Filter 4: Exclude already-liked products ---
    liked_ids = constraints.get("liked_product_ids") or []
    if liked_ids:
        candidates = candidates[~candidates["product_id"].isin(liked_ids)]

    if candidates.empty:
        return _EMPTY_RECS.copy()

    # --- Score: cosine similarity on filtered subset ---
    # Guard against products that are in metadata but not in the vector index
    if product_index is not None:
        valid_mask = candidates["product_id"].isin(product_index)
        candidates = candidates[valid_mask].reset_index(drop=True)

    if candidates.empty:
        return _EMPTY_RECS.copy()

    is_cold_start = np.linalg.norm(user_vector) < 1e-9

    if not is_cold_start and product_index is not None:
        candidate_indices = [product_index[pid] for pid in candidates["product_id"]]
        candidate_vectors = product_vectors[candidate_indices]
        sims = score_similarity(user_vector, candidate_vectors)
    else:
        sims = np.zeros(len(candidates), dtype=np.float32)

    candidates = candidates.copy()
    candidates["similarity"] = sims

    # --- Rank ---
    if is_cold_start:
        # Cold start fallback: return highest-priced (proxy for popular/premium) products
        # Probably not ideal but better than random. Might be better to use some kindof review metric
        recs = candidates.sort_values("price", ascending=False).head(top_n)
    else:
        recs = candidates.sort_values("similarity", ascending=False).head(top_n)

    return recs[["product_id", "brand", "category", "price", "similarity"]].reset_index(drop=True)


def recommend(liked_product_ids, explicit_prefs, constraints, top_n=10):
    """
    End-to-end recommendation pipeline. Loads artifacts from disk on each call.
    Note to self: Might change later if we want to call frequently and need to cache artifacts in memory.

    Args:
        liked_product_ids: list[str]
        explicit_prefs:    dict (see build_user_vector for keys)
        constraints:       dict (see rank_products for keys)
        top_n:             int

    Returns:
        DataFrame with columns [product_id, brand, category, price, similarity]
    """
    vectors, product_index, metadata_df = load_artifacts()
    tokens_df = pd.read_csv(TOKENS_PATH, dtype={"product_id": str})

    user_vector = build_user_vector(
        liked_product_ids, explicit_prefs, vectors, product_index
    )

    # Pass liked_product_ids into constraints so ranked results exclude them
    merged_constraints = dict(constraints)
    if liked_product_ids:
        existing = merged_constraints.get("liked_product_ids") or []
        merged_constraints["liked_product_ids"] = list(set(existing) | set(liked_product_ids))

    return rank_products(
        user_vector=user_vector,
        product_vectors=vectors,
        metadata_df=metadata_df,
        constraints=merged_constraints,
        top_n=top_n,
        tokens_df=tokens_df,
        product_index=product_index,
    )
