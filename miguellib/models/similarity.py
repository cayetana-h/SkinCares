import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT / "artifacts" / "feature_schema.json"


def _build_default_weights():
    """
    Build a weight vector from feature_schema.json.
    Up-weights group and category dims relative to the TF-IDF block so
    high-level structure contributes meaningfully despite fewer dims.
    Returns None if the schema has not been generated yet.
    """
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    n = schema["total_features"]
    tfidf_end = len(schema["tfidf"])
    groups_start = tfidf_end
    groups_end = groups_start + len(schema["groups"])
    cat_start = groups_end
    cat_end = schema["price_index"]
    price_dim = schema["price_index"]

    w = np.ones(n, dtype=np.float32)
    w[groups_start:groups_end] = 3.0   # group dims × 3
    w[cat_start:cat_end] = 5.0         # category dims × 5
    w[price_dim] = 0.5                 # price is a soft signal
    return w


# Built once at import time; None if schema not yet generated.
DEFAULT_WEIGHTS = _build_default_weights()


def score_similarity(user_vector, product_vectors, weights=None, dims_mask=None):
    """
    Compute weighted cosine similarity between a user vector and all product vectors.

    The weight vector is applied via element-wise scaling: both the user vector
    and every product vector are multiplied by sqrt(weights) before computing
    standard cosine similarity. This yields sum(u_i * v_i * w_i) in the dot
    product, i.e., a true weighted cosine.

    Args:
        user_vector:     np.ndarray shape (total_features,)
        product_vectors: np.ndarray shape (N, total_features)
        weights:         np.ndarray shape (total_features,) or None (uses DEFAULT_WEIGHTS)
        dims_mask:       np.ndarray of bool shape (total_features,) or None.
                         If provided, only the selected dims are used.

    Returns:
        np.ndarray shape (N,), dtype float32, values in [-1, 1].
        Returns zeros if user_vector is a zero vector (cold start).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if weights is None:
        # Schema not generated yet — fall back to unweighted cosine
        weights = np.ones(len(user_vector), dtype=np.float32)

    # Apply optional dim subset
    if dims_mask is not None:
        user_vector = user_vector[dims_mask]
        product_vectors = product_vectors[:, dims_mask]
        weights = weights[dims_mask]

    # Cold start: zero user vector -> undefined cosine sim, return zeros
    if np.linalg.norm(user_vector) < 1e-9:
        return np.zeros(len(product_vectors), dtype=np.float32)

    # Weighted scaling: multiply by sqrt(w) on both sides so the weighted dot product is sum(u_i * v_i * w_i)
    w_sqrt = np.sqrt(weights)
    user_w = user_vector * w_sqrt          # shape (D,)
    products_w = product_vectors * w_sqrt  # shape (N, D), broadcasts

    sims = sklearn_cosine(user_w.reshape(1, -1), products_w).flatten()
    return sims.astype(np.float32)
