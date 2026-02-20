from pathlib import Path

import numpy as np
import pandas as pd

import miguellib.models.user_profile as _up

ROOT = Path(__file__).resolve().parent.parent.parent

# Human-readable labels for ingredient group names
_GROUP_LABELS = {
    "active":           "active",
    "antioxidant":      "antioxidant",
    "chelating_agent":  "chelating",
    "colorant":         "color",
    "emollient":        "moisturizing",
    "exfoliant":        "exfoliating",
    "humectant":        "hydrating",
    "irritant_flag":    "fragrance/irritant",
    "occlusive":        "barrier-protecting",
    "ph_adjuster":      "pH-balancing",
    "preservative":     "preservative",
    "solvent":          "solvent",
    "soothing_agent":   "soothing",
    "sunscreen_filter": "sun protection",
    "texture_agent":    "texture",
}

# Module-level reverse TF-IDF vocab cache: dim index -> token name
_tfidf_reverse_vocab = None


def _get_tfidf_reverse_vocab():
    global _tfidf_reverse_vocab
    if _tfidf_reverse_vocab is not None:
        return _tfidf_reverse_vocab
    if not _up.TFIDF_PATH.exists():
        return {}
    _up._load_tfidf_vocab()
    _tfidf_reverse_vocab = {v: k for k, v in _up._tfidf_vocab.items()}
    return _tfidf_reverse_vocab


def _get_group_overlap(user_vector, product_vector):
    """
    Return (group_name, user_val, product_val) tuples for groups where both
    the user vector and product vector have a meaningful non-zero value.
    Sorted by product_val descending.
    """
    _up._load_group_info()
    overlaps = []
    for i, grp in enumerate(_up._group_names):
        dim = _up.GROUPS_START + i
        u_val = float(user_vector[dim])
        p_val = float(product_vector[dim])
        if u_val > 0.01 and p_val > 0.0:
            overlaps.append((grp, u_val, p_val))
    overlaps.sort(key=lambda x: x[2], reverse=True)
    return overlaps


def _get_top_shared_ingredients(user_vector, product_vector, top_k=3):
    """
    Return up to top_k ingredient token names (TF-IDF dims) with the highest
    co-activation between user vector and product vector.
    """
    rev_vocab = _get_tfidf_reverse_vocab()
    if not rev_vocab:
        return []

    alignment = user_vector[: _up.TFIDF_END] * product_vector[: _up.TFIDF_END]
    top_indices = np.argsort(alignment)[::-1]

    results = []
    seen = set()
    for idx in top_indices:
        if alignment[idx] < 1e-6 or len(results) >= top_k:
            break
        token = rev_vocab.get(int(idx), "")
        if not token or token in seen:
            continue
        seen.add(token)
        results.append(token)

    return results


def _get_category_match(user_vector, product_row):
    """Return True if the user vector shows preference for the product's category."""
    _up._load_cat_info()
    cat = product_row.get("category", "")
    if cat in _up._cat_dim:
        return float(user_vector[_up._cat_dim[cat]]) > 0.3
    return False


def _get_price_tier(product_row, metadata_df):
    """Return 'budget-friendly', 'mid-range', or 'premium' based on price percentile."""
    price = pd.to_numeric(product_row.get("price"), errors="coerce")
    if pd.isna(price):
        return None
    prices = metadata_df["price"].dropna()
    if prices.empty:
        return None
    percentile = float((prices < price).mean())
    if percentile < 0.33:
        return "budget-friendly"
    elif percentile < 0.67:
        return "mid-range"
    return "premium"


def _build_explanation(user_vector, product_vector, product_row, metadata_df, explicit_prefs):
    """Build a single explanation string for one recommended product."""
    skin_type = (explicit_prefs.get("skin_type") or "").lower().strip()
    preferred_ings = {i.lower().strip() for i in (explicit_prefs.get("preferred_ingredients") or [])}

    group_overlaps = _get_group_overlap(user_vector, product_vector)
    top_groups = [g for g, _, _ in group_overlaps[:2]]
    shared_ings = _get_top_shared_ingredients(user_vector, product_vector, top_k=3)
    cat_match = _get_category_match(user_vector, product_row)
    price_tier = _get_price_tier(product_row, metadata_df)
    category = product_row.get("category", "product")

    price_val = pd.to_numeric(product_row.get("price"), errors="coerce")
    price_str = f"${price_val:.0f}" if not pd.isna(price_val) else ""

    parts = []

    # Primary: group overlap
    if len(top_groups) >= 2:
        g1 = _GROUP_LABELS.get(top_groups[0], top_groups[0])
        g2 = _GROUP_LABELS.get(top_groups[1], top_groups[1])
        parts.append(f"recommended because it shares your preference for {g1} and {g2} ingredients")
    elif len(top_groups) == 1:
        g1 = _GROUP_LABELS.get(top_groups[0], top_groups[0])
        ing_str = shared_ings[0] if shared_ings else "key ingredients"
        parts.append(f"recommended because it is rich in {g1} ingredients like {ing_str}")
    else:
        parts.append("recommended based on overall ingredient similarity to products you liked")

    # Preferred ingredient call-out
    matching_pref = [i for i in shared_ings if i in preferred_ings]
    if matching_pref:
        parts.append(f"contains {matching_pref[0]}, which matches your ingredient preferences")

    # Skin type suitability
    if skin_type in _up.SKIN_TYPE_PREFS and top_groups:
        boost_groups, _ = _up.SKIN_TYPE_PREFS[skin_type]
        if top_groups[0] in boost_groups:
            parts.append(f"well-suited for {skin_type} skin")

    # Category match
    if cat_match:
        parts.append(f"a {category} — one of your preferred routine steps")

    # Price tier
    if price_tier and price_str:
        parts.append(f"priced {price_tier} at {price_str}")

    if len(parts) == 1:
        explanation = parts[0] + "."
    else:
        primary = parts[0]
        secondary = "; ".join(parts[1:])
        explanation = f"{primary}. Also: {secondary}."

    return explanation[0].upper() + explanation[1:]


def explain_recommendations(recs_df, user_profile, metadata_df, product_vectors=None, product_index=None):
    """
    Add an 'explanation' column to recs_df describing why each product was recommended.

    Args:
        recs_df:         DataFrame from rank_products (columns: product_id, brand, category, price, similarity)
        user_profile:    dict with keys:
                           user_vector (np.ndarray shape 534)
                           liked_product_ids (list[str])
                           explicit_prefs (dict)
        metadata_df:     DataFrame with columns [product_id, brand, category, price]
        product_vectors: np.ndarray shape (N, 534) — required for per-product explanations
        product_index:   dict mapping product_id (str) -> row index (int)

    Returns:
        Copy of recs_df with an added 'explanation' column (str).
    """
    user_vector = user_profile.get("user_vector")
    explicit_prefs = user_profile.get("explicit_prefs") or {}

    is_cold_start = user_vector is None or np.linalg.norm(user_vector) < 1e-9

    explanations = []

    for _, row in recs_df.iterrows():
        if is_cold_start:
            price_val = pd.to_numeric(row.get("price"), errors="coerce")
            price_str = f" at ${price_val:.0f}" if not pd.isna(price_val) else ""
            explanation = f"A well-regarded {row.get('category', 'product')}{price_str}."
            explanations.append(explanation)
            continue

        pid = row["product_id"]
        if product_vectors is not None and product_index is not None and pid in product_index:
            prod_vec = product_vectors[product_index[pid]]
            explanation = _build_explanation(
                user_vector, prod_vec, row, metadata_df, explicit_prefs
            )
        else:
            explanation = "Recommended based on your preferences."

        explanations.append(explanation)

    result = recs_df.copy()
    result["explanation"] = explanations
    return result
