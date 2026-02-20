import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

GROUPS_PATH = ROOT / "features" / "ingredient_groups.json"
TFIDF_PATH = ROOT / "artifacts" / "tfidf.joblib"
SCHEMA_PATH = ROOT / "artifacts" / "feature_schema.json"

TFIDF_START = 0
TFIDF_END = None    # set by _init_layout() from feature_schema.json
GROUPS_START = None
TOTAL_DIMS = None

_schema = None  # cached parsed schema


def _init_layout():
    global _schema, TFIDF_END, GROUPS_START, TOTAL_DIMS
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"feature_schema.json not found at {SCHEMA_PATH}. "
            "Run models/vectorizer.py first."
        )
    with open(SCHEMA_PATH) as f:
        _schema = json.load(f)
    TFIDF_END = len(_schema["tfidf"])
    GROUPS_START = TFIDF_END
    TOTAL_DIMS = _schema["total_features"]


_init_layout()

# Skin type -> (groups to boost, groups to suppress)
SKIN_TYPE_PREFS = {
    "dry":         (["humectant", "emollient", "occlusive", "soothing_agent"],
                    ["exfoliant", "irritant_flag"]),
    "oily":        (["exfoliant", "active", "ph_adjuster"],
                    ["occlusive", "emollient"]),
    "sensitive":   (["soothing_agent", "humectant"],
                    ["irritant_flag", "exfoliant", "active"]),
    "combination": (["humectant"],
                    ["irritant_flag"]),
    "normal":      (["active", "antioxidant"],
                    []),
}

# Module-level caches — populated lazily on first call
_group_map = None       # ingredient -> group name
_group_names = None     # sorted list of unique group names
_group_dim = None       # group name -> absolute dim index
_cat_dim = None         # category name -> absolute dim index
_tfidf_vocab = None     # token -> dim index (0-511)


def _load_group_info():
    global _group_map, _group_names, _group_dim
    if _group_names is not None:
        return
    with open(GROUPS_PATH) as f:
        _group_map = json.load(f)
    # Filter out empty-string group names (ingredient_groups.json may be partially filled)
    _group_names = sorted(g for g in set(_group_map.values()) if g)
    _group_dim = {name: GROUPS_START + i for i, name in enumerate(_group_names)}


def _load_cat_info():
    global _cat_dim
    if _cat_dim is not None:
        return
    if _schema is not None:
        cat_names = [c.replace("cat_", "") for c in _schema["categories"]]
        cat_start = _schema["price_index"] - len(_schema["categories"])
    elif SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            s = json.load(f)
        cat_names = [c.replace("cat_", "") for c in s["categories"]]
        cat_start = s["price_index"] - len(s["categories"])
    else:
        # Alphabetical fallback matching sklearn fit order on the known label set
        cat_names = ["Cleanser", "Eye cream", "Face Mask", "Moisturizer", "Sun protect", "Treatment"]
        cat_start = GROUPS_START + 1  # 1 group dim when all groups are empty
    _cat_dim = {name: cat_start + i for i, name in enumerate(cat_names)}


def _load_tfidf_vocab():
    global _tfidf_vocab
    if _tfidf_vocab is not None:
        return
    vec = joblib.load(TFIDF_PATH)
    _tfidf_vocab = vec.vocabulary_  # token -> dim index (0-511)


def build_user_vector(liked_product_ids, explicit_prefs, product_vectors, product_index):
    """
    Build a user preference vector from liked products and explicit preferences.

    Args:
        liked_product_ids: list[str] — product IDs the user has liked
        explicit_prefs:    dict with optional keys:
                             skin_type (str): "dry" | "oily" | "sensitive" | "combination" | "normal"
                             budget (float): max price in dollars
                             preferred_ingredients (list[str]): ingredient names to boost
                             preferred_categories (list[str]): category names to boost
                             banned_ingredients (list[str]): used downstream by ranker, not here
        product_vectors:   np.ndarray shape (N, total_features)
        product_index:     dict mapping product_id (str) -> row index (int)

    Returns:
        np.ndarray shape (total_features,), dtype float32
    """
    _load_group_info()
    _load_cat_info()

    # --- Step 1: Base vector from liked products ---
    valid_ids = [pid for pid in (liked_product_ids or []) if pid in product_index]
    if valid_ids:
        indices = [product_index[pid] for pid in valid_ids]
        base_vector = product_vectors[indices].mean(axis=0).astype(np.float32)
    else:
        base_vector = np.zeros(TOTAL_DIMS, dtype=np.float32)

    # --- Step 2: Skin type -> group dim boost/suppress ---
    skin_type = (explicit_prefs.get("skin_type") or "").lower().strip()
    if skin_type in SKIN_TYPE_PREFS:
        boost_groups, suppress_groups = SKIN_TYPE_PREFS[skin_type]
        # Use catalog mean per group dim as reference magnitude for boosts
        catalog_group_mean = product_vectors[:, GROUPS_START:GROUPS_START + len(_group_names)].mean(axis=0)
        BOOST_FACTOR = 0.3
        SUPPRESS_FACTOR = 0.5
        for grp in boost_groups:
            if grp not in _group_dim:
                continue
            dim = _group_dim[grp]
            group_offset = dim - GROUPS_START
            delta = float(catalog_group_mean[group_offset]) * BOOST_FACTOR
            base_vector[dim] += delta
        for grp in suppress_groups:
            if grp not in _group_dim:
                continue
            base_vector[_group_dim[grp]] *= SUPPRESS_FACTOR

    # --- Step 3: preferred_ingredients -> TF-IDF dim boost ---
    preferred_ingredients = explicit_prefs.get("preferred_ingredients") or []
    if preferred_ingredients and TFIDF_PATH.exists():
        _load_tfidf_vocab()
        INGREDIENT_BOOST = 0.15
        for ing in preferred_ingredients:
            token = ing.lower().strip()
            if token in _tfidf_vocab:
                base_vector[_tfidf_vocab[token]] += INGREDIENT_BOOST

    # --- Step 4: preferred_categories -> category dim boost ---
    preferred_categories = explicit_prefs.get("preferred_categories") or []
    CAT_BOOST = 0.5
    for cat in preferred_categories:
        if cat in _cat_dim:
            base_vector[_cat_dim[cat]] += CAT_BOOST

    # --- Step 5: budget -> price dim (cold start only) ---
    budget = explicit_prefs.get("budget")
    price_dim = TOTAL_DIMS - 1
    if budget is not None and not valid_ids:
        budget = float(budget)
        if budget <= 20:
            base_vector[price_dim] = 0.15
        elif budget <= 50:
            base_vector[price_dim] = 0.35
        elif budget <= 100:
            base_vector[price_dim] = 0.60
        else:
            base_vector[price_dim] = 0.85

    return base_vector
