from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from miguellib.ml_system.artifacts import load_artifacts
from miguellib.ml_system.candidate_source import get_candidates, get_candidates
from miguellib.ml_system.feedback_update import UserState, compute_user_vector
from miguellib.ml_system.reranker import rerank_candidates


def recommend_with_feedback(
    user_state: UserState,
    metadata_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    constraints: Dict[str, Any],
    top_n: int = 10,
    candidate_k: int = 200,
) -> pd.DataFrame:
    """
    Pipeline:
      1) candidate list from ranker (constraint-aware)
      2) rerank with Model 3 feedback-updated user_vector
      3) return ordered DataFrame
    """
    product_vectors, product_index, _, schema = load_artifacts()

    user_vec = compute_user_vector(user_state, schema=schema)
    candidates = get_candidates(
        user_vector=user_vec,   
        product_vectors=product_vectors,
        metadata_df=metadata_df,
        constraints=constraints,
        tokens_df=tokens_df,
        product_index=product_index,
        k=candidate_k,
    )

    if not candidates:
        return pd.DataFrame(columns=["product_id", "brand", "category", "price", "similarity"])

    reranked_ids = rerank_candidates(
        user_vector=user_vec,
        candidate_ids=candidates,
        product_vectors=product_vectors,
        product_index=product_index,
        top_n=top_n,
    )

    
    out = metadata_df[metadata_df["product_id"].astype(str).isin(reranked_ids)].copy()
    out["product_id"] = out["product_id"].astype(str)
    out = out.set_index("product_id").loc[reranked_ids].reset_index()

    return out