from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from miguellib.ml_system.reranker import mock_candidates_similarity_seed
from miguellib.models.recommender_ranker import rank_products


def get_candidates_mock(
    seed_id: str,
    product_vectors: np.ndarray,
    product_index: Dict[str, int],
    index_to_id: Dict[int, str],
    k: int = 200,
) -> List[str]:
    return mock_candidates_similarity_seed(
        seed_product_id=str(seed_id),
        product_vectors=product_vectors,
        product_index=product_index,
        index_to_id=index_to_id,
        k=k,
    )


def get_candidates(
    user_vector: np.ndarray,
    product_vectors: np.ndarray,
    metadata_df: pd.DataFrame,
    constraints: Dict[str, Any],
    tokens_df: Optional[pd.DataFrame],
    product_index: Dict[str, int],
    k: int = 200,
) -> List[str]:
    """
    Uses rank_products() to generate a constrained, ranked candidate list.
    """
    recs_df = rank_products(
        user_vector=user_vector,
        product_vectors=product_vectors,
        metadata_df=metadata_df,
        constraints=constraints,
        top_n=k,
        tokens_df=tokens_df,
        product_index=product_index,
    )
    if recs_df.empty:
        return []
    return recs_df["product_id"].astype(str).tolist()