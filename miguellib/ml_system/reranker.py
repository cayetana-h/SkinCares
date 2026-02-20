from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rerank_candidates(
    user_vector: np.ndarray,
    candidate_ids: List[str],
    product_vectors: np.ndarray,
    product_index: Dict[str, int],
    top_n: int = 10,
) -> List[str]:
    idxs = [product_index[pid] for pid in candidate_ids if pid in product_index]
    if not idxs:
        return []

    X = product_vectors[idxs]
    scores = cosine_similarity(user_vector.reshape(1, -1), X).flatten()
    order = np.argsort(scores)[::-1]

    
    valid_candidate_ids = [pid for pid in candidate_ids if pid in product_index]
    return [valid_candidate_ids[i] for i in order[: min(top_n, len(order))]]


def mock_candidates_similarity_seed(
    seed_product_id: str,
    product_vectors: np.ndarray,
    product_index: Dict[str, int],
    index_to_id: Dict[int, str],
    k: int = 200,
) -> List[str]:
    seed_product_id = str(seed_product_id)
    seed_idx = product_index[seed_product_id]
    seed_vec = product_vectors[seed_idx].reshape(1, -1)

    sims = cosine_similarity(seed_vec, product_vectors).flatten()
    order = np.argsort(sims)[::-1]

    order = [i for i in order if i != seed_idx]
    top_idxs = order[: min(k, len(order))]

    return [index_to_id[i] for i in top_idxs]