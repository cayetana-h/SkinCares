# miguellib/ml_system/feedback_update.py

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class UserState:
    dim: int
    base_profile: np.ndarray = field(init=False)
    liked_vectors: List[np.ndarray] = field(default_factory=list)
    disliked_vectors: List[np.ndarray] = field(default_factory=list)
    reason_history: List[List[str]] = field(default_factory=list)
    interactions: int = 0

    def __post_init__(self):
        self.base_profile = np.zeros(self.dim, dtype=np.float32)


def update_user_state(
    state: UserState,
    reaction: str,
    product_vector: np.ndarray,
    reason_tags: Optional[List[str]] = None,
):
    reaction = reaction.lower()

    if reaction == "like":
        state.liked_vectors.append(product_vector)
    elif reaction in ["dislike", "irritation"]:
        state.disliked_vectors.append(product_vector)

    state.reason_history.append(reason_tags or [])
    state.interactions += 1
    return state


def apply_reason_penalties(
    user_vector: np.ndarray,
    reason_tags: List[str],
    schema: dict,
    penalty_strength: float = 1.5,
) -> np.ndarray:

    if not schema or "groups" not in schema:
        return user_vector

    tfidf_len = len(schema["tfidf"])
    group_names = schema["groups"]
    updated = user_vector.copy()

    for reason in reason_tags:
        reason = reason.lower()

        if "irritat" in reason and "group_irritant" in group_names:
            idx = tfidf_len + group_names.index("group_irritant")
            updated[idx] -= penalty_strength

        if "greasy" in reason and "group_occlusive" in group_names:
            idx = tfidf_len + group_names.index("group_occlusive")
            updated[idx] -= penalty_strength

        if "not moisturizing" in reason and "group_humectant" in group_names:
            idx = tfidf_len + group_names.index("group_humectant")
            updated[idx] += penalty_strength

    return updated


def compute_user_vector(
    state: UserState,
    schema: Optional[dict] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    gamma: float = 0.8,
) -> np.ndarray:

    base = state.base_profile
    liked_avg = np.mean(state.liked_vectors, axis=0) if state.liked_vectors else 0
    disliked_avg = np.mean(state.disliked_vectors, axis=0) if state.disliked_vectors else 0

    user_vec = alpha * base + beta * liked_avg - gamma * disliked_avg
    user_vec = np.asarray(user_vec, dtype=np.float32)

    if schema:
        for reasons in state.reason_history:
            user_vec = apply_reason_penalties(user_vec, reasons, schema)

    if np.linalg.norm(user_vec) == 0:
        return base.copy()

    return user_vec
