
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "artifacts").exists():
            return p
    raise FileNotFoundError("Could not find project root (folder containing 'artifacts/').")


def load_artifacts():
    root = find_project_root()

    vectors = np.load(root / "artifacts" / "product_vectors.npy")

    with open(root / "artifacts" / "product_index.json", "r", encoding="utf-8") as f:
        product_index = json.load(f)

    schema = None
    schema_path = root / "artifacts" / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    index_to_id = {v: k for k, v in product_index.items()}
    return vectors, product_index, index_to_id, schema


class UserState:
    """Tracks user interactions and preferences."""

    def __init__(self, dim: int):
        self.dim = dim

        
        self.liked_vectors: List[np.ndarray] = []
        self.disliked_vectors: List[np.ndarray] = []
        self.irritation_vectors: List[np.ndarray] = []

        
        self.liked_reasons: List[str] = []
        self.disliked_reasons: List[str] = []
        self.irritation_reasons: List[str] = []

        
        self.interactions: int = 0
        self.liked_count: int = 0
        self.disliked_count: int = 0
        self.irritation_count: int = 0

    def add_liked(self, vec: np.ndarray, reasons: List[str]):
        self.liked_vectors.append(vec)
        self.liked_reasons.extend(reasons)
        self.interactions += 1
        self.liked_count += 1

    def add_disliked(self, vec: np.ndarray, reasons: List[str]):
        self.disliked_vectors.append(vec)
        self.disliked_reasons.extend(reasons)
        self.interactions += 1
        self.disliked_count += 1

    def add_irritation(self, vec: np.ndarray, reasons: List[str]):
        self.irritation_vectors.append(vec)
        self.irritation_reasons.extend(reasons)
        self.interactions += 1
        self.irritation_count += 1


def update_user_state(
    user: UserState,
    reaction: str,
    product_vec: np.ndarray,
    reason_tags: Optional[List[str]] = None,
):
    """
    Update user state based on a single interaction.

    Design choice:
    - "irritation" is treated as a strong negative, so we:
        (1) record it in irritation_vectors (for stronger penalty in user vector)
        (2) ALSO count it as a disliked interaction (so summaries + metrics match intuition)
    """
    if reason_tags is None:
        reason_tags = []

    reaction = (reaction or "").lower().strip()

    if reaction == "like":
        user.add_liked(product_vec, reason_tags)

    elif reaction == "dislike":
        user.add_disliked(product_vec, reason_tags)

    elif reaction == "irritation":
        
        user.add_disliked(product_vec, reason_tags)
        user.add_irritation(product_vec, reason_tags)

    else:
        
        return user

    return user


def compute_user_vector(user: UserState, schema: Optional[Dict] = None) -> np.ndarray:
    """
    Compute user preference vector from feedback.

    Current weighting:
      +2.0 * mean(liked vectors)
      -1.0 * mean(disliked vectors)
      -2.0 * mean(irritation vectors)

    Note: Because irritation is also counted in disliked_vectors, it contributes to both
    the general negative signal and the stronger irritation-specific penalty.
    """
    user_vec = np.zeros(user.dim, dtype=np.float32)

    
    if user.liked_vectors:
        liked_avg = np.mean(user.liked_vectors, axis=0)
        user_vec += 2.0 * liked_avg

    
    if user.disliked_vectors:
        disliked_avg = np.mean(user.disliked_vectors, axis=0)
        user_vec -= 1.0 * disliked_avg

    
    if user.irritation_vectors:
        irritation_avg = np.mean(user.irritation_vectors, axis=0)
        user_vec -= 2.0 * irritation_avg

    
    norm = np.linalg.norm(user_vec)
    if norm > 1e-9:
        user_vec = user_vec / norm

    return user_vec