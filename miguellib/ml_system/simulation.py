# miguellib/ml_system/simulation.py

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List, Dict, Any

import pandas as pd

from miguellib.ml_system.artifacts import load_artifacts, find_project_root
from miguellib.ml_system.feedback_update import (
    UserState,
    update_user_state,
    compute_user_vector,
)
from miguellib.ml_system.reranker import (
    rerank_candidates,
    mock_candidates_similarity_seed,
)


def load_metadata() -> pd.DataFrame:
    """
    Loads products_clean.csv and returns a DataFrame indexed by product_id (as string).
    """
    root = find_project_root()
    path = root / "miguellib" / "datasets" / "datasets" / "products_clean.csv"
    df = pd.read_csv(path, dtype={"product_id": str})

    # Make sure common fields exist (don't crash if missing)
    for col in ["brand", "category", "price"]:
        if col not in df.columns:
            df[col] = ""

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.set_index("product_id", drop=False)
    return df


def format_product_line(pid: str, meta: pd.DataFrame) -> str:
    """
    Returns a single human-readable line for a product_id.
    """
    pid = str(pid)
    if pid in meta.index:
        row = meta.loc[pid]
        brand = str(row.get("brand", ""))
        category = str(row.get("category", ""))
        price = row.get("price", None)
        price_str = f"${price:.2f}" if pd.notna(price) else "NA"
        return f"{pid} | {brand} | {category} | {price_str}"
    return f"{pid} | (metadata missing)"


def pretty_list(product_ids: List[str], meta: pd.DataFrame, n: int = 10) -> str:
    lines = []
    for i, pid in enumerate(product_ids[:n], start=1):
        lines.append(f"{i:>2}. {format_product_line(pid, meta)}")
    return "\n".join(lines)


def compute_overlap(a: List[str], b: List[str]) -> int:
    return len(set(a) & set(b))


def run_simulation(
    top_n: int = 10,
    candidate_k: int = 200,
    seed_id: str | None = None,
) -> None:
    # ---- Load artifacts ----
    product_vectors, product_index, index_to_id, schema = load_artifacts()
    dim = product_vectors.shape[1]
    meta = load_metadata()

    print(f"Loaded vectors: {product_vectors.shape} (products={product_vectors.shape[0]}, dim={dim})")

    # ---- Choose a seed product to generate realistic candidates ----
    if seed_id is None:
        seed_id = next(iter(product_index.keys()))
    seed_id = str(seed_id)

    if seed_id not in product_index:
        # fall back gracefully
        print(f"Seed id {seed_id} not found in index. Falling back to first product in index.")
        seed_id = next(iter(product_index.keys()))

    print(f"\nSeed product for candidate generation:")
    print(f"  {format_product_line(seed_id, meta)}")

    candidates = mock_candidates_similarity_seed(
        seed_product_id=seed_id,
        product_vectors=product_vectors,
        product_index=product_index,
        index_to_id=index_to_id,
        k=candidate_k,
    )

    print(f"\nCandidate pool size: {len(candidates)} (generated via similarity-to-seed)")

    # ---- Initialize user ----
    user = UserState(dim=dim)

    # ---- BEFORE feedback ----
    user_vec_before = compute_user_vector(user, schema=schema)
    ranked_before = rerank_candidates(
        user_vector=user_vec_before,
        candidate_ids=candidates,
        product_vectors=product_vectors,
        product_index=product_index,
        top_n=top_n,
    )

    print("\n=== BEFORE FEEDBACK ===")
    print(pretty_list(ranked_before, meta, n=top_n))

    # ---- Interaction plan (edit these reasons to test your penalty logic) ----
    # IMPORTANT: The “reasons” strings can be anything; your penalty logic checks substrings like "irritat".
    interaction_plan = [
        ("like", candidates[0], ["hydrated_well"]),
        ("like", candidates[3], ["absorbed_quickly"]),
        ("dislike", candidates[10], ["too_greasy"]),
        ("irritation", candidates[15], ["irritated_my_skin"]),
        ("like", candidates[25], ["good_price_to_quality"]),
    ]

    print("\nApplied interactions:")
    for reaction, pid, reasons in interaction_plan:
        print(f"  - {reaction.upper():<10} {format_product_line(pid, meta)} | reasons={reasons}")

    # ---- Apply interactions ----
    for reaction, pid, reasons in interaction_plan:
        pid = str(pid)
        vec = product_vectors[product_index[pid]]
        update_user_state(user, reaction, vec, reason_tags=reasons)

    # ---- AFTER feedback ----
    user_vec_after = compute_user_vector(user, schema=schema)
    ranked_after = rerank_candidates(
        user_vector=user_vec_after,
        candidate_ids=candidates,
        product_vectors=product_vectors,
        product_index=product_index,
        top_n=top_n,
    )

    print("\n=== AFTER FEEDBACK ===")
    print(pretty_list(ranked_after, meta, n=top_n))

    # ---- Summary stats ----
    ov = compute_overlap(ranked_before, ranked_after)
    print(f"\nTop-{top_n} overlap BEFORE vs AFTER: {ov}/{top_n}")

    # Optional: show how many likes/dislikes recorded
    print("\nUser state summary:")
    print(f"  interactions: {user.interactions}")
    print(f"  liked count:  {len(user.liked_vectors)}")
    print(f"  disliked count: {len(user.disliked_vectors)}")


def main():
    parser = argparse.ArgumentParser(description="Model 3 simulation (works without Diana).")
    parser.add_argument("--top_n", type=int, default=10, help="How many products to print in the ranked list.")
    parser.add_argument("--candidate_k", type=int, default=200, help="Candidate pool size (similarity-to-seed).")
    parser.add_argument("--seed_id", type=str, default=None, help="Optional seed product_id for candidate generation.")
    args = parser.parse_args()

    run_simulation(top_n=args.top_n, candidate_k=args.candidate_k, seed_id=args.seed_id)


if __name__ == "__main__":
    main()
