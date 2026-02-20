from __future__ import annotations

import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from miguellib.ml_system.artifacts import load_artifacts, find_project_root
from miguellib.ml_system.feedback_update import UserState, update_user_state, compute_user_vector
from miguellib.ml_system.reranker import rerank_candidates
from miguellib.models.recommender_ranker import rank_products


def load_metadata(root) -> pd.DataFrame:
    path = root / "miguellib" / "datasets" / "datasets" / "products_clean.csv"
    df = pd.read_csv(path, dtype={"product_id": str})

    for col in ["brand", "category", "price"]:
        if col not in df.columns:
            df[col] = ""

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def load_tokens(root) -> pd.DataFrame:
    path = root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv"
    return pd.read_csv(path, dtype={"product_id": str})


def format_product(pid: str, meta_indexed: pd.DataFrame) -> str:
    pid = str(pid)
    if pid not in meta_indexed.index:
        return f"{pid} | (metadata missing)"
    row = meta_indexed.loc[pid]
    price = row.get("price", None)
    price_str = f"${price:.2f}" if pd.notna(price) else "NA"
    return f"{pid} | {row.get('brand','')} | {row.get('category','')} | {price_str}"


def pretty_list(product_ids: List[str], meta_indexed: pd.DataFrame, n: int = 10) -> str:
    return "\n".join(
        [f"{i:>2}. {format_product(pid, meta_indexed)}" for i, pid in enumerate(product_ids[:n], start=1)]
    )


def run_simulation(
    top_n: int = 10,
    candidate_k: int = 200,
    budget: float | None = 100.0,
    categories: List[str] | None = None,
):
    # ---- Load artifacts ----
    product_vectors, product_index, _, schema = load_artifacts()
    dim = product_vectors.shape[1]

    root = find_project_root()
    meta = load_metadata(root)
    tokens_df = load_tokens(root)

    # Index for pretty printing
    meta_idx = meta.copy()
    meta_idx["product_id"] = meta_idx["product_id"].astype(str)
    meta_idx = meta_idx.set_index("product_id", drop=False)

    print(f"Loaded vectors: {product_vectors.shape} (products={product_vectors.shape[0]}, dim={dim})")

    # ---- Initialize user ----
    user = UserState(dim=dim)

    # ---- Constraints for ranker ----
    constraints: Dict[str, Any] = {}
    if budget is not None:
        constraints["budget"] = float(budget)
    if categories:
        constraints["categories"] = categories
    constraints["banned_ingredients"] = []
    constraints["liked_product_ids"] = []

    # ---- Build user vector (cold start) and get candidate pool ----
    user_vec_before = compute_user_vector(user, schema=schema)

    candidates_df = rank_products(
        user_vector=user_vec_before,
        product_vectors=product_vectors,
        metadata_df=meta,
        constraints=constraints,
        top_n=candidate_k,
        tokens_df=tokens_df,
        product_index=product_index,
    )

    if candidates_df.empty:
        print("No candidates returned from ranker. Try loosening constraints.")
        return

    candidate_ids = candidates_df["product_id"].astype(str).tolist()
    print(f"\nCandidate pool size: {len(candidate_ids)} (rank_products)")
    print(f"Constraints: {constraints}")

    # ---- BEFORE feedback: rerank candidate pool ----
    ranked_before = rerank_candidates(
        user_vector=user_vec_before,
        candidate_ids=candidate_ids,
        product_vectors=product_vectors,
        product_index=product_index,
        top_n=top_n,
    )

    print("\n=== BEFORE FEEDBACK ===")
    print(pretty_list(ranked_before, meta_idx, n=top_n))

    # ---- Interaction plan (edit to match your UX reasons) ----
    interaction_plan = [
        ("like", candidate_ids[0], ["hydrated_well"]),
        ("like", candidate_ids[3], ["absorbed_quickly"]),
        ("dislike", candidate_ids[10], ["too_greasy"]),
        ("irritation", candidate_ids[15], ["irritated_my_skin"]),
        ("like", candidate_ids[25], ["good_price_to_quality"]),
    ]

    print("\nApplied interactions:")
    for reaction, pid, reasons in interaction_plan:
        print(f"  - {reaction.upper():<10} {format_product(pid, meta_idx)} | reasons={reasons}")

    # ---- Apply interactions ----
    for reaction, pid, reasons in interaction_plan:
        pid = str(pid)
        if pid not in product_index:
            continue
        vec = product_vectors[product_index[pid]]
        update_user_state(user, reaction, vec, reason_tags=reasons)

    # ---- AFTER feedback: recompute user vector ----
    user_vec_after = compute_user_vector(user, schema=schema)

    # ---- Exclude liked products from AFTER recommendations ----
    liked_ids = {
        str(pid) for reaction, pid, _ in interaction_plan
        if reaction.lower() == "like"
    }

    candidate_ids_after = [pid for pid in candidate_ids if str(pid) not in liked_ids]

    # Optional: print excluded liked ids for clarity
    print(f"\nExcluded liked products from AFTER ranking: {sorted(liked_ids)}")

    ranked_after = rerank_candidates(
        user_vector=user_vec_after,
        candidate_ids=candidate_ids_after,
        product_vectors=product_vectors,
        product_index=product_index,
        top_n=top_n,
    )

    print("\n=== AFTER FEEDBACK ===")
    print(pretty_list(ranked_after, meta_idx, n=top_n))

    overlap = len(set(ranked_before) & set(ranked_after))
    print(f"\nTop-{top_n} overlap BEFORE vs AFTER: {overlap}/{top_n}")

    print("\nUser state summary:")
    print(f"  interactions: {user.interactions}")
    print(f"  liked count:  {len(user.liked_vectors)}")
    print(f"  disliked count: {len(user.disliked_vectors)}")


def main():
    p = argparse.ArgumentParser(description="Model 3 simulation using Diana candidate generation (no mock).")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--candidate_k", type=int, default=200)
    p.add_argument("--budget", type=float, default=100.0)
    p.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=["Moisturizer"],
        help="Allowed categories (space-separated). Example: --categories Moisturizer Cleanser",
    )
    args = p.parse_args()

    cats = args.categories if args.categories else None
    run_simulation(
        top_n=args.top_n,
        candidate_k=args.candidate_k,
        budget=args.budget,
        categories=cats,
    )


if __name__ == "__main__":
    main()