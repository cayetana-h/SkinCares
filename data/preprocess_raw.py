import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "processed" / "cosmetics_processed_clean_tokens.csv"

OUT_CLEAN  = ROOT / "data" / "processed" / "products_clean.csv"
OUT_TOKENS = ROOT / "data" / "processed" / "products_tokens.csv"


def run_preprocessing():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Building products_clean.csv...")
    clean_df = (
        df[["Brand", "Label", "Price"]]
        .rename(columns={
            "Brand": "brand",
            "Label": "category",
            "Price": "price",
        })
        .assign(product_id=df.index.astype(str))
    )

    print("Building products_tokens.csv...")
    tokens_df = (
        df[["ingredient_tokens"]]
        .assign(product_id=df.index.astype(str))
    )

    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(OUT_CLEAN, index=False)
    tokens_df.to_csv(OUT_TOKENS, index=False)

    print("Saved:")
    print(" →", OUT_CLEAN)
    print(" →", OUT_TOKENS)


if __name__ == "__main__":
    run_preprocessing()