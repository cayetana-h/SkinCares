from pathlib import Path
import pandas as pd

from .utils import (
    clean_ingredients,
    ingredient_tokens,
    apply_synonyms_to_tokens,
    apply_canon_to_tokens,
)


def build_clean_tokenized_ingredients(
    df: pd.DataFrame, synonyms_path: str = "synonyms.json"
) -> pd.DataFrame:
    """
    Cleans and standardizes ingredient lists.

    Overwrites df["Ingredients"] with the cleaned + canonical ingredient string
    Adds df["ingredient_tokens"] with the final list of tokens (synonyms + canon applied)
    """
    df_out = df.copy()

    # 1) Clean raw string
    df_out["Ingredients"] = df_out["Ingredients"].apply(clean_ingredients)

    # 2) Tokenize
    df_out["ingredient_tokens"] = df_out["Ingredients"].apply(ingredient_tokens)

    # 3) Synonyms
    df_out["ingredient_tokens"] = df_out["ingredient_tokens"].apply(
        lambda toks: apply_synonyms_to_tokens(toks, synonyms_path=synonyms_path)
    )

    # 4) Canon
    df_out["ingredient_tokens"] = df_out["ingredient_tokens"].apply(
        apply_canon_to_tokens
    )

    # 5) Overwrite Ingredients with canonical joined tokens (final “fixed” Ingredients)
    df_out["Ingredients"] = df_out["ingredient_tokens"].apply(lambda xs: ", ".join(xs))

    return df_out


def run_pipeline3(
    raw_path: str, processed_path: str = None, synonyms_path: str = "synonyms.json"
) -> pd.DataFrame:
    """
    Load -> clean/tokenize/canon -> save.
    """
    raw_path = Path(raw_path)

    df = pd.read_csv(raw_path)

    df_processed = build_clean_tokenized_ingredients(df, synonyms_path=synonyms_path)

    if processed_path is None:
        processed_path = raw_path.parent / "cosmetics_processed_clean_tokens.csv"
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df_processed.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to: {processed_path}")

    return df_processed
