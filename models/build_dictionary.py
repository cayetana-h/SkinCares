import pandas as pd
from collections import Counter
import json
from pathlib import Path

# Config

TOKENS_PATH = Path("data/processed/products_tokens.csv")
FREQ_OUTPUT = Path("artifacts/top_ingredients.csv")
DICT_OUTPUT = Path("features/ingredient_groups.json")

TOP_K = 80   # how many frequent ingredients to inspect


# Load tokens

def load_ingredient_tokens():
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TOKENS_PATH}")

    df = pd.read_csv(TOKENS_PATH)

    if "ingredient_tokens" not in df.columns:
        raise ValueError("Expected column 'ingredient_tokens' in products_tokens.csv")

    all_ingredients = []

    for row in df["ingredient_tokens"].dropna():
        parts = [x.strip() for x in row.split(",") if x.strip()]
        all_ingredients.extend(parts)

    return all_ingredients


# Frequency extraction

def build_frequency_table(ingredients):
    counter = Counter(ingredients)
    most_common = counter.most_common(TOP_K)

    freq_df = pd.DataFrame(most_common, columns=["ingredient", "count"])
    return freq_df


# Save outputs

def save_frequency_csv(freq_df):
    FREQ_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    freq_df.to_csv(FREQ_OUTPUT, index=False)
    print(f"Saved ingredient frequency → {FREQ_OUTPUT}")


def create_dictionary_template(freq_df):
    DICT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    ingredient_dict = {row["ingredient"]: "" for _, row in freq_df.iterrows()}

    with open(DICT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(ingredient_dict, f, indent=2)

    print(f"Created editable dictionary → {DICT_OUTPUT}")


# Main helper pipeline

def run_dictionary_builder():
    print("Loading ingredient tokens...")
    ingredients = load_ingredient_tokens()

    print("Computing frequencies...")
    freq_df = build_frequency_table(ingredients)

    print("Saving frequency table...")
    save_frequency_csv(freq_df)

    print("Creating dictionary template...")
    create_dictionary_template(freq_df)

    print("\nDone!")
    print("Review artifacts/top_ingredients.csv")
    print("Fill in features/ingredient_groups.json")


# Entry point

if __name__ == "__main__":
    run_dictionary_builder()
