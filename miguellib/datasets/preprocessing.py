from pathlib import Path
import pandas as pd
from .utils import (
    load_df,
    validate_data,
    standardize_data,
    flag_non_ingredients,
    apply_review_actions
)

"""
This module implements the full preprocessing pipeline for the cosmetics dataset, including:
    1. Loading raw dataset
    2. Validating raw dataset
    3. Cleaning / standardizing ingredients
    4. Flagging suspicious ingredient entries
    5. Applying manual review actions (used xls file with review decisions)
    6. Re-standardizing after manual review
    7. Re-validating the cleaned dataset
    8. Saving the processed dataset to a new file 
"""

def run_pipeline(raw_path, processed_path=None, reviewed_path=None, flag_non_ingredients_rows=True):
    """
    Full preprocessing pipeline for the cosmetics dataset.
    """
    # step 1
    df = load_df(raw_path)
    
    # step 2
    validate_data(df)
    
    # step 3
    df = standardize_data(df)
    
    # step 4
    flagged = None
    if flag_non_ingredients_rows:
        flagged = flag_non_ingredients(df)
    
    # step 5
    if reviewed_path is not None:
        df = apply_review_actions(df, reviewed_path)
        print("Manual review actions applied.")

    # step 6
    df = standardize_data(df)

    # step 7
    validate_data(df)
    
    # step 8
    if processed_path is None:
        processed_path = Path(raw_path).parent / "cosmetics_processed.csv"

    df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to: {processed_path}")
    
    return df, flagged
