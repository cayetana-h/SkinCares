from pathlib import Path
import pandas as pd
from .utils import (
    load_df,
    validate_data,
    standardize_data,
    flag_non_ingredients
)

"""
This module implements the full preprocessing pipeline for the cosmetics dataset, including:
    1. Loading raw dataset
    2. Validating raw dataset
    3. Cleaning / standardizing ingredients
    4. Flagging suspicious ingredient entries
    5. Validating cleaned dataset
    6. Saving processed dataset 
"""

def run_pipeline(raw_path, processed_path=None, flag_non_ingredients_rows=True):
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
    if flag_non_ingredients_rows:
        flagged = flag_non_ingredients(df)
        if not flagged.empty:
            print("Review the flagged rows before proceeding.")
    
    # step 5
    validate_data(df)
    
    # step 6
    if flagged.empty:
        if processed_path is None:
            processed_path = Path(raw_path).parent / "cosmetics_processed.csv"
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to: {processed_path}")
    else:
        print("Dataset not saved. Suspicious ingredient entries need review first.")
    
    return df
