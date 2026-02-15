import pandas as pd

def load_df(path):
    """To load a csv file and return a pandas DataFrame."""
    df = pd.read_csv(path)
    return df

def validate_data(df):
    """To validate the df for expected structure and values."""

    required_columns = ["Label", "Brand", "Name", "Price", "Rank", "Ingredients", "Combination", "Dry", "Normal", "Oily", "Sensitive"]

    # Check #1: All required columns are present
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check #2: Rank values are between 0 and 5
    if not df["Rank"].between(0, 5).all():
        raise ValueError("Rank values must be between 0 and 5.")

    # Check #3: Price values are non-negative
    if (df["Price"] < 0).any():
        raise ValueError("Price cannot be negative.")

    # Check #4: Skin type columns must contain only 0 or 1
    skin_cols = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
    for col in skin_cols:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")

    # Check #5: Label column includes only expected categories
    allowed_labels = [
        "Moisturizer", "Cleanser", "Face Mask",
        "Treatment", "Eye cream", "Sun protect"
    ]
    if not df["Label"].isin(allowed_labels).all():
        raise ValueError("Unexpected category found in Label column.")

    # Check #6: No null values in required columns
    if df[required_columns].isnull().any().any():
        raise ValueError("Null values detected in required columns.")
    
    # Check #7: No duplicate products 
    duplicates = df.duplicated(subset=["Brand", "Name"]).sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} duplicate products detected by Brand+Name.")

    # Check #8: Ingredients column is populated
    empty_ingredients = (df["Ingredients"].str.strip() == "").sum()
    if empty_ingredients > 0:
        raise ValueError(f"{empty_ingredients} products have empty ingredient lists.")

    print("Validation passed.")

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    To standardize the dataset by cleaning up the Ingredients column, standardizing Brand names, 
    and ensuring consistent formatting for name and label
    """
    df = df.copy()
    
    # Standizing ingredients: lowercase, remove extra spaces, ensure consistent comma separation
    def clean_ingredients(ing):
        if not isinstance(ing, str):
            return ing
        ing = ing.strip().lower()  
        ing = ", ".join([x.strip() for x in ing.split(",")])  
        return ing
    
    df['Ingredients'] = df['Ingredients'].apply(clean_ingredients)
    
    # applying title case to Brand names and stripping extra spaces
    df['Brand'] = df['Brand'].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    
    # removing extra spaces from Name and Label columns
    for col in ['Name', 'Label']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Ensuring Price and Rank are numeric 
    for col in ['Price', 'Rank']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def flag_non_ingredients(df, min_length=15):
    """
    To flag rows where the Ingredients column seems suspicious (e.g., too short, missing commas, contains non-ingredient phrases).
    """
    flagged = df[
        df["Ingredients"].isna() | # ingredients is null
        (df["Ingredients"].str.strip().str.len() < min_length) | # ingredients is too short to be valid
        (~df["Ingredients"].str.contains(",")) | # ingredients does not contain commas 
        (df["Ingredients"].str.lower().str.contains("no info|^\\*\\*|visit")) # ingredients contains phrases that suggest it's not a valid ingredient list
    ]
    
    if flagged.empty:
        print("No suspicious entries found in Ingredients.")
    else:
        print(f"Found {len(flagged)} suspicious entries in Ingredients:\n")
        for idx, row in flagged.iterrows():
            print(f"Index {idx} | Brand: {row['Brand']} | Name: {row['Name']}")
            print(f"Ingredients: {row['Ingredients']}\n")
    
    return flagged
