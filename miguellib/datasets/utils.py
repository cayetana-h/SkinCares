import pandas as pd
import re
from typing import List, Tuple, Pattern, Callable, Optional

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
        ing = ", ".join([re.sub(r'\s+', ' ', x.strip()) for x in ing.split(",")]) 
        ing = re.sub(r'(\w)-\s+(\w)', r'\1-\2', ing)  
        return ing
    
    df['Ingredients'] = df['Ingredients'].apply(clean_ingredients)

    df['Ingredients'] = df['Ingredients'].str.replace(r'\s+', ' ', regex=True)
    
    # applying title case to Brand names and stripping extra spaces
    df['Brand'] = df['Brand'].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    
    # removing extra spaces from Name and Label columns
    for col in ['Name', 'Label']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].str.replace(r'[™®]', '', regex=True)
    
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


def apply_review_actions(df, review_path):
    """
    To apply the review actions from the review file to the dataset
    """
    if str(review_path).endswith((".xlsx", ".xls")):
        df_review = pd.read_excel(review_path)
    else:
        df_review = pd.read_csv(review_path)
    
    df_review.columns = df_review.columns.str.strip()

    required_cols = ["Brand", "Name", "Action", "Fill_in_Ingredients"]
    missing = set(required_cols) - set(df_review.columns)
    if missing:
        raise ValueError(f"Review file missing columns: {missing}")

    df = standardize_data(df)
    df_review = standardize_data(df_review)

    df = df.merge(
        df_review[required_cols],
        on=['Brand', 'Name'],
        how='left'
    )

    replace_mask = df['Action'] == 'replace'
    df.loc[replace_mask, 'Ingredients'] = df.loc[replace_mask, 'Fill_in_Ingredients']

    df = df[df['Action'] != 'remove'].copy()
    df = df.drop(columns=['Action', 'Fill_in_Ingredients'])

    return df

def clean_ingredients(ing: Optional[str]) -> str:
    """
    Clean a raw INCI ingredient string.

    Keeps meaningful parenthetical content (e.g., "(ci 77891)"),
    while removing common wrappers/prefixes (e.g., "+/-", "may contain", bullets, stray stars),
    normalizing whitespace/dashes, and dropping obvious marketing/non-ingredient fragments.

    """
    if not isinstance(ing, str) or not ing.strip():
        return ""

    parts = [p.strip() for p in ing.split(",") if p.strip()]
    cleaned: list[str] = []

    for raw in parts:
        t = raw.lower().strip()
        t = re.sub(r"\s+", " ", t)
        t = t.replace("–", "-").replace("—", "-")

        # Remove leading bullets/stars/plus/minus markers
        t = re.sub(r"^[\*\u2022]+\s*", "", t)     
        t = re.sub(r"^[\+\-]+\s*", "", t)        

        # Remove leading wrappers 
        t = re.sub(
            r"^[\[\(]?\s*(?:\+\/-|\+|-|may contain|peut contenir)\s*[\]\):\-]*\s*",
            "",
            t,
        )
        t = re.sub(r"^(?:may contain|peut contenir)\s*:?\s*", "", t)

        # Trim outer brackets/braces 
        t = t.strip("[]{}").strip()

        # "(-)-alpha-bisabolol" -> "alpha-bisabolol"
        t = re.sub(r"^\(\s*[-+]\s*\)\s*-\s*", "", t)

       
        m = re.match(r"^\(([^()]{1,80})\)$", t)
        if m:
            t = m.group(1).strip()

        # Remove common footer / disclaimer fragments
        t = re.sub(r"\s*please be aware.*$", "", t)
        t = re.sub(r"\s*ingredient lists may change.*$", "", t)
        t = re.sub(r"\s*please refer.*$", "", t)

        # Remove broken tail segments 
        t = re.sub(r"\[\s*\+\/-\s*:.*$", "", t)
        t = re.sub(r"\(\s*\+\/-\s*\)\s*:.*$", "", t)
        t = re.sub(r"\bmay contain\s*:.*$", "", t)

        t = t.strip(" .;:-")

        # If parentheses are unbalanced, drop trailing after "("
        if t.count("(") > t.count(")"):
            t = t.split("(", 1)[0].strip()

        # Final tail cleanup
        t = re.sub(r"[\*\.;:]+$", "", t).strip()
        t = t.strip(" -_")

        # Drop obvious marketing / non-ingredient fragments
        if re.search(r"\bpercent\b", t) and any(w in t for w in ["off", "save", "discount", "free shipping"]):
            continue
        if len(t) > 120 and (":" in t or "helps" in t or "step" in t):
            continue
        if "division:" in t or "active ingredients:" in t:
            continue
        if t.startswith("and ") or t.startswith("all products are"):
            continue

        if not t:
            continue

        cleaned.append(t)

    return ", ".join(cleaned)

    from typing import List, Optional

import re
from typing import Optional

def normalize_ingredient_token(token: Optional[str]) -> str:
    """
    Normalize a single ingredient token so matching/deduping works reliably.

    - lowercases
    - trims whitespace
    - normalizes weird dashes
    - collapses repeated spaces
    - trims surrounding punctuation/brackets
    """
    if not isinstance(token, str) or not token.strip():
        return ""

    t = token.strip().lower()
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)

   
    t = t.strip(" \t\n\r[]{}")


    t = t.strip(" .;:-")

    return t

def ingredient_tokens(ing: Optional[str]) -> List[str]:
    """
    Convert a cleaned Ingredients string into a list of normalized ingredient tokens.

    Steps:
    Split by commas
    Normalize each token using normalize_ingredient_token()
    Drop empty/invalid tokens 
    Deduplicate while preserving original order

    """
    if not isinstance(ing, str) or not ing.strip():
        return []

    parts = [p.strip() for p in ing.split(",") if p.strip()]
    tokens = [normalize_ingredient_token(p) for p in parts]


    tokens = [t for t in tokens if t and t != "nan"]

    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out

    import json
from pathlib import Path
from typing import List, Dict, Optional, Callable

def apply_synonyms_to_tokens(
    tokens: List[str],
    synonyms_path: str = "synonyms.json",
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Apply a synonyms mapping to a list of ingredient tokens.

    Loads synonyms from a JSON file once and caches them.
    Normalizes keys/values so they match token normalization.
    If a synonym maps to "" it will drop that token.
    Deduplicates while preserving order.
    """
    if not isinstance(tokens, list):
        return []

    
    if normalize_fn is None:
        normalize_fn = normalize_ingredient_token  # must exist in utils or be imported

    # Load + cache synonyms (cache is per file path)
    cache_key = str(Path(synonyms_path).resolve())
    if not hasattr(apply_synonyms_to_tokens, "_syn_cache"):
        apply_synonyms_to_tokens._syn_cache = {}

    if cache_key not in apply_synonyms_to_tokens._syn_cache:
        p = Path(synonyms_path)
        raw: Dict[str, str] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        apply_synonyms_to_tokens._syn_cache[cache_key] = {
            normalize_fn(k): (normalize_fn(v) if v else "")
            for k, v in raw.items()
        }

    syn: Dict[str, str] = apply_synonyms_to_tokens._syn_cache[cache_key]

    # Map tokens
    mapped: List[str] = []
    for t in tokens:
        t_norm = normalize_fn(t)
        t_final = syn.get(t_norm, t_norm)
        if t_final:  # drop "" mappings
            mapped.append(t_final)

    # Deduplicate 
    seen = set()
    out: List[str] = []
    for t in mapped:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out


CANON_RULES: List[Tuple[str, List[str]]] = [
    # water / base-water
    ("water", [
        r".*:\s*water\s*$",
        r".*ingredients:\s*water\s*$",
        r".*\bbase:\s*water\s*$",
        r".*\bbase concentrate.*water\s*$",
        r".*\bmask:\s*water\s*$",
        r".*\bcleanser:\s*water\s*$",
        r".*\beye.*cream:\s*water\s*$",

        r"^\s*aqua\s*$",
        r"^\s*water\s*$",
        r"^\s*eau\s*$",
        r"aqua\s*/\s*water\s*/\s*eau",
        r"water\s*/\s*aqua\s*/\s*eau",
        r"aqua\s*\(water\)",
        r"water\s*\(aqua\)",
        r"purified\s+water",
    ]),

    ("fragrance", [
        r"^\s*parfum\s*$",
        r"^\s*fragrance\s*$",
        r"parfum\s*\(fragrance\)",
        r"fragrance\s*\(parfum\)",
        r"fragrance\s*/\s*parfum",
        r"parfum\s*/\s*fragrance",
        r"natural\s+fragrance",
        r"\baroma\b",
        r"\bflavor\b",
    ]),

    ("vitamin e", [
        r"\btocopherol\b",
        r"\btocopheryl\s+acetate\b",
        r"\btocopheryl\s+succinate\b",
        r"\btocotrienols\b",
    ]),

    # pigments
    ("mica", [r"\bmica\b", r"\bci\s*77019\b"]),

    ("titanium dioxide", [
        r"titanium\s+dioxide",
        r"\bci\s*77891\b",
        r"\bci77891\b",
        r"\bci7789\b",
    ]),

    ("iron oxides", [
        r"iron\s+oxides",
        r"\bci\s*77491\b",
        r"\bci\s*77492\b",
        r"\bci\s*77499\b",
        r"\b77491\b",
        r"\b77492\b",
        r"\b77499\b",
    ]),

    # citrus (collapse variations -> canonical token)
    ("orange extract", [
        r"citrus\s+sinensis",
        r"aurantium\s+dulcis",
        r"\borange\b.*\b(peel|fruit|flower|leaf|oil|water|extract|powder|wax)\b",
        r"\b(sweet|blood)\s+orange\b",
    ]),
    ("lemon extract", [
        r"citrus\s+limon",
        r"medica\s+limonum",
        r"\blemon\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
    ("lime extract", [
        r"citrus\s+aurantifolia",
        r"\blime\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
    ("grapefruit extract", [
        r"citrus\s+paradisi",
        r"citrus\s+grandis",
        r"\b(grapefruit|pomelo)\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
    ("bergamot extract", [
        r"citrus\s+aurantium\s+bergamia",
        r"aurantium\s+bergamia",
        r"\bbergamot\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
    ("mandarin/tangerine extract", [
        r"citrus\s+reticulata",
        r"citrus\s+nobilis",
        r"citrus\s+tangerina",
        r"\b(mandarin|tangerine)\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
    ("yuzu extract", [
        r"citrus\s+junos",
        r"\byuzu\b.*\b(peel|fruit|oil|water|extract|powder)\b",
    ]),
]

# Compile rules once 

CANON_RULES_COMPILED: List[Tuple[str, List[Pattern]]] = [
    (canon, [re.compile(pat, flags=re.IGNORECASE) for pat in pats])
    for canon, pats in CANON_RULES
]

def apply_canon_to_tokens(
    tokens: List[str],
    canon_rules_compiled: List[Tuple[str, List[Pattern]]] = CANON_RULES_COMPILED,
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Canonicalize ingredient tokens by collapsing many variants into one canonical token.

    Uses compiled rules.
    If a token matches any pattern, it is replaced by the canonical token.
    Deduplicates while preserving order.
    """
    if not isinstance(tokens, list):
        return []

    if normalize_fn is None:
        normalize_fn = lambda s: s.strip().lower()

    out: List[str] = []
    seen = set()

    for tok in tokens:
        t = normalize_fn(tok)
        if not t:
            continue

        canon = None
        for canon_token, patterns in canon_rules_compiled:
            if any(p.search(t) for p in patterns):
                canon = canon_token
                break

        final_tok = canon if canon else t

        if final_tok not in seen:
            out.append(final_tok)
            seen.add(final_tok)

    return out