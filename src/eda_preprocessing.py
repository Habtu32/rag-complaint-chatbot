"""Simple EDA & preprocessing for CFPB consumer complaints.

This simplified script focuses on clarity and ease-of-use:
- Load CSV, drop rows without narratives
- Add a simple word count column (`n_words`) and report basic stats
- Map products to project categories, clean text, and save filtered CSV

Usage:
    python -m src.eda_preprocessing --input data/raw/cfpb_complaints.csv --output data/processed/filtered_complaints.csv
"""

from pathlib import Path
import argparse
import re
import pandas as pd

# Minimal product mapping used by the project
PRODUCT_MAPPING = {
    "Credit Card": ["Credit card", "Credit card or prepaid card", "Prepaid card"],
    "Personal Loan": ["Personal loan", "Consumer Loan", "Student loan", "Vehicle loan or lease"],
    "Savings Account": ["Checking or savings account", "Bank account or service"],
    "Money Transfers": ["Money transfers", "Money transfer, virtual currency, or money service"],
}


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def map_product(name: str) -> str:
    for k, vals in PRODUCT_MAPPING.items():
        if name in vals:
            return k
    return "Other"


def preprocess(df: pd.DataFrame) -> (pd.DataFrame, dict):
    # Drop rows with missing narratives
    total = len(df)
    df = df[df["Consumer complaint narrative"].notna()].copy()
    kept = len(df)

    # Word counts
    df["n_words"] = df["Consumer complaint narrative"].str.split().str.len().fillna(0).astype(int)

    # Basic stats
    stats = {
        "total_rows": total,
        "kept_rows": kept,
        "pct_removed": round((1 - kept / total) * 100, 1) if total else 0.0,
        "median_words": int(df["n_words"].median()),
        "max_words": int(df["n_words"].max()),
    }

    # Map products and clean text
    df["product_category"] = df["Product"].apply(map_product)
    df["clean_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    # Keep only target categories
    targets = set(PRODUCT_MAPPING.keys())
    df_filtered = df[df["product_category"].isin(targets)].copy()

    return df_filtered, stats


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Simple EDA & preprocessing for CFPB complaints")
    parser.add_argument("--input", required=True, help="Path to raw CFPB CSV")
    parser.add_argument("--output", required=True, help="Path to save filtered CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_data(input_path)
    df_filtered, stats = preprocess(df)

    print("\nSummary:")
    print(f"  - Rows loaded: {stats['total_rows']:,}")
    print(f"  - Rows kept (with narratives): {stats['kept_rows']:,} ({100 - stats['pct_removed']}% kept)")
    print(f"  - Median words per narrative: {stats['median_words']}")
    print(f"  - Longest narrative (words): {stats['max_words']}")

    save_data(df_filtered, output_path)

    print("\nNote: BNPL was not explicitly identifiable in raw product labels; review mapping if needed.")


if __name__ == "__main__":
    main()