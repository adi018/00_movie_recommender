"""Data preprocessing module for book data."""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "books.csv"
PROCESSED_CSV = DATA_DIR / "books_processed.csv"


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw books data from CSV."""
    if filepath is None:
        filepath = INPUT_CSV
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for recommender system."""
    df = df.copy()
    
    # Select relevant columns
    relevant_cols = ['isbn13', 'title', 'authors', 'categories', 'description', 'average_rating']
    df = df[[col for col in relevant_cols if col in df.columns]]
    
    # Drop rows with missing critical columns
    df = df.dropna(subset=['title', 'description'])
    
    # Fill missing values in other columns
    df['authors'] = df['authors'].fillna('Unknown')
    df['categories'] = df['categories'].fillna('General')
    df['average_rating'] = df['average_rating'].fillna(0.0)
    
    # Clean text: remove extra whitespace and special characters
    text_cols = ['title', 'authors', 'categories', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Ensure description is not empty
    df = df[df['description'].str.len() > 10]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data: {len(df)} valid records retained")
    return df


def combine_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple text fields into single feature for embedding."""
    df = df.copy()
    
    # Combine title, authors, categories, and description into single text field
    df['combined_text'] = (
        df['title'].fillna('') + ' | ' +
        df['authors'].fillna('') + ' | ' +
        df['categories'].fillna('') + ' | ' +
        df['description'].fillna('')
    )
    
    return df


def preprocess_pipeline(input_filepath: str = None, output_filepath: str = None) -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    if input_filepath is None:
        input_filepath = INPUT_CSV
    if output_filepath is None:
        output_filepath = PROCESSED_CSV
    
    # Load raw data
    df = load_raw_data(input_filepath)
    
    # Clean data
    df = clean_data(df)
    
    # Combine text features
    df = combine_text_features(df)
    
    # Save processed data
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved processed data to {output_filepath}")
    
    return df


if __name__ == "__main__":
    df = preprocess_pipeline()
    print(f"\nPreprocessing complete!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst row sample:\n{df.iloc[0]}")
