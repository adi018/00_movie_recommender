"""Model training pipeline."""

import os
import json
import numpy as np
from pathlib import Path
import pandas as pd
import joblib

from recommender.preprocess import preprocess_pipeline, PROCESSED_CSV
from recommender.embeddings import build_embeddings
from recommender.models import BookRecommenderModel

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model artifact paths
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_PKL_PATH = MODELS_DIR / "recommender_model.pkl"


def train_model(processed_data_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
    """Train the recommender model."""
    
    # Use processed data if available, otherwise preprocess
    if processed_data_path and os.path.exists(processed_data_path):
        print(f"Loading processed data from {processed_data_path}")
        df = pd.read_csv(processed_data_path)
    else:
        print("Preprocessing data...")
        df = preprocess_pipeline()
    
    # Extract texts and metadata
    texts = df['combined_text'].tolist()
    metadata = []
    
    for idx, row in df.iterrows():
        metadata.append({
            'isbn': str(row.get('isbn13', 'N/A')),
            'title': str(row['title']),
            'authors': str(row['authors']),
            'categories': str(row['categories']),
            'rating': float(row.get('average_rating', 0.0))
        })
    
    # Generate embeddings
    embeddings, generator = build_embeddings(texts, embedding_model)
    
    # Create model
    model = BookRecommenderModel(embeddings, metadata)
    
    # Save artifacts
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {METADATA_PATH}")
    
    model.save(MODEL_PKL_PATH)
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING RECOMMENDER MODEL")
    print("=" * 60)
    
    model = train_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {MODEL_PKL_PATH}")
