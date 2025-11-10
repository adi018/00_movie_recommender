"""Model class definition for book recommender."""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class BookRecommenderModel:
    """Book recommender using semantic embeddings."""
    
    def __init__(self, embeddings: np.ndarray = None, metadata: list = None):
        """Initialize model with embeddings and metadata."""
        self.embeddings = embeddings
        self.metadata = metadata
        self.n_books = len(metadata) if metadata else 0
        if metadata:
            print(f"Model initialized with {self.n_books} books")
    
    def recommend(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """Recommend top-k books for a query embedding."""
        # Compute cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build recommendations
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'index': int(idx),
                'title': self.metadata[idx]['title'],
                'authors': self.metadata[idx]['authors'],
                'categories': self.metadata[idx]['categories'],
                'rating': self.metadata[idx]['rating'],
                'similarity_score': float(similarities[idx])
            })
        
        return recommendations
    
    def save(self, filepath):
        """Save model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load model from disk using joblib."""
        return joblib.load(filepath)
