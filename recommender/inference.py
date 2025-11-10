"""Inference engine for recommendations."""

import json
import os
import sys
from pathlib import Path
import joblib

from recommender.embeddings import EmbeddingsGenerator
from recommender.models import BookRecommenderModel

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PKL_PATH = PROJECT_ROOT / "models" / "recommender_model.pkl"


class RecommenderEngine:
    """Main inference engine for book recommendations."""
    
    def __init__(self, model_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the recommender engine."""
        if model_path is None:
            model_path = MODEL_PKL_PATH
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
        
        self.model = BookRecommenderModel.load(model_path)
        
        # Initialize embeddings generator
        self.embedding_generator = EmbeddingsGenerator(embedding_model)
    
    def recommend(self, query: str, k: int = 5) -> list:
        """Get book recommendations for a query."""
        # Generate embedding for query
        query_embedding = self.embedding_generator.encode_single(query)
        
        # Get recommendations
        recommendations = self.model.recommend(query_embedding, k=k)
        
        return recommendations
    
    def recommend_with_explain(self, query: str, k: int = 5) -> dict:
        """Get recommendations with explanation."""
        recommendations = self.recommend(query, k=k)
        
        return {
            'query': query,
            'num_recommendations': len(recommendations),
            'recommendations': recommendations
        }


def test_inference():
    """Test inference on sample queries."""
    engine = RecommenderEngine()
    
    test_queries = [
        "I love science fiction with space exploration and futuristic worlds",
        "Looking for mystery thrillers with complex plots",
        "Adventure fantasy with magic and dragons"
    ]
    
    print("=" * 70)
    print("TESTING INFERENCE ENGINE")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        recommendations = engine.recommend(query, k=3)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Authors: {rec['authors']}")
            print(f"   Categories: {rec['categories']}")
            print(f"   Rating: {rec['rating']:.2f}")
            print(f"   Similarity Score: {rec['similarity_score']:.4f}")


if __name__ == "__main__":
    test_inference()
