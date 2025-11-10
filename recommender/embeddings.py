"""Embeddings generation using sentence transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class EmbeddingsGenerator:
    """Generate semantic embeddings for book descriptions."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model."""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts: list) -> np.ndarray:
        """Encode texts to embeddings."""
        print(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding


def build_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """Build embeddings for a list of texts."""
    generator = EmbeddingsGenerator(model_name)
    embeddings = generator.encode(texts)
    return embeddings, generator


if __name__ == "__main__":
    # Test embeddings
    test_texts = [
        "Fantasy adventure with dragons and magic",
        "Science fiction space opera",
        "Mystery thriller in London"
    ]
    embeddings, gen = build_embeddings(test_texts)
    print(f"\nTest embeddings shape: {embeddings.shape}")
