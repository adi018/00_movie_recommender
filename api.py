"""FastAPI server for Book Recommender Model."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.inference import RecommenderEngine

# Initialize FastAPI app
app = FastAPI(
    title="📚 LLM Book Recommender API",
    description="AI-powered book recommendation system using semantic embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender engine (global)
try:
    engine = RecommenderEngine()
except Exception as e:
    print(f"Error initializing recommender engine: {e}")
    engine = None


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    """Request model for book recommendations."""
    query: str
    k: Optional[int] = 5
    
    class Config:
        example = {
            "query": "Science fiction with space exploration",
            "k": 5
        }


class RecommendationItem(BaseModel):
    """Single book recommendation."""
    index: int
    title: str
    authors: str
    categories: str
    rating: float
    similarity_score: float


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    query: str
    num_recommendations: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    total_books: int


# Routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info."""
    return {
        "title": "📚 LLM Book Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender engine not initialized"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "total_books": engine.model.n_books
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get book recommendations for a query.
    
    Args:
        query: Natural language query about books
        k: Number of recommendations (1-20, default: 5)
    
    Returns:
        List of recommended books with similarity scores
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender engine not initialized"
        )
    
    # Validate k parameter
    if not 1 <= request.k <= 20:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 20"
        )
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        recommendations = engine.recommend(request.query, k=request.k)
        
        return {
            "query": request.query,
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.post("/batch-recommend", tags=["Recommendations"])
async def batch_recommendations(queries: List[str], k: Optional[int] = 5):
    """Get recommendations for multiple queries.
    
    Args:
        queries: List of query strings
        k: Number of recommendations per query (default: 5)
    
    Returns:
        List of recommendation results
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender engine not initialized"
        )
    
    if not 1 <= k <= 20:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 20"
        )
    
    if not queries or len(queries) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one query is required"
        )
    
    if len(queries) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 queries allowed per request"
        )
    
    try:
        results = []
        for query in queries:
            if query.strip():
                recommendations = engine.recommend(query, k=k)
                results.append({
                    "query": query,
                    "num_recommendations": len(recommendations),
                    "recommendations": recommendations
                })
        
        return {
            "total_queries": len(queries),
            "successful": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


@app.get("/model-info", tags=["General"])
async def model_info():
    """Get information about the loaded model."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender engine not initialized"
        )
    
    return {
        "total_books": engine.model.n_books,
        "embedding_dimension": engine.embedding_generator.embedding_model.get_sentence_embedding_dimension(),
        "embedding_model": "all-MiniLM-L6-v2",
        "serialization_format": "joblib"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
