# 📚 FastAPI Model Server - Documentation

## Overview

The FastAPI server provides REST API endpoints for the LLM Book Recommender system, offering programmatic access to book recommendations alongside the Streamlit UI.

## Quick Start

### Run FastAPI Server Directly
```bash
python api.py
```
API will be available at: **http://localhost:8000**

### Run with Docker Compose (Both Streamlit + API)
```bash
docker-compose up --build
```

- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)

## Endpoints

### 1. **Health Check**
**GET** `/health`

Check if the API and model are healthy.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_books": 6538
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. **Get Recommendations**
**POST** `/recommend`

Get book recommendations for a single query.

**Request Body:**
```json
{
  "query": "Science fiction with space exploration",
  "k": 5
}
```

**Parameters:**
- `query` (string, required): Natural language query about books
- `k` (integer, optional): Number of recommendations (1-20, default: 5)

**Response:**
```json
{
  "query": "Science fiction with space exploration",
  "num_recommendations": 3,
  "recommendations": [
    {
      "index": 2747,
      "title": "Rocket Ship Galileo",
      "authors": "Robert Anson Heinlein",
      "categories": "Fiction",
      "rating": 3.71,
      "similarity_score": 0.5522
    },
    {
      "index": 2738,
      "title": "Starplex",
      "authors": "Robert J. Sawyer",
      "categories": "Fiction",
      "rating": 3.83,
      "similarity_score": 0.5433
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Fantasy with dragons",
    "k": 5
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "query": "Mystery thriller",
        "k": 3
    }
)
recommendations = response.json()
for rec in recommendations["recommendations"]:
    print(f"{rec['title']} by {rec['authors']} (Score: {rec['similarity_score']:.2%})")
```

---

### 3. **Batch Recommendations**
**POST** `/batch-recommend`

Get recommendations for multiple queries in one request.

**Query Parameters:**
- `queries` (list of strings): List of queries
- `k` (integer, optional): Recommendations per query (1-20, default: 5)

**Request Body:**
```json
[
  "Science fiction with space exploration",
  "Fantasy with dragons",
  "Mystery thriller"
]
```

**Response:**
```json
{
  "total_queries": 3,
  "successful": 3,
  "results": [
    {
      "query": "Science fiction with space exploration",
      "num_recommendations": 2,
      "recommendations": [...]
    },
    {
      "query": "Fantasy with dragons",
      "num_recommendations": 2,
      "recommendations": [...]
    },
    {
      "query": "Mystery thriller",
      "num_recommendations": 2,
      "recommendations": [...]
    }
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/batch-recommend?k=3" \
  -H "Content-Type: application/json" \
  -d '["Fantasy with dragons", "Science fiction", "Mystery"]'
```

**Python Example:**
```python
import requests

queries = [
    "Science fiction",
    "Fantasy",
    "Mystery"
]

response = requests.post(
    "http://localhost:8000/batch-recommend?k=3",
    json=queries
)

results = response.json()
for result in results["results"]:
    print(f"Query: {result['query']}")
    for rec in result["recommendations"]:
        print(f"  - {rec['title']}")
```

---

### 4. **Model Information**
**GET** `/model-info`

Get technical information about the loaded model.

**Response:**
```json
{
  "total_books": 6538,
  "embedding_dimension": 384,
  "embedding_model": "all-MiniLM-L6-v2",
  "serialization_format": "joblib"
}
```

**Example:**
```bash
curl http://localhost:8000/model-info
```

---

### 5. **API Root**
**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "title": "📚 LLM Book Recommender API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "recommend": "/recommend",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

---

### 6. **Interactive API Documentation**
**GET** `/docs`

Access Swagger UI for interactive API exploration and testing.
- URL: http://localhost:8000/docs

**GET** `/openapi.json`

Access OpenAPI schema for the API.

---

## Error Handling

The API returns appropriate HTTP status codes:

| Status | Error | Example |
|--------|-------|---------|
| 200 | Success | Valid recommendation returned |
| 400 | Bad Request | Invalid `k` value or empty query |
| 503 | Service Unavailable | Model not loaded |
| 500 | Internal Server Error | Unexpected error during processing |

**Example Error Response:**
```json
{
  "detail": "k must be between 1 and 20"
}
```

---

## Integration Examples

### JavaScript/Fetch
```javascript
// Single recommendation
fetch('http://localhost:8000/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Science fiction books",
    k: 5
  })
})
.then(res => res.json())
.then(data => {
  console.log(`Query: ${data.query}`);
  data.recommendations.forEach(book => {
    console.log(`- ${book.title} (${book.similarity_score.toFixed(2)})`);
  });
});
```

### cURL Examples

**Get single recommendation:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Adventure books", "k": 3}'
```

**Get health status:**
```bash
curl http://localhost:8000/health
```

**Get batch recommendations:**
```bash
curl -X POST "http://localhost:8000/batch-recommend?k=3" \
  -H "Content-Type: application/json" \
  -d '["Fantasy", "Mystery", "Science Fiction"]'
```

---

## Performance

- **Recommendation Generation**: ~100-200ms per query
- **Batch Processing**: ~50-150ms per query
- **Concurrent Requests**: Supported (async)
- **Model Loading**: ~2-3 seconds on startup

---

## Environment Variables

None required - all settings are automatically configured.

---

## Running Both Services

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```

Provides:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- Both share the same model and data

### Option 2: Run Separately
```bash
# Terminal 1: Streamlit
streamlit run app.py

# Terminal 2: FastAPI
python api.py
```

### Option 3: Docker Individual Containers
```bash
# Build image
docker build -t book-recommender .

# Run API
docker run -p 8000:8000 -v ./models:/app/models book-recommender python api.py

# Run Streamlit (in another terminal)
docker run -p 8501:8501 -v ./models:/app/models book-recommender streamlit run app.py
```

---

## Monitoring

Check logs for the API service:
```bash
# With Docker Compose
docker-compose logs -f api

# Running directly
# Logs appear in terminal
```

---

## CORS Support

The API supports CORS requests from any origin. Safe for cross-domain requests.

---

## Rate Limiting

Currently no rate limiting. Add if needed for production deployment.

---

## Future Enhancements

- [ ] User preferences and saved queries
- [ ] Caching of popular queries
- [ ] Query analytics and logging
- [ ] Authentication/API keys
- [ ] Rate limiting
- [ ] Batch export (CSV, JSON)
- [ ] Recommendation explanations
- [ ] A/B testing support

---

## Support

For issues or questions:
1. Check `/docs` endpoint for interactive documentation
2. Review error messages in API response
3. Check server logs for detailed error information

