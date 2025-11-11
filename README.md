# 📚 LLM Book Recommender System - Complete Guide

A production-ready **AI-powered boo6. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   
   The app will open at `http://localhost:8501`

7. **Run the FastAPI server** (Optional, in another terminal)
   ```bash
   python api.py
   ```
   
   API will be available at `http://localhost:8000`
   Documentation at `http://localhost:8000/docs`

## 📡 FastAPI REST API

The system now includes a professional FastAPI server for programmatic access!

### Quick API Examples

**Get Recommendations:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Science fiction with space exploration", "k": 5}'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Batch Recommendations:**
```bash
curl -X POST "http://localhost:8000/batch-recommend?k=3" \
  -H "Content-Type: application/json" \
  -d '["Fantasy with dragons", "Mystery thriller"]'
```

### API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /recommend` - Single recommendation
- `POST /batch-recommend` - Multiple recommendations
- `GET /model-info` - Model specifications
- `GET /docs` - Swagger UI (interactive testing)

**See API_DOCUMENTATION.md for complete API reference**

## 🐳 Docker Deployment

### Using Docker Compose (Recommended) - Both Streamlit & API

1. **Build and run**
   ```bash
   docker-compose up --build
   ```

2. **Access services**
   - Streamlit UI: http://localhost:8501
   - FastAPI: http://localhost:8000
   - API Docs: http://localhost:8000/docsn system** using semantic embeddings and transformer models. Access your recommender through:
- 🎨 **Streamlit UI** - Beautiful interactive interface
- 📡 **FastAPI REST API** - Programmatic access for developers
- 🐳 **Docker** - Container deployment

**Tech Stack:**
- Sentence Transformers (all-MiniLM-L6-v2)
- 6,538 books with 384-dimensional embeddings
- Cosine similarity ranking
- Streamlit + FastAPI
- Docker containerization

## 🎯 Features

- **Semantic Understanding**: Uses pre-trained sentence transformers to understand book content and user queries at a semantic level
- **Fast Similarity Search**: Implements cosine similarity for efficient recommendation computation
- **Beautiful UI**: Streamlit-powered interactive web interface
- **Dockerized Deployment**: Easy containerization for production deployment
- **Scalable Architecture**: Can handle thousands of books efficiently

## 📁 Project Structure

```
.
├── data/
│   ├── books.csv                 # Original dataset
│   └── books_processed.csv       # Cleaned and processed data
├── recommender/
│   ├── __init__.py
│   ├── preprocess.py             # Data cleaning and preprocessing
│   ├── embeddings.py             # Embedding generation
│   ├── train.py                  # Model training pipeline
│   └── inference.py              # Inference engine
├── models/
│   ├── embeddings.npy            # Stored embeddings
│   ├── metadata.json             # Book metadata
│   └── recommender_model.pkl     # Trained model
├── app.py                        # Streamlit UI
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- Docker (optional, for containerization)

### Local Installation

1. **Clone the repository**
   ```bash
   cd /path/to/project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data**
   ```bash
   python -m recommender.preprocess
   ```

4. **Train the model** (first time only)
   ```bash
   python -m recommender.train
   ```
   
   This will:
   - Load and process the raw book data
   - Generate semantic embeddings using Sentence Transformers
   - Save the trained model and embeddings to `models/`

5. **Test the inference engine**
   ```bash
   python -m recommender.inference
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   
   The app will open at `http://localhost:8501`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and run**
   ```bash
   docker-compose up --build
   ```

2. **Access the app**
   - Open your browser and go to `http://localhost:8501`

3. **Stop the service**
   ```bash
   docker-compose down
   ```

### Using Docker CLI

1. **Build the image**
   ```bash
   docker build -t book-recommender .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     book-recommender
   ```

## 📖 How It Works

### 1. Data Preprocessing (`recommender/preprocess.py`)

- Loads the raw books.csv dataset
- Cleans and validates data
- Combines relevant fields (title, authors, categories, description) into a single text feature
- Removes duplicate/invalid entries
- Saves processed data to `books_processed.csv`

### 2. Embeddings Generation (`recommender/embeddings.py`)

- Uses Sentence Transformers (all-MiniLM-L6-v2 by default)
- Converts book descriptions into 384-dimensional semantic embeddings
- Enables semantic understanding of content

### 3. Model Training (`recommender/train.py`)

- Loads preprocessed data
- Generates embeddings for all books
- Creates a `BookRecommenderModel` instance
- Saves:
  - **embeddings.npy**: Raw embedding vectors
  - **metadata.json**: Book information (title, authors, rating, etc.)
  - **recommender_model.pkl**: Trained model for inference

### 4. Inference (`recommender/inference.py`)

- Loads saved model and embeddings
- Encodes user query using the same embedding model
- Computes cosine similarity between query and all books
- Returns top-k most similar books with scores

### 5. Streamlit UI (`app.py`)

- Interactive query interface
- Adjustable number of recommendations (1-20)
- Displays book details: title, authors, categories, rating, similarity score
- Real-time search results

## 🛠️ Configuration

### Model Selection

To use a different embedding model, modify the `embedding_model` parameter:

```python
# In train.py
model = train_model(embedding_model="distiluse-base-multilingual-cased-v2")

# In app.py (after retraining)
engine = RecommenderEngine(embedding_model="your-model-name")
```

Available models: See [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)

### Number of Recommendations

In the Streamlit UI sidebar, adjust the slider for 1-20 recommendations.

## 📊 Performance Metrics

- **Dataset Size**: 6,810 books
- **Embedding Dimension**: 384 (MiniLM model)
- **Similarity Computation**: O(n) where n = number of books
- **Query Response Time**: ~100-500ms

## 🔧 Troubleshooting

### Model not found error
```bash
# Retrain the model
python -m recommender.train
```

### Out of memory during training
- Reduce batch size or use a smaller embedding model
- Process data in chunks

### Streamlit cache issues
```bash
# Clear Streamlit cache
streamlit cache clear
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web UI framework |
| sentence-transformers | 2.2.2 | Semantic embeddings |
| scikit-learn | 1.3.2 | Similarity computations |
| pandas | 2.1.3 | Data processing |
| numpy | 1.24.3 | Numerical computations |

## 🎓 Implementation Details

### Similarity Metric

Cosine similarity is used to compare embeddings:
```
similarity = (A · B) / (||A|| × ||B||)
```

### Embedding Model

- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Training Data**: SNLI + Multi-Genre NLI
- **Inference Time**: ~5ms per text

## 🚀 Future Enhancements

- [ ] Hybrid recommendation (content + collaborative)
- [ ] User ratings integration
- [ ] Real-time feedback loop for model improvement
- [ ] Advanced filtering (genre, language, year)
- [ ] User history and personalization
- [ ] API endpoint for programmatic access
- [ ] Multi-language support
- [ ] Explanation generation using LLMs

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ for book lovers and AI enthusiasts.

## 📞 Support

For issues and questions, please open a GitHub issue or contact the maintainers.

---

**Happy reading! 📚**
