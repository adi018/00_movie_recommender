LLM Book Recommender (local)

This small project provides a local book recommender using local embeddings (sentence-transformers when available, with a TF-IDF fallback) and a Streamlit UI.

Prerequisites
- Docker (optional) or Python 3.12+

Quick start (local)

1. Create and activate a virtualenv

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies

   pip install -r app/requirements.txt

3. Run Streamlit

   streamlit run app/streamlit_app.py

Quick start (Docker)

1. Build image

   docker build -t book-recommender .

2. Run (recommended)

   docker compose up --build

Or run directly:

   docker run -p 8501:8501 book-recommender

Notes
- Embeddings are cached to `app/embeddings.npy` and `app/meta.json` after first run.
 - The app uses `all-MiniLM-L6-v2` by default; override with environment variable `SENTENCE_MODEL`.
 - This project doesn't call OpenAI by default. Local LLM explanations are supported via the `ollama` CLI. If you have `ollama` installed and local models available, enable the "Ollama explanations (local)" checkbox in the Streamlit UI. The app calls the `ollama` CLI, so make sure `ollama` is on your PATH.
 - Automatic pulling of models during generation has been disabled. To fetch models you must either:
    - Use the Pull button next to the manual model input in the sidebar, or
    - Use the "Model actions" expander (appears only when pulling may be required), or
    - Pull via the host: `ollama pull <model>`

Files
- `app/recommender.py` - core recommender that computes embeddings and searches
- `app/streamlit_app.py` - Streamlit UI
- `Dockerfile` and `docker-compose.yml` - Docker run configs
- `data/books.csv` - dataset (already present)

