import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# We'll use sentence-transformers for local embeddings to avoid forcing an OpenAI embedding requirement for offline use.
try:
    from sentence_transformers import SentenceTransformer
    USE_TRANSFORMER = True
except Exception:
    SentenceTransformer = None
    USE_TRANSFORMER = False

from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'books.csv'))
EMBED_CACHE = os.path.join(BASE_DIR, 'embeddings.npy')
META_CACHE = os.path.join(BASE_DIR, 'meta.json')
VECTORIZER_CACHE = os.path.join(BASE_DIR, 'vectorizer.joblib')

MODEL_NAME = os.environ.get('SENTENCE_MODEL', 'all-MiniLM-L6-v2')

class BookRecommender:
    def __init__(self):
        # instantiate transformer model only if sentence-transformers imported successfully
        if USE_TRANSFORMER and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(MODEL_NAME)
            except Exception:
                # if model init fails, fall back to None and use TF-IDF
                self.model = None
        else:
            self.model = None
        self.books = None
        self.embeddings = None
        self.meta = None
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(DATA_PATH)
        # Keep relevant columns and fillna
        df = df[['isbn13','title','subtitle','authors','categories','description','published_year']]
        df = df.fillna('')
        # Create a text field to embed
        df['text'] = df.apply(lambda r: ' '.join([str(r.title), str(r.subtitle), str(r.authors), str(r.categories), str(r.description)]), axis=1)
        self.books = df

        if os.path.exists(EMBED_CACHE) and os.path.exists(META_CACHE):
            try:
                self.embeddings = np.load(EMBED_CACHE)
                with open(META_CACHE, 'r') as f:
                    self.meta = json.load(f)
                # basic sanity check
                if len(self.embeddings) != len(df):
                    print('Embedding cache length mismatch; rebuilding embeddings')
                    self._build_embeddings()
            except Exception as e:
                print('Failed to load caches, rebuilding embeddings:', e)
                self._build_embeddings()
        else:
            self._build_embeddings()

    def _build_embeddings(self):
        texts = self.books['text'].tolist()
        print(f'Computing embeddings for {len(texts)} books...')
        if USE_TRANSFORMER and self.model is not None:
            print(f' Using sentence-transformers model: {MODEL_NAME}')
            embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(EMBED_CACHE, embs)
            self.embeddings = embs
        else:
            # fallback: use TF-IDF vectorizer
            print(' Using TF-IDF fallback embeddings (no sentence-transformers available)')
            vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            embs = X.toarray()
            np.save(EMBED_CACHE, embs)
            joblib.dump(vectorizer, VECTORIZER_CACHE)
            self.embeddings = embs

        meta = self.books[['isbn13','title','authors','published_year']].to_dict(orient='records')
        with open(META_CACHE, 'w') as f:
            json.dump(meta, f)
        self.meta = meta

    def recommend(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # compute query embedding depending on available backend
        if USE_TRANSFORMER and self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True)
        else:
            # load vectorizer if exists
            if os.path.exists(VECTORIZER_CACHE):
                vectorizer = joblib.load(VECTORIZER_CACHE)
                q_emb = vectorizer.transform([query]).toarray()
            else:
                # unlikely path, fallback to simple bag-of-words via TfidfVectorizer on-the-fly
                tmp_vec = TfidfVectorizer(max_features=50000, stop_words='english')
                # fit on book texts then transform query
                tmp_vec.fit(self.books['text'].tolist())
                q_emb = tmp_vec.transform([query]).toarray()
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            row = self.books.iloc[i]
            results.append({
                'isbn13': row['isbn13'],
                'title': row['title'],
                'subtitle': row['subtitle'],
                'authors': row['authors'],
                'categories': row['categories'],
                'description': row['description'],
                'published_year': row['published_year'],
                'score': float(sims[i])
            })
        return results

    def explain_with_ollama(self, recs: List[Dict[str, Any]], query: str, model: str = None, timeout: int = 30, auto_pull: bool = True, structured: bool = False, short: bool = True) -> List[Dict[str, Any]]:
        """Use local Ollama (CLI) to generate brief explanations for each recommendation.

        This method calls the `ollama` CLI installed on the host. It builds a small prompt
        combining the user query and the book metadata and calls `ollama generate`.

        Args:
            recs: list of recommendation dicts (as returned by recommend)
            query: original user query
            model: name of the local Ollama model (if None, read OLLAMA_MODEL env var or default 'llama2')
            timeout: seconds to wait for model generation

        Returns:
            The same list of recs with an added 'explanation' field per item when generation succeeds.
        """
        # Prefer explicit model param; otherwise try environment; otherwise auto-detect a local model
        requested = (model or os.environ.get('OLLAMA_MODEL') or '').strip()

        local_models = self._get_local_ollama_models()
        chosen_model = None
        if requested:
            # try to find exact or prefix match in local models
            for m in local_models:
                if m == requested or m.startswith(requested) or requested.startswith(m):
                    chosen_model = m
                    break
        if not chosen_model:
            # pick a sensible default from installed models
            chosen_model = self._select_local_ollama_model(local_models)

        if not chosen_model:
            # nothing installed locally
            if requested and auto_pull:
                ok, status = self._ensure_ollama_model(requested)
                if not ok:
                    msg = '(ollama CLI not found or failed to pull model: ' + str(status) + ')'
                    for r in recs:
                        r['explanation'] = msg
                    return recs
                chosen_model = requested
            else:
                msg = '(no local ollama models found; install or enable auto-pull to fetch a model)'
                for r in recs:
                    r['explanation'] = msg
                return recs

        model = chosen_model
        for r in recs:
            # Build a focused prompt. If structured=True request a strict JSON object.
            base = (
                f"User query: {query}\n"
                f"Book title: {r.get('title')}\n"
                f"Authors: {r.get('authors')}\n"
                f"Categories: {r.get('categories')}\n"
                f"Description: {r.get('description')}\n\n"
            )
            if structured:
                instruction = (
                    "You are an assistant that returns a single JSON object explaining why the book matches the user's query. "
                    "Return exactly one valid JSON object and nothing else. The JSON must have these keys: "
                    "'reason' (a short 1-2 sentence explanation), 'themes' (an array of 2-5 short theme keywords), "
                    "and optionally 'confidence' (a number from 0 to 1). Do not include additional commentary or markdown."
                )
                prompt = base + instruction
            else:
                if short:
                    # very tight short prompt for precision
                    prompt = base + "In one concise sentence, explain precisely why this book matches the user's request. Use plain language, avoid hedging and filler."
                else:
                    prompt = base + "In 2-3 concise sentences, explain precisely why this book is a good match for the user's query. Be direct and avoid filler."
            try:
                # Prefer passing prompt via stdin to avoid shell quoting issues
                # Use `ollama run <model>` to execute the model. The model will read prompt from stdin.
                cmd = ["ollama", "run", model]
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0:
                    explanation = (proc.stdout or '').strip()
                    if not explanation and proc.stderr:
                        explanation = proc.stderr.strip()
                    if len(explanation) > 20000:
                        explanation = explanation[:20000] + '...'
                    r['explanation'] = explanation or '(no explanation produced)'
                    # If structured output was requested, try to parse JSON from response
                    if structured:
                        parsed = None
                        try:
                            # find the first JSON object in the text
                            s = explanation
                            start = s.find('{')
                            end = s.rfind('}')
                            if start != -1 and end != -1 and end > start:
                                candidate = s[start:end+1]
                                parsed = json.loads(candidate)
                        except Exception:
                            parsed = None
                        r['explanation_json'] = parsed
                else:
                    r['explanation'] = f"(failed to generate explanation; exit {proc.returncode}: {proc.stderr.strip()})"
            except FileNotFoundError:
                r['explanation'] = '(ollama CLI not found; install and ensure `ollama` is on PATH)'
            except subprocess.TimeoutExpired:
                r['explanation'] = '(ollama generation timed out)'
            except Exception as e:
                # catch-all to avoid breaking the recommender
                r['explanation'] = f'(error running ollama: {e})'
        return recs

    def _ensure_ollama_model(self, model: str, pull_timeout: int = 600):
        """Check if an Ollama model exists locally; if not, attempt to pull it.

        Returns (True, 'present'|'pulled') on success. On failure returns (False, reason).
        Reasons: 'ollama_not_found', 'pull_failed: stderr', 'pull_timed_out', or other messages.
        """
        try:
            # List local models
            proc = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=30)
            out = (proc.stdout or '')
            local = self._parse_ollama_list_output(out)
            if model in local:
                return True, 'present'
            # attempt to pull the model
            pull_proc = subprocess.run(['ollama', 'pull', model], capture_output=True, text=True, timeout=pull_timeout)
            # combine stdout/stderr and sanitize ANSI control sequences
            raw_out = (pull_proc.stdout or '') + '\n' + (pull_proc.stderr or '')
            clean_out = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', raw_out)
            clean_out = clean_out.strip()
            if pull_proc.returncode == 0:
                return True, 'pulled'
            else:
                # detect common known failure: Ollama CLI is outdated and cannot pull model manifest
                low = clean_out.lower()
                if 'requires a newer version' in low or 'requires a newer version of ollama' in low or 'pull model manifest: 412' in low:
                    return False, f'ollama_outdated: {clean_out}'
                return False, f'pull_failed: {clean_out}'
        except FileNotFoundError:
            return False, 'ollama_not_found'
        except subprocess.TimeoutExpired:
            return False, 'pull_timed_out'
        except Exception as e:
            return False, f'error: {e}'

    def _get_local_ollama_models(self):
        """Return a list of model names installed in local Ollama (strings)."""
        try:
            proc = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=30)
            out = proc.stdout or ''
            return self._parse_ollama_list_output(out)
        except Exception:
            return []

    def _parse_ollama_list_output(self, out: str):
        """Parse `ollama list` plain text output and return list of names (first column).

        Expects output like:
        NAME    ID   SIZE   MODIFIED
        llama3.2:latest  ...
        """
        lines = [l.strip() for l in (out or '').splitlines() if l.strip()]
        models = []
        if not lines:
            return models
        # skip header line if it looks like one
        start = 0
        if 'NAME' in lines[0].upper() and 'ID' in lines[0].upper():
            start = 1
        for line in lines[start:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models

    def _select_local_ollama_model(self, local_models: list):
        """Select best model from local_models using a priority list."""
        if not local_models:
            return None
        priority = [
            'llama3.2:latest',
            'llama3.2',
            'llava:7b',
            'llava',
            'Dragon:latest',
            'deepseek-r1:1.5b',
            'nomic-embed-text:latest',
        ]
        for p in priority:
            for m in local_models:
                if m == p or m.startswith(p) or p.startswith(m):
                    return m
        # otherwise return first available
        return local_models[0]


if __name__ == '__main__':
    r = BookRecommender()
    while True:
        q = input('Query (or q to quit): ')
        if q.lower() in ('q','quit','exit'):
            break
        recs = r.recommend(q, k=5)
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} — {rec['authors']} ({rec['published_year']}) score={rec['score']:.4f}")
