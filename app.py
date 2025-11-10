"""Streamlit UI for Book Recommender System."""

import streamlit as st
from pathlib import Path
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.inference import RecommenderEngine
from recommender.train import MODEL_PKL_PATH

# Page configuration
st.set_page_config(
    page_title="📚 LLM Book Recommender",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { max-width: 1200px; }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    with st.spinner("Loading recommender model..."):
        try:
            st.session_state.engine = RecommenderEngine()
        except FileNotFoundError:
            st.error("❌ Model not found! Please run training first: `python -m recommender.train`")
            st.stop()

# Header
st.title("📚 LLM-Based Book Recommender System")
st.markdown("**Discover your next favorite book using AI-powered semantic search**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    num_recommendations = st.slider("Number of recommendations", 1, 20, 5)
    
    st.markdown("---")
    st.markdown("**About**")
    st.info(
        "This recommender system uses sentence transformers to generate semantic "
        "embeddings of book descriptions. Your query is compared to all books "
        "using cosine similarity to find the best matches."
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Find Books")
    query = st.text_area(
        "Enter your query or describe the book you're looking for:",
        placeholder="E.g., 'I love fantasy novels with strong female protagonists and magic systems'",
        height=100
    )

with col2:
    st.subheader("📊 Stats")
    total_books = st.session_state.engine.model.n_books
    st.metric("Books in Database", total_books)

# Search button
if st.button("🚀 Find Recommendations", use_container_width=True):
    if not query.strip():
        st.warning("⚠️ Please enter a query first!")
    else:
        with st.spinner("Searching for recommendations..."):
            try:
                recommendations = st.session_state.engine.recommend(query, k=num_recommendations)
                
                st.success(f"✅ Found {len(recommendations)} recommendations!")
                
                # Display recommendations
                st.subheader("📖 Top Recommendations")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            ### {i}. {rec['title']}
                            **Authors:** {rec['authors']}  
                            **Categories:** {rec['categories']}  
                            **Rating:** ⭐ {rec['rating']:.2f}/5.0
                            """)
                        
                        with col2:
                            st.metric("Similarity", f"{rec['similarity_score']:.1%}")
                        
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error during recommendation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 Powered by Sentence Transformers & Streamlit</p>
    <p style='font-size: 12px; color: #888;'>LLM Book Recommender System v1.0</p>
</div>
""", unsafe_allow_html=True)
